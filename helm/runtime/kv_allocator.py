import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

class KVAllocator:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, page_size: int = 64, dtype=torch.bfloat16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        
        self.gpu_free_pool = []
        self.cpu_free_pool = []
        
        self.gpu_total_pages = 0
        self.cpu_total_pages = 0
        
        self._next_page_id = 0

    def _create_page(self, device: torch.device):
        from helm.runtime.kv_cache import KVPage # Import inside to avoid circular dependency
        
        # Batch size 1, num_kv_heads, page_size, head_dim
        shape = (1, self.num_kv_heads, self.page_size, self.head_dim)
        k_tensor = torch.empty(shape, dtype=self.dtype, device=device)
        v_tensor = torch.empty(shape, dtype=self.dtype, device=device)
        
        page = KVPage(
            page_id=self._next_page_id,
            layer_id=-1,
            start_token=0,
            used_tokens=0,
            capacity_tokens=self.page_size,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            device=device,
            state='FREE'
        )
        self._next_page_id += 1
        
        if device.type == 'cuda':
            self.gpu_total_pages += 1
        else:
            self.cpu_total_pages += 1
            
        return page

    def reserve_batch(self, batch_size: int, device: torch.device):
        """Allocate a batch of free pages on the specified device."""
        for _ in range(batch_size):
            page = self._create_page(device)
            if device.type == 'cuda':
                self.gpu_free_pool.append(page)
            else:
                self.cpu_free_pool.append(page)

    def allocate(self, device: torch.device):
        """Get a free page from the pool, allocating if necessary."""
        if device.type == 'cuda':
            if not self.gpu_free_pool:
                self.reserve_batch(8, device) # reserve small batch on GPU
            page = self.gpu_free_pool.pop()
            page.state = 'GPU'
        else:
            if not self.cpu_free_pool:
                self.reserve_batch(16, device) # reserve larger batch on CPU
            page = self.cpu_free_pool.pop()
            page.state = 'CPU'
            
        # Defensively reset metadata
        page.layer_id = -1
        page.start_token = 0
        page.used_tokens = 0
        page.device = device
            
        return page

    def free(self, page):
        """
        Recycle a page back to the free pool.
        Note: The underlying tensors are NOT zeroed out for performance.
        Unused token slots contain undefined data and should never be read.
        """
        page.state = 'FREE'
        page.layer_id = -1
        page.start_token = 0
        page.used_tokens = 0
        if page.device.type == 'cuda':
            self.gpu_free_pool.append(page)
        else:
            self.cpu_free_pool.append(page)

    def move_page(self, page, target_device: torch.device):
        """
        Move a page between CPU and GPU.
        Allocates a new page from the target pool, copies valid tokens,
        recycles the old page, and returns the new page.
        """
        if page.device == target_device:
            return page
            
        new_page = self.allocate(target_device)
        
        used = page.used_tokens
        if used > 0:
            # Synchronous copies: non_blocking=True would cause a race —
            # the subsequent copy_() reads the destination CPU tensor before
            # the GPU→CPU DMA transfer has completed, corrupting V data.
            new_page.k_tensor[:, :, :used, :].copy_(
                page.k_tensor[:, :, :used, :]
            )
            new_page.v_tensor[:, :, :used, :].copy_(
                page.v_tensor[:, :, :used, :]
            )
            
        new_page.layer_id = page.layer_id
        new_page.start_token = page.start_token
        new_page.used_tokens = used
        new_page.device = target_device

        # GPU→CPU eviction: release VRAM immediately.  Recycling back to the
        # pool (via free()) keeps tensors alive in VRAM even though no KV data
        # lives there, causing the free pool to grow unboundedly — OOM at long
        # decode lengths.
        #
        # We replace the GPU tensors with zero-element CPU placeholders rather
        # than using `del`.  Both release the VRAM reference, but the placeholder
        # approach preserves the attribute so that any stale reference to this
        # page (e.g. an in-flight pages list) raises a clear error rather than
        # an opaque AttributeError.  The 'DEAD' state gates the page out of
        # every active processing path (eviction scan, streaming attention,
        # clear()).
        if page.device.type == 'cuda' and target_device.type != 'cuda':
            placeholder_dtype = page.k_tensor.dtype
            page.k_tensor = torch.empty(0, dtype=placeholder_dtype)
            page.v_tensor = torch.empty(0, dtype=placeholder_dtype)
            page.state = 'DEAD'
            self.gpu_total_pages -= 1
        else:
            self.free(page)

        return new_page

    def report_usage(self) -> dict:
        bytes_per_page = 2 * (1 * self.num_kv_heads * self.page_size * self.head_dim * torch.finfo(self.dtype).bits // 8)
        
        gpu_used_pages = self.gpu_total_pages - len(self.gpu_free_pool)
        cpu_used_pages = self.cpu_total_pages - len(self.cpu_free_pool)
        
        return {
            "gpu_pool_size": self.gpu_total_pages,
            "cpu_pool_size": self.cpu_total_pages,
            "active_gpu_pages": gpu_used_pages,
            "active_cpu_pages": cpu_used_pages,
            "gpu_free_pages": len(self.gpu_free_pool),
            "cpu_free_pages": len(self.cpu_free_pool),
            "gpu_reserved_bytes": self.gpu_total_pages * bytes_per_page,
            "cpu_reserved_bytes": self.cpu_total_pages * bytes_per_page,
            "gpu_used_bytes": gpu_used_pages * bytes_per_page,
            "cpu_used_bytes": cpu_used_pages * bytes_per_page,
        }
