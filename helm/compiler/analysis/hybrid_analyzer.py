import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx

from ..IR.graph import HelmGraph, HelmNode

logger = logging.getLogger(__name__)

_DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
}


@dataclass(frozen=True)
class HybridAnalysisSummary:
    num_nodes: int
    num_nodes_with_shapes: int
    total_activation_bytes: int
    total_param_bytes: int
    total_flops_prefill: int
    total_flops_decode: int
    total_kv_bytes_per_token: int


class HybridAnalyzer:
    """
    Pass: Hybrid Cost Analysis

    Responsibilities:
    - run the FX GraphModule on example inputs to capture per-node outputs
    - annotate node shapes and activation sizes
    - annotate per-module parameter bytes
    - estimate block-level FLOPs for prefill/decode
    - estimate KV-cache growth per token

    Notes:
    - This is intentionally a planner-oriented estimator, not an exact profiler.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        helm_graph: HelmGraph,
        model: torch.nn.Module,
    ):
        self.gm = gm
        self.helm_graph = helm_graph
        self.model = model

        self.config = self._build_model_config()

    # ============================================================
    # Public API
    # ============================================================

    def run(self, example_inputs: Any) -> HybridAnalysisSummary:
        self._reset_annotations()
        node_to_output = self._propagate_shapes(example_inputs)
        self._annotate_node_shapes_and_activations(node_to_output)
        self._annotate_module_costs()
        summary = self._build_summary()
        return summary

    def export_summary(self, path: str) -> None:
        data = []
        for node in self.helm_graph.nodes:
            data.append(
                {
                    "node_id": node.id,
                    "node_name": node.name,
                    "fx_node_name": node.fx_node_name,
                    "op_type": node.op_type,
                    "target": str(node.target),
                    "module_path": node.module_path,
                    "layer_id": node.layer_id,
                    "input_shapes": node.input_shapes,
                    "output_shapes": node.output_shapes,
                    "activation_bytes": node.activation_bytes,
                    "param_bytes": node.param_bytes,
                    "flops_prefill": node.flops_prefill,
                    "flops_decode": node.flops_decode,
                    "kv_bytes_per_token": node.kv_bytes_per_token,
                    "output_dtype": (
                        str(node.output_dtype)
                        if node.output_dtype is not None
                        else None
                    ),
                }
            )

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            raise RuntimeError(
                f"HybridAnalyzer: failed to write summary to '{path}'."
            ) from exc

    # ============================================================
    # Setup / Reset
    # ============================================================

    def _reset_annotations(self) -> None:
        for node in self.helm_graph.nodes:
            node.input_shapes = []
            node.output_shapes = []
            node.flops_prefill = 0
            node.flops_decode = 0
            node.activation_bytes = 0
            node.param_bytes = 0
            node.bytes_read = 0
            node.bytes_written = 0
            node.kv_bytes_per_token = 0
            node.output_dtype = None
            node.batch_size = 1
            node.sequence_length = 1

    def _build_model_config(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            logger.warning(
                "HybridAnalyzer: model has no .config attribute — "
                "block-level FLOPs and KV estimates will be unavailable. "
                "Only shape and activation annotations will be populated."
            )
            dtype = self._infer_model_dtype()
            dtype_size = _DTYPE_BYTES.get(dtype)
            if dtype_size is None:
                raise ValueError(
                    f"HybridAnalyzer: unsupported dtype '{dtype}'. Add it to _DTYPE_BYTES."
                )
            return {
                "hidden_size": 0,
                "num_attention_heads": 0,
                "num_key_value_heads": 0,
                "intermediate_size": 0,
                "vocab_size": 0,
                "dtype_size": dtype_size,
            }

        # transformers >= 4.56 exposes `config.dtype`; reading the legacy
        # `torch_dtype` attribute emits a deprecation warning, so only fall
        # back to it on configs that store it as a plain attribute.
        dtype = getattr(cfg, "dtype", None)
        if dtype is None:
            dtype = getattr(cfg, "__dict__", {}).get("torch_dtype")
        if dtype is None:
            dtype = self._infer_model_dtype()

        dtype_size = _DTYPE_BYTES.get(dtype)
        if dtype_size is None:
            raise ValueError(
                f"HybridAnalyzer: unsupported dtype '{dtype}'. Add it to _DTYPE_BYTES."
            )

        hidden_size = getattr(cfg, "hidden_size", 0)
        intermediate_size = getattr(cfg, "intermediate_size", 0)
        num_heads = getattr(cfg, "num_attention_heads", 0)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        vocab_size = getattr(cfg, "vocab_size", 0)
        return {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "dtype_size": dtype_size,
        }

    def _infer_model_dtype(self) -> torch.dtype:
        try:
            first_param = next(self.model.parameters())
        except StopIteration:
            raise RuntimeError(
                "HybridAnalyzer: model has no parameters — cannot infer dtype. "
                "Pass a model with at least one parameter or set config.dtype."
            )
        return first_param.dtype

    # ============================================================
    # Shape Propagation
    # ============================================================

    def _propagate_shapes(self, example_inputs: Any) -> Dict[fx.Node, Any]:
        """
        Executes the GraphModule via an FX interpreter and captures every node output.
        """

        node_to_output: Dict[fx.Node, Any] = {}

        def _submodule_device(submod: torch.nn.Module) -> Optional[torch.device]:
            param = next(submod.parameters(recurse=True), None)
            if param is not None:
                return param.device
            buf = next(submod.buffers(recurse=True), None)
            return buf.device if buf is not None else None

        def _align_to_device(value: Any, device: torch.device) -> Any:
            # Submodule outputs in transformer FX graphs are often containers
            # (rotary returns a (cos, sin) tuple; some layers return dicts).
            # Recurse so device alignment reaches every leaf tensor.
            if torch.is_tensor(value):
                return value.to(device) if value.device != device else value
            if isinstance(value, tuple):
                return tuple(_align_to_device(v, device) for v in value)
            if isinstance(value, list):
                return [_align_to_device(v, device) for v in value]
            if isinstance(value, dict):
                return {k: _align_to_device(v, device) for k, v in value.items()}
            return value

        class ShapeInterpreter(fx.Interpreter):
            def run_node(self_, n: fx.Node):
                result = super().run_node(n)
                node_to_output[n] = result
                return result

            def call_module(self_, target, args, kwargs):
                # Multi-device models (e.g. Accelerate-dispatched with offload)
                # have submodules on different devices and rely on pre-forward
                # hooks to shuttle activations. paper_bench removes those hooks
                # before tracing, so during shape propagation we replicate the
                # behavior: move tensor args to the submodule's own device.
                submod = self_.fetch_attr(target)
                target_dev = _submodule_device(submod)
                if target_dev is not None:
                    args = tuple(_align_to_device(a, target_dev) for a in args)
                    kwargs = {k: _align_to_device(v, target_dev) for k, v in kwargs.items()}
                return submod(*args, **kwargs)

        interp = ShapeInterpreter(self.gm)

        try:
            with torch.no_grad():
                if isinstance(example_inputs, dict):
                    interp.run(**example_inputs)
                elif isinstance(example_inputs, (tuple, list)):
                    interp.run(*example_inputs)
                else:
                    interp.run(example_inputs)
        except Exception as exc:
            raise RuntimeError(
                f"HybridAnalyzer: shape propagation failed on inputs "
                f"(type={type(example_inputs).__name__}). "
                f"Check that example_inputs match the model signature."
            ) from exc

        return node_to_output

    def _annotate_node_shapes_and_activations(
        self, node_to_output: Dict[fx.Node, Any]
    ) -> None:
        for helm_node in self.helm_graph.nodes:
            fx_node = helm_node.fx_node
            if fx_node not in node_to_output:
                raise RuntimeError(
                    f"HybridAnalyzer: no output captured for node {helm_node.name!r} "
                    f"(op={fx_node.op!r}). Shape propagation may not have run, or "
                    f"the graph was modified after tracing."
                )

            out = node_to_output[fx_node]

            helm_node.output_shapes = self._extract_shapes(out)
            helm_node.activation_bytes = self._activation_size_bytes(out)
            helm_node.output_dtype = self._extract_dtype(out)

            for input_fx_node in fx_node.all_input_nodes:
                if input_fx_node in node_to_output:
                    inp = node_to_output[input_fx_node]
                    helm_node.input_shapes.extend(self._extract_shapes(inp))

            batch_size, sequence_length = self._infer_batch_and_sequence(out)
            helm_node.batch_size = batch_size
            helm_node.sequence_length = sequence_length

            primary_shape = (
                helm_node.output_shapes[0] if helm_node.output_shapes else []
            )
            for edge in helm_node.out_edges:
                edge.tensor_shape = list(primary_shape)
                edge.tensor_bytes = helm_node.activation_bytes

    # ============================================================
    # Cost Annotation
    # ============================================================

    def _annotate_module_costs(self) -> None:
        seen_param_ids: Set[int] = set()
        for helm_node in self.helm_graph.nodes:
            fx_node = helm_node.fx_node

            if fx_node.op == "call_module":
                try:
                    submodule = self.gm.get_submodule(fx_node.target)
                except AttributeError as exc:
                    raise RuntimeError(
                        f"HybridAnalyzer: submodule '{fx_node.target}' not found in "
                        f"GraphModule. The graph may have been modified after tracing."
                    ) from exc
                # Count each underlying parameter tensor once across the full graph.
                # This keeps model-level memory accounting honest for tied weights.
                helm_node.param_bytes = self._module_param_bytes(
                    submodule, seen_param_ids
                )

                # Conservative default I/O traffic model
                helm_node.bytes_written = helm_node.activation_bytes
                helm_node.bytes_read = self._input_bytes(helm_node)

                if self._is_transformer_block(fx_node.target):
                    b = helm_node.batch_size
                    s = max(helm_node.sequence_length, 1)

                    stats = self._estimate_block_costs(batch_size=b, seq_len=s)
                    helm_node.flops_prefill = stats["flops_prefill"]
                    helm_node.flops_decode = stats["flops_decode"]
                    helm_node.kv_bytes_per_token = stats["kv_bytes_per_token"]

                elif self._is_lm_head(fx_node.target):
                    b = helm_node.batch_size
                    s = max(helm_node.sequence_length, 1)
                    h = self.config["hidden_size"]
                    v = self.config["vocab_size"]

                    if h > 0 and v > 0:
                        helm_node.flops_prefill = int(2 * b * s * h * v)
                        helm_node.flops_decode = int(2 * b * 1 * h * v)

            else:
                # non-module nodes still carry activation metadata
                helm_node.bytes_written = helm_node.activation_bytes
                helm_node.bytes_read = self._input_bytes(helm_node)

    def _estimate_block_costs(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """
        Coarse transformer-block estimator.

        Assumptions:
        - decoder-only transformer block
        - fused attention treated analytically
        - decode uses one query token attending over `seq_len` context
        """

        b = max(batch_size, 1)
        s = max(seq_len, 1)
        h = self.config["hidden_size"]
        i = self.config["intermediate_size"]

        if h <= 0 or i <= 0:
            return {
                "flops_prefill": 0,
                "flops_decode": 0,
                "kv_bytes_per_token": 0,
            }

        kv_heads = max(self.config["num_key_value_heads"], 1)
        attn_heads = max(self.config["num_attention_heads"], 1)
        head_dim = h // attn_heads
        kv_dim = kv_heads * head_dim  # total KV projection output width (< h for GQA)
        dtype_size = self.config["dtype_size"]

        # Prefill:
        # 4 projections (Q, K, V, O): Q+O cost B*S*H*H MACs each, K+V cost B*S*H*kv_dim MACs each
        # Total projections MACs = 2*B*S*H*(H+kv_dim); ×2 for FLOPs/MAC = 4×
        attn_linear = 4 * b * s * h * (h + kv_dim)

        # QK^T + AV: each costs B*S^2*H MACs; ×2 for FLOPs/MAC = 4×
        attn_scores = 4 * b * s * s * h

        # MLP: gate/up/down (3 matmuls × 2 FLOPs per MAC = 6×)
        mlp = 6 * b * s * h * i

        flops_prefill = int(attn_linear + attn_scores + mlp)

        # Decode: one new token attends over s context
        attn_linear_decode = 4 * b * 1 * h * (h + kv_dim)
        attn_scores_decode = 4 * b * 1 * s * h
        mlp_decode = 6 * b * 1 * h * i

        flops_decode = int(attn_linear_decode + attn_scores_decode + mlp_decode)

        kv_bytes_per_token = int(2 * kv_heads * head_dim * dtype_size)

        return {
            "flops_prefill": flops_prefill,
            "flops_decode": flops_decode,
            "kv_bytes_per_token": kv_bytes_per_token,
        }

    # ============================================================
    # Summary
    # ============================================================

    def _build_summary(self) -> HybridAnalysisSummary:
        return HybridAnalysisSummary(
            num_nodes=len(self.helm_graph.nodes),
            num_nodes_with_shapes=sum(
                1 for n in self.helm_graph.nodes if n.output_shapes
            ),
            total_activation_bytes=sum(
                n.activation_bytes for n in self.helm_graph.nodes
            ),
            total_param_bytes=sum(n.param_bytes for n in self.helm_graph.nodes),
            total_flops_prefill=sum(n.flops_prefill for n in self.helm_graph.nodes),
            total_flops_decode=sum(n.flops_decode for n in self.helm_graph.nodes),
            # Sum of per-token KV bytes across all transformer layers —
            # i.e., the total KV cache growth per new token across the full model.
            total_kv_bytes_per_token=sum(
                n.kv_bytes_per_token for n in self.helm_graph.nodes
            ),
        )

    # ============================================================
    # Helpers
    # ============================================================

    def _module_param_bytes(
        self, module: torch.nn.Module, seen_param_ids: Optional[Set[int]] = None
    ) -> int:
        total = 0
        for param in module.parameters():
            param_id = id(param)
            if seen_param_ids is not None and param_id in seen_param_ids:
                continue
            total += param.numel() * param.element_size()
            if seen_param_ids is not None:
                seen_param_ids.add(param_id)
        return total

    def _is_transformer_block(self, target: str) -> bool:
        # Match only whole indexed block paths, including root-level names like
        # "layers.0" and nested paths like "model.layers.0" or "encoder.block.0".
        return bool(re.search(r"(?:^|\.)(layers|blocks|h|block|layer)\.\d+$", target))

    def _is_lm_head(self, target: str) -> bool:
        # Match the same output-head aliases the importer understands, while still
        # requiring the final path segment to be a head module.
        return bool(
            re.search(
                r"(?:^|\.)(lm_head|output_head|output_layer|embed_out|logits|cls)$",
                target,
            )
        )

    def _activation_size_bytes(self, obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(self._activation_size_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(self._activation_size_bytes(v) for v in obj.values())
        return 0

    def _extract_shapes(self, obj: Any) -> List[List[int]]:
        """
        Returns a flat list of tensor shapes found in the object.
        """
        shapes: List[List[int]] = []

        def visit(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                shapes.append(list(x.shape))
            elif isinstance(x, (list, tuple)):
                for item in x:
                    visit(item)
            elif isinstance(x, dict):
                for v in x.values():
                    visit(v)

        visit(obj)
        return shapes

    def _extract_dtype(self, obj: Any) -> Optional[torch.dtype]:
        if isinstance(obj, torch.Tensor):
            return obj.dtype
        if isinstance(obj, (list, tuple)):
            for item in obj:
                dt = self._extract_dtype(item)
                if dt is not None:
                    return dt
        if isinstance(obj, dict):
            for v in obj.values():
                dt = self._extract_dtype(v)
                if dt is not None:
                    return dt
        return None

    def _infer_batch_and_sequence(self, obj: Any) -> Tuple[int, int]:
        shapes = self._extract_shapes(obj)
        if not shapes:
            return 1, 1

        # Rank-3 tensors are [B, S, H] — the standard transformer hidden state.
        for shape in shapes:
            if len(shape) == 3:
                return int(shape[0]), int(shape[1])

        # Rank-4 tensors are [B, num_heads, S, head_dim] — attention outputs.
        # shape[1] is num_heads, NOT seq_len; use shape[2] for sequence length.
        for shape in shapes:
            if len(shape) == 4:
                return int(shape[0]), int(shape[2])

        # Rank-2 tensors [B, H] — decode step (single token), seq_len=1.
        for shape in shapes:
            if len(shape) == 2:
                return int(shape[0]), 1

        return 1, 1

    def _numel_from_shape(self, shape: List[int]) -> int:
        n = 1
        for dim in shape:
            n *= int(dim)
        return n

    def _input_bytes(self, node: HelmNode) -> int:
        # Use the model's dominant dtype size (from config) rather than the
        # node's output dtype, since input and output dtypes can differ
        # (e.g. int32 input_ids into an embedding whose output is fp16).
        dtype_size = self.config["dtype_size"]
        return sum(
            self._numel_from_shape(shape) * dtype_size
            for shape in node.input_shapes
            if self._is_plain_shape(shape)
        )

    def _is_plain_shape(self, shape: Any) -> bool:
        return isinstance(shape, list) and all(isinstance(x, int) for x in shape)
