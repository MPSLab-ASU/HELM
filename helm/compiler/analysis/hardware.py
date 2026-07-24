import logging

import psutil
import torch

from ..graph import HelmGraph

logger = logging.getLogger(__name__)

class HardwareAnalyzer:
    """
    Pass: Hardware Detection
    Detects available hardware (CPU, GPU, RAM) and attaches metadata to the HelmGraph.
    """
    def __init__(self, graph: HelmGraph):
        self.graph = graph

    def run(self):
        logger.info("[HardwareAnalyzer] Detecting System Resources...")

        meta = {}

        # 1. CPU Info
        meta['cpu_count_physical'] = psutil.cpu_count(logical=False)
        meta['cpu_count_logical'] = psutil.cpu_count(logical=True)

        mem = psutil.virtual_memory()
        meta['system_ram_total_gb'] = mem.total / (1024**3)
        meta['system_ram_available_gb'] = mem.available / (1024**3)

        # 2. GPU Info
        if torch.cuda.is_available():
            meta['gpu_available'] = True
            meta['gpu_count'] = torch.cuda.device_count()

            gpu_info = []
            for i in range(meta['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor
                })
            meta['gpus'] = gpu_info
        else:
            meta['gpu_available'] = False

        self.graph.hardware_meta = meta

        logger.info("  CPU Cores: %s (Phys) / %s (Log)", meta['cpu_count_physical'], meta['cpu_count_logical'])
        logger.info("  System RAM: %.2f GB / %.2f GB", meta['system_ram_available_gb'], meta['system_ram_total_gb'])

        if meta['gpu_available']:
            for idx, gpu in enumerate(meta['gpus']):
                logger.info("  GPU %d: %s (%.2f GB VRAM, %d SMs)",
                            idx, gpu['name'], gpu['total_memory_gb'], gpu['multi_processor_count'])
        else:
            logger.info("  GPU: None detected.")
