import torch
from lightllm.utils.log_utils import init_logger
from .mem_manager import MemoryManager

logger = init_logger(__name__)
    

class BiBMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy=False)

    @torch.no_grad()
    def alloc(self, need_size):
        logger.warn(f'you should not use alloc of BiBMemoryManager, as all memory slots are pre-allocated.')
        raise NotImplemented
