import os
from typing import List

import torch
import torch.distributed as dist
from sortedcontainers import SortedSet

from lightllm.common.cpu_memory_manager import CPUMemoryManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.router.dynamic_prompt.radix_cache import (MemoryType,
                                                               RadixCache,
                                                               TreeNode)

def init_manager(mem_manager: MemoryManager, cpu_manager: CPUMemoryManager):
    for i in range(len(mem_manager.kv_buffer)):
        mem_manager.kv_buffer[i] = torch.randn_like(mem_manager.kv_buffer[i])
        cpu_manager.cpu_kv_pool[i] = torch.randn_like(cpu_manager.cpu_kv_pool[i])

def init_torch_dist_with_nccl():
    port = 23145
    setting = {
        "nccl_port": port,
    }
    rank_id = 0
    world_size = 1

    dist.init_process_group(
        "nccl",
        init_method=f'tcp://127.0.0.1:{setting["nccl_port"]}',
        rank=rank_id,
        world_size=world_size,
    )
    os.environ["_NCCL_PORT_"] = str(setting["nccl_port"])
    torch.cuda.set_device(rank_id)



def main():
    total_cpu_size = 20
    total_gpu_size = 20
    head_num = 32
    head_dim = 128
    layer_num = 32
    dtype = torch.half

    init_torch_dist_with_nccl()

    gpu_manager = MemoryManager(
        size=total_gpu_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
    )
    cpu_manager = CPUMemoryManager(gpu_manager, total_cpu_size, 10)
    init_manager(gpu_manager, cpu_manager)


    src_idx = torch.randperm(total_gpu_size)[:8].cuda()
    dst_idx = cpu_manager.offload(src_idx=src_idx)

    print(dst_idx, src_idx)
    # take layer -1 
    dst_data = cpu_manager.cpu_kv_pool[-1][dst_idx]
    src_data = gpu_manager.kv_buffer[-1][src_idx]

    torch.cuda.synchronize()
    error = torch.abs(dst_data.detach().clone().cpu() - src_data.detach().clone().cpu())
    print(error.max())

    dst_idx = torch.randperm(total_gpu_size)[:8].cuda()
    src_idx = torch.randperm(total_cpu_size)[:8]
    cpu_manager.resume(src_idx=src_idx, dst_idx=dst_idx)

    dst_data = gpu_manager.kv_buffer[-1][dst_idx]
    src_data = cpu_manager.cpu_kv_pool[-1][src_idx]

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    error = torch.abs(dst_data.detach().clone().cpu() - src_data.detach().clone().cpu())
    print(error.max())

if __name__ == "__main__":
    main()
    # a = torch.rand(100, 20)
    # idx = [1, 3, 5, 2]
    # b = torch.rand(4, 20)
    # off = a[0].data_ptr() - a.data_ptr()
    # a[idx] = b
    # print(a[idx] - b)
    #


