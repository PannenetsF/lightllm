import os
from typing import List

import torch
import torch.distributed as dist
from sortedcontainers import SortedSet

from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.router.dynamic_prompt.radix_cache import (MemoryType,
                                                               RadixCache,
                                                               TreeNode)


def validate_sorted_set(s: SortedSet):
    xs = []
    for x in s:
        xs.append(x)
    for x in xs:
        s.discard(x)
        s.add(x)


def valid(radix: RadixCache):
    try:
        validate_sorted_set(radix.coldhot_evict_queue)
    except:
        raise ValueError("coldhot_evict_queue is not a SortedSet")
    try:
        validate_sorted_set(radix.evict_tree_set)
    except:
        raise ValueError("evict_tree_set is not a SortedSet")
    try:
        validate_sorted_set(radix.free_tree_set)
    except:
        raise ValueError("free_tree_set is not a SortedSet")


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


g_data_dict = {}


def insert(mem_manager: MemoryManager, radix_cache: RadixCache, key: List[int]):
    size = len(key)
    indices = mem_manager.alloc(size)
    assert indices is not None, f"cannot allocate {size} tokens"
    assert size == len(
        indices
    ), f"allocated {len(indices)} tokens, but need {size} tokens, idx={indices}"
    key = torch.tensor(key, dtype=torch.int32, device="cpu")
    valid(radix_cache)
    plen, node = radix_cache.insert(key, indices)

    cuda_data = radix_cache.mem_manager.kv_buffer[0][indices[plen:]]
    for k, v in zip(key[plen:], cuda_data):
        if k not in g_data_dict:
            g_data_dict[k.item()] = v.detach().cpu()

    valid(radix_cache)
    cpu_mem = radix_cache.cpu_mem_manager
    mem_manager.free(indices[:plen])
    status = f"gpu can_use_mem_size: {mem_manager.can_use_mem_size} cpu can_use_mem_size: {cpu_mem.cpu_can_use_mem_size}"
    return status


def free(mem_manager: MemoryManager, radix_cache: RadixCache, token_num: int):
    radix_cache.free_radix_tree_to_get_enough_token(token_num)
    status = f"mem_manager can_use_mem_size: {mem_manager.can_use_mem_size} cpu can_use_mem_size: {radix_cache.cpu_mem_manager.cpu_can_use_mem_size}"
    return status


def evict(mem_manager: MemoryManager, radix_cache: RadixCache, token_num: int):
    radix_cache.free_radix_cache_to_get_enough_token(token_num)
    status = f"mem_manager can_use_mem_size: {mem_manager.can_use_mem_size} cpu can_use_mem_size: {radix_cache.cpu_mem_manager.cpu_can_use_mem_size}"
    return status


def match_prefix(
    mem_manager: MemoryManager, radix_cache: RadixCache, tokens: List[int]
):
    tokens = torch.tensor(tokens, dtype=torch.int32, device="cpu")

    prefix = radix_cache.match_prefix(tokens)
    return prefix


def make_sure(mem_manager: MemoryManager, radix_cache: RadixCache, tokens: List[int]):
    token_num = len(tokens)
    gpu_left = mem_manager.can_use_mem_size
    gpu_used = mem_manager.mem_state.shape[0] - gpu_left
    total_size = mem_manager.mem_state.shape[0]
    cpu_left = radix_cache.cpu_mem_manager.cpu_can_use_mem_size
    cpu_used = radix_cache.cpu_mem_manager.cpu_mem_state.shape[0] - cpu_left
    # 3 cases
    # 1. insert to gpu directly
    # 2. offload tokens from gpu to cpu directly, then insert to gpu
    # 3. clean cpu first, then offload, then insert
    # 4. impossible

    if token_num > total_size:
        assert (
            False
        ), f"Should not insert {token_num} tokens, since the total size is {total_size}"
    if gpu_left >= token_num:
        # case 1
        pass
    elif gpu_used <= cpu_left and token_num <= total_size:
        offload_tokens = token_num - gpu_left
        evict(mem_manager, radix_cache, token_num)
        # case 2
    else:
        cpu_clean_tokens = gpu_used - cpu_left
        offload_tokens = token_num - gpu_left
        # cpu
        free(mem_manager, radix_cache, gpu_used)
        # gpu
        evict(mem_manager, radix_cache, token_num)
        # case 3
    return insert(mem_manager, radix_cache, tokens)


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
    radix_cache = RadixCache(
        unique_name="test",
        total_cpu_token_num=total_cpu_size,
        total_token_num=total_gpu_size,
        tp_id=0,
        mem_manager=gpu_manager,
    )

    for i in range(layer_num):
        radix_cache.cpu_mem_manager.cpu_kv_pool[i] = torch.zeros_like(
            radix_cache.cpu_mem_manager.cpu_kv_pool[i]
        )
        radix_cache.mem_manager.kv_buffer[i] = torch.rand_like(
            radix_cache.mem_manager.kv_buffer[i]
        )

    tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    status = insert(gpu_manager, radix_cache, tokens)
    print(radix_cache)

    tokens = [0, 1, 8, 9]
    status = insert(gpu_manager, radix_cache, tokens)
    print(radix_cache)

    token_needed = len(tokens)
    status = evict(gpu_manager, radix_cache, token_needed)
    print(radix_cache)

    tokens = [16 + i for i in range(16)]
    import pdb; pdb.set_trace()
    status = make_sure(gpu_manager, radix_cache, tokens)
    print(radix_cache)

    cpu_nodes = []
    all_nodes = []
    queue = [radix_cache.root_node]
    while True:
        # print(queue)
        if len(queue) == 0:
            break
        node = queue.pop(0)
        children = node.children.values()
        queue.extend(children)
        all_nodes.extend(children)

    cpu_nodes: List[TreeNode] = [
        node for node in all_nodes if node.mem_type == MemoryType.CPU
    ]
    print(cpu_nodes)

    for node in cpu_nodes:
        keys = node.token_id_key
        cpu_idx = node.token_mem_index_value
        for key, idx in zip(keys, cpu_idx):
            key = key.item()
            token_kv = radix_cache.cpu_mem_manager.cpu_kv_pool[0][idx]
            assert key in g_data_dict, f"{key} not in g_data_dict {g_data_dict.keys()}"
            abs_err = torch.abs( token_kv - torch.tensor(g_data_dict[key], dtype=dtype)).max()
            assert (
                abs_err < 1e-5
            ).all(), f"{abs_err}"


    print(radix_cache)


if __name__ == "__main__":
    main()
