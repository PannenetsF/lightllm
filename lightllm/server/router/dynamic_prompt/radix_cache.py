# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
from contextlib import contextmanager
from enum import Enum
import torch
import heapq
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple
from lightllm.common.cpu_memory_manager import CPUMemoryManager
from sortedcontainers import SortedSet

from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray, SharedLinkedListManager, SharedTreeInfoNode
# from .shared_arr import SharedArray, SharedTreeInfoNode, SharedLinkedListManager
from lightllm.common.mem_manager import MemoryManager


class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()


class MemoryType(Enum):
    CPU = 0
    GPU = 1 
    NOT_VALID = 2

class TreeNode:
    def __init__(self, shared_idx_manager):
        self.shared_idx_manager: SharedLinkedListManager = shared_idx_manager
        self.children = {}  # 这里的键 为 token_id_key 的第一个元素
        self.parent: TreeNode = None
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None  # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.shared_idx_node: SharedTreeInfoNode = self.shared_idx_manager.alloc()
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

        self.hot_counter = 0 # used to record the hot degree of the node, if the node is hot, it cannot be evicted
        self.evict_time: float = -1
        self.mem_type = MemoryType.NOT_VALID

    def __repr__(self):
        return f"TreeNode({self.token_id_key}, {self.mem_type})"

    def get_free_compare_key(self):
        # 1. cpu first 
        # 2. least recent used
        # note all nodes to be freed are leaves, so we do not need to consider the ref counter 
        return (0 if self.mem_type == MemoryType.CPU else 1, self.time_id)

    def get_compare_key(self):
        # return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)
        # need to consider the hot degree now 
        return (0 if self.mem_type == MemoryType.GPU else 1,
                0 if self.hot_counter == 0 else 1, 
                0 if self.ref_counter == 0 else 1, 
                len(self.children), self.time_id)

    def split_node(self, prefix_len):
        split_parent_node = TreeNode(self.shared_idx_manager)
        split_parent_node.parent = self.parent
        split_parent_node.parent.children[self.token_id_key[0].item()] = split_parent_node
        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        split_parent_node.children = {}
        split_parent_node.children[self.token_id_key[prefix_len].item()] = self
        split_parent_node.ref_counter = self.ref_counter
        split_parent_node.hot_counter = self.hot_counter
        split_parent_node.evict_time = self.evict_time
        split_parent_node.mem_type = self.mem_type

        split_parent_node.shared_idx_node.set_parent_idx(self.shared_idx_node.get_parent_idx())
        new_len = len(split_parent_node.token_mem_index_value)
        split_parent_node.shared_idx_node.set_node_value_len(new_len)
        split_parent_node.shared_idx_node.set_node_prefix_total_len(
            split_parent_node.get_parent_prefix_total_len() + new_len
        )

        self.token_id_key = self.token_id_key[prefix_len:]
        self.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        self.parent = split_parent_node
        self.shared_idx_node.set_parent_idx(split_parent_node.shared_idx_node.get_idx())
        new_len = len(self.token_mem_index_value)
        self.shared_idx_node.set_node_value_len(new_len)
        self.shared_idx_node.set_node_prefix_total_len(self.get_parent_prefix_total_len() + new_len)

        return split_parent_node

    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode(self.shared_idx_manager)
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0].item()
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self
        child.mem_type = self.mem_type

        # 更新shared 信息
        child.shared_idx_node.set_parent_idx(self.shared_idx_node.get_idx())
        new_len = len(child.token_mem_index_value)
        child.shared_idx_node.set_node_value_len(new_len)
        child.shared_idx_node.set_node_prefix_total_len(child.get_parent_prefix_total_len() + new_len)
        return child

    def remove_child(self, child_node):
        del self.children[child_node.token_id_key[0].item()]
        child_node.parent = None
        return

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        children_node = self.children.values()
        return (self.mem_type == MemoryType.GPU and MemoryType.GPU not in  [x.mem_type for x in children_node])

    def get_parent_prefix_total_len(self):
        return self.parent.shared_idx_node.get_node_prefix_total_len()


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


@contextmanager
def modify_object_in_sets(set_list: List[SortedSet], obj_x: TreeNode):
    sets_containing_x = [s for s in set_list if obj_x in s]
    
    # Remove the object from the sets it is in
    for s in sets_containing_x:
        s.remove(obj_x)
    
    try:
        yield obj_x
    finally:
        # Re-insert the object into the sets it was in
        for s in sets_containing_x:
            s.add(obj_x)


class RadixCache:
    """
    unique_name 主要用于解决单机，多实列部署时的shm冲突

    The tree would be like 
    ROOT
    |-- GPU
    |   `-- GPU
    |       `-- GPU
    |           |-- CPU
    |           `-- CPU
    |               `-- NV
    |-- GPU
    |   `-- GPU
    |       |-- CPU
    |       `-- NV
    |-- CPU
    |   `-- CPU
    |-- CPU
    `-- NV
    """

    def __init__(self, unique_name, total_token_num, total_cpu_token_num, tp_id, mem_manager: Optional[MemoryManager] = None, mov_buf_size=8192):
        self.mem_manager: Optional[MemoryManager] = mem_manager
        self.cpu_mem_manager = CPUMemoryManager(mem_manager=mem_manager, size=total_cpu_token_num, mov_buf_size=mov_buf_size)
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.shared_idx_manager = SharedLinkedListManager(unique_name, total_token_num + total_cpu_token_num, tp_id)

        self.root_node = TreeNode(self.shared_idx_manager)
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉
        self.root_node.hot_counter = 1  # 初始化为 1 保证永远不会被 evict 掉
        self.root_node.mem_type = MemoryType.GPU # root node is always in GPU, or its children cannot be in GPU

        self.evict_tree_set = SortedSet(key=lambda x: x.get_compare_key())  # 自定义比较器
        self.evict_tree_set.add(self.root_node)

        self.free_tree_set = SortedSet(key=lambda x: x.get_free_compare_key())  # used to free the tree nodes
        self.free_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(f"{unique_name}_tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num.arr[0] = 0
        self.tree_capacity = total_token_num + total_cpu_token_num

        self.session2leaf = {} # track the session id to the leaf node
        self.leaf2session = {}
        self.coldhot_evict_queue = SortedSet(key=lambda x: x.evict_time)


    def _update_session_info(self, src, tgt):
        if src in self.leaf2session:
            sid = self.leaf2session.pop(src)
            self.leaf2session[tgt] = sid
            self.session2leaf[sid] = tgt

    def keep_session(self, session_id, key, interval=5):
        # find the leaf node for the key 
        leaf_node = self._match_prefix_helper(self.root_node, key, [], update_refs=False)
        assert leaf_node.mem_type is MemoryType.GPU, "the leaf node should be in GPU"
        self.session2leaf[session_id] = leaf_node
        self.leaf2session[leaf_node] = session_id
        self._set_eviction_time(leaf_node, time.time(), interval)

    def _set_eviction_time(self, leaf_node: TreeNode, now: float, interval: float):
        node = leaf_node
        while node is not self.root_node:
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                if node.evict_time < 0:
                    node.evict_time = now + interval
                else:
                    node.evict_time = max(node.evict_time, now + interval)
                node.hot_counter += 1
            self.coldhot_evict_queue.add(node)
            node = node.parent

    def _cold_hot_evict(self):
        now = time.time()
        cold_nodes = []
        for node in self.coldhot_evict_queue:
            if node.evict_time < now:
                cold_nodes.append(node)
            else:
                break
        for node in cold_nodes:
            # there is no need to evict them explicitly, they will be evicted when the tree is full
            # we just want to make sure they are alive when we still need them 
            self.coldhot_evict_queue.discard(node)
            with modify_object_in_sets([self.evict_tree_set, self.free_tree_set], node) as node:
                node.hot_counter = 0
                node.evict_time = -1
        

    def _resume_helper(self, resume_idx, resume_len, nodes):
        # though resuming has different cases 
        #   1. cpu -> gpu 
        #   2. gpu -> cpu, cpu -> gpu 
        #   3. swap 
        # there is no need to consider too many case in real application, as the corner case is rare
        # so we just consider the case 1 and 2
        if self.mem_manager.can_use_mem_size >= resume_idx.shape[0]:
            # case 1 
            gpu_alloc_idx = self.mem_manager.alloc(resume_idx.shape[0])
            self.cpu_mem_manager.resume(resume_idx, gpu_alloc_idx)
        else:
            # case 2, if cannot do the swap, drop the GPU cache 
            max_slot = self.cpu_mem_manager.cpu_mem_state.shape[0] - sum(resume_len)
            gpu_off = resume_idx.shape[0] - self.mem_manager.can_use_mem_size
            real_off = 0
            for node in self.evict_tree_set:
                real_off += len(node.token_mem_index_value)
                if real_off >= gpu_off:
                    break
            can_do_swap = max_slot >= max(gpu_off, real_off)
            if can_do_swap:
                # do cache drop 
                self.free_radix_tree_to_keep_nodes(gpu_off, nodes)
                # off load gpu to cpu 
                off_idx = self.free_radix_cache_to_get_enough_token(resume_idx.shape[0])
                self.cpu_mem_manager.offload(off_idx)
                # resume the cpu nodes
                gpu_alloc_idx = self.mem_manager.alloc(resume_idx.shape[0])
                self.cpu_mem_manager.resume(resume_idx, gpu_alloc_idx)
            else:
                # drop the GPU cache 
                self.free_radix_cache_to_get_enough_token(resume_idx.shape[0], do_offload=False)
                gpu_alloc_idx = self.mem_manager.alloc(resume_idx.shape[0])
                self.cpu_mem_manager.resume(resume_idx, gpu_alloc_idx)
        gpu_idx_list = []
        start = 0
        for l in resume_len:
            end = start + l
            gpu_idx_list.append(gpu_alloc_idx[start:end])
            start = end
        return gpu_idx_list

        # if self.mem_manager.can_use_mem_size < resume_idx.shape[0]:
        #     gpu_need_size = resume_idx.shape[0] 
        #     gpu_off_size = self.mem_manager.can_use_mem_size - gpu_need_size
        #     import pdb; pdb.set_trace()
        #     if self.cpu_mem_manager.cpu_can_use_mem_size < gpu_off_size:
        #         self.free_radix_tree_to_get_enough_cpu_token(gpu_off_size)
        #     self.free_radix_cache_to_get_enough_token(resume_idx.shape[0])
        # gpu_alloc_idx = self.mem_manager.alloc(resume_idx.shape[0])
        # self.cpu_mem_manager.resume(resume_idx, gpu_alloc_idx)
        # # split the resume_idx to slices with the resume_len
        # gpu_idx_list = []
        # start = 0
        # for l in resume_len:
        #     end = start + l
        #     gpu_idx_list.append(gpu_alloc_idx[start:end])
        #     start = end
        # return gpu_idx_list
        # 
        #
    def resume_from_cpu(self, leaf_node: TreeNode):
        node: TreeNode = leaf_node 
        resume_idx = []
        resume_len = []
        nodes = []
        
        while node is not self.root_node:
            if node.mem_type == MemoryType.GPU:
                break
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                resume_idx.append(node.token_mem_index_value)
                resume_len.append(len(node.token_mem_index_value))
                nodes.append(node)
            node = node.parent
        if len(resume_idx) == 0:
            return
        resume_len = torch.tensor(resume_len) 
        resume_idx = torch.concat(resume_idx)
        gpu_idx_list = self._resume_helper(resume_idx, resume_len, nodes)
        for node, gpu_idx in zip(nodes, gpu_idx_list):
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                node.token_mem_index_value = gpu_idx
                node.mem_type = MemoryType.GPU




    def insert(self, key, value=None):
        if value is None:
            value = key

        assert len(key) == len(value) and len(key) >= 1, f'len_key = {len(key)} len_value = {len(value)}'
        self._cold_hot_evict()
        prefil_len, tail_node = self._insert_helper(self.root_node, key, value)
        self.resume_from_cpu(tail_node)
        return prefil_len, tail_node 

    def _insert_helper(self, node: TreeNode, key, value):
        if node.is_leaf():
            self.evict_tree_set.discard(node)
            self.free_tree_set.discard(node)

        try:
            first_key_id = key[0].item()
            if first_key_id in node.children.keys():
                child: TreeNode = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
                if prefix_len == len(key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                        self.free_tree_set.discard(child)
                    with modify_object_in_sets([self.coldhot_evict_queue], child) as child:
                        child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                        self.free_tree_set.add(child)
                    return prefix_len, child

                elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                        self.free_tree_set.discard(child)

                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    if child in self.coldhot_evict_queue:
                        self.coldhot_evict_queue.discard(child)
                    split_parent_node = child.split_node(prefix_len)
                    new_node = split_parent_node.add_and_return_new_child(key, value)
                    if child in self.coldhot_evict_queue:
                        self.coldhot_evict_queue.add(child)
                        self.coldhot_evict_queue.add(split_parent_node)
                    # update total token num
                    self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                        self.free_tree_set.add(split_parent_node)
                    if new_node.is_leaf():
                        self.evict_tree_set.add(new_node)
                        self.free_tree_set.add(new_node)

                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                        self.free_tree_set.add(child)
                    return prefix_len, new_node
                elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                    _, new_node = self._insert_helper(child, key[prefix_len:], value[prefix_len:])
                    return prefix_len, new_node
                else:
                    assert False, "can not run to here"

            else:
                with modify_object_in_sets([self.coldhot_evict_queue], node) as node:
                    new_node = node.add_and_return_new_child(key, value)
                # update total token num
                self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)
                    self.free_tree_set.add(new_node)
                return 0, new_node
        finally:
            with modify_object_in_sets([self.coldhot_evict_queue], node) as node:
                node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)
                self.free_tree_set.add(node)

        assert len(key) != 0
    def match_prefix(self, key, update_refs=False):
        ans_value_list = []
        ans_node_list = []

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

        valid(self)
        tree_node = self._match_prefix_helper(self.root_node, key, ans_node_list, update_refs=update_refs)
        valid(self)
        if tree_node != self.root_node:
            if len(ans_node_list) != 0:
                self.resume_from_cpu(tree_node)
                ans_value_list = [node.token_mem_index_value for node in ans_node_list]
                value = torch.concat(ans_value_list)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            return tree_node, len(value), value
        else:
            self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _match_prefix_helper(self, node: TreeNode, key, ans_node_list: list, update_refs=False) -> TreeNode:
        if node.is_leaf():
            self.evict_tree_set.discard(node)
            self.free_tree_set.discard(node)

        if update_refs:
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                node.ref_counter += 1
                # from 0 to 1 need update refs token num
                if node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        try:
            if len(key) == 0:
                return node

            first_key_id = key[0].item()
            if first_key_id not in node.children.keys():
                return node
            else:
                child = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
                if prefix_len == len(child.token_id_key):
                    ans_node_list.append(child)
                    return self._match_prefix_helper(child, key[prefix_len:], ans_node_list, update_refs=update_refs)
                elif prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                        self.free_tree_set.discard(child)

                    split_parent_node = child.split_node(prefix_len)
                    ans_node_list.append(split_parent_node)

                    if update_refs:
                        with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], split_parent_node) as split_parent_node:
                            split_parent_node.ref_counter += 1
                            # from 0 to 1 need update refs token num
                            if split_parent_node.ref_counter == 1:
                                self.refed_tokens_num.arr[0] += len(split_parent_node.token_mem_index_value)

                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                        self.free_tree_set.add(child)
                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                        self.free_tree_set.add(split_parent_node)

                    return split_parent_node
                else:
                    assert False, "error state"
        finally:
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)
                self.free_tree_set.add(node)

    def _offload_helper(self, mem_idx, off_len):
        if len(mem_idx) > self.cpu_mem_manager.cpu_can_use_mem_size:
            self.free_radix_tree_to_get_enough_cpu_token(len(mem_idx))
        off_idx = self.cpu_mem_manager.offload(mem_idx)
        self.mem_manager.free(mem_idx)
        start = 0
        off_idx_list = []
        for l in off_len:
            end = start + l
            off_idx_list.append(off_idx[start:end])
            start = end
        return off_idx_list
    
    def offload_to_cpu(self, nodes: List[TreeNode]):
        mem_idx = []
        off_len = []
        for node in nodes:
            idx = node.token_mem_index_value
            mem_idx.append(idx)
            off_len.append(len(idx))
        mem_idx = torch.concat(mem_idx).cuda()
        off_idx = self._offload_helper(mem_idx, off_len)
        for node, idx in zip(nodes, off_idx):
            with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], node) as node:
                node.token_mem_index_value = idx
                node.mem_type = MemoryType.CPU

    def evict(self, need_remove_tokens, evict_callback, do_offload=True):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        offload_nodes = []
        while num_evicted < need_remove_tokens:
            node: TreeNode = self.evict_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            # update total token num
            assert node.mem_type == MemoryType.GPU, "the node should be in GPU"
            parent_node: TreeNode = node.parent
            # parent_node.remove_child(node)
            # if parent_node.is_leaf():
            #     self.evict_tree_set.add(parent_node)

            # 回收 shared 链表资源
            # not need to free the shared idx node, because it will be reused as CPU nodes 
            # self.shared_idx_manager.free(node.shared_idx_node.get_idx())
            offload_nodes.append(node)
        if do_offload:
            self.offload_to_cpu(offload_nodes)
        else:
            free_idx = []
            for node in offload_nodes:
                self.shared_idx_manager.free(node.shared_idx_node.get_idx())
                self.coldhot_evict_queue.discard(node)
                self.free_tree_set.discard(node)
                free_idx.append(node.token_mem_index_value)
                p_node = node.parent
                with modify_object_in_sets([self.coldhot_evict_queue, self.evict_tree_set, self.free_tree_set], p_node) as p_node:
                    p_node.remove_child(node)
                    if p_node.is_leaf():
                        self.evict_tree_set.add(p_node)
                        self.free_tree_set.add(p_node)
            free_idx = torch.concat(free_idx)
            self.mem_manager.free(free_idx)
        return

    def clear_tree_nodes(self):
        """
        该函数只在测试时调用
        """
        while True:
            node: TreeNode = self.evict_tree_set.pop(0)
            if node != self.root_node:
                parent_node: TreeNode = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)

                self.shared_idx_manager.free(node.shared_idx_node.get_idx())
            else:
                break

        self.tree_total_tokens_num.arr[0] = 0
        self.refed_tokens_num.arr[0] = 0
        return

    def dec_node_ref_counter(self, node):
        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent
        return

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_tree_currenct_capacity(self):
        return self.tree_capacity - self.get_tree_total_tokens_num()

    def get_cpu_tree_currenct_capacity(self):
        return self.cpu_mem_manager.cpu_can_use_mem_size

    def print_tree(self, node, indent=0):
        ss = []
        s = " " * indent + f"shared_idx: {node.shared_idx_node.get_idx()} mem_type: {node.mem_type} node_token_id_key: {node.token_id_key}"
            # k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} 
        for _, child in node.children.items():
            ss.append(self.print_tree(child, indent=indent + 2))
        return '\n'.join([s] + ss)

    def __repr__(self):
        return self.print_tree(self.root_node)

    def print_self(self, indent=0):
        self._print_helper(self.root_node, indent)

    def _print_helper(self, node: TreeNode, indent):
        print(
            " " * indent,
            f"shared_idx: {node.shared_idx_node.get_idx()} p_idx: {node.shared_idx_node.get_parent_idx()} \
            refs: {node.ref_counter} hot_refs: {node.hot_counter} mem_type: {node.mem_type} \
            time_id: {node.time_id} prefix_total_len: {node.shared_idx_node.get_node_prefix_total_len()} \
            node_value_len: {node.shared_idx_node.get_node_value_len()}",
        )
            # k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} 
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num, do_offload=True):
        # free gpu memory
        assert self.mem_manager is not None
        self._cold_hot_evict()
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem, do_offload)
            mem_index = torch.concat(release_mems)
        return
    


    def free(self, token_num, free_callback, kept=[]):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < token_num:
            assert False, f"""can not free tree tokens {token_num},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        tmp_pop = []

        while num_evicted < token_num:
            node: TreeNode = self.free_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node
            ), "error evict tree node state"
            if node in kept:
                tmp_pop.append(node)
                continue
            num_evicted += len(node.token_mem_index_value)
            free_callback(node.token_mem_index_value, node.mem_type == MemoryType.CPU)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            self.evict_tree_set.discard(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)
                self.free_tree_set.add(parent_node)

            # 回收 shared 链表资源
            self.shared_idx_manager.free(node.shared_idx_node.get_idx())
        for node in tmp_pop:
            self.free_tree_set.add(node)
        return


    def free_radix_tree_to_keep_nodes(self, needed_token_num, kept_nodes):
        # free cpu memory, might remove some tree nodes.
        self._cold_hot_evict()
        if needed_token_num > self.get_tree_currenct_capacity():
            need_evict_token_num = needed_token_num - self.get_tree_currenct_capacity()
            evict_nodes = []
            def release_mem(mem_index, is_cpu):
                evict_nodes.append(mem_index) if is_cpu else None
            self.free(need_evict_token_num, release_mem, kept_nodes)
            if len(evict_nodes) > 0:
                for node in evict_nodes:
                    if node in kept_nodes:
                        kept_nodes.remove(node)
        return

    def free_radix_tree_to_get_enough_cpu_token(self, need_token_num):
        # free cpu memory, might remove some tree nodes.
        self._cold_hot_evict()
        if need_token_num > self.cpu_mem_manager.cpu_mem_state.shape[0]:
            raise ValueError("not enough cpu memory")
        if need_token_num > self.get_cpu_tree_currenct_capacity():
            need_evict_token_num = need_token_num - self.get_cpu_tree_currenct_capacity()
            
            cpu_mems = []
            gpu_mems = []
            def release_mem(mem_index, is_cpu):
                if is_cpu:
                    assert mem_index.device == 'cpu', f'{mem_index.device}'
                else:
                    assert mem_index.device == 'cuda', f'{mem_index.device}'
                cpu_mems.append(mem_index) if is_cpu else gpu_mems.append(mem_index)

            self.free(need_evict_token_num, release_mem)
            if len(cpu_mems):
                cpu_mem_index = torch.concat(cpu_mems)
                # free the cpu memory 
                self.cpu_mem_manager.free(cpu_mem_index)
            if len(gpu_mems):
                gpu_mem_index = torch.concat(gpu_mems).cuda()
                # free the gpu memory 
                self.mem_manager.free(gpu_mem_index)


    def free_radix_tree_to_get_enough_token(self, need_token_num):
        # free cpu memory, might remove some tree nodes.
        self._cold_hot_evict()
        if need_token_num > self.get_tree_currenct_capacity():
            need_evict_token_num = need_token_num - self.get_tree_currenct_capacity()
            
            cpu_mems = []
            gpu_mems = []
            def release_mem(mem_index, is_cpu):
                if is_cpu:
                    assert mem_index.device == 'cpu', f'{mem_index.device}'
                else:
                    assert mem_index.device == 'cuda', f'{mem_index.device}'
                cpu_mems.append(mem_index) if is_cpu else gpu_mems.append(mem_index)

            self.free(need_evict_token_num, release_mem)
            if len(cpu_mems):
                cpu_mem_index = torch.concat(cpu_mems)
                # free the cpu memory 
                self.cpu_mem_manager.free(cpu_mem_index)
            if len(gpu_mems):
                gpu_mem_index = torch.concat(gpu_mems).cuda()
                # free the gpu memory 
                self.mem_manager.free(gpu_mem_index)



class RadixCacheReadOnlyClient:
    """
    router 端只读用的客户端，用于从共享内存中读取树结构中的信息，用于进行prompt cache 的调度估计。
    """

    def __init__(self, unique_name, total_token_num, tp_id):
        self.shared_idx_manager = SharedLinkedListManager(unique_name, total_token_num, tp_id)
        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(f"{unique_name}_tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_unrefed_tokens_num(self):
        return self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0]

    def get_shared_node(self, idx):
        return self.shared_idx_manager.get_shared_node(idx)

    def get_all_parent_shared_nodes(self, idx):
        node = self.shared_idx_manager.get_shared_node(idx)
        ans_list = [node]
        while node.get_parent_idx() != -1:
            node = self.shared_idx_manager.get_shared_node(node.get_parent_idx())
            ans_list.append(node)
        return ans_list


# ///////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    def test0():
        tree = RadixCache("unique_name", 100, 0)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 0
        ans = tree.match_prefix(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        print(ans)

    test0()
    exit(0)

    # test 1
    def test1():
        tree = RadixCache("unique_name", 100, 0)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 0
        tree.print_self()
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 5
        tree.print_self()
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 8
        tree.print_self()

        assert tree.get_refed_tokens_num() == 0
        assert tree.get_tree_total_tokens_num() == 13

        # print("evict")
        tree.evict(9, lambda x: x)
        tree.print_self()
        assert tree.get_refed_tokens_num() == 0 and tree.get_tree_total_tokens_num() == 0

    test1()

    # test 2
    def test2():
        tree = RadixCache("unique_name", 100, 1)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 9], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 8], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 7 and size == 7 and len(values) == 7
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 6 and size == 6 and len(values) == 6
        print(ans)
        return

    # test2()

    # test 3
    def test3():
        tree = RadixCache("unique_name", 100, 2)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=True
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        assert tree.get_refed_tokens_num() == 5 and tree.get_tree_total_tokens_num() == 13

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=True
       )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 6 and size == 6 and len(values) == 6
        assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 13

        tree.print_self()
        tree.evict(2, lambda x: x)
        assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 8
        tree.print_self()

        tree.dec_node_ref_counter(tree_node)
        tree.print_self()
        print(ans)
        return

    test3()

    def test4():

        tree = RadixCache("unique_name", 100, 2)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree.clear_tree_nodes()
        assert tree.shared_idx_manager.can_alloc_num() == 100
        print(ans)
        return

    test4()
