import torch
from lightllm.server.router.dynamic_prompt.radix_cache import (MemoryManager,
                                                               RadixCache,
                                                               TreeNode, match)


class ColdHotRadixCache(RadixCache):
    def __init__(
        self, unique_name, total_token_num, tp_id, mem_manager: MemoryManager = None
    ):
        super().__init__(unique_name, total_token_num, tp_id, mem_manager)
        self.held_nodes = set()
        self.node_hold_counter = {}
        self.session_to_end_node = {}


    def hold_request(self, session_id, input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu")
        tail_node, size, values = self.match_prefix(
            input_ids, update_refs=False
        ) # for now, we don't need to update refs, as the cache is held
        if session_id in self.session_to_end_node:
            self._cool(self.session_to_end_node[session_id]) # make sure only 1 reference is kept
        self._hold(tail_node)
        self.session_to_end_node[session_id] = tail_node
        print('>> hold session for ', session_id)

    def cool_request(self, session_id, input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu")
        tail_node, size, values = self.match_prefix(
            input_ids, update_refs=True
        ) # for now, we need to update refs, as the cache is cooled/evicted
        self._cool(tail_node)
        self.session_to_end_node.pop(session_id)

    def _hold(self, node: TreeNode):
        while node is not None:
            if node not in self.node_hold_counter:
                self.node_hold_counter[node] = 0
            self.node_hold_counter[node] += 1
            node = node.parent

    def _cool(self, node: TreeNode):
        while node is not None:
            if node in self.node_hold_counter:
                self.node_hold_counter[node] -= 1
                if self.node_hold_counter[node] == 0:
                    del self.node_hold_counter[node]
                    self.held_nodes.discard(node)
            node = node.parent


if __name__ == "__main__":
    import torch

    # test 1
    def test1():
        tree = ColdHotRadixCache("unique_name", 100, 0)
        print(">> INIT with {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}")
        ans = tree.insert(
            torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"
            )
        )
        assert ans == 0
        tree.print_self()
        print(">> INSERT {0, 1, 2, 3, 4, 7, 8, 9}")
        ans = tree.insert(
            torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu")
        )
        assert ans == 5
        tree.print_self()
        print(">> INSERT {0, 1, 2, 3, 4, 7, 8, 9}")
        ans = tree.insert(
            torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu")
        )
        assert ans == 8
        tree.print_self()

        print(">> INSERT {0, 1, 3, 5}")
        ans = tree.insert(torch.tensor([0, 1, 3, 5], dtype=torch.int64, device="cpu"))
        tree.print_self()

        print(">> HOLD {0, 1, 3, 5}")
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 3, 5], dtype=torch.int64, device="cpu"),
            update_refs=True,
        )
        tree._hold(tree_node)
        print(">> EVICT 9 tokens")
        tree.evict(9, lambda x: x)
        tree.print_self()

        print(">> COOL some tokens")
        tree_node = tree.match_prefix(
            torch.tensor([0, 1, 3, 5], dtype=torch.int64, device="cpu"),
            update_refs=False,
                )
        # print the tree_node 
        # print(tree_node)
        print('>> print the tree_node')
        tree._print_helper(tree_node[0], 2)
        tree._print_helper(tree_node[0].parent, 2)
        tree._cool(tree_node[0])
        # import pdb; pdb.set_trace()
        tree.dec_node_ref_counter(tree_node[0])
        tree._print_helper(tree_node[0], 2)
        tree._print_helper(tree_node[0].parent, 2)
        tree.evict(4, lambda x: x)
        tree.print_self()

    test1()
