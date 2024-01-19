from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_bib.infer_struct import LlamaInferStateInfoBiB
from lightllm.models.llama_bib.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferBiB


class LlamaTpPartModelBib(LlamaTpPartModel):
    transformer_layer_infer_class = LlamaTransformerLayerInferBiB
    infer_state_class = LlamaInferStateInfoBiB
    def __init__(self, kvargs):
        super().__init__(kvargs)
        assert 'bib_route' in self.mode, f'bib model should only be used under bib_route'
        self.bib_size = kvargs["bib_size"]
        self.chunk_size = kvargs["chunk_size"]

    def _prefill(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len,
                 multimodal_params):
        infer_state = self.infer_state_class(self.bib_size, self.chunk_size)
        infer_state.is_prefill = True
        infer_state.return_all_prompt_logprobs = self.return_all_prompt_logprobs
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        infer_state.key_buffer = self.mem_manager.key_buffer
        infer_state.value_buffer = self.mem_manager.value_buffer

        alloc_mem = self.mem_manager.alloc_contiguous(infer_state.total_token_num)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]

        else:
            raise ValueError(f'Please check the router strategy, it should not allocate non-cont mem')

        # init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
        #                           max_len_in_batch, infer_state.mem_index)

        infer_state.init_bib_state(self, input_ids, b_req_idx, b_start_loc, b_seq_len)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len,
                multimodal_params):
        infer_state = self.infer_state_class(self.bib_size, self.chunk_size)
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        infer_state.mem_is_contiguous = True
        infer_state.key_buffer = self.mem_manager.key_buffer
        infer_state.value_buffer = self.mem_manager.value_buffer

        infer_state.mem_index = infer_state.b_start_loc + infer_state.b_seq_len - 1
        assert True
        infer_state.init_bib_state(self, input_ids, b_req_idx, b_start_loc, b_seq_len)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics
