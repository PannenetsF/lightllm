from functools import partial
from typing import Tuple

import torch

from lightllm.common.basemodel import InferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.bib_mha_decoding import bib_decoding
from lightllm.models.llama.triton_kernel.context_flashattention_nopad_bib import context_attention_fwd_bib
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.rotary_emb_bib import rotary_emb_fwd_bib
from lightllm.models.llama_bib.infer_struct import LlamaInferStateInfoBiB


class LlamaTransformerLayerInferBiB(LlamaTransformerLayerInfer):
    def _pre_cache_kv(self, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        if infer_state.is_prefill:
            cache_k = infer_state.key_buffer[self.layer_num_]
            cache_v = infer_state.value_buffer[self.layer_num_]
        else:
            cache_idx = infer_state.mem_index
            cache_k = infer_state.key_buffer[self.layer_num_][cache_idx]
            cache_v = infer_state.value_buffer[self.layer_num_][cache_idx]
        return cache_k, cache_v

    def _mm_kv(self, input, weight, cache, b_seq_len, b_start_loc, is_prefill):
        if is_prefill:
            output = torch.mm(input.view(-1, self.embed_dim_), weight)
            assert output.shape[0] == b_seq_len.sum()
            cur_start = 0
            for start, leng in zip(b_start_loc, b_seq_len):
                cache.view(-1, self.tp_k_head_num_ * self.head_dim_)[start:start + leng] = output[
                                                                                           cur_start:cur_start + leng]
                cur_start += leng
        else:
            torch.mm(input.view(-1, self.embed_dim_), weight, out=cache.view(-1, self.tp_k_head_num_ * self.head_dim_))

    def _rotary_emb_kv(self, kv: torch.Tensor, infer_state: LlamaInferStateInfoBiB):
        if infer_state.is_prefill:
            rotary_emb_fwd_bib(kv, infer_state.position_cos_kv, infer_state.position_sin_kv,
                               infer_state.block_to_request, infer_state.block_to_start, infer_state.block_to_length,
                               infer_state.block_to_chunk, infer_state.block_num,
                               chunk_size=infer_state.chunk_size)
        else:
            rotary_emb_fwd(kv.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos_kv,
                           infer_state.position_sin_kv)

    def _get_qkv(self, input, cache_k, cache_v, infer_state: LlamaInferStateInfoBiB,
                 layer_weight: LlamaTransformerLayerWeight):
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos,
                       infer_state.position_sin)
        self._mm_kv(input, layer_weight.k_weight_, cache_k, infer_state.b_seq_len, infer_state.b_start_loc,
                    infer_state.is_prefill)
        self._rotary_emb_kv(cache_k, infer_state)
        self._mm_kv(input, layer_weight.v_weight_, cache_v, infer_state.b_seq_len, infer_state.b_start_loc,
                    infer_state.is_prefill)
        return q, cache_k, cache_v

    def _copy_kv_to_mem_cache(self, cache_k, cache_v, target_idx, mem_manager):
        destindex_copy_kv(cache_k, target_idx, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(cache_v, target_idx, mem_manager.value_buffer[self.layer_num_])

    def _post_cache_kv(self, cache_k, cache_v, infer_state: InferStateInfo, layer_weight):
        if infer_state.is_prefill:
            pass
        else:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.mem_index, infer_state.mem_manager)

    def _context_attention_kernel_bib(self, q, k, v, infer_state: LlamaInferStateInfoBiB, layer_weight,
                                      out=None) -> torch.Tensor:
        o_tensor = torch.empty_like(q) if out is None else out
        context_attention_fwd_bib(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                                  k.view(-1, self.tp_k_head_num_, self.head_dim_),
                                  v.view(-1, self.tp_v_head_num_, self.head_dim_),
                                  o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                                  infer_state.b_start_loc,
                                  infer_state.b_seq_len,
                                  infer_state.b_q_start_loc,
                                  infer_state.max_len_in_batch)
        return o_tensor

    def _token_decode_attention_bib(self, q, infer_state: LlamaInferStateInfoBiB, layer_weight, out=None):
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        score = torch.empty_like(q)
        bib_decoding(
            q_tensor=q,
            k_tensor=infer_state.key_buffer[self.layer_num_],
            v_tensor=infer_state.value_buffer[self.layer_num_],
            score_tensor=score,
            request_to_block=infer_state.request_to_block,
            block_to_request=infer_state.block_to_request,
            block_to_start=infer_state.block_to_start,
            block_to_length=infer_state.block_to_length,
            block_to_chunk=infer_state.block_to_chunk,
            sm_scale=1 / self.head_dim_ ** 0.5,
            deno_tensor=infer_state.deno_tensor,
            nume_tensor=infer_state.nume_tensor,
            CHUNK_SIZE=infer_state.chunk_size,
            debug=(infer_state.b_seq_len, infer_state.b_start_loc)
        )
        torch.cuda.synchronize()
        to_save = [
            q, infer_state.key_buffer[self.layer_num_], infer_state.value_buffer[self.layer_num_],
            infer_state.b_seq_len, infer_state.b_start_loc,
            score
        ]
        if hasattr(self, 'cnt__'):
            self.cnt__ += 1
        else:
            self.cnt__ = 0
        to_save = [x.clone().detach() if isinstance(x, torch.Tensor) else x for x in to_save]
        torch.save(to_save, f'bib_token_{self.cnt__}.pth')
        return score

    def _bind_attention(self):
        super()._bind_attention()
        self._context_attention_kernel = partial(LlamaTransformerLayerInferBiB._context_attention_kernel_bib, self)
        self._token_attention_kernel = partial(LlamaTransformerLayerInferBiB._token_decode_attention_bib, self)
