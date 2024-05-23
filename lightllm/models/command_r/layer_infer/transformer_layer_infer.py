from functools import partial

from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class CommandRTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)

    def _bind_norm(self):
        self._att_norm = partial(BloomTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(BloomTransformerLayerInfer._ffn_norm, self)
        return

