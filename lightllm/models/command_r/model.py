from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.models.command_r.layer_infer.transformer_layer_infer import CommandRTransformerLayerInfer
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.splitfuse_infer_struct import LlamaSplitFuseInferStateInfo
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class CommandRTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = BloomPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    transformer_layer_infer_class = CommandRTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo
    splitfuse_infer_state_class = LlamaSplitFuseInferStateInfo
