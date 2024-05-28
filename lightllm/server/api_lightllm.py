import collections
from typing import AsyncGenerator
from uuid import  uuid3, NAMESPACE_DNS
from fastapi import BackgroundTasks, Request
from fastapi.responses import Response, StreamingResponse
from .sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .metrics import monitor
import json


_session_id_cnt = 0
_session_pool = set()
def generate_session_id() -> str:
    global _session_id_cnt
    id = uuid3(NAMESPACE_DNS, "lightllm_session" + str(_session_id_cnt))
    _session_id_cnt += 1 
    hex_str = id.hex
    _session_pool.add(hex_str)
    return hex_str


async def lightllm_generate(request: Request, g_id_gen, httpserver_manager) -> Response:
    monitor.counter_inc("lightllm_request_count")

    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    session_id = request_dict.pop("session_id", generate_session_id())
    assert session_id in _session_pool, f"session_id should be in session_pool, or it is illegal. id={session_id} set={_session_pool}"
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    group_request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(
        prompt, sampling_params, group_request_id, multimodal_params, request=request, session_id=session_id
    )

    # Non-streaming case
    final_output_dict = collections.defaultdict(list)
    count_output_tokens_dict = collections.defaultdict(lambda: 0)
    tokens_dict = collections.defaultdict(list)
    finish_reason_dict = {}
    prompt_logprobs = None
    prompt_token_ids = None
    is_first_metadata = True
    async for sub_req_id, request_output, metadata, finish_status in results_generator:
        # when set "--return_all_prompt_logprobs", the first token metadata will contains
        # prompt_logprobs and prompt_token_ids
        if is_first_metadata:
            prompt_logprobs = metadata.get("prompt_logprobs", None)
            prompt_token_ids = metadata.get("prompt_token_ids", None)
            if prompt_logprobs is not None:
                del metadata["prompt_logprobs"]
            if prompt_token_ids is not None:
                del metadata["prompt_token_ids"]
            is_first_metadata = False

        count_output_tokens_dict[sub_req_id] += 1
        final_output_dict[sub_req_id].append(request_output)
        if return_details:
            tokens_dict[sub_req_id].append(metadata)
            metadata["text"] = request_output

        if finish_status.is_finished():
            finish_reason_dict[sub_req_id] = finish_status
    n = sampling_params.n
    sub_ids = list(final_output_dict.keys())[:n]
    final_output_list = ["".join(final_output_dict[sub_id]) for sub_id in sub_ids]
    count_output_tokens_list = [count_output_tokens_dict[sub_id] for sub_id in sub_ids]
    finish_reson_list = [finish_reason_dict[sub_id].get_finish_reason() for sub_id in sub_ids]
    tokens_list = [tokens_dict[sub_id] for sub_id in sub_ids]
    only_one = len(sub_ids) == 1

    ret_data_format = lambda data_list: data_list[0] if only_one else data_list

    ret = {
        "generated_text": final_output_list,
        "count_output_tokens": ret_data_format(count_output_tokens_list),
        "finish_reason": ret_data_format(finish_reson_list),
        "session_id": session_id,
    }
    if return_details:
        ret["tokens"] = ret_data_format(tokens_list)
    if prompt_token_ids is not None:
        ret["prompt_token_ids"] = prompt_token_ids
    if prompt_logprobs is not None:
        ret["prompt_logprobs"] = prompt_logprobs
    monitor.counter_inc("lightllm_request_success")
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


async def lightllm_generate_stream(request: Request, g_id_gen, httpserver_manager) -> Response:
    monitor.counter_inc("lightllm_request_count")

    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    _ = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    if sampling_params.best_of != 1:
        raise Exception("stream api only support best_of == 1")

    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    group_request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(
        prompt, sampling_params, group_request_id, multimodal_params, request=request
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for _, request_output, metadata, finish_status in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": metadata.get("special", False),
                    "count_output_tokens": metadata.get("count_output_tokens", 0),
                },
                "generated_text": None,
                "finished": finish_status.is_finished(),
                "finish_reason": finish_status.get_finish_reason(),
                "details": None,
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(group_request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)
    monitor.counter_inc("lightllm_request_success")
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)
