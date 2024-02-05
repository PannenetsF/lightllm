import argparse
import logging

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from lightllm.models.llama.model import LlamaTpPartModel as normal_model
from lightllm.models.llama_bib.model import LlamaTpPartModelBib as bib_model


def prepare_bib(model_part, batch_size, input_len, bib_size):
    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len[0] = input_len
    for i in range(1, batch_size):
        b_seq_len[i] = int(input_len * (1 + i / 10))
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1] + bib_size

    total_token_num = b_seq_len.sum()
    return total_token_num, b_req_idx, b_start_loc, b_seq_len


def prepare_normal(model_part, batch_size, input_len):
    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len[0] = input_len
    for i in range(1, batch_size):
        b_seq_len[i] = int(input_len * (1 + i / 10))
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1] + 0
    total_token_num = b_seq_len.sum()
    return total_token_num, b_req_idx, b_start_loc, b_seq_len


def tppart_model_infer(model_kvargs, batch_size, input_len, output_len, test_data_all, mode='normal'):
    if mode == 'normal':
        model_class = normal_model
        prepare_fn = lambda: prepare_normal(model_part, batch_size, input_len)
    elif mode == 'bib':
        model_class = bib_model
        bib_size = model_kvargs["bib_size"]
        assert bib_size >= output_len
        prepare_fn = lambda: prepare_bib(model_part, batch_size, input_len, bib_size)
    else:
        raise NotImplementedError(f'not support mode {mode}')
    dist.barrier()
    torch.cuda.empty_cache()
    model_part = model_class(model_kvargs)
    model_part.layers_num = 1
    # for i in range(model_part.layers_num):
    #     model_part.layers_infer[i]._token_attention = lambda *args, **kwargs: args[0]
    # model_part.post_infer.token_forward = lambda *args, **kwargs: args[0].to(torch.float32)
    #
    logging.info(f'prepared model {model_class}')

    total_token_num, b_req_idx, b_start_loc, b_seq_len = prepare_fn()
    print(b_seq_len)
    test_data = test_data_all[:total_token_num]
    test_data = torch.from_numpy(test_data).cuda()

    logits_result = []
    logics = model_part.forward(batch_size, total_token_num, b_seq_len.max(), test_data, b_req_idx, b_start_loc,
                                b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    logits_result.append(prob_out.detach().cpu())
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in tqdm(range(output_len)):
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1,
                                    torch.from_numpy(predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len,
                                    is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        logits_result.append(prob_out.detach().cpu())
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    assert len(logits_result) == output_len + 1

    def _clone(x):
        r = []
        for xx in x:
            if isinstance(xx, torch.Tensor):
                r.append(xx.clone().detach())
        return r

    return logits_result
            # , _clone(model_part.layers_infer[0].cache_me))


def parse_args():
    # input_len, output_len, batch_size, bib_size
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--oup')
    parser.add_argument('--bs')
    parser.add_argument('--bib')
    parser.add_argument('--model')
    parser.add_argument('--skip', action='store_true')
    args = parser.parse_args()
    return args


def tensor_cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


if __name__ == '__main__':
    args = parse_args()
    input_len = int(args.inp)
    output_len = int(args.oup)
    batch_size = int(args.bs)
    bib_size = int(args.bib)
    model_dir = args.model

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    rank_id = 0
    world_size = 1
    p = '28765' if args.skip else '12456'
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:' + p, rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    test_data_length = (input_len + output_len) * batch_size * 10
    test_data_all = np.random.randint(100, 2000, test_data_length, dtype=np.int32)

    model_kvargs = {"tp_rank": 0, "world_size": 1, "weight_dir": model_dir,
                    "max_total_token_num": ((input_len + output_len + bib_size) * batch_size) * 5, "load_way": "HF",
                    "mode": [],
                    "max_req_num": batch_size, "max_seq_length": (input_len + output_len + bib_size) * 5, }

    if not args.skip:
        n_res = tppart_model_infer(model_kvargs, batch_size, input_len, output_len, test_data_all,
                                   mode='normal')

    model_kvargs['bib_size'] = bib_size
    model_kvargs['chunk_size'] = 8
    model_kvargs['mode'] = ['bib_route']

    b_res = tppart_model_infer(model_kvargs, batch_size, input_len, output_len, test_data_all,
                               mode='bib')


    def cmp(x, y):
        shape = 1
        xx = x.reshape(shape, -1)
        yy = y.reshape(shape, -1)
        cos = torch.nn.functional.cosine_similarity(xx, yy, dim=-1)
        err = (xx - yy).abs().max(dim=-1)
        return cos, err, (xx).abs().mean()


    # assert len(n_res) == len(b_res)
    for idx, (n, b) in enumerate(zip(n_res, b_res)):
        sim = cmp(n, b)
        print(f'iter {idx} sim = {sim}')
        print(f'iter {idx} norm nan = {n.isnan().sum()} bib nan = {b.isnan().sum()}')