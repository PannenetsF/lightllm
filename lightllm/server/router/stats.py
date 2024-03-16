import os.path
import time

from lightllm.server.io_struct import Batch, ReqRunStatus
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class Stats:

    def __init__(self, log_status, log_stats_interval) -> None:
        self.log_stats = log_status
        self.log_stats_interval = log_stats_interval
        self.last_log_time = time.time()
        self.all_tokens = 0
        self.output_tokens = 0
        self.prompt_tokens = 0
        return
    
    def count_prompt_tokens(self, run_batch):
        if self.log_stats:
            tokens = run_batch.input_tokens()
            self.prompt_tokens += tokens
            self.all_tokens += tokens
        return
    
    def count_output_tokens(self, run_batch):
        if self.log_stats:
            tokens = len(run_batch.reqs)
            self.output_tokens += tokens
            self.all_tokens += tokens
        return

    def print_stats(self):
        if not self.log_stats:
            return

        now = time.time()
        if now - self.last_log_time > self.log_stats_interval:
            logger.debug(f"Avg tokens(prompt+generate) throughput: {self.all_tokens/(now-self.last_log_time):8.3f} tokens/s\n"
                         f"Avg prompt tokens throughput:           {self.prompt_tokens/(now-self.last_log_time):8.3f} tokens/s\n"
                         f"Avg generate tokens throughput:         {self.output_tokens/(now-self.last_log_time):8.3f} tokens/s")
            self.all_tokens = 0
            self.output_tokens = 0
            self.prompt_tokens = 0
            self.last_log_time = now
        return


class Statistics:
    def __init__(self, save_name):
        if save_name is None:
            self.enable = False
        else:
            self.enable = True
            self.save_name = save_name
            if os.path.exists(self.save_name):
                os.remove(self.save_name)
        self.data = []
        # todo: support multi process
        logger.info(f"bib statistics is enabled, save to {self.save_name}")

    def add(self, batch: Batch):
        if not self.enable:
            return
        time_now = time.time()
        running_reqs = [req for req in batch.reqs if req.req_status == ReqRunStatus.RUNNING]
        running_req_lengths = [req.cur_kv_len for req in running_reqs]
        self.data.append((time_now, {'batch_size': len(running_reqs), 'running_lengths': running_req_lengths}))

        self.save()

    def save(self):
        _data = self.data
        self.data = []
        with open(self.save_name, 'a') as f:
            for d in _data:
                f.write(f"{d}\n")