import time

from lightllm.server.io_struct import Batch
from lightllm.server.router.req_queue import ReqQueue, ReqRunStatus

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class BibStatistics:
    def __init__(self, save_name):
        if save_name is None:
            self.enable = False
        else:
            self.enable = True
            self.save_name = save_name
        self.data = []
        # todo: support multi process
        logger.info(f"bib statistics is enabled, save to {self.save_name}")

    def add(self, batch: Batch):
        if not self.enable:
            return
        time_now = time.time()
        running_reqs = [req for req in batch.reqs if req.req_status == ReqRunStatus.RUNNING]
        running_req_lengths = [req.input_len + req.cur_kv_len for req in running_reqs]
        self.data.append((time_now, {'batch_size': len(running_reqs), 'running_lengths': running_req_lengths}))

        self.save()

    def save(self):
        _data = self.data
        self.data = []
        with open(self.save_name, 'a') as f:
            for d in _data:
                f.write(f"{d}\n")
