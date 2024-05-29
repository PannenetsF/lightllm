import threading
from uuid import NAMESPACE_DNS, uuid3


class SessionIDGenerator:
    def __init__(self):
        self.current_id = 0
        self.lock = threading.Lock()

    def generate_session_id(self):
        with self.lock:
            id = self.current_id
            self.current_id += 1
        hex = uuid3(NAMESPACE_DNS, f"lightllm_session_{id}").hex
        return hex
