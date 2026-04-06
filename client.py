import requests


class EcommerceEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str):
        return requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30).json()

    def step(self, action: dict):
        return requests.post(f"{self.base_url}/step", json=action, timeout=30).json()

    def state(self):
        return requests.get(f"{self.base_url}/state", timeout=30).json()

    def tasks(self):
        return requests.get(f"{self.base_url}/tasks", timeout=30).json()

    def baseline(self):
        return requests.get(f"{self.base_url}/baseline", timeout=120).json()
