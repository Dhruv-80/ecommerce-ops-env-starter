import requests


class EcommerceEnvClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, json_payload: dict | None = None, timeout: int | None = None):
        response = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=json_payload,
            timeout=timeout if timeout is not None else self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self):
        return self._request("GET", "/health")

    def reset(self, task_id: str):
        return self._request("POST", "/reset", json_payload={"task_id": task_id})

    def step(self, action: dict):
        return self._request("POST", "/step", json_payload=action)

    def state(self):
        return self._request("GET", "/state")

    def tasks(self):
        return self._request("GET", "/tasks")

    def grader(self):
        return self._request("POST", "/grader")

    def baseline(self):
        return self._request("GET", "/baseline", timeout=120)
