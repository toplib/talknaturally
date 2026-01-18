import requests
import json
from .config import BASE_URL, MODEL, TIMEOUT


class LLMClient:
    def __init__(self, base_url=BASE_URL, model=MODEL):
        self.base_url = base_url
        self.model = model

    def chat_stream(self, messages, temperature=0.7):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        with requests.post(url, json=payload, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                if line.startswith(b"data: "):
                    data = line[6:]

                    if data == b"[DONE]":
                        break

                    event = json.loads(data)

                    delta = (
                        event["choices"][0]
                        .get("delta", {})
                        .get("content")
                    )

                    if delta:
                        yield delta
