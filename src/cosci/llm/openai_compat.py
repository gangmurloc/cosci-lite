"""OpenAI-compatible chat backend (vLLM, Ollama, LM Studio, llama.cpp server)."""
from __future__ import annotations

import time
from typing import Any

import requests

from ..utils import strip_think
from .base import LLMBackend, LLMError


class OpenAICompatBackend(LLMBackend):
    type_name = "openai_compatible"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base_url = str(self.cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.json_mode = str(self.cfg.get("json_mode", "schema")).lower()
        self._no_reasoning = False   # set once the server is seen spending the whole answer on reasoning

    def _payload(self, prompt: str, system: str, schema: dict[str, Any] | None, role: str) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.cfg.get("model"),
            "messages": messages,
            "temperature": float(self.cfg.get("temperature", 0.7)),
            "max_tokens": int(self.cfg.get("max_tokens", 6000)),
        }
        if role in ("compare", "debate", "review", "novelty", "dedupe"):
            payload["temperature"] = float(self.cfg.get("judge_temperature", 0.2))
        if schema is not None and self.json_mode == "schema":
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": role or "output", "schema": schema}}
        elif schema is not None and self.json_mode == "object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.cfg.get("extra_body") or {})
        if self._no_reasoning:
            payload["reasoning_effort"] = "none"
        return payload

    def _call(self, prompt: str, system: str, schema: dict[str, Any] | None, role: str):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.get('api_key', 'EMPTY')}"}
        connect_wait = float(self.cfg.get("connect_wait", 180))
        waited, delay, soft = 0.0, 2.0, 0
        last = ""
        while True:
            payload = self._payload(prompt, system, schema, role)
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=int(self.cfg.get("timeout", 600)))
            except requests.ConnectionError as e:
                # server restarting or not up yet: keep trying for up to connect_wait seconds in total
                last = f"connection error: {e}"
                if waited >= connect_wait:
                    break
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 30.0)
                continue
            except requests.RequestException as e:
                last = f"request error: {e}"
                soft += 1
                if soft >= 3:
                    break
                time.sleep(2 * soft)
                continue
            if r.status_code == 400 and "response_format" in payload and self.json_mode != "none":
                # server does not support this response_format -> degrade once and retry
                self.json_mode = "object" if self.json_mode == "schema" else "none"
                last = f"400 with response_format; degrading json_mode to {self.json_mode}"
                continue
            if r.status_code >= 500 or r.status_code == 429:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                soft += 1
                if soft >= 3:
                    break
                time.sleep(2 * soft)
                continue
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            d = r.json()
            try:
                msg = d["choices"][0]["message"]
                text = msg.get("content") or ""
            except (KeyError, IndexError) as e:
                raise LLMError(f"unexpected response: {str(d)[:300]}") from e
            if not text.strip() and (msg.get("reasoning") or msg.get("reasoning_content")):
                # A thinking model (e.g. Qwen on Ollama) used the whole token budget on reasoning and gave no
                # answer. Ollama ignores vLLM's chat_template_kwargs, so ask for no reasoning and retry.
                if self._no_reasoning:
                    raise LLMError("empty answer: the model spent max_tokens on reasoning even with "
                                   "reasoning_effort=none; raise max_tokens or disable thinking in extra_body")
                self._no_reasoning = True
                continue
            return strip_think(text), None, {"usage": d.get("usage", {})}
        raise LLMError(f"{self.base_url}: {last}")

    def close(self) -> None:
        """Ollama with `unload_on_exit: true`: take the model off the GPU now instead of after keep_alive.

        Lets a run hold the GPU with a long keep_alive (no idle gaps in which another job grabs the card and
        collides with the reload) and still free it the moment the run ends.
        """
        if not self.cfg.get("unload_on_exit"):
            return
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        model = self.cfg.get("model")
        try:
            loaded = requests.get(f"{root}/api/ps", timeout=10).json().get("models") or []
            # only if it is in memory: an unload request must never be what loads the model
            if any(model in (m.get("name"), m.get("model")) for m in loaded):
                requests.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except (requests.RequestException, ValueError):
            pass   # the server is gone or is not Ollama; keep_alive will expire on its own
