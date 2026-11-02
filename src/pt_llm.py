"""Async chat client for Fireworks + OpenRouter with disk cache, retries and accounting."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential_jitter

from pt_prompt import ROOT

load_dotenv(ROOT / ".env", override=False)

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")  # any OpenAI-compatible router
CACHE_DIR = Path("/tmp/pawtriage/cache") if os.environ.get("VERCEL") else ROOT / "results" / "cache"


def provider_for(model: str) -> str:
    return "fireworks" if model.startswith("accounts/") else "openrouter"


class LLM:
    def __init__(self, concurrency: int = 16, cache_name: str = "llm"):
        self.sem = asyncio.Semaphore(concurrency)
        self.clients: dict[str, AsyncOpenAI] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{cache_name}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in open(self.cache_path):
                try:
                    d = json.loads(line)
                    self.cache[d["k"]] = d["v"]
                except Exception:
                    pass

    def client(self, provider: str) -> AsyncOpenAI:
        if provider not in self.clients:
            if provider == "fireworks":
                key = os.environ.get("FIREWORKS_API_KEY")
                url = FIREWORKS_URL
            else:
                key = os.environ.get("OPENROUTER_API_KEY")
                url = OPENROUTER_URL
            if not key:
                raise RuntimeError(f"Missing API key for {provider}; fill ~/pawtriage/.env")
            self.clients[provider] = AsyncOpenAI(api_key=key, base_url=url, timeout=180, max_retries=0)
        return self.clients[provider]

    @staticmethod
    def _key(model, messages, temperature, max_tokens, extra) -> str:
        h = hashlib.sha256(json.dumps([model, messages, temperature, max_tokens, extra], sort_keys=True).encode())
        return h.hexdigest()

    async def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024,
                   cache_key: str | None = None, json_mode: bool = False, use_cache: bool = True) -> dict:
        k = self._key(model, cache_key or messages, temperature, max_tokens, json_mode)
        if use_cache and k in self.cache:
            return {**self.cache[k], "cached": True}
        async with self.sem:
            res = await self._call(model, messages, temperature, max_tokens, json_mode)
        if use_cache:
            self.cache[k] = res
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"k": k, "v": res}) + "\n")
        return {**res, "cached": False}

    @retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=2, max=60), reraise=True,
           retry=retry_if_not_exception_type(BadRequestError))
    async def _call(self, model, messages, temperature, max_tokens, json_mode) -> dict:
        kw = {}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if provider_for(model) == "fireworks":
            # GLM-5.3 is thinking-only ("none" is rejected); keep thinking minimal.
            kw["extra_body"] = {"reasoning_effort": os.environ.get("PT_REASONING_EFFORT", "low")}
        t0 = time.perf_counter()
        r = await self.client(provider_for(model)).chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kw)
        dt = (time.perf_counter() - t0) * 1000
        u = r.usage
        return {"text": r.choices[0].message.content or "", "latency_ms": round(dt),
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0, "model": model}


def run(coro):
    """Run a coroutine from a notebook or script."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import nest_asyncio  # pragma: no cover
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)
