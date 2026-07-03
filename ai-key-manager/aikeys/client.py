"""OpenAI-compatible chat client with key rotation and provider fallback.

Given a routing plan (from ``router.build_plan``), try each (provider, key)
attempt until one returns a completion. Rate-limit (429), quota, and auth
(401/403) failures advance to the next attempt; other errors are recorded
and also fall through so a single flaky provider never blocks the request.

Optional features:
  * ``state`` — records success/failure stats and puts rate-limited keys on a
    cooldown so future requests prefer fresh keys.
  * ``on_delta`` — when provided, the request is streamed and each text chunk
    is passed to the callback as it arrives (token-by-token output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .router import Attempt, build_plan
from .providers import Provider
from .state import cooldown_for_status

# Status codes that mean "this key/provider can't serve us right now" — try
# the next one rather than giving up.
FALLBACK_STATUS = {401, 402, 403, 429, 500, 502, 503, 529}


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    masked_key: str
    attempts: int


@dataclass
class ChatError(Exception):
    message: str
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        detail = "\n  ".join(self.failures)
        return f"{self.message}\n  {detail}" if detail else self.message


def _post_chat(
    attempt: Attempt,
    messages: list[dict],
    model: str | None,
    timeout: float,
    transport: httpx.BaseTransport | None,
) -> httpx.Response:
    """Non-streaming request. Also used by the `test` command."""
    prov = attempt.provider
    url = prov.base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model or prov.model, "messages": messages}
    headers = {
        "Authorization": f"Bearer {attempt.key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, transport=transport) as client:
        return client.post(url, json=payload, headers=headers)


def _run_attempt(
    attempt: Attempt,
    messages: list[dict],
    model: str | None,
    timeout: float,
    transport: httpx.BaseTransport | None,
    on_delta: Callable[[str], None] | None,
) -> tuple[int, str]:
    """Perform one request. Returns (status_code, text_or_error_reason).

    On success returns (200, full_text). On failure returns (status, reason).
    Streams via ``on_delta`` when provided.
    """
    if on_delta is None:
        resp = _post_chat(attempt, messages, model, timeout, transport)
        if resp.status_code == 200:
            return 200, _extract_text(resp.json())
        return resp.status_code, _error_reason(resp)

    prov = attempt.provider
    url = prov.base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model or prov.model, "messages": messages, "stream": True}
    headers = {
        "Authorization": f"Bearer {attempt.key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, transport=transport) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()  # load body so _error_reason can parse it
                return resp.status_code, _error_reason(resp)
            chunks: list[str] = []
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                delta = _extract_delta(obj)
                if delta:
                    chunks.append(delta)
                    on_delta(delta)
            return 200, "".join(chunks)


def _extract_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some providers return content parts
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def _extract_delta(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def chat(
    providers: dict[str, Provider],
    prompt: str | list[dict],
    only: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
    state=None,
    now: float | None = None,
    on_delta: Callable[[str], None] | None = None,
) -> ChatResult:
    """Send a chat request, rotating keys/providers until one succeeds.

    ``prompt`` may be a plain string (treated as a single user message) or a
    full OpenAI-style ``messages`` list. ``transport`` is injectable so tests
    can supply a mock without touching the network.
    """
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    plan = build_plan(providers, only=only, state=state, now=now)
    if not plan:
        raise ChatError(
            "No usable provider. Add a key with `aikeys add-key <provider> <KEY>` "
            "or set the matching env var (e.g. GROQ_API_KEY)."
        )

    failures: list[str] = []
    try:
        for i, attempt in enumerate(plan, start=1):
            prov_name = attempt.provider.name
            label = f"{prov_name} [{attempt.masked_key}]"
            try:
                status, text = _run_attempt(
                    attempt, messages, model, timeout, transport, on_delta
                )
            except httpx.HTTPError as exc:
                failures.append(f"{label}: network error: {exc}")
                if state is not None:
                    state.record_failure(prov_name, attempt.key, 0, now)
                continue

            if status == 200:
                if state is not None:
                    state.record_success(prov_name, attempt.key, now)
                return ChatResult(
                    text=text,
                    provider=prov_name,
                    model=model or attempt.provider.model,
                    masked_key=attempt.masked_key,
                    attempts=i,
                )

            failures.append(f"{label}: HTTP {status}: {text}")
            if state is not None:
                state.record_failure(prov_name, attempt.key, status, now)
                if status in FALLBACK_STATUS:
                    state.set_cooldown(
                        prov_name, attempt.key, cooldown_for_status(status), now
                    )
            # Whether or not it's a known fallback status, move on and try the
            # next provider — a different provider may accept the request.
    finally:
        if state is not None:
            state.save()

    raise ChatError("All providers/keys failed.", failures)


def _error_reason(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return err.get("message", str(err))[:200]
        if err:
            return str(err)[:200]
    return str(data)[:200]
