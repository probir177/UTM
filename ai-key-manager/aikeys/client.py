"""OpenAI-compatible chat client with key rotation and provider fallback.

Given a routing plan (from ``router.build_plan``), try each (provider, key)
attempt until one returns a completion. Rate-limit (429), quota, and auth
(401/403) failures advance to the next attempt; other errors are recorded
and also fall through so a single flaky provider never blocks the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .router import Attempt, build_plan
from .providers import Provider

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
    prov = attempt.provider
    url = prov.base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model or prov.model, "messages": messages}
    headers = {
        "Authorization": f"Bearer {attempt.key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, transport=transport) as client:
        return client.post(url, json=payload, headers=headers)


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


def chat(
    providers: dict[str, Provider],
    prompt: str | list[dict],
    only: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """Send a chat request, rotating keys/providers until one succeeds.

    ``prompt`` may be a plain string (treated as a single user message) or a
    full OpenAI-style ``messages`` list. ``transport`` is injectable so tests
    can supply a mock without touching the network.
    """
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    plan = build_plan(providers, only=only)
    if not plan:
        raise ChatError(
            "No usable provider. Add a key with `aikeys add-key <provider> <KEY>` "
            "or set the matching env var (e.g. GROQ_API_KEY)."
        )

    failures: list[str] = []
    for i, attempt in enumerate(plan, start=1):
        label = f"{attempt.provider.name} [{attempt.masked_key}]"
        try:
            resp = _post_chat(attempt, messages, model, timeout, transport)
        except httpx.HTTPError as exc:
            failures.append(f"{label}: network error: {exc}")
            continue

        if resp.status_code == 200:
            text = _extract_text(resp.json())
            return ChatResult(
                text=text,
                provider=attempt.provider.name,
                model=model or attempt.provider.model,
                masked_key=attempt.masked_key,
                attempts=i,
            )

        reason = _error_reason(resp)
        failures.append(f"{label}: HTTP {resp.status_code}: {reason}")
        if resp.status_code in FALLBACK_STATUS:
            continue
        # Unexpected client error (e.g. 400 bad model) — still try the next
        # provider, since a different provider may accept the request.

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
