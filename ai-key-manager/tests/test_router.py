"""Tests for routing order, key rotation, and provider fallback.

These use a mock httpx transport so nothing touches the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from aikeys.client import ChatError, chat
from aikeys.providers import Provider
from aikeys.router import build_plan, mask_key, usable_providers


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep any state.save() during tests inside a temp dir, not ~/.config."""
    monkeypatch.setenv("AIKEYS_CONFIG", str(tmp_path / "config.yaml"))


def make_providers():
    return {
        "groq": Provider(
            name="groq",
            base_url="https://groq.test/v1",
            model="m-groq",
            env="GROQ_API_KEY",
            priority=10,
            keys=["gsk_aaa", "gsk_bbb"],
        ),
        "gemini": Provider(
            name="gemini",
            base_url="https://gemini.test/v1",
            model="m-gemini",
            env="GEMINI_API_KEY",
            priority=20,
            keys=["AIza_ccc"],
        ),
        "disabled": Provider(
            name="disabled",
            base_url="https://x.test/v1",
            model="m",
            env="",
            priority=1,
            enabled=False,
            keys=["k"],
        ),
        "nokey": Provider(
            name="nokey",
            base_url="https://y.test/v1",
            model="m",
            env="",
            priority=2,
            keys=[],
        ),
    }


def test_mask_key():
    assert mask_key("gsk_abcdef1234") == "gsk_…1234"
    assert mask_key("short") == "s…t"
    assert mask_key("") == "(empty)"


def test_usable_providers_orders_by_priority_and_skips_unusable():
    provs = usable_providers(make_providers())
    names = [p.name for p in provs]
    # disabled (no) and nokey (no keys) excluded; groq(10) before gemini(20)
    assert names == ["groq", "gemini"]


def test_build_plan_expands_every_key_in_order():
    plan = build_plan(make_providers())
    assert [(a.provider.name, a.key) for a in plan] == [
        ("groq", "gsk_aaa"),
        ("groq", "gsk_bbb"),
        ("gemini", "AIza_ccc"),
    ]


def test_build_plan_only_filter():
    plan = build_plan(make_providers(), only="gemini")
    assert [a.provider.name for a in plan] == ["gemini"]


class Recorder:
    """Builds a mock transport that returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen: list[tuple[str, str]] = []  # (url, auth key)

    def transport(self):
        def handler(request: httpx.Request) -> httpx.Response:
            auth = request.headers.get("authorization", "")
            key = auth.replace("Bearer ", "")
            self.seen.append((str(request.url), key))
            status, body = self._responses.pop(0)
            return httpx.Response(status, json=body)

        return httpx.MockTransport(handler)


def ok_body(text="hi"):
    return {"choices": [{"message": {"content": text}}]}


def err_body(msg):
    return {"error": {"message": msg}}


def test_first_key_succeeds():
    rec = Recorder([(200, ok_body("hello"))])
    result = chat(make_providers(), "hi", transport=rec.transport())
    assert result.text == "hello"
    assert result.provider == "groq"
    assert result.attempts == 1
    assert rec.seen[0][1] == "gsk_aaa"


def test_rotates_to_next_key_on_429():
    rec = Recorder([(429, err_body("rate limited")), (200, ok_body("ok"))])
    result = chat(make_providers(), "hi", transport=rec.transport())
    assert result.text == "ok"
    assert result.provider == "groq"
    assert result.attempts == 2
    assert [k for _, k in rec.seen] == ["gsk_aaa", "gsk_bbb"]


def test_falls_back_to_next_provider_when_keys_exhausted():
    rec = Recorder(
        [
            (429, err_body("rl")),
            (401, err_body("bad key")),
            (200, ok_body("from gemini")),
        ]
    )
    result = chat(make_providers(), "hi", transport=rec.transport())
    assert result.provider == "gemini"
    assert result.text == "from gemini"
    assert [k for _, k in rec.seen] == ["gsk_aaa", "gsk_bbb", "AIza_ccc"]


def test_all_fail_raises_with_details():
    rec = Recorder([(429, err_body("rl")), (429, err_body("rl")), (403, err_body("no"))])
    with pytest.raises(ChatError) as exc:
        chat(make_providers(), "hi", transport=rec.transport())
    assert len(exc.value.failures) == 3


def test_no_providers_raises():
    empty = {
        "p": Provider(name="p", base_url="", model="", env="", keys=[]),
    }
    with pytest.raises(ChatError):
        chat(empty, "hi")


def test_400_still_tries_next_provider():
    # A non-fallback status (400) on groq keys should still let gemini answer.
    rec = Recorder(
        [
            (400, err_body("bad model")),
            (400, err_body("bad model")),
            (200, ok_body("gemini saves the day")),
        ]
    )
    result = chat(make_providers(), "hi", transport=rec.transport())
    assert result.provider == "gemini"


# ---- cooldown / state ------------------------------------------------

from aikeys.state import State, key_id


def test_build_plan_moves_cooling_keys_to_the_end():
    provs = make_providers()
    state = State()
    # Put groq's first key on cooldown; it should be tried last.
    state.set_cooldown("groq", "gsk_aaa", 60, now=1000)
    plan = build_plan(provs, state=state, now=1000)
    order = [(a.provider.name, a.key) for a in plan]
    assert order == [
        ("groq", "gsk_bbb"),
        ("gemini", "AIza_ccc"),
        ("groq", "gsk_aaa"),  # cooling → last resort
    ]


def test_expired_cooldown_is_not_skipped():
    state = State()
    state.set_cooldown("groq", "gsk_aaa", 60, now=1000)
    # 61s later the cooldown has expired.
    assert state.in_cooldown("groq", "gsk_aaa", now=1061) is False


def test_success_records_stats_and_clears_cooldown():
    provs = make_providers()
    state = State()
    state.set_cooldown("groq", "gsk_aaa", 60, now=1000)
    # Only the cooling key exists in a single-provider plan to force its use.
    rec = Recorder([(200, ok_body("ok"))])
    result = chat(
        provs, "hi", only="groq", transport=rec.transport(), state=state, now=1000
    )
    assert result.text == "ok"
    entry = state.entries[key_id("groq", "gsk_bbb")]
    assert entry["success"] == 1
    # The used key's cooldown is cleared on success.
    assert state.in_cooldown("groq", "gsk_bbb", now=1000) is False


def test_429_sets_cooldown_on_the_failing_key():
    provs = make_providers()
    state = State()
    rec = Recorder([(429, err_body("rate limited")), (200, ok_body("ok"))])
    chat(provs, "hi", transport=rec.transport(), state=state, now=2000)
    # The first key got rate limited → cooldown active.
    assert state.in_cooldown("groq", "gsk_aaa", now=2000) is True
    assert state.entries[key_id("groq", "gsk_aaa")]["fail"] == 1


# ---- streaming -------------------------------------------------------


def sse(*deltas):
    """Build an OpenAI-style SSE body from content deltas."""
    lines = []
    for d in deltas:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


class StreamRecorder:
    def __init__(self, status, body_text):
        self.status = status
        self.body_text = body_text

    def transport(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if self.status == 200:
                return httpx.Response(200, text=self.body_text)
            return httpx.Response(self.status, json=err_body("nope"))

        return httpx.MockTransport(handler)


def test_streaming_calls_on_delta_and_returns_full_text():
    got: list[str] = []
    rec = StreamRecorder(200, sse("Hel", "lo ", "world"))
    result = chat(
        make_providers(),
        "hi",
        only="groq",
        transport=rec.transport(),
        on_delta=got.append,
    )
    assert "".join(got) == "Hello world"
    assert result.text == "Hello world"
