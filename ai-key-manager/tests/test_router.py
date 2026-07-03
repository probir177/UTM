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
