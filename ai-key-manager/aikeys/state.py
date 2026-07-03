"""Persistent per-key runtime state: cooldowns and usage stats.

Stored at ``~/.config/aikeys/state.json`` (next to the config). This is what
makes free keys last: when a key is rate limited, it's put on a short cooldown
so future requests skip it and prefer a fresh key/provider, coming back to it
only once the cooldown expires.

Keys are never stored in the state file — each key is referenced by a short
SHA-256 fingerprint, so the raw secret stays only in the config.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .config import config_path

# Cooldown lengths (seconds) by failure category.
COOLDOWN_RATE_LIMIT = 60      # 429 / 5xx overload — try again soon
COOLDOWN_AUTH = 3600          # 401/402/403 — key likely exhausted or invalid


def state_path() -> Path:
    return config_path().parent / "state.json"


def key_id(provider: str, key: str) -> str:
    """Stable, non-reversible id for a (provider, key) pair."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{provider}:{digest}"


def cooldown_for_status(status: int) -> int:
    if status in (401, 402, 403):
        return COOLDOWN_AUTH
    return COOLDOWN_RATE_LIMIT


def _now(now: float | None) -> float:
    return time.time() if now is None else now


class State:
    def __init__(self, entries: dict | None = None):
        # key_id -> {cooldown_until, success, fail, last_status, last_used}
        self.entries: dict[str, dict] = entries or {}

    @classmethod
    def load(cls) -> "State":
        path = state_path()
        if path.exists():
            try:
                return cls(json.loads(path.read_text()))
            except (ValueError, OSError):
                return cls()
        return cls()

    def save(self) -> Path:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    def _entry(self, kid: str) -> dict:
        return self.entries.setdefault(
            kid,
            {"cooldown_until": 0, "success": 0, "fail": 0, "last_status": None, "last_used": 0},
        )

    # ---- cooldown -----------------------------------------------------
    def in_cooldown(self, provider: str, key: str, now: float | None = None) -> bool:
        entry = self.entries.get(key_id(provider, key))
        return bool(entry and entry.get("cooldown_until", 0) > _now(now))

    def cooldown_remaining(self, provider: str, key: str, now: float | None = None) -> float:
        entry = self.entries.get(key_id(provider, key))
        if not entry:
            return 0.0
        return max(0.0, entry.get("cooldown_until", 0) - _now(now))

    def set_cooldown(self, provider: str, key: str, seconds: float, now: float | None = None) -> None:
        self._entry(key_id(provider, key))["cooldown_until"] = _now(now) + seconds

    def clear_cooldown(self, provider: str, key: str) -> None:
        entry = self.entries.get(key_id(provider, key))
        if entry:
            entry["cooldown_until"] = 0

    # ---- stats --------------------------------------------------------
    def record_success(self, provider: str, key: str, now: float | None = None) -> None:
        entry = self._entry(key_id(provider, key))
        entry["success"] += 1
        entry["last_status"] = 200
        entry["last_used"] = _now(now)
        entry["cooldown_until"] = 0

    def record_failure(self, provider: str, key: str, status: int, now: float | None = None) -> None:
        entry = self._entry(key_id(provider, key))
        entry["fail"] += 1
        entry["last_status"] = status
        entry["last_used"] = _now(now)

    def stats_for(self, provider: str, key: str) -> dict:
        return self.entries.get(
            key_id(provider, key),
            {"cooldown_until": 0, "success": 0, "fail": 0, "last_status": None, "last_used": 0},
        )
