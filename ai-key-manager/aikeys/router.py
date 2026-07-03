"""Routing logic: decide the order in which providers and keys are tried.

Pure, side-effect-free logic so it can be unit tested without any network.
The client (``client.py``) walks the plan returned here and stops at the
first attempt that succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass

from .providers import Provider


@dataclass
class Attempt:
    """A single (provider, key) pair to try, in order."""

    provider: Provider
    key: str
    key_index: int

    @property
    def masked_key(self) -> str:
        return mask_key(self.key)


def mask_key(key: str) -> str:
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return key[0] + "…" + key[-1]
    return f"{key[:4]}…{key[-4:]}"


def usable_providers(providers: dict[str, Provider]) -> list[Provider]:
    """Enabled providers that have at least one key and a base_url, sorted.

    Sort is by ``priority`` (low first = cheapest/free first), then name for
    a stable, predictable order.
    """
    ready = [
        p
        for p in providers.values()
        if p.enabled and p.keys and p.base_url and p.model
    ]
    return sorted(ready, key=lambda p: (p.priority, p.name))


def build_plan(
    providers: dict[str, Provider],
    only: str | None = None,
) -> list[Attempt]:
    """Return the ordered list of attempts.

    For each provider (cheapest first), every key is queued in turn so that
    rate-limited or dead keys fall through to the next key, then the next
    provider.

    ``only`` restricts the plan to a single named provider.
    """
    plan: list[Attempt] = []
    for prov in usable_providers(providers):
        if only and prov.name != only.lower():
            continue
        for idx, key in enumerate(prov.keys):
            plan.append(Attempt(provider=prov, key=key, key_index=idx))
    return plan
