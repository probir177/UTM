"""Load and save the local aikeys configuration.

The config is a YAML file at ``~/.config/aikeys/config.yaml`` (overridable
with the ``AIKEYS_CONFIG`` environment variable). It stores, per provider,
the list of keys plus routing settings. Keys never leave this machine and
are never committed to the repo.

Environment variables (e.g. ``GROQ_API_KEY``) are honoured as an additional
key source at load time, so the tool works even before ``add-key`` is run.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .providers import BUILTIN_PROVIDERS, Provider, builtin_provider


def config_path() -> Path:
    override = os.environ.get("AIKEYS_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    return Path(base).expanduser() / "aikeys" / "config.yaml"


class Config:
    """In-memory view of provider configuration."""

    def __init__(self, providers: dict[str, Provider]):
        self.providers = providers

    # ---- construction -------------------------------------------------
    @classmethod
    def default(cls) -> "Config":
        providers = {name: builtin_provider(name) for name in BUILTIN_PROVIDERS}
        return cls(providers)  # type: ignore[arg-type]

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, seeded with built-in providers.

        Missing file is fine — you get the built-in providers with no keys.
        Environment-variable keys are merged in on top.
        """
        cfg = cls.default()
        path = config_path()
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            for name, data in (raw.get("providers") or {}).items():
                cfg._merge_provider(name, data)
        cfg._merge_env_keys()
        return cfg

    def _merge_provider(self, name: str, data: dict) -> None:
        prov = self.providers.get(name)
        if prov is None:
            # Unknown/custom provider defined entirely in the config file.
            prov = Provider(
                name=name,
                base_url=data.get("base_url", ""),
                model=data.get("model", ""),
                env=data.get("env", ""),
                priority=int(data.get("priority", 100)),
                signup=data.get("signup", ""),
            )
            self.providers[name] = prov
        # Override built-in defaults with anything present in the file.
        if "base_url" in data:
            prov.base_url = data["base_url"]
        if "model" in data:
            prov.model = data["model"]
        if "env" in data:
            prov.env = data["env"]
        if "priority" in data:
            prov.priority = int(data["priority"])
        if "enabled" in data:
            prov.enabled = bool(data["enabled"])
        if "signup" in data:
            prov.signup = data["signup"]
        for key in data.get("keys") or []:
            if key and key not in prov.keys:
                prov.keys.append(key)

    def _merge_env_keys(self) -> None:
        for prov in self.providers.values():
            if not prov.env:
                continue
            val = os.environ.get(prov.env)
            if val and val not in prov.keys:
                prov.keys.append(val)

    # ---- persistence --------------------------------------------------
    def save(self) -> Path:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"providers": {n: p.to_dict() for n, p in self.providers.items()}}
        path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True))
        try:
            path.chmod(0o600)  # keys are secrets; keep them owner-only
        except OSError:
            pass
        return path

    # ---- mutation -----------------------------------------------------
    def add_key(self, provider_name: str, key: str) -> Provider:
        name = provider_name.lower()
        prov = self.providers.get(name)
        if prov is None:
            prov = builtin_provider(name)
            if prov is None:
                raise KeyError(
                    f"Unknown provider '{provider_name}'. Add it to the config "
                    "file with a base_url and model, then retry."
                )
            self.providers[name] = prov
        if key not in prov.keys:
            prov.keys.append(key)
        return prov

    def remove_key(self, provider_name: str, key: str) -> bool:
        prov = self.providers.get(provider_name.lower())
        if prov and key in prov.keys:
            prov.keys.remove(key)
            return True
        return False
