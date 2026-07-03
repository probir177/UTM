"""Built-in provider registry.

Nearly all of these providers expose an OpenAI-compatible
``/chat/completions`` endpoint, so a single client works for all of them.
Adding a new provider is just adding an entry here (or in the user's config).

Each provider defines:
    base_url    OpenAI-compatible API base (ends with the version path).
    model       A sensible free/default model to use when none is given.
    env         Environment variable checked as a key fallback.
    priority    Lower is tried first. Free tiers get low numbers.
    key_prefix  Optional expected key prefix (used only for a friendly hint).
    signup      Where to get a real key for free (shown in help/README).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    env: str
    priority: int = 100
    key_prefix: str = ""
    signup: str = ""
    enabled: bool = True
    keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "priority": self.priority,
            "base_url": self.base_url,
            "model": self.model,
            "env": self.env,
            "signup": self.signup,
            "keys": list(self.keys),
        }


# Built-in defaults. Priorities put the most generous free tiers first.
BUILTIN_PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        env="GROQ_API_KEY",
        priority=10,
        key_prefix="gsk_",
        signup="https://console.groq.com/keys",
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.0-flash",
        env="GEMINI_API_KEY",
        priority=20,
        key_prefix="AIza",
        signup="https://aistudio.google.com/app/apikey",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct:free",
        env="OPENROUTER_API_KEY",
        priority=30,
        key_prefix="sk-or-",
        signup="https://openrouter.ai/keys",
    ),
    "mistral": Provider(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        model="mistral-small-latest",
        env="MISTRAL_API_KEY",
        priority=40,
        signup="https://console.mistral.ai/api-keys",
    ),
    "huggingface": Provider(
        name="huggingface",
        base_url="https://router.huggingface.co/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        env="HF_TOKEN",
        priority=50,
        key_prefix="hf_",
        signup="https://huggingface.co/settings/tokens",
    ),
    "cohere": Provider(
        name="cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        model="command-r-08-2024",
        env="COHERE_API_KEY",
        priority=60,
        signup="https://dashboard.cohere.com/api-keys",
    ),
    "together": Provider(
        name="together",
        base_url="https://api.together.xyz/v1",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        env="TOGETHER_API_KEY",
        priority=70,
        signup="https://api.together.ai/settings/api-keys",
    ),
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        env="DEEPSEEK_API_KEY",
        priority=80,
        key_prefix="sk-",
        signup="https://platform.deepseek.com/api_keys",
    ),
}


def builtin_provider(name: str) -> Provider | None:
    """Return a fresh copy of a built-in provider definition, if known."""
    proto = BUILTIN_PROVIDERS.get(name.lower())
    if proto is None:
        return None
    return Provider(
        name=proto.name,
        base_url=proto.base_url,
        model=proto.model,
        env=proto.env,
        priority=proto.priority,
        key_prefix=proto.key_prefix,
        signup=proto.signup,
    )
