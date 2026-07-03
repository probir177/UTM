"""aikeys — manage your own AI provider API keys and route chat requests.

This package helps you keep multiple legitimate free-tier API keys (Gemini,
Groq, OpenRouter, Mistral, HuggingFace, Cohere, and any custom provider),
rotate between them when one is rate limited, and automatically send each
request to the cheapest/free provider available.

It does NOT generate, crack, or forge keys. You bring keys you obtained
yourself from each provider's own website.
"""

__version__ = "0.1.0"
