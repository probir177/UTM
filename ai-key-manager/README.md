# aikeys — AI API key manager & smart router

A small command-line tool that manages **your own** free-tier API keys for AI
providers (Gemini, Groq, OpenRouter, Mistral, HuggingFace, Cohere, and any
other OpenAI-compatible provider), **rotates** between multiple keys when one
hits a rate limit, and automatically **routes** each request to the
free/cheapest provider available.

> **This tool does not generate, crack, or forge keys.** There is no such
> thing as a working "generated" API key — providers validate every key
> against their own database, so a made-up string never works and tools that
> claim otherwise are scams or malware. What `aikeys` does is let you use the
> **real, free keys you get yourself** from each provider more efficiently:
> pool several of them, rotate on rate limits, and fall back across providers
> so you rarely run out.

## How it works

Almost every provider offers an OpenAI-compatible `/chat/completions`
endpoint, so one client drives all of them. `aikeys` keeps an ordered list of
providers (cheapest/free first). For each request it tries the first
provider's first key; on a rate limit (`429`), quota, or auth error it moves
to that provider's next key, then to the next provider — until one answers.

### Smart cooldown (makes free keys last)

When a key hits a rate limit, `aikeys` puts it on a short **cooldown** and
prefers fresh keys on the next request, coming back to it only after the
cooldown expires (60s for rate limits, 1h for auth/quota errors). This spreads
load across all your keys and providers so you rarely run out of free quota.
State lives at `~/.config/aikeys/state.json` and stores only a hashed
fingerprint of each key — never the key itself. See usage with `aikeys stats`.

### Streaming

Replies stream token-by-token by default. Pass `--no-stream` to wait for the
full reply instead (useful when piping output).

## Get free keys (legitimately)

Sign up (free) and paste the key with `aikeys add-key`:

| Provider    | Get a free key |
|-------------|----------------|
| Groq        | https://console.groq.com/keys |
| Gemini      | https://aistudio.google.com/app/apikey |
| OpenRouter  | https://openrouter.ai/keys |
| Mistral     | https://console.mistral.ai/api-keys |
| HuggingFace | https://huggingface.co/settings/tokens |
| Cohere      | https://dashboard.cohere.com/api-keys |
| Together    | https://api.together.ai/settings/api-keys |
| DeepSeek    | https://platform.deepseek.com/api_keys |

## Install

```bash
cd ai-key-manager
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # gives you the `aikeys` command
```

(Without `pip install -e .` you can still run it as `python -m aikeys.cli`.)

## Usage

```bash
# See the built-in providers and the order they're tried in
aikeys providers

# Add one or more of YOUR OWN keys (rotates automatically)
aikeys add-key groq gsk_xxxxxxxx
aikeys add-key groq gsk_another_one
aikeys add-key gemini AIzaxxxxxxxx

# List stored keys (masked)
aikeys keys

# Check which keys actually work right now
aikeys test

# See per-key usage: successes, failures, and active cooldowns
aikeys stats

# Ask something — streams from the free/cheapest working provider
aikeys chat "Explain quantum entanglement in one line"

# Wait for the whole reply instead of streaming
aikeys chat "Summarize this" --no-stream

# Force a specific provider or model
aikeys chat "hello" --provider gemini --model gemini-2.0-flash

# Interactive chat loop (keeps conversation history)
aikeys chat -i

# Machine-friendly output (reply only)
aikeys chat "2+2?" --quiet
```

Every reply shows which provider and key served it, e.g.
`— groq (llama-3.3-70b-versatile) via gsk_…1234, attempt #1`.

## Adding any other provider

Any OpenAI-compatible provider works — add it to
`~/.config/aikeys/config.yaml` (see `config.example.yaml`):

```yaml
providers:
  together:
    enabled: true
    priority: 70
    base_url: https://api.together.xyz/v1
    model: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
    env: TOGETHER_API_KEY
    keys:
      - your_together_key
```

## Where keys are stored

`~/.config/aikeys/config.yaml` (override with `AIKEYS_CONFIG`), written with
owner-only (`600`) permissions. Keys are **never** committed. Environment
variables like `GROQ_API_KEY` or `GEMINI_API_KEY` are also picked up
automatically as an extra key source.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest        # router/rotation/fallback tests, no network needed
```
