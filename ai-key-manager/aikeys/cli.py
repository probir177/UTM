"""Command-line interface for aikeys.

Commands:
    aikeys providers                 list known providers and priority order
    aikeys add-key <provider> <KEY>  store a key locally
    aikeys remove-key <provider> <KEY>
    aikeys keys                      list stored keys (masked)
    aikeys test                      ping every key, report working/dead
    aikeys chat "prompt"             auto-routed chat (rotation + fallback)
    aikeys chat -i                   interactive chat loop
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .client import ChatError, chat
from .config import Config, config_path
from .router import build_plan, mask_key, usable_providers
from .state import State

app = typer.Typer(
    add_completion=False,
    help="Manage your own AI provider keys and route chat to the free/cheapest one.",
    no_args_is_help=True,
)
console = Console()


def _load() -> Config:
    return Config.load()


@app.command()
def providers() -> None:
    """List known providers, their priority, model, and key count."""
    cfg = _load()
    table = Table(title="Providers (lower priority tried first)")
    table.add_column("Provider", style="bold")
    table.add_column("Priority", justify="right")
    table.add_column("Keys", justify="right")
    table.add_column("Model")
    table.add_column("Enabled", justify="center")
    for prov in sorted(cfg.providers.values(), key=lambda p: (p.priority, p.name)):
        table.add_row(
            prov.name,
            str(prov.priority),
            str(len(prov.keys)),
            prov.model,
            "✓" if prov.enabled else "-",
        )
    console.print(table)


@app.command("add-key")
def add_key(
    provider: str = typer.Argument(..., help="e.g. groq, gemini, openrouter"),
    key: str = typer.Argument(..., help="Your own API key from that provider"),
) -> None:
    """Store a key for a provider (saved locally, owner-only permissions)."""
    cfg = _load()
    try:
        prov = cfg.add_key(provider, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    path = cfg.save()
    console.print(
        f"[green]Added[/green] key {mask_key(key)} to "
        f"[bold]{prov.name}[/bold] ({len(prov.keys)} key(s)). Saved to {path}"
    )


@app.command("remove-key")
def remove_key(
    provider: str = typer.Argument(...),
    key: str = typer.Argument(..., help="Full key to remove"),
) -> None:
    """Remove a stored key."""
    cfg = _load()
    if cfg.remove_key(provider, key):
        cfg.save()
        console.print(f"[green]Removed[/green] key from {provider}.")
    else:
        console.print(f"[yellow]No matching key found for {provider}.[/yellow]")
        raise typer.Exit(1)


@app.command()
def keys() -> None:
    """List stored keys (masked)."""
    cfg = _load()
    table = Table(title="Stored keys")
    table.add_column("Provider", style="bold")
    table.add_column("#", justify="right")
    table.add_column("Key (masked)")
    total = 0
    for prov in sorted(cfg.providers.values(), key=lambda p: (p.priority, p.name)):
        for i, key in enumerate(prov.keys):
            table.add_row(prov.name, str(i), mask_key(key))
            total += 1
    if total == 0:
        console.print(
            "[yellow]No keys stored yet.[/yellow] Add one with "
            "`aikeys add-key <provider> <KEY>`. Free keys:"
        )
        for prov in sorted(cfg.providers.values(), key=lambda p: p.priority):
            if prov.signup:
                console.print(f"  • {prov.name}: {prov.signup}")
        return
    console.print(table)


@app.command()
def test(
    provider: str = typer.Option(None, "--provider", "-p", help="Test one provider only"),
) -> None:
    """Ping each stored key with a tiny prompt and report which ones work."""
    cfg = _load()
    plan = build_plan(cfg.providers, only=provider)  # test every key, cooling or not
    if not plan:
        console.print("[yellow]No usable keys to test.[/yellow]")
        raise typer.Exit(1)
    from .client import _post_chat, _error_reason, FALLBACK_STATUS
    from .state import cooldown_for_status

    import httpx

    state = State.load()
    messages = [{"role": "user", "content": "ping"}]
    ok = dead = 0
    for attempt in plan:
        name = attempt.provider.name
        label = f"{name} [{attempt.masked_key}]"
        try:
            resp = _post_chat(attempt, messages, None, 30.0, None)
        except httpx.HTTPError as exc:
            console.print(f"[red]✗[/red] {label}: {exc}")
            state.record_failure(name, attempt.key, 0)
            dead += 1
            continue
        if resp.status_code == 200:
            console.print(f"[green]✓[/green] {label}: working")
            state.record_success(name, attempt.key)
            ok += 1
        else:
            console.print(
                f"[red]✗[/red] {label}: HTTP {resp.status_code}: {_error_reason(resp)}"
            )
            state.record_failure(name, attempt.key, resp.status_code)
            if resp.status_code in FALLBACK_STATUS:
                state.set_cooldown(name, attempt.key, cooldown_for_status(resp.status_code))
            dead += 1
    state.save()
    console.print(f"\n[bold]{ok} working, {dead} not working.[/bold]")


@app.command()
def chat(
    prompt: str = typer.Argument(None, help="Your message. Omit with -i for chat loop."),
    interactive: bool = typer.Option(False, "-i", "--interactive", help="Chat loop"),
    provider: str = typer.Option(None, "--provider", "-p", help="Force one provider"),
    model: str = typer.Option(None, "--model", "-m", help="Override the model"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Print only the reply"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the full reply"),
) -> None:
    """Send a prompt; auto-route to the free/cheapest provider with fallback."""
    cfg = _load()
    state = State.load()
    if interactive:
        _chat_loop(cfg, state, provider, model, quiet, no_stream)
        return
    if not prompt:
        console.print("[red]Provide a prompt or use -i for interactive mode.[/red]")
        raise typer.Exit(1)
    _one_shot(cfg, state, prompt, provider, model, quiet, no_stream)


def _stream_printer():
    """Return an on_delta callback that writes raw tokens as they arrive."""

    def emit(delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()

    return emit


def _footer(result) -> str:
    return (
        f"[dim]— {result.provider} ({result.model}) "
        f"via {result.masked_key}, attempt #{result.attempts}[/dim]"
    )


def _one_shot(cfg: Config, state, prompt, provider, model, quiet, no_stream) -> None:
    from .client import chat as run_chat

    on_delta = None if no_stream else _stream_printer()
    try:
        result = run_chat(
            cfg.providers, prompt, only=provider, model=model,
            state=state, on_delta=on_delta,
        )
    except ChatError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if on_delta is not None:
        print()  # newline after the streamed reply
        if not quiet:
            console.print(_footer(result))
    elif quiet:
        print(result.text)
    else:
        console.print(result.text)
        console.print("\n" + _footer(result))


def _chat_loop(cfg: Config, state, provider, model, quiet, no_stream) -> None:
    from .client import chat as run_chat

    console.print("[dim]Interactive chat. Type 'exit' or Ctrl-D to quit.[/dim]")
    history: list[dict] = []
    while True:
        try:
            user = console.input("[bold cyan]you ›[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if user.strip().lower() in {"exit", "quit"}:
            return
        if not user.strip():
            continue
        history.append({"role": "user", "content": user})
        on_delta = None if no_stream else _stream_printer()
        if on_delta is not None:
            console.print("[bold green]ai ›[/bold green] ", end="")
        try:
            result = run_chat(
                cfg.providers, history, only=provider, model=model,
                state=state, on_delta=on_delta,
            )
        except ChatError as exc:
            console.print(f"[red]{exc}[/red]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": result.text})
        if on_delta is not None:
            print()
        else:
            console.print(f"[bold green]ai ›[/bold green] {result.text}")
        if not quiet:
            console.print(
                f"[dim]— {result.provider} ({result.model}) via {result.masked_key}[/dim]"
            )


@app.command()
def stats() -> None:
    """Show per-key usage: successes, failures, and any active cooldown."""
    cfg = _load()
    state = State.load()
    table = Table(title="Key usage & cooldowns")
    table.add_column("Provider", style="bold")
    table.add_column("Key", no_wrap=True)
    table.add_column("OK", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Last", justify="right")
    table.add_column("Cooldown", justify="right")
    rows = 0
    for prov in sorted(cfg.providers.values(), key=lambda p: (p.priority, p.name)):
        for key in prov.keys:
            s = state.stats_for(prov.name, key)
            remaining = state.cooldown_remaining(prov.name, key)
            cooldown = f"{int(remaining)}s" if remaining > 0 else "-"
            table.add_row(
                prov.name,
                mask_key(key),
                str(s["success"]),
                str(s["fail"]),
                str(s["last_status"] or "-"),
                cooldown,
            )
            rows += 1
    if rows == 0:
        console.print("[yellow]No keys stored yet.[/yellow]")
        return
    console.print(table)


@app.command()
def version() -> None:
    """Show the aikeys version and config file location."""
    console.print(f"aikeys {__version__}")
    console.print(f"config: {config_path()}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
