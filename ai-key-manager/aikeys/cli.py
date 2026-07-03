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
    plan = build_plan(cfg.providers, only=provider)
    if not plan:
        console.print("[yellow]No usable keys to test.[/yellow]")
        raise typer.Exit(1)
    from .client import _post_chat  # reuse the same request path

    import httpx

    messages = [{"role": "user", "content": "ping"}]
    ok = dead = 0
    for attempt in plan:
        label = f"{attempt.provider.name} [{attempt.masked_key}]"
        try:
            resp = _post_chat(attempt, messages, None, 30.0, None)
        except httpx.HTTPError as exc:
            console.print(f"[red]✗[/red] {label}: {exc}")
            dead += 1
            continue
        if resp.status_code == 200:
            console.print(f"[green]✓[/green] {label}: working")
            ok += 1
        else:
            from .client import _error_reason

            console.print(
                f"[red]✗[/red] {label}: HTTP {resp.status_code}: {_error_reason(resp)}"
            )
            dead += 1
    console.print(f"\n[bold]{ok} working, {dead} not working.[/bold]")


@app.command()
def chat(
    prompt: str = typer.Argument(None, help="Your message. Omit with -i for chat loop."),
    interactive: bool = typer.Option(False, "-i", "--interactive", help="Chat loop"),
    provider: str = typer.Option(None, "--provider", "-p", help="Force one provider"),
    model: str = typer.Option(None, "--model", "-m", help="Override the model"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Print only the reply"),
) -> None:
    """Send a prompt; auto-route to the free/cheapest provider with fallback."""
    cfg = _load()
    if interactive:
        _chat_loop(cfg, provider, model, quiet)
        return
    if not prompt:
        console.print("[red]Provide a prompt or use -i for interactive mode.[/red]")
        raise typer.Exit(1)
    _one_shot(cfg, prompt, provider, model, quiet)


def _one_shot(cfg: Config, prompt, provider, model, quiet) -> None:
    from .client import chat as run_chat

    try:
        result = run_chat(cfg.providers, prompt, only=provider, model=model)
    except ChatError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if quiet:
        print(result.text)
    else:
        console.print(result.text)
        console.print(
            f"\n[dim]— {result.provider} ({result.model}) "
            f"via {result.masked_key}, attempt #{result.attempts}[/dim]"
        )


def _chat_loop(cfg: Config, provider, model, quiet) -> None:
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
        try:
            result = run_chat(cfg.providers, history, only=provider, model=model)
        except ChatError as exc:
            console.print(f"[red]{exc}[/red]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": result.text})
        console.print(f"[bold green]ai ›[/bold green] {result.text}")
        if not quiet:
            console.print(
                f"[dim]— {result.provider} ({result.model}) via {result.masked_key}[/dim]"
            )


@app.command()
def version() -> None:
    """Show the aikeys version and config file location."""
    console.print(f"aikeys {__version__}")
    console.print(f"config: {config_path()}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
