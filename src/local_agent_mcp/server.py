from typing import Optional

from mcp.server.fastmcp import FastMCP, Context

from .config import Config
from .ollama_client import OllamaClient
from .executor import run_execution

CONFIG = Config.from_env()
mcp = FastMCP("local-agent-mcp")


def _client() -> OllamaClient:
    return OllamaClient(CONFIG.ollama_host)


def shape_models_list(client) -> dict:
    """Pure shaping helper (unit-tested): wrap client.list_models() in a dict."""
    return {"models": client.list_models()}


@mcp.tool()
async def local_execute(
    task: str,
    working_dir: str,
    command_allowlist: list[str],
    model: Optional[str] = None,
    max_turns: int = 3,
    ctx: Context = None,
) -> dict:
    """Dispatch one detailed task to a local model that executes it within `working_dir`.

    The local agent gets four tools: read_file, list_files, write_file, run_command.
    run_command ONLY runs commands matching `command_allowlist` glob patterns — SCOPE
    THESE TIGHTLY per task and NEVER use '*'. Empty allowlist = files-only. The loop is
    hard-capped at 3 turns, after which control returns to you regardless. Returns a
    structured report (status, summary, files_changed, commands_run, transcript).
    """
    def on_progress(msg: str) -> None:
        if ctx is not None:
            ctx.session.send_log_message(level="info", data=msg)

    return run_execution(
        client=_client(),
        model=model or CONFIG.default_model,
        task=task,
        working_dir=working_dir,
        command_allowlist=command_allowlist,
        max_turns=max_turns,
        command_timeout=CONFIG.command_timeout,
        on_progress=on_progress,
    )


@mcp.tool()
async def local_consult(
    prompt: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """One-shot reasoning with a local model. No tools, no file/command access.

    Use for second opinions, poking holes in an approach, or offloading pure reasoning.
    Include any needed code/plan in `context`.
    """
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    msg = _client().chat(model or CONFIG.default_model, messages, tools=None)
    return {"response": msg.get("content", "")}


@mcp.tool()
async def local_models(
    action: str,
    name: Optional[str] = None,
    ctx: Context = None,
) -> dict:
    """Manage local Ollama models. action='list' returns installed models;
    action='pull' downloads `name` (long-running; progress is reported)."""
    client = _client()
    if action == "list":
        return shape_models_list(client)
    if action == "pull":
        if not name:
            return {"error": "pull requires a model name"}

        def on_progress(msg: str) -> None:
            if ctx is not None:
                ctx.session.send_log_message(level="info", data=msg)

        return client.pull_model(name, on_progress=on_progress)
    return {"error": f"Unknown action: {action!r} (use 'list' or 'pull')"}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
