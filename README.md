# local-agent-mcp

An MCP server that lets Claude Code offload work to a local Ollama model.

## Tools

- `local_execute(task, working_dir, command_allowlist, model?, max_turns?)` —
  bounded autonomous execution (≤3 turns) of a detailed task. The local agent can read,
  list, and write files within `working_dir`, and run commands that match
  `command_allowlist` glob patterns. Returns a structured report.
- `local_consult(prompt, context?, model?)` — one-shot reasoning, no tools.
- `local_models(action, name?)` — `list` installed models or `pull` a new one.

## Requirements

- [Ollama](https://ollama.com) running locally with at least one model pulled
  (e.g. `ollama pull qwen3-coder:30b`).
- Python 3.11+.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

## Register with Claude Code

Registered at **user scope** so it is available in every project on this machine:

```bash
claude mcp add local-agent -s user -- C:/Users/gysem/vibecoding/.venv/Scripts/python.exe -m local_agent_mcp.server
```

Optional environment overrides: `OLLAMA_HOST`, `DEFAULT_MODEL`, `COMMAND_TIMEOUT`.

## Safety

- File operations are confined to `working_dir` (symlink/`..` escapes are refused).
- `run_command` runs only commands matching the per-call allowlist. **Scope patterns
  tightly; never use `*`.** An empty allowlist disables commands entirely.
- Each command has a wall-clock timeout (default 120s).
- The execution loop is hard-capped at 3 turns; control then returns to Claude.
