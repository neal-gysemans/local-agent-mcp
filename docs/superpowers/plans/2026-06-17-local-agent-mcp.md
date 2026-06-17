# Local Agent MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python MCP server that lets Claude Code offload work to a local Ollama model — bounded autonomous execution, one-shot reasoning consults, and model management — with safety guardrails and progress notifications.

**Architecture:** A FastMCP stdio server exposes three tools (`local_execute`, `local_consult`, `local_models`). `local_execute` runs a bounded agent loop (≤3 turns) that gives the local model four inner tools (read/list/write files, run allowlisted commands), all confined to a working directory. The server talks to Ollama over HTTP via the `ollama` Python client.

**Tech Stack:** Python 3.14, FastMCP (`mcp` package), `ollama` Python client, `pytest` for tests. Project root is `C:\Users\gysem\vibecoding` (the repo root itself — code, tests, and docs all live here; created already with the spec/plan under `docs/`).

**Spec:** `docs/superpowers/specs/2026-06-17-local-agent-mcp-design.md`

---

## File Structure

```
vibecoding/                # repo root = C:\Users\gysem\vibecoding
  pyproject.toml
  README.md
  docs/superpowers/         # spec + this plan (already present)
  src/local_agent_mcp/
    __init__.py
    config.py          # env-var configuration (OLLAMA_HOST, DEFAULT_MODEL)
    safety.py          # path confinement + command allowlist matching
    fs_tools.py        # inner toolset: read_file, list_files, write_file, run_command
    ollama_client.py   # Ollama wrapper: chat-with-tools, list_models, pull_model
    executor.py        # the bounded agent loop for local_execute
    server.py          # FastMCP server: registers the 3 tools + progress wiring
  tests/
    test_safety.py
    test_fs_tools.py
    test_executor.py
    test_models.py
```

**Responsibilities:**
- `config.py` — read env vars once, expose a `Config` dataclass. No logic.
- `safety.py` — pure functions: `resolve_within(working_dir, path)` and `command_allowed(command, allowlist)`. No I/O. Fully unit-tested.
- `fs_tools.py` — the four inner tools the local model calls. Uses `safety.py`. Returns plain dicts/strings.
- `ollama_client.py` — thin wrapper over the `ollama` client so the loop and tests can mock one seam.
- `executor.py` — orchestrates the loop: build messages → call Ollama → dispatch tool calls to `fs_tools` → feed back → stop at turn cap → assemble report. Emits progress via an injected callback.
- `server.py` — FastMCP tool definitions; wires `Context.report_progress` into the executor callback.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/local_agent_mcp/__init__.py`
- Create: `.gitignore`

(The repo root `C:\Users\gysem\vibecoding` already exists, with `docs/` inside it.)

- [ ] **Step 1: Create source/test directories and init git**

```bash
cd /c/Users/gysem/vibecoding
mkdir -p src/local_agent_mcp tests
git init
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "local-agent-mcp"
version = "0.1.0"
description = "MCP server that lets Claude offload work to a local Ollama model"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.2.0",
    "ollama>=0.4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
local-agent-mcp = "local_agent_mcp.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/local_agent_mcp"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 4: Write empty `src/local_agent_mcp/__init__.py`**

```python
"""Local Agent MCP server package."""
```

- [ ] **Step 5: Create venv and install**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```
Expected: installs `mcp`, `ollama`, `pytest` without error.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold local-agent-mcp project"
```

---

## Task 2: Config

**Files:**
- Create: `src/local_agent_mcp/config.py`

- [ ] **Step 1: Write `config.py`**

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_host: str
    default_model: str
    command_timeout: int  # seconds, per run_command

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            default_model=os.environ.get("DEFAULT_MODEL", "qwen3-coder:30b"),
            command_timeout=int(os.environ.get("COMMAND_TIMEOUT", "120")),
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/local_agent_mcp/config.py
git commit -m "feat: add env-var config"
```

---

## Task 3: Safety — path confinement + allowlist

**Files:**
- Create: `src/local_agent_mcp/safety.py`
- Test: `tests/test_safety.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_safety.py
import os
import pytest
from local_agent_mcp.safety import resolve_within, command_allowed, PathEscapeError


def test_resolve_within_allows_child(tmp_path):
    target = resolve_within(str(tmp_path), "sub/file.txt")
    assert target.startswith(os.path.realpath(str(tmp_path)))


def test_resolve_within_rejects_parent_escape(tmp_path):
    with pytest.raises(PathEscapeError):
        resolve_within(str(tmp_path), "../outside.txt")


def test_resolve_within_rejects_absolute_outside(tmp_path):
    with pytest.raises(PathEscapeError):
        resolve_within(str(tmp_path), "C:/Windows/system32/x.txt")


def test_command_allowed_prefix_glob():
    assert command_allowed("pytest -q", ["pytest*"]) is True
    assert command_allowed("go build ./...", ["go build*"]) is True


def test_command_allowed_rejects_unlisted():
    assert command_allowed("rm -rf /", ["pytest*"]) is False


def test_command_allowed_empty_list_denies_all():
    assert command_allowed("pytest", []) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_agent_mcp.safety'`

- [ ] **Step 3: Write `safety.py`**

```python
import fnmatch
import os


class PathEscapeError(Exception):
    """Raised when a path resolves outside the allowed working directory."""


def resolve_within(working_dir: str, path: str) -> str:
    """Resolve `path` relative to `working_dir` and ensure it stays inside it.

    Resolves symlinks and `..` before checking. Raises PathEscapeError on escape.
    Returns the resolved absolute path as a string.
    """
    base = os.path.realpath(working_dir)
    candidate = os.path.realpath(os.path.join(base, path))
    # commonpath raises ValueError across drives; treat that as an escape too.
    try:
        common = os.path.commonpath([base, candidate])
    except ValueError:
        raise PathEscapeError(f"{path!r} resolves outside {working_dir!r}")
    if common != base:
        raise PathEscapeError(f"{path!r} resolves outside {working_dir!r}")
    return candidate


def command_allowed(command: str, allowlist: list[str]) -> bool:
    """Return True if `command` matches any glob pattern in `allowlist`.

    An empty allowlist denies everything (files-only mode).
    """
    return any(fnmatch.fnmatch(command, pattern) for pattern in allowlist)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/local_agent_mcp/safety.py tests/test_safety.py
git commit -m "feat: add path confinement and command allowlist"
```

---

## Task 4: Inner filesystem/command toolset

**Files:**
- Create: `src/local_agent_mcp/fs_tools.py`
- Test: `tests/test_fs_tools.py`

The four tools the local model calls. Each takes `working_dir` plus its own args and
returns a JSON-serializable result. `run_command` enforces the allowlist and a timeout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fs_tools.py
import pytest
from local_agent_mcp.fs_tools import read_file, list_files, write_file, run_command


def test_write_then_read(tmp_path):
    wd = str(tmp_path)
    res = write_file(wd, "a.txt", "hello")
    assert res["ok"] is True
    assert read_file(wd, "a.txt")["content"] == "hello"


def test_read_missing_file_returns_error(tmp_path):
    res = read_file(str(tmp_path), "nope.txt")
    assert res["ok"] is False
    assert "not found" in res["error"].lower()


def test_write_outside_is_refused(tmp_path):
    res = write_file(str(tmp_path), "../escape.txt", "x")
    assert res["ok"] is False
    assert "outside" in res["error"].lower()


def test_list_files_returns_relative_paths(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_text("2")
    names = set(list_files(str(tmp_path), ".")["files"])
    assert "x.txt" in names
    assert "sub/y.txt" in names


def test_run_command_allowed(tmp_path):
    res = run_command(str(tmp_path), "python -c \"print('hi')\"",
                      allowlist=["python*"], timeout=30)
    assert res["ok"] is True
    assert res["exit_code"] == 0
    assert "hi" in res["stdout"]


def test_run_command_denied(tmp_path):
    res = run_command(str(tmp_path), "rm -rf /", allowlist=["pytest*"], timeout=30)
    assert res["ok"] is False
    assert "not allowed" in res["error"].lower()
    assert "ran" not in res  # never executed


def test_run_command_timeout(tmp_path):
    res = run_command(str(tmp_path), "python -c \"import time; time.sleep(5)\"",
                      allowlist=["python*"], timeout=1)
    assert res["ok"] is False
    assert "timed out" in res["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fs_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_agent_mcp.fs_tools'`

- [ ] **Step 3: Write `fs_tools.py`**

```python
import os
import subprocess

from .safety import resolve_within, command_allowed, PathEscapeError


def read_file(working_dir: str, path: str) -> dict:
    try:
        full = resolve_within(working_dir, path)
    except PathEscapeError as e:
        return {"ok": False, "error": str(e)}
    if not os.path.isfile(full):
        return {"ok": False, "error": f"File not found: {path}"}
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return {"ok": True, "content": f.read()}


def list_files(working_dir: str, subdir: str) -> dict:
    try:
        base = resolve_within(working_dir, subdir)
    except PathEscapeError as e:
        return {"ok": False, "error": str(e)}
    if not os.path.isdir(base):
        return {"ok": False, "error": f"Directory not found: {subdir}"}
    root = os.path.realpath(working_dir)
    files = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel.replace(os.sep, "/"))
    return {"ok": True, "files": sorted(files)}


def write_file(working_dir: str, path: str, content: str) -> dict:
    try:
        full = resolve_within(working_dir, path)
    except PathEscapeError as e:
        return {"ok": False, "error": str(e)}
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    rel = os.path.relpath(full, os.path.realpath(working_dir)).replace(os.sep, "/")
    return {"ok": True, "path": rel}


def run_command(working_dir: str, command: str, allowlist: list[str],
                timeout: int) -> dict:
    if not command_allowed(command, allowlist):
        return {"ok": False,
                "error": f"Command not allowed by allowlist: {command!r}"}
    try:
        proc = subprocess.run(
            command, shell=True, cwd=working_dir, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Command timed out after {timeout}s"}
    return {
        "ok": True,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fs_tools.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/local_agent_mcp/fs_tools.py tests/test_fs_tools.py
git commit -m "feat: add inner filesystem/command toolset"
```

---

## Task 5: Ollama client wrapper

**Files:**
- Create: `src/local_agent_mcp/ollama_client.py`

Thin seam over the `ollama` package so the executor depends on a small surface and tests
can substitute a fake. No tests of its own (it is pure delegation); it is exercised via
mocking in Task 6 and Task 7.

- [ ] **Step 1: Write `ollama_client.py`**

```python
from typing import Any, Callable, Optional

import ollama


class OllamaClient:
    """Thin wrapper over the ollama Python client."""

    def __init__(self, host: str):
        self._client = ollama.Client(host=host)

    def chat(self, model: str, messages: list[dict],
             tools: Optional[list[dict]] = None) -> dict:
        """One chat round. Returns the response 'message' dict, which may contain
        'content' and/or 'tool_calls'."""
        resp = self._client.chat(model=model, messages=messages, tools=tools)
        # ollama returns an object with a .message; normalize to a plain dict.
        msg = resp["message"]
        return {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls") or [],
        }

    def list_models(self) -> list[dict]:
        data = self._client.list()
        out = []
        for m in data.get("models", []):
            out.append({
                "name": m.get("model") or m.get("name"),
                "size": m.get("size"),
                "modified_at": str(m.get("modified_at", "")),
            })
        return out

    def pull_model(self, name: str,
                   on_progress: Optional[Callable[[str], None]] = None) -> dict:
        last_status = ""
        for chunk in self._client.pull(name, stream=True):
            status = chunk.get("status", "")
            if status and status != last_status:
                last_status = status
                if on_progress:
                    on_progress(status)
        return {"status": last_status or "completed", "name": name}
```

- [ ] **Step 2: Commit**

```bash
git add src/local_agent_mcp/ollama_client.py
git commit -m "feat: add Ollama client wrapper"
```

---

## Task 6: Bounded agent loop (executor)

**Files:**
- Create: `src/local_agent_mcp/executor.py`
- Test: `tests/test_executor.py`

The core. Builds the tool definitions, runs the loop against an injected `chat`-capable
client, dispatches tool calls to `fs_tools`, enforces the 3-turn cap, emits progress, and
returns the structured report.

- [ ] **Step 1: Write the failing tests (with a fake client)**

```python
# tests/test_executor.py
from local_agent_mcp.executor import run_execution


class FakeClient:
    """Returns a scripted sequence of assistant messages, one per chat() call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    def chat(self, model, messages, tools=None):
        self.calls += 1
        return self._scripted.pop(0)


def _tool_call(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_writes_file_then_completes(tmp_path):
    client = FakeClient([
        {"role": "assistant", "content": "",
         "tool_calls": [_tool_call("write_file", {"path": "a.txt", "content": "hi"})]},
        {"role": "assistant", "content": "Done. Wrote a.txt.", "tool_calls": []},
    ])
    report = run_execution(
        client=client, model="m", task="write a.txt",
        working_dir=str(tmp_path), command_allowlist=[],
        max_turns=3, command_timeout=30,
    )
    assert report["status"] == "completed"
    assert "a.txt" in report["files_changed"]
    assert (tmp_path / "a.txt").read_text() == "hi"


def test_stops_at_turn_cap(tmp_path):
    # Always asks to write; never emits a final answer.
    looping = {"role": "assistant", "content": "",
               "tool_calls": [_tool_call("write_file",
                                          {"path": "a.txt", "content": "x"})]}
    client = FakeClient([looping, looping, looping, looping, looping])
    report = run_execution(
        client=client, model="m", task="loop forever",
        working_dir=str(tmp_path), command_allowlist=[],
        max_turns=3, command_timeout=30,
    )
    assert report["status"] == "turn_limit_reached"
    assert client.calls == 3


def test_max_turns_clamped_to_three(tmp_path):
    looping = {"role": "assistant", "content": "",
               "tool_calls": [_tool_call("list_files", {"subdir": "."})]}
    client = FakeClient([looping] * 10)
    report = run_execution(
        client=client, model="m", task="loop",
        working_dir=str(tmp_path), command_allowlist=[],
        max_turns=99, command_timeout=30,
    )
    assert client.calls == 3
    assert report["status"] == "turn_limit_reached"


def test_progress_events_emitted(tmp_path):
    events = []
    client = FakeClient([
        {"role": "assistant", "content": "",
         "tool_calls": [_tool_call("write_file", {"path": "a.txt", "content": "y"})]},
        {"role": "assistant", "content": "done", "tool_calls": []},
    ])
    run_execution(
        client=client, model="m", task="t", working_dir=str(tmp_path),
        command_allowlist=[], max_turns=3, command_timeout=30,
        on_progress=lambda msg: events.append(msg),
    )
    assert any("write_file" in e for e in events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_agent_mcp.executor'`

- [ ] **Step 3: Write `executor.py`**

```python
from typing import Callable, Optional

from . import fs_tools

HARD_TURN_CAP = 3

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file within the working directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to working dir"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files under a subdirectory of the working directory.",
        "parameters": {"type": "object", "properties": {
            "subdir": {"type": "string", "description": "Subdir relative to working dir; use '.' for root"}},
            "required": ["subdir"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 text file within the working directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command. Only commands matching the allowlist run.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}},
            "required": ["command"]}}},
]


def _system_prompt(task: str, allowlist: list[str], max_turns: int) -> str:
    allow = ", ".join(allowlist) if allowlist else "(none — no commands permitted)"
    return (
        "You are a local execution agent. Implement EXACTLY the task below using the "
        "provided tools. Stay within the working directory.\n\n"
        f"TASK:\n{task}\n\n"
        f"Allowed command patterns: {allow}\n"
        f"You have at most {max_turns} tool-using turns. When the task is done, or on "
        "your final turn, reply with a plain-text summary (no tool calls) describing "
        "what you changed and what, if anything, remains."
    )


def _dispatch(name: str, args: dict, working_dir: str,
              allowlist: list[str], timeout: int) -> dict:
    if name == "read_file":
        return fs_tools.read_file(working_dir, args.get("path", ""))
    if name == "list_files":
        return fs_tools.list_files(working_dir, args.get("subdir", "."))
    if name == "write_file":
        return fs_tools.write_file(working_dir, args.get("path", ""),
                                   args.get("content", ""))
    if name == "run_command":
        return fs_tools.run_command(working_dir, args.get("command", ""),
                                    allowlist, timeout)
    return {"ok": False, "error": f"Unknown tool: {name}"}


def run_execution(*, client, model: str, task: str, working_dir: str,
                  command_allowlist: list[str], max_turns: int,
                  command_timeout: int,
                  on_progress: Optional[Callable[[str], None]] = None) -> dict:
    turns = min(max_turns, HARD_TURN_CAP)

    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    messages = [{"role": "system",
                 "content": _system_prompt(task, command_allowlist, turns)}]
    files_changed: list[str] = []
    commands_run: list[dict] = []
    transcript: list[dict] = []

    for turn in range(1, turns + 1):
        msg = client.chat(model=model, messages=messages, tools=TOOL_DEFS)
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content", ""),
                         "tool_calls": tool_calls})

        if not tool_calls:
            progress(f"turn {turn}: final summary")
            return {
                "status": "completed",
                "summary": msg.get("content", ""),
                "files_changed": files_changed,
                "commands_run": commands_run,
                "transcript": transcript,
            }

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {}) or {}
            result = _dispatch(name, args, working_dir, command_allowlist,
                               command_timeout)
            progress(f"turn {turn}: {name} -> {'ok' if result.get('ok') else 'error'}")
            transcript.append({"turn": turn, "tool": name, "args": args,
                               "result": result})
            if name == "write_file" and result.get("ok"):
                files_changed.append(result["path"])
            if name == "run_command" and result.get("ok"):
                commands_run.append({"command": args.get("command", ""),
                                     "exit_code": result["exit_code"],
                                     "stdout": result["stdout"],
                                     "stderr": result["stderr"]})
            messages.append({"role": "tool", "name": name, "content": str(result)})

    progress(f"turn limit ({turns}) reached")
    return {
        "status": "turn_limit_reached",
        "summary": "Turn limit reached before the agent reported completion.",
        "files_changed": files_changed,
        "commands_run": commands_run,
        "transcript": transcript,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/local_agent_mcp/executor.py tests/test_executor.py
git commit -m "feat: add bounded agent loop executor"
```

---

## Task 7: FastMCP server with the three tools

**Files:**
- Create: `src/local_agent_mcp/server.py`
- Test: `tests/test_models.py`

Registers `local_execute`, `local_consult`, `local_models`. Wires MCP progress into the
executor callback. For testability, the Ollama-touching logic for `local_models list` is
delegated to the client wrapper, which we mock here.

- [ ] **Step 1: Write the failing test for the models-list shaping**

```python
# tests/test_models.py
from local_agent_mcp.server import shape_models_list


class FakeClient:
    def list_models(self):
        return [{"name": "qwen3-coder:30b", "size": 123, "modified_at": "2026-01-01"}]


def test_shape_models_list():
    out = shape_models_list(FakeClient())
    assert out["models"][0]["name"] == "qwen3-coder:30b"
    assert out["models"][0]["size"] == 123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_agent_mcp.server'`

- [ ] **Step 3: Write `server.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS (all tests from Tasks 3, 4, 6, 7)

- [ ] **Step 6: Verify the server imports and starts**

Run: `.venv/Scripts/python.exe -c "from local_agent_mcp.server import mcp; print('ok')"`
Expected: prints `ok` with no import error.

- [ ] **Step 7: Commit**

```bash
git add src/local_agent_mcp/server.py tests/test_models.py
git commit -m "feat: add FastMCP server with execute/consult/models tools"
```

---

## Task 8: README and Claude Code registration

**Files:**
- Create: `local-agent-mcp/README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
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

```bash
claude mcp add local-agent -s user -- C:/Users/gysem/vibecoding/.venv/Scripts/python.exe -m local_agent_mcp.server
```

Optional environment overrides: `OLLAMA_HOST`, `DEFAULT_MODEL`, `COMMAND_TIMEOUT`.

## Safety

- File operations are confined to `working_dir` (symlink/`..` escapes are refused).
- `run_command` runs only commands matching the per-call allowlist. **Scope patterns
  tightly; never use `*`.** An empty allowlist disables commands entirely.
- Each command has a wall-clock timeout (default 120s).
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README and registration instructions"
```

---

## Task 9: Manual smoke test (run by user, real Ollama)

**Not automated — depends on locally installed models. Run after Tasks 1–8.**

- [ ] **Step 1: Register the server**

```bash
claude mcp add local-agent -s user -- C:/Users/gysem/vibecoding/.venv/Scripts/python.exe -m local_agent_mcp.server
```

- [ ] **Step 2: Smoke `local_models list`** — from Claude, call `local_models(action="list")`. Expected: your installed models appear.

- [ ] **Step 3: Smoke `local_consult`** — call `local_consult(prompt="Name one risk in retrying a failed DB write without idempotency.")`. Expected: a short reasoned answer.

- [ ] **Step 4: Smoke `local_execute`** — create a throwaway dir, then call:

```
local_execute(
  task="Create hello.py that prints 'hello from local agent' and nothing else.",
  working_dir="C:/Users/gysem/vibecoding/scratch-smoke",
  command_allowlist=["python hello.py"]
)
```
Expected: `hello.py` created; if the model runs it, `commands_run` shows the output. Report `status` is `completed` or `turn_limit_reached`.

- [ ] **Step 5: Confirm allowlist denial** — call `local_execute` with `command_allowlist=[]` and a task that asks to run a command. Expected: the transcript shows the `run_command` refusal, no command executed.

---

## Notes for the implementer

- **Windows venv paths:** use `.venv/Scripts/python.exe` (not `bin/`). The Bash tool accepts forward slashes.
- **Ollama response shape:** the `ollama` client returns objects that behave like dicts; `ollama_client.chat` normalizes them. If `tool_calls` arrive as objects rather than dicts in your version, adjust the normalization there — the executor only depends on `msg["tool_calls"][i]["function"]["name"]` / `["arguments"]`.
- **Tool-result messages:** Ollama expects tool outputs as `{"role": "tool", ...}`; we stringify the result dict. Models in this class read that fine.
- **Don't loosen the 3-turn cap.** `HARD_TURN_CAP = 3` in `executor.py` is the safety bound; `max_turns` is clamped to it.
