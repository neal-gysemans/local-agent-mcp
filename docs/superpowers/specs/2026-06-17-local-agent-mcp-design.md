# Local Agent MCP Server — Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan

## Purpose

Let Claude Code offload mechanical, well-specified work to a local LLM (via Ollama),
saving Claude/API tokens. Two use cases:

1. **Bounded execution** — Claude writes a detailed implementation plan and dispatches
   a single step to a local model that executes it (edits files, runs allowlisted
   commands), instead of dispatching to a Haiku-level subagent.
2. **One-shot reasoning consult** — Claude asks the local model a question / gets a
   second opinion, with no tool or file access.

**Core principle:** Claude does the thinking and the verification; the local model does
the mechanical execution against a tight spec. The quality of Claude's plan is what makes
this reliable.

## Architecture

A single Python MCP server (`local-agent-mcp`), built with FastMCP, connected to Claude
Code over stdio. It talks to a local Ollama instance over HTTP.

```
Claude Code ──(MCP/stdio)──> local-agent-mcp ──(HTTP)──> Ollama ──> qwen3-coder:30B etc.
                                   │
                                   └── executes local-agent tool calls
                                       (read/write files, run allowlisted commands)
                                       within a working directory
```

**Stack:** Python + FastMCP + the Ollama Python client.

**Configuration** (env vars in the MCP server definition):

- `OLLAMA_HOST` — default `http://localhost:11434`
- `DEFAULT_MODEL` — default `qwen3-coder:30b`

## Tools Exposed to Claude

### `local_execute`

Runs a bounded autonomous agent loop against one detailed task.

| Param | Type | Required | Notes |
|---|---|---|---|
| `task` | string | yes | The detailed step/spec Claude wrote. Tighter is better. |
| `working_dir` | string | yes | Absolute path the agent is scoped to. All file ops confined here. |
| `command_allowlist` | string[] | yes | Glob/prefix patterns, e.g. `["pytest*", "go build*"]`. Empty = files-only (no commands). |
| `model` | string | no | Overrides `DEFAULT_MODEL`. |
| `max_turns` | int | no | Default 3, **hard-capped at 3**. Values above 3 are clamped. |

**Returns** a structured report:

- `status` — one of `completed`, `turn_limit_reached`, `blocked`
- `summary` — what was done / what remains
- `files_changed[]` — paths written
- `commands_run[]` — each with command + captured output + exit code
- `transcript` — the loop's tool calls and results

The tool description (visible to Claude on every call) MUST state the hard rule:
**scope `command_allowlist` patterns tightly per task; never use `*`.**

### `local_consult`

One-shot reasoning. No tools, no file/command access.

| Param | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | The question / thing to reason about. |
| `context` | string | no | Code, plan, or prior reasoning to include. |
| `model` | string | no | Overrides `DEFAULT_MODEL`. |

**Returns** `{ response }`.

### `local_models`

Inspect and manage locally available Ollama models so Claude can choose one before
dispatching.

| Param | Type | Required | Notes |
|---|---|---|---|
| `action` | string | yes | `list` or `pull`. |
| `name` | string | for `pull` | Model tag to pull, e.g. `qwen3-coder:30b`. Ignored for `list`. |

- `list` → calls Ollama `GET /api/tags`; returns `models[]` with `name`, `size`, and
  `modified_at` for each installed model.
- `pull` → calls Ollama `POST /api/pull` for `name`. Pulls can be multi-GB / long-running;
  progress is surfaced via the same progress-notification mechanism as `local_execute`
  (see Streaming below). Returns final `status` and the resolved model name.

**Returns** `{ models[] }` for `list`, or `{ status, name }` for `pull`.

## The Agent Loop (inside `local_execute`)

Uses Ollama's native tool-calling. The server hands the local model four inner tools:

- `read_file(path)` — read a file within `working_dir`.
- `list_files(subdir)` — list files within `working_dir`.
- `write_file(path, content)` — create/overwrite a file within `working_dir`.
- `run_command(command)` — runs **only if it matches `command_allowlist`**; otherwise
  returns a refusal string the model can see and adapt to.

**Loop:**

1. Send system prompt (role + `task` + "you have ≤N turns, then you must report") plus
   the four tool definitions to Ollama.
2. Model returns tool calls → server executes them → feeds results back as tool messages.
3. Repeat until the model emits a final answer **or** the turn cap is reached.
4. Server assembles the structured report and returns it to Claude.

**Turn definition:** one model round that issues tool calls. Hard stop at 3. On the final
allowed turn, the system prompt instructs the model to summarize what's done and what
remains rather than start new work.

## Streaming (progress notifications)

`local_execute` and `local_models` `pull` emit progress as work advances, using FastMCP's
`Context.report_progress` / log-message mechanism. Examples:

- `local_execute`: one progress event per loop step — e.g. "turn 1: ran `pytest` (passed)",
  "turn 2: wrote `auth.py`".
- `local_models pull`: periodic download progress from Ollama's streaming pull response.

**Limitation (honest framing):** MCP returns a tool's *result* as a single payload — there
is no token-by-token streaming of the local model's output *into* Claude's reasoning
mid-call. "Streaming" here means in-flight progress notifications about loop/pull state,
not incremental result content. The final structured report is still returned whole when
the call completes.

**Rationale for the cap:** local models in this class (qwen3-coder:30B and peers) handle
~3–5 tool turns reliably; beyond that, error accumulates (misreading output, repeating
failed approaches, premature success claims). A cap of 3 keeps the loop honest and returns
control to Claude for verification.

## Safety Enforcement

- **Path confinement:** every file op resolves the absolute path (resolving symlinks and
  `..`) and verifies it falls inside `working_dir`. Outside → refused, surfaced to the model.
- **Command allowlist:** `run_command` matches the command against the per-call patterns.
  No match → not executed; refusal text returned to the model. Empty allowlist = files-only.
- **Working directory as cwd:** commands run with `working_dir` as their working directory.
- **Timeouts:** each `run_command` has a wall-clock timeout (default 120s) so a hung
  command cannot stall the loop.

**Known limitation / risk (v1):** the allowlist is pattern-based. A sloppy pattern (e.g.
`*`) defeats it. Mitigation: the `local_execute` tool description instructs Claude to scope
patterns tightly every call. No OS-level sandbox is added in v1 — we rely on the allowlist
plus path confinement. This is an accepted trade-off for v1 and a candidate for hardening
later.

## Testing

- **Unit (no Ollama):** path confinement (reject `../`, symlink escapes), allowlist
  matching (allow/deny cases), turn-cap enforcement/clamping, command timeout handling.
- **Integration (mocked Ollama):** fake Ollama responses that issue tool calls → assert the
  loop executes them, feeds results back, and stops at the turn cap. Assert the structured
  report shape. Assert `local_models list` parses `/api/tags` and that progress events are
  emitted during the loop.
- **Smoke (real Ollama, manual, run by user):** one real `local_execute` against a
  throwaway directory with qwen3-coder, plus one `local_consult`. Not in CI — depends on
  locally installed models.

## Out of Scope (v1)

- Stateful / multi-turn consult sessions (consult is one-shot only).
- OS-level sandboxing beyond path confinement + allowlist.
- Token-by-token streaming of local-model output into Claude's reasoning (MCP limitation;
  v1 provides progress notifications instead — see Streaming).
