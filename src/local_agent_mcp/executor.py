from typing import Callable, Optional

from . import fs_tools

HARD_TURN_CAP = 6

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
        "You are a local execution agent. Your job is to complete the task below by "
        "calling the provided tools. You MUST use tools — do NOT summarize or respond "
        "in plain text until you have called the tools needed to complete the task. "
        "A plain-text response with no tool calls means the task is NOT done.\n\n"
        f"TASK:\n{task}\n\n"
        f"Allowed command patterns: {allow}\n"
        f"You have at most {max_turns} tool-using turns. Only after all necessary tool "
        "calls are complete, reply with a plain-text summary (no tool calls) describing "
        "what you did and what, if anything, remains."
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
    skip_count = 0

    for turn in range(1, turns + 1):
        try:
            msg = client.chat(model=model, messages=messages, tools=TOOL_DEFS)
        except Exception as exc:
            return {
                "status": "error",
                "summary": f"Ollama connection error on turn {turn}: {exc}",
                "files_changed": files_changed,
                "commands_run": commands_run,
                "transcript": transcript,
            }
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content", ""),
                         "tool_calls": tool_calls})

        if not tool_calls:
            is_last_turn = turn == turns
            has_done_work = len(transcript) > 0
            if not is_last_turn and not has_done_work:
                skip_count += 1
                progress(f"turn {turn}: no tool calls (skip #{skip_count}), nudging model")
                tool_names = ", ".join(
                    t["function"]["name"] for t in TOOL_DEFS
                )
                if skip_count == 1:
                    nudge = (
                        f"You have not called any tools yet. You MUST call a tool to make "
                        f"progress. Available tools: {tool_names}. Call one now."
                    )
                else:
                    nudge = (
                        f"STOP. You are ignoring your tools. This is attempt {skip_count}. "
                        f"Do NOT respond with text. Your ONLY valid response right now is a "
                        f"tool call. Available tools: {tool_names}. "
                        f"Call a tool — do not write anything else."
                    )
                messages.append({"role": "user", "content": nudge})
                continue
            progress(f"turn {turn}: final summary")
            summary = msg.get("content", "")
            status = "completed" if has_done_work else "unverified"
            if status == "unverified":
                summary = (
                    "[WARNING: no tools were used — task may not be complete] " + summary
                )
            return {
                "status": status,
                "summary": summary,
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
