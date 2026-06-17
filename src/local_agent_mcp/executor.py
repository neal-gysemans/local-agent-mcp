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
