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
