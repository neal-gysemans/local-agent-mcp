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
