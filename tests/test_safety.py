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
