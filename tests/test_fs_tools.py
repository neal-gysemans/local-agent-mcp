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
