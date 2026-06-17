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
