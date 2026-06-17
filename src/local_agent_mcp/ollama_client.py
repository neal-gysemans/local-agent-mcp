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
