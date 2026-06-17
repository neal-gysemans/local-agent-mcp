from local_agent_mcp.server import shape_models_list


class FakeClient:
    def list_models(self):
        return [{"name": "qwen3-coder:30b", "size": 123, "modified_at": "2026-01-01"}]


def test_shape_models_list():
    out = shape_models_list(FakeClient())
    assert out["models"][0]["name"] == "qwen3-coder:30b"
    assert out["models"][0]["size"] == 123
