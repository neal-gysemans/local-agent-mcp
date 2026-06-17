import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_host: str
    default_model: str
    command_timeout: int  # seconds, per run_command

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            default_model=os.environ.get("DEFAULT_MODEL", "qwen3-coder:30b"),
            command_timeout=int(os.environ.get("COMMAND_TIMEOUT", "120")),
        )
