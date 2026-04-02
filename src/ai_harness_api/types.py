from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class ClientConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    executable_path: str | None = None
    model: str | None = None
    timeout: float | None = None
    cwd: str | None = None
    additional_args: list[str] = []
    session_id: str | None = None
    persist_session: bool = False
    allow_all_tools: bool = False  # Bypass all tool-approval prompts

    @field_validator('timeout')
    @classmethod
    def timeout_must_be_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError('timeout must be > 0')
        return v

    @field_validator('cwd')
    @classmethod
    def cwd_must_be_nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError('cwd must be a non-empty string')
        return v

    @field_validator('executable_path')
    @classmethod
    def executable_path_must_be_nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError('executable_path must be a non-empty string')
        return v


class RunOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    cwd: str | None = None
    executable_path: str | None = None
    model: str | None = None
    timeout: float | None = None
    additional_args: list[str] = []
    session_id: str | None = None
    persist_session: bool = False
    allow_all_tools: bool = False  # Bypass all tool-approval prompts for this call only

    @field_validator('timeout')
    @classmethod
    def timeout_must_be_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError('timeout must be > 0')
        return v

    @field_validator('cwd')
    @classmethod
    def cwd_must_be_nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError('cwd must be a non-empty string')
        return v

    @field_validator('executable_path')
    @classmethod
    def executable_path_must_be_nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError('executable_path must be a non-empty string')
        return v


class ResolvedOptions(BaseModel):
    """Internal — not exported from the package public API."""

    model_config = ConfigDict(frozen=True)

    executable: str
    model: str
    timeout: float | None
    cwd: str
    additional_args: list[str]
    session_id: str | None
    persist_session: bool
    allow_all_tools: bool  # True when tool-approval bypass flag should be passed to the CLI


class AiUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None


class AiResponseMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    model: str | None = None
    usage: AiUsage | None = None
    stderr: str | None = None
    fallback: bool = False
    timed_out: bool = False
    error_code: str | None = None
    message: str | None = None
    session_id: str | None = None
    rate_limited: bool = False
    rate_limit_reset_at: datetime | None = None


class AiResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    status: Literal['success', 'error']
    metadata: AiResponseMetadata | None = None


class AiChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: str
    type: Literal['content', 'metadata', 'status', 'error']
    role: str | None = None
    raw: Any = None


class SpawnOptions(BaseModel):
    """Internal — not exported from the package public API."""

    model_config = ConfigDict(frozen=True)

    executable: str
    args: list[str]
    cwd: str
