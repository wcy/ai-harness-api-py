from __future__ import annotations

from typing import Callable, Literal, TypedDict

from ai_harness_api.backends.claude import ClaudeClient
from ai_harness_api.backends.codex import CodexClient
from ai_harness_api.backends.gemini import GeminiClient
from ai_harness_api.base import AiCliClient
from ai_harness_api.types import ClientConfig

BackendName = Literal['claude', 'codex', 'gemini']


class BackendEntry(TypedDict):
    create: Callable[[ClientConfig], AiCliClient]


BACKENDS: dict[BackendName, BackendEntry] = {
    'claude': {'create': lambda c: ClaudeClient(c)},
    'codex': {'create': lambda c: CodexClient(c)},
    'gemini': {'create': lambda c: GeminiClient(c)},
}
