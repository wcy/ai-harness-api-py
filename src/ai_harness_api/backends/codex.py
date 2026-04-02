from __future__ import annotations

import json

from ai_harness_api.base import AiCliClient
from ai_harness_api.types import (
    AiChunk,
    AiResponse,
    AiResponseMetadata,
    ResolvedOptions,
)


def _try_parse_json(line: str) -> dict | None:
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None


def _build_codex_args(prompt: str, resolved: ResolvedOptions) -> list[str]:
    if resolved.session_id is not None:
        args = ['exec', 'resume', resolved.session_id, '--json']
        args.extend(['--model', resolved.model])
        args.append(prompt)
    elif resolved.persist_session:
        args = ['exec', '--json']
        if resolved.cwd:
            args.extend(['-C', resolved.cwd])
        args.extend(['--model', resolved.model])
        args.append(prompt)
    else:
        args = ['exec', '--ephemeral', '--json']
        if resolved.cwd:
            args.extend(['-C', resolved.cwd])
        args.extend(['--model', resolved.model])
        args.append(prompt)
    if resolved.allow_all_tools:
        args.append('--full-auto')
    args.extend(resolved.additional_args)
    return args


class CodexClient(AiCliClient):
    """Backend client for the Codex CLI."""

    _default_executable = 'codex'
    _default_model = 'gpt-5.4-mini'

    def _build_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        return _build_codex_args(prompt, options)

    def _build_stream_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        return _build_codex_args(prompt, options)

    def _parse_response(self, stdout: str, options: ResolvedOptions) -> AiResponse:
        lines = [line for line in stdout.split('\n') if line.strip()]
        events = [e for e in (_try_parse_json(line) for line in lines) if e is not None]

        thread_started = next((e for e in events if e.get('type') == 'thread.started'), None)
        session_id = thread_started.get('thread_id') if thread_started else None

        responses = [
            e for e in events
            if e.get('type') == 'item.completed'
            and isinstance(e.get('item'), dict)
            and e['item'].get('type') == 'agent_message'
        ]
        error_events = [e for e in events if e.get('type') == 'error']

        if responses:
            last = responses[-1]
            return AiResponse(
                content=last['item'].get('text', ''),
                status='success',
                metadata=AiResponseMetadata(
                    backend='codex',
                    model=last.get('model'),
                    session_id=session_id,
                ),
            )

        if error_events:
            return AiResponse(
                content='',
                status='error',
                metadata=AiResponseMetadata(
                    backend='codex',
                    message=error_events[0].get('message', 'codex reported an error'),
                ),
            )

        raw = stdout.strip()
        if raw:
            return AiResponse(
                content=raw,
                status='success',
                metadata=AiResponseMetadata(backend='codex', fallback=True),
            )

        return AiResponse(
            content='',
            status='error',
            metadata=AiResponseMetadata(backend='codex', message='no output'),
        )

    def _parse_chunk(self, line: str) -> AiChunk | None:
        obj = _try_parse_json(line)
        if obj is None:
            return None

        event_type = obj.get('type')

        if event_type == 'item.completed':
            item = obj.get('item')
            if isinstance(item, dict) and item.get('type') == 'agent_message':
                return AiChunk(
                    delta=item.get('text', ''),
                    type='content',
                    role='assistant',
                    raw=obj,
                )
        if event_type == 'error':
            return AiChunk(delta='', type='status', raw=obj)

        return None
