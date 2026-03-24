from __future__ import annotations

import json

from ai_harness_api.base import AiCliClient
from ai_harness_api.types import (
    AiChunk,
    AiResponse,
    AiResponseMetadata,
    AiUsage,
    ResolvedOptions,
)


class ClaudeClient(AiCliClient):
    """Backend client for the Claude Code CLI."""

    _default_executable = 'claude'
    _default_model = 'haiku'

    def _build_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        args = ['-p', prompt]
        if options.session_id is not None:
            args.extend(['--resume', options.session_id])
        elif not options.persist_session:
            args.append('--no-session-persistence')
        args.extend(['--output-format', 'json'])
        args.extend(['--model', options.model])
        args.extend(options.additional_args)
        return args

    def _build_stream_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        args = ['-p', prompt]
        if options.session_id is not None:
            args.extend(['--resume', options.session_id])
        elif not options.persist_session:
            args.append('--no-session-persistence')
        args.extend(['--output-format', 'stream-json'])
        args.append('--verbose')
        args.extend(['--model', options.model])
        args.extend(options.additional_args)
        return args

    def _parse_response(self, stdout: str, options: ResolvedOptions) -> AiResponse:
        trimmed = stdout.strip()

        try:
            obj = json.loads(trimmed)
            content = obj.get('result') or obj.get('content') or obj.get('text') or ''
            usage_raw = obj.get('usage')
            usage = (
                AiUsage(
                    input_tokens=usage_raw.get('input_tokens'),
                    output_tokens=usage_raw.get('output_tokens'),
                )
                if usage_raw
                else None
            )
            return AiResponse(
                content=content,
                status='success',
                metadata=AiResponseMetadata(
                    backend='claude',
                    model=obj.get('model'),
                    usage=usage,
                    session_id=obj.get('session_id'),
                ),
            )

        except (json.JSONDecodeError, AttributeError):
            if trimmed:
                return AiResponse(
                    content=trimmed,
                    status='success',
                    metadata=AiResponseMetadata(backend='claude', fallback=True),
                )
            return AiResponse(
                content='',
                status='error',
                metadata=AiResponseMetadata(
                    backend='claude', message='empty or unparseable output'
                ),
            )

    def _parse_chunk(self, line: str) -> AiChunk | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'Unparseable stream line: {line}')

        event_type = obj.get('type')

        if event_type == 'assistant':
            content_blocks = obj.get('message', {}).get('content', [])
            text = ''.join(
                block.get('text', '')
                for block in content_blocks
                if block.get('type') == 'text'
            )
            if text:
                return AiChunk(delta=text, type='content', raw=obj)
            return AiChunk(delta='', type='status', raw=obj)

        if event_type == 'system':
            return AiChunk(delta='', type='metadata', raw=obj)

        if event_type == 'result':
            return AiChunk(delta='', type='metadata', raw=obj)

        return None
