from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from ai_harness_api.base import AiCliClient
from ai_harness_api.types import (
    AiChunk,
    AiResponse,
    AiResponseMetadata,
    AiUsage,
    ResolvedOptions,
)


class RateLimitError(ValueError):
    def __init__(self, metadata: AiResponseMetadata) -> None:
        super().__init__('rate_limited')
        self.metadata = metadata


def _detect_rate_limit(text: str, backend: str = 'claude') -> AiResponseMetadata | None:
    if not re.search(r'hit your limit.*resets\s+\d+[ap]m', text, re.IGNORECASE | re.DOTALL):
        # Broader check: hit your limit + resets (without requiring digit+am/pm)
        if not re.search(r'hit your limit.*resets', text, re.IGNORECASE | re.DOTALL):
            return None

    match = re.search(r'resets\s+(\d+)([ap]m)', text, re.IGNORECASE)
    if not match:
        return AiResponseMetadata(backend=backend, rate_limited=True,
                                  rate_limit_reset_at=None, message='rate_limited')

    hour = int(match.group(1))
    ampm = match.group(2).lower()
    hour_24 = hour
    if ampm == 'pm' and hour != 12:
        hour_24 = hour + 12
    elif ampm == 'am' and hour == 12:
        hour_24 = 0

    now_utc = datetime.now(timezone.utc)
    reset_dt = now_utc.replace(hour=hour_24, minute=0, second=0, microsecond=0)
    if reset_dt <= now_utc:
        reset_dt += timedelta(days=1)
    reset_dt += timedelta(seconds=60)

    return AiResponseMetadata(backend=backend, rate_limited=True,
                              rate_limit_reset_at=reset_dt, message='rate_limited')


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
        if options.allow_all_tools:
            args.append('--dangerously-skip-permissions')
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
        if options.allow_all_tools:
            args.append('--dangerously-skip-permissions')
        args.extend(options.additional_args)
        return args

    def _parse_response(self, stdout: str, options: ResolvedOptions) -> AiResponse:
        trimmed = stdout.strip()

        # Rate limit check — takes priority over JSON parsing
        rate_limit_meta = _detect_rate_limit(trimmed)
        if rate_limit_meta is not None:
            return AiResponse(content='', status='error', metadata=rate_limit_meta)

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
        # Rate limit plain-text detection (before JSON parsing)
        rate_limit_meta = _detect_rate_limit(line)
        if rate_limit_meta is not None:
            raise RateLimitError(rate_limit_meta)

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
