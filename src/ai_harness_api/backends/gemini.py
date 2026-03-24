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


def _find_main_model(stats_models: dict | None) -> dict | None:
    """Find the primary model entry from the stats['models'] dict."""
    for name, entry in (stats_models or {}).items():
        if 'main' in (entry.get('roles') or {}):
            return {'name': name, 'tokens': entry.get('tokens')}
    return None


class GeminiClient(AiCliClient):
    """Backend client for the Gemini CLI."""

    _default_executable = 'gemini'
    _default_model = 'gemini-3-flash-preview'

    def _build_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        args = ['-p', prompt]
        args.extend(['-o', 'json'])
        args.extend(['--model', options.model])
        if options.session_id is not None:
            args.extend(['--resume', options.session_id])
        args.extend(options.additional_args)
        return args

    def _build_stream_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        args = ['-p', prompt]
        args.extend(['-o', 'stream-json'])
        args.extend(['--model', options.model])
        if options.session_id is not None:
            args.extend(['--resume', options.session_id])
        args.extend(options.additional_args)
        return args

    def _parse_response(self, stdout: str, options: ResolvedOptions) -> AiResponse:
        trimmed = stdout.strip()

        obj = None
        try:
            obj = json.loads(trimmed)
        except (json.JSONDecodeError, AttributeError):
            json_start = trimmed.find('{')
            if json_start >= 0:
                try:
                    obj = json.loads(trimmed[json_start:])
                except (json.JSONDecodeError, AttributeError):
                    pass

        if obj is not None:
            text = obj.get('response')

            if not isinstance(text, str):
                if trimmed:
                    return AiResponse(
                        content=trimmed,
                        status='success',
                        metadata=AiResponseMetadata(backend='gemini', fallback=True),
                    )
                return AiResponse(
                    content='',
                    status='error',
                    metadata=AiResponseMetadata(
                        backend='gemini', message='no text in response'
                    ),
                )

            main_model = _find_main_model((obj.get('stats') or {}).get('models'))
            usage = None
            if main_model:
                tokens = main_model.get('tokens') or {}
                usage = AiUsage(
                    input_tokens=tokens.get('prompt'),
                    output_tokens=tokens.get('candidates'),
                )

            return AiResponse(
                content=text,
                status='success',
                metadata=AiResponseMetadata(
                    backend='gemini',
                    model=main_model['name'] if main_model else None,
                    usage=usage,
                    session_id=obj.get('session_id'),
                ),
            )

        # obj is None — could not parse any JSON from output
        if trimmed:
            return AiResponse(
                content=trimmed,
                status='success',
                metadata=AiResponseMetadata(backend='gemini', fallback=True),
            )
        return AiResponse(
            content='',
            status='error',
            metadata=AiResponseMetadata(
                backend='gemini', message='empty or unparseable output'
            ),
        )

    def _parse_chunk(self, line: str) -> AiChunk | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'Unparseable stream line: {line}')

        if obj.get('type') == 'message' and obj.get('role') == 'assistant':
            return AiChunk(
                delta=obj.get('content', ''),
                type='content',
                role='assistant',
                raw=obj,
            )

        if obj.get('type') == 'result':
            return AiChunk(delta='', type='metadata', raw=obj)

        return None
