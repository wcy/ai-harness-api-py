from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator

from ai_harness_api.cli.backends import BACKENDS
from ai_harness_api.types import AiChunk, ClientConfig


def _add_subcommand(subparsers: argparse._SubParsersAction, name: str) -> None:
    sub = subparsers.add_parser(name, help=f'Run the {name} backend')
    sub.add_argument('prompt', help='Prompt to send to the AI backend')
    sub.add_argument('--model', default=None)
    sub.add_argument('--timeout', type=float, default=None)
    sub.add_argument('--cwd', default=None)
    sub.add_argument('--executable-path', dest='executable_path', default=None)
    sub.add_argument('--new-session', dest='new_session', action='store_true', default=False)
    sub.add_argument('--session-id', dest='session_id', default=None)
    sub.add_argument('--allow-all-tools', dest='allow_all_tools', action='store_true', default=False)
    sub.add_argument('--output', dest='output', choices=['json', 'jsonl', 'text'], default='json')


def _validate_flags(args: argparse.Namespace) -> None:
    if args.timeout is not None and args.timeout <= 0:
        print(
            f'error: --timeout must be > 0, got {args.timeout}',
            file=sys.stderr,
        )
        sys.exit(1)
    if args.cwd is not None and not args.cwd:
        print('error: --cwd must be a non-empty string', file=sys.stderr)
        sys.exit(1)
    if args.executable_path is not None and not args.executable_path:
        print('error: --executable-path must be a non-empty string', file=sys.stderr)
        sys.exit(1)
    if args.new_session is True and args.session_id is not None:
        print('error: --new-session and --session-id are mutually exclusive', file=sys.stderr)
        sys.exit(1)


def _render_json(iterable: Iterator[AiChunk], new_session: bool = False) -> int:
    message: list[str] = []
    session_id = None
    input_tokens = None
    output_tokens = None
    exit_code = 0

    try:
        for chunk in iterable:
            if chunk.type == 'content':
                message.append(chunk.delta)
            elif chunk.type == 'metadata':
                raw = chunk.raw or {}
                if new_session is True and 'session_id' in raw:
                    session_id = raw['session_id']
                usage = raw.get('usage') or {}
                if 'input_tokens' in usage:
                    input_tokens = usage['input_tokens']
                if 'output_tokens' in usage:
                    output_tokens = usage['output_tokens']
            elif chunk.type == 'status':
                raw = chunk.raw or {}
                error_msg = raw.get('error')
                if error_msg:
                    print(f'error: {error_msg}', file=sys.stderr)
                    exit_code = 1
                    break
            elif chunk.type == 'error':
                sys.stderr.write(chunk.delta)
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        exit_code = 1

    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    result = {
        'message': ''.join(message),
        'session_id': session_id,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
    }
    sys.stdout.write(json.dumps(result) + '\n')
    return exit_code


def _render_jsonl(iterable: Iterator[AiChunk]) -> int:
    exit_code = 0

    try:
        for chunk in iterable:
            line = {'delta': chunk.delta, 'type': chunk.type, 'role': chunk.role, 'raw': chunk.raw}
            sys.stdout.write(json.dumps(line) + '\n')
            if chunk.type == 'status':
                raw = chunk.raw or {}
                error_msg = raw.get('error')
                if error_msg:
                    print(f'error: {error_msg}', file=sys.stderr)
                    exit_code = 1
                    break
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        exit_code = 1

    sys.stdout.write('\n')
    return exit_code


def _render_text(iterable: Iterator[AiChunk], new_session: bool = False) -> int:
    last_role: str | None = None
    session_id: str | None = None
    exit_code = 0

    try:
        for chunk in iterable:
            if chunk.type == 'content':
                if chunk.role is not None and chunk.role != last_role and last_role is not None:
                    sys.stdout.write('\n')
                last_role = chunk.role if chunk.role is not None else last_role
                sys.stdout.write(chunk.delta)
            elif chunk.type == 'metadata':
                raw = chunk.raw or {}
                if new_session is True and 'session_id' in raw:
                    session_id = raw['session_id']
            elif chunk.type == 'status':
                raw = chunk.raw or {}
                error_msg = raw.get('error')
                if error_msg:
                    print(f'error: {error_msg}', file=sys.stderr)
                    exit_code = 1
                    break
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        exit_code = 1

    sys.stdout.write('\n')
    if new_session is True and session_id is not None:
        sys.stderr.write(f'session_id: {session_id}\n')
    return exit_code


def _run_backend(args: argparse.Namespace) -> int:
    config = ClientConfig(
        model=args.model,
        timeout=args.timeout,
        cwd=args.cwd,
        executable_path=args.executable_path,
        persist_session=args.new_session or args.session_id is not None,
        session_id=args.session_id,
        allow_all_tools=args.allow_all_tools,
    )

    entry = BACKENDS[args.backend]
    client = entry['create'](config)

    try:
        iterable = client.stream_sync(args.prompt)
        if args.output == 'json':
            return _render_json(iterable, new_session=args.new_session)
        elif args.output == 'jsonl':
            return _render_jsonl(iterable)
        else:
            return _render_text(iterable, new_session=args.new_session)
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='aicli',
        description='Run AI CLI backends from the terminal',
    )
    subparsers = parser.add_subparsers(dest='backend', required=True)

    for name in ('claude', 'codex', 'gemini'):
        _add_subcommand(subparsers, name)

    args = parser.parse_args()
    _validate_flags(args)
    sys.exit(_run_backend(args))
