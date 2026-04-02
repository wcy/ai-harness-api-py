from __future__ import annotations

import json
import random
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_harness_api.cli.backends import BACKENDS
from ai_harness_api.cli.main import (
    _render_json,
    _render_jsonl,
    _render_text,
    _run_backend,
    _validate_flags,
    main,
)
from ai_harness_api.types import AiChunk, ClientConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]):
    """Parse args using the real argparse setup."""
    import argparse
    from ai_harness_api.cli.main import _add_subcommand

    parser = argparse.ArgumentParser(prog='aicli')
    subparsers = parser.add_subparsers(dest='backend', required=True)
    for name in ('claude', 'codex', 'gemini'):
        _add_subcommand(subparsers, name)
    return parser.parse_args(argv)


def _make_chunks(chunks):
    """Return an iterator or raise-on-call mock."""
    if isinstance(chunks, Exception):
        def bad_gen():
            raise chunks
            yield  # make it a generator
        return bad_gen()
    return iter(chunks)


# ---------------------------------------------------------------------------
# Flag parsing — flags map to ClientConfig
# ---------------------------------------------------------------------------

class TestFlagParsing:
    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_model_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--model', 'claude-opus-4-6'])
        assert args.model == 'claude-opus-4-6'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_timeout_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--timeout', '5.0'])
        assert args.timeout == 5.0

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_cwd_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--cwd', '/tmp'])
        assert args.cwd == '/tmp'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_executable_path_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--executable-path', '/usr/local/bin/claude'])
        assert args.executable_path == '/usr/local/bin/claude'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_new_session_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--new-session'])
        assert args.new_session is True

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_new_session_default_false(self, backend):
        args = _parse_args([backend, 'prompt'])
        assert args.new_session is False

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_session_id_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--session-id', 'abc123'])
        assert args.session_id == 'abc123'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_session_id_default_none(self, backend):
        args = _parse_args([backend, 'prompt'])
        assert args.session_id is None

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_allow_all_tools_flag(self, backend):
        args = _parse_args([backend, 'prompt', '--allow-all-tools'])
        assert args.allow_all_tools is True

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_allow_all_tools_default_false(self, backend):
        args = _parse_args([backend, 'prompt'])
        assert args.allow_all_tools is False

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_additional_args_not_accepted(self, backend):
        with pytest.raises(SystemExit) as exc:
            _parse_args([backend, 'prompt', '--additional-args=--foo'])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Flag parsing — --output flag
# ---------------------------------------------------------------------------

class TestOutputFlagParsing:
    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_default_output_is_json(self, backend):
        args = _parse_args([backend, 'prompt'])
        assert args.output == 'json'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_output_json(self, backend):
        args = _parse_args([backend, 'prompt', '--output', 'json'])
        assert args.output == 'json'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_output_jsonl(self, backend):
        args = _parse_args([backend, 'prompt', '--output', 'jsonl'])
        assert args.output == 'jsonl'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_output_text(self, backend):
        args = _parse_args([backend, 'prompt', '--output', 'text'])
        assert args.output == 'text'

    @pytest.mark.parametrize('backend', ['claude', 'codex', 'gemini'])
    def test_output_xml_exits_2(self, backend):
        with pytest.raises(SystemExit) as exc:
            _parse_args([backend, 'prompt', '--output', 'xml'])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------

class TestValidateFlags:
    def test_timeout_zero_exits_1(self):
        args = _parse_args(['claude', 'hi', '--timeout', '5.0'])
        args.timeout = 0
        with pytest.raises(SystemExit) as exc:
            _validate_flags(args)
        assert exc.value.code == 1

    def test_timeout_negative_exits_1(self):
        args = _parse_args(['claude', 'hi', '--timeout', '5.0'])
        args.timeout = -1
        with pytest.raises(SystemExit) as exc:
            _validate_flags(args)
        assert exc.value.code == 1

    def test_valid_timeout_passes(self):
        args = _parse_args(['claude', 'hi', '--timeout', '5.0'])
        _validate_flags(args)  # should not raise

    def test_timeout_type_error_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse_args(['claude', 'hi', '--timeout', 'abc'])
        assert exc.value.code == 2

    def test_missing_prompt_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            _parse_args(['claude'])
        assert exc.value.code == 2

    def test_new_session_and_session_id_mutually_exclusive(self, capsys):
        args = _parse_args(['claude', 'hi', '--new-session', '--session-id', 'abc'])
        with pytest.raises(SystemExit) as exc:
            _validate_flags(args)
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert 'error' in captured.err.lower()


# ---------------------------------------------------------------------------
# _render_text unit tests
# ---------------------------------------------------------------------------

class TestRenderText:
    def test_content_chunks_written_to_stdout(self, capsys):
        chunks = [
            AiChunk(type='content', delta='Hello'),
            AiChunk(type='content', delta=' world'),
        ]
        code = _render_text(iter(chunks))
        captured = capsys.readouterr()
        assert 'Hello world' in captured.out
        assert captured.out.endswith('\n')
        assert code == 0

    def test_metadata_chunks_skipped_no_session_id_on_stderr(self, capsys):
        chunks = [
            AiChunk(type='metadata', delta=''),
            AiChunk(type='content', delta='ok'),
        ]
        code = _render_text(iter(chunks))
        captured = capsys.readouterr()
        assert 'ok' in captured.out
        assert 'session_id' not in captured.err
        assert code == 0

    def test_metadata_session_id_written_to_stderr(self, capsys):
        chunks = [
            AiChunk(type='content', delta='hi'),
            AiChunk(type='metadata', delta='', raw={'session_id': 'sess-42'}),
        ]
        code = _render_text(iter(chunks), new_session=True)
        captured = capsys.readouterr()
        assert 'hi' in captured.out
        assert 'session_id: sess-42\n' in captured.err
        assert code == 0

    def test_metadata_no_session_id_no_stderr_line(self, capsys):
        chunks = [
            AiChunk(type='content', delta='hi'),
            AiChunk(type='metadata', delta='', raw={'usage': {'input_tokens': 5}}),
        ]
        code = _render_text(iter(chunks), new_session=True)
        captured = capsys.readouterr()
        assert 'session_id' not in captured.err
        assert code == 0

    def test_last_metadata_session_id_wins(self, capsys):
        chunks = [
            AiChunk(type='metadata', delta='', raw={'session_id': 'first'}),
            AiChunk(type='metadata', delta='', raw={'session_id': 'second'}),
        ]
        code = _render_text(iter(chunks), new_session=True)
        captured = capsys.readouterr()
        assert 'session_id: second\n' in captured.err
        assert code == 0

    def test_session_id_suppressed_when_new_session_false(self, capsys):
        chunks = [
            AiChunk(type='content', delta='Hello'),
            AiChunk(type='metadata', delta='', raw={'session_id': 'abc123'}),
        ]
        code = _render_text(iter(chunks), new_session=False)
        captured = capsys.readouterr()
        assert captured.out == 'Hello\n'
        assert 'session_id' not in captured.err
        assert code == 0

    def test_status_error_chunk_triggers_exit_1(self, capsys):
        chunks = [AiChunk(type='status', delta='', raw={'error': 'timeout'})]
        code = _render_text(iter(chunks))
        captured = capsys.readouterr()
        assert 'timeout' in captured.err or 'error' in captured.err.lower()
        assert code == 1

    def test_trailing_newline_always_written(self, capsys):
        code = _render_text(iter([]))
        captured = capsys.readouterr()
        assert captured.out == '\n'

    def test_stream_exception_triggers_exit_1(self, capsys):
        code = _render_text(_make_chunks(RuntimeError('spawn failed')))
        captured = capsys.readouterr()
        assert 'spawn failed' in captured.err
        assert code == 1


# ---------------------------------------------------------------------------
# _render_json unit tests
# ---------------------------------------------------------------------------

class TestRenderJson:
    def test_content_chunks_assembled_into_message(self, capsys):
        chunks = [
            AiChunk(type='content', delta='Hello'),
            AiChunk(type='content', delta=' world'),
        ]
        code = _render_json(iter(chunks))
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['message'] == 'Hello world'
        assert obj['session_id'] is None
        assert obj['input_tokens'] is None
        assert obj['output_tokens'] is None
        assert obj['total_tokens'] is None
        assert code == 0

    def test_metadata_chunk_populates_token_counts_and_session_id(self, capsys):
        chunks = [
            AiChunk(type='content', delta='Hi'),
            AiChunk(type='metadata', delta='', raw={
                'session_id': 'abc',
                'usage': {'input_tokens': 10, 'output_tokens': 5},
            }),
        ]
        code = _render_json(iter(chunks), new_session=True)
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['session_id'] == 'abc'
        assert obj['input_tokens'] == 10
        assert obj['output_tokens'] == 5
        assert obj['total_tokens'] == 15
        assert code == 0

    def test_session_id_null_when_new_session_false(self, capsys):
        chunks = [
            AiChunk(type='content', delta='Hi'),
            AiChunk(type='metadata', delta='', raw={
                'session_id': 'abc',
                'usage': {'input_tokens': 10, 'output_tokens': 5},
            }),
        ]
        code = _render_json(iter(chunks), new_session=False)
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['session_id'] is None
        assert obj['input_tokens'] == 10
        assert obj['output_tokens'] == 5
        assert code == 0

    def test_total_tokens_null_when_only_input_tokens(self, capsys):
        chunks = [
            AiChunk(type='metadata', delta='', raw={
                'usage': {'input_tokens': 10},
            }),
        ]
        code = _render_json(iter(chunks))
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['total_tokens'] is None

    def test_last_metadata_chunk_wins(self, capsys):
        chunks = [
            AiChunk(type='metadata', delta='', raw={
                'session_id': 'first',
                'usage': {'input_tokens': 1, 'output_tokens': 1},
            }),
            AiChunk(type='metadata', delta='', raw={
                'session_id': 'second',
                'usage': {'input_tokens': 20, 'output_tokens': 10},
            }),
        ]
        code = _render_json(iter(chunks), new_session=True)
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['session_id'] == 'second'
        assert obj['input_tokens'] == 20
        assert obj['output_tokens'] == 10
        assert obj['total_tokens'] == 30

    def test_status_error_chunk_triggers_exit_1(self, capsys):
        chunks = [AiChunk(type='status', delta='', raw={'error': 'timeout'})]
        code = _render_json(iter(chunks))
        captured = capsys.readouterr()
        assert 'timeout' in captured.err or 'error' in captured.err.lower()
        assert code == 1
        # stdout still receives a JSON object
        obj = json.loads(captured.out.strip())
        assert 'message' in obj

    def test_stream_exception_triggers_exit_1(self, capsys):
        code = _render_json(_make_chunks(RuntimeError('spawn failed')))
        captured = capsys.readouterr()
        assert 'spawn failed' in captured.err
        assert code == 1


# ---------------------------------------------------------------------------
# _render_jsonl unit tests
# ---------------------------------------------------------------------------

class TestRenderJsonl:
    def test_each_chunk_emitted_as_json_line(self, capsys):
        chunks = [
            AiChunk(type='content', delta='A', role='assistant'),
            AiChunk(type='metadata', delta='', raw={'x': 1}),
        ]
        code = _render_jsonl(iter(chunks))
        captured = capsys.readouterr()
        lines = [l for l in captured.out.split('\n') if l.strip()]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {'delta': 'A', 'type': 'content', 'role': 'assistant', 'raw': None}
        second = json.loads(lines[1])
        assert second == {'delta': '', 'type': 'metadata', 'role': None, 'raw': {'x': 1}}
        assert code == 0

    def test_status_error_chunk_emitted_then_exit_1(self, capsys):
        chunks = [AiChunk(type='status', delta='', raw={'error': 'fail'})]
        code = _render_jsonl(iter(chunks))
        captured = capsys.readouterr()
        # The status chunk should be emitted as a JSON line
        lines = [l for l in captured.out.split('\n') if l.strip()]
        assert len(lines) >= 1
        obj = json.loads(lines[0])
        assert obj['type'] == 'status'
        # stderr contains error message
        assert 'fail' in captured.err or 'error' in captured.err.lower()
        assert code == 1

    def test_stream_exception_triggers_exit_1(self, capsys):
        code = _render_jsonl(_make_chunks(RuntimeError('spawn failed')))
        captured = capsys.readouterr()
        assert 'spawn failed' in captured.err
        assert code == 1


# ---------------------------------------------------------------------------
# _run_backend — stream exception caught at call time
# ---------------------------------------------------------------------------

class TestRunBackend:
    def _make_args(self, backend: str = 'claude', prompt: str = 'hello', output: str = 'text', **kwargs):
        import argparse
        defaults = dict(
            backend=backend,
            prompt=prompt,
            model=None,
            timeout=None,
            cwd=None,
            executable_path=None,
            new_session=False,
            session_id=None,
            allow_all_tools=False,
            output=output,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _mock_client(self, chunks):
        client = MagicMock()
        if isinstance(chunks, Exception):
            client.stream_sync.side_effect = chunks
        else:
            client.stream_sync.return_value = iter(chunks)
        return client

    def test_content_chunks_written_to_stdout(self, capsys):
        args = self._make_args()
        chunks = [
            AiChunk(type='content', delta='Hello'),
            AiChunk(type='content', delta=' world'),
        ]
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        assert 'Hello world' in captured.out
        assert captured.out.endswith('\n')
        assert code == 0

    def test_metadata_chunks_silently_skipped(self, capsys):
        args = self._make_args()
        chunks = [
            AiChunk(type='metadata', delta=''),
            AiChunk(type='content', delta='ok'),
        ]
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        assert 'ok' in captured.out
        assert code == 0

    def test_status_error_chunk_triggers_exit_1(self, capsys):
        args = self._make_args()
        chunks = [AiChunk(type='status', delta='', raw={'error': 'timeout'})]
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        assert 'timeout' in captured.err or 'error' in captured.err.lower()
        assert code == 1

    def test_stream_exception_triggers_exit_1(self, capsys):
        args = self._make_args()
        mock_client = self._mock_client(RuntimeError('spawn failed'))
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        assert 'spawn failed' in captured.err
        assert code == 1

    def test_trailing_newline_always_written(self, capsys):
        args = self._make_args()
        chunks = []
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            _run_backend(args)
        captured = capsys.readouterr()
        assert captured.out == '\n'

    def test_json_output_dispatches_to_json_renderer(self, capsys):
        args = self._make_args(output='json')
        chunks = [AiChunk(type='content', delta='hi')]
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert obj['message'] == 'hi'
        assert code == 0

    def test_jsonl_output_dispatches_to_jsonl_renderer(self, capsys):
        args = self._make_args(output='jsonl')
        chunks = [AiChunk(type='content', delta='hi')]
        mock_client = self._mock_client(chunks)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': lambda c: mock_client}
        }):
            code = _run_backend(args)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.split('\n') if l.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj['type'] == 'content'
        assert code == 0

    def test_default_session_is_ephemeral(self):
        """No flags → ephemeral call; persist_session is False."""
        captured_config = {}

        def capture_create(c):
            captured_config['config'] = c
            return self._mock_client([])

        args = self._make_args()
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': capture_create}
        }):
            _run_backend(args)

        assert captured_config['config'].persist_session is False

    def test_new_session_flag_sets_persist_session(self):
        """--new-session → persist_session is True."""
        captured_config = {}

        def capture_create(c):
            captured_config['config'] = c
            return self._mock_client([])

        args = self._make_args(new_session=True)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': capture_create}
        }):
            _run_backend(args)

        assert captured_config['config'].persist_session is True

    def test_session_id_flag_sets_persist_session(self):
        """--session-id → persist_session is True so --no-session-persistence is not added."""
        captured_config = {}

        def capture_create(c):
            captured_config['config'] = c
            return self._mock_client([])

        args = self._make_args(session_id='abc-123')
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': capture_create}
        }):
            _run_backend(args)

        assert captured_config['config'].persist_session is True

    def test_allow_all_tools_flag_sets_allow_all_tools(self):
        """--allow-all-tools → ClientConfig.allow_all_tools == True."""
        captured_config = {}

        def capture_create(c):
            captured_config['config'] = c
            return self._mock_client([])

        args = self._make_args(allow_all_tools=True)
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': capture_create}
        }):
            _run_backend(args)

        assert captured_config['config'].allow_all_tools is True

    def test_allow_all_tools_absent_leaves_false(self):
        """No --allow-all-tools flag → ClientConfig.allow_all_tools == False."""
        captured_config = {}

        def capture_create(c):
            captured_config['config'] = c
            return self._mock_client([])

        args = self._make_args()
        with patch.dict('ai_harness_api.cli.backends.BACKENDS', {
            'claude': {'create': capture_create}
        }):
            _run_backend(args)

        assert captured_config['config'].allow_all_tools is False


# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

class TestBackendRegistry:
    def test_all_backends_registered(self):
        assert 'claude' in BACKENDS
        assert 'codex' in BACKENDS
        assert 'gemini' in BACKENDS

    def test_claude_create_returns_client(self):
        config = ClientConfig()
        client = BACKENDS['claude']['create'](config)
        assert hasattr(client, 'stream_sync')

    def test_codex_create_returns_client(self):
        config = ClientConfig()
        client = BACKENDS['codex']['create'](config)
        assert hasattr(client, 'stream_sync')

    def test_gemini_create_returns_client(self):
        config = ClientConfig()
        client = BACKENDS['gemini']['create'](config)
        assert hasattr(client, 'stream_sync')

    def test_create_passes_config(self):
        from ai_harness_api.backends.claude import ClaudeClient
        config = ClientConfig(model='test-model')
        client = BACKENDS['claude']['create'](config)
        assert isinstance(client, ClaudeClient)
        assert client._default_config.model == 'test-model'


# ---------------------------------------------------------------------------
# End-to-end integration tests — require real 'claude' binary on PATH
# ---------------------------------------------------------------------------

_AICLI = str(Path(sys.executable).parent / 'aicli')


class TestE2ESessionResume:
    def test_session_id_resumes_session(self):
        """--session-id resumes a session: second invocation recalls a number from context."""
        # First invocation: create a new session and ask the model to remember a random number.
        # The prompt explicitly forbids memory/persistence tools so the test exercises
        # session context recall, not tool-based recall.
        n = random.randint(10000, 99999)
        result1 = subprocess.run(
            [
                _AICLI,
                'claude', '--new-session',
                f'Remember the number {n}. Do not use memory tools or write it down — keep it in context only.',
            ],
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0, (
            f'First invocation failed (rc={result1.returncode}):\nstdout={result1.stdout}\nstderr={result1.stderr}'
        )
        first_output = json.loads(result1.stdout.strip())
        session_id = first_output['session_id']
        assert session_id, (
            f'Expected a non-empty session_id in first response, got: {first_output!r}'
        )

        # Second invocation: resume the session and ask for the remembered number.
        result2 = subprocess.run(
            [
                _AICLI,
                'claude', '--session-id', session_id,
                'What number did I ask you to remember?',
            ],
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, (
            f'Second invocation failed (rc={result2.returncode}):\nstdout={result2.stdout}\nstderr={result2.stderr}'
        )
        second_output = json.loads(result2.stdout.strip())
        response_text = second_output['message']
        assert str(n) in response_text, (
            f'Expected "{n}" in second response message, got: {response_text!r}'
        )
