from __future__ import annotations

import random

import pytest

from ai_harness_api.backends.gemini import GeminiClient
from ai_harness_api.types import (
    ClientConfig,
    ResolvedOptions,
    RunOptions,
)


def _make_resolved(
    model: str = 'gemini-3-flash-preview',
    executable: str = 'gemini',
    cwd: str = '/tmp',
    timeout: float | None = None,
    additional_args: list[str] | None = None,
    session_id: str | None = None,
    persist_session: bool = False,
) -> ResolvedOptions:
    return ResolvedOptions(
        executable=executable,
        model=model,
        timeout=timeout,
        cwd=cwd,
        additional_args=additional_args or [],
        session_id=session_id,
        persist_session=persist_session,
    )


# ---------------------------------------------------------------------------
# Command Builder — _build_args
# ---------------------------------------------------------------------------

class TestBuildArgs:
    def setup_method(self):
        self.client = GeminiClient()

    def test_minimal_run(self):
        args = self.client._build_args('hello', _make_resolved())
        assert args == ['-p', 'hello', '-o', 'json', '--model', 'gemini-3-flash-preview']

    def test_with_model(self):
        args = self.client._build_args('hello', _make_resolved(model='gemini-2.0-flash'))
        assert '--model' in args
        idx = args.index('--model')
        assert args[idx + 1] == 'gemini-2.0-flash'

    def test_with_additional_args(self):
        args = self.client._build_args('hello', _make_resolved(additional_args=['--safety-settings', 'none']))
        assert args[-2] == '--safety-settings'
        assert args[-1] == 'none'

    def test_stream_uses_stream_json(self):
        args = self.client._build_stream_args('hello', _make_resolved())
        assert '-o' in args
        idx = args.index('-o')
        assert args[idx + 1] == 'stream-json'

    def test_no_resume_flag_by_default(self):
        args = self.client._build_args('hello', _make_resolved())
        assert '--resume' not in args

    def test_session_id_adds_resume_flag(self):
        args = self.client._build_args('hello', _make_resolved(session_id='5'))
        assert '--resume' in args
        idx = args.index('--resume')
        assert args[idx + 1] == '5'

    def test_stream_session_id_adds_resume_flag(self):
        args = self.client._build_stream_args('hello', _make_resolved(session_id='latest'))
        assert '--resume' in args
        idx = args.index('--resume')
        assert args[idx + 1] == 'latest'

    def test_arg_order(self):
        args = self.client._build_args('hello', _make_resolved())
        assert args[0] == '-p'
        assert args[1] == 'hello'
        assert args[2] == '-o'
        assert args[3] == 'json'
        assert args[4] == '--model'


# ---------------------------------------------------------------------------
# Codec — _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.client = GeminiClient()
        self.resolved = _make_resolved()

    def test_gemini_cli_wrapper_format(self):
        stdout = '{"response":"Hello","stats":{"models":{"gemini-1.5":{"tokens":{"prompt":10,"candidates":5},"roles":{"main":{}}}}}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Hello'
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.backend == 'gemini'
        assert resp.metadata.model == 'gemini-1.5'
        assert resp.metadata.usage is not None
        assert resp.metadata.usage.input_tokens == 10
        assert resp.metadata.usage.output_tokens == 5

    def test_response_absent(self):
        stdout = '{"session_id":"abc"}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.fallback is True

    def test_response_not_a_string(self):
        stdout = '{"response":42}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.fallback is True

    def test_json_parse_fails_nonempty_stdout(self):
        stdout = 'plain text response'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.fallback is True
        assert resp.content == 'plain text response'

    def test_empty_stdout(self):
        resp = self.client._parse_response('', self.resolved)
        assert resp.status == 'error'

    def test_parse_response_extracts_session_id(self):
        stdout = '{"response":"Hi","session_id":"g-sess-1","stats":{}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id == 'g-sess-1'

    def test_parse_response_no_session_id_is_none(self):
        stdout = '{"response":"Hi","stats":{}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id is None


# ---------------------------------------------------------------------------
# Codec — _parse_chunk
# ---------------------------------------------------------------------------

class TestParseChunk:
    def setup_method(self):
        self.client = GeminiClient()

    def test_assistant_message(self):
        line = '{"type":"message","role":"assistant","content":"hello"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == 'hello'
        assert chunk.type == 'content'
        assert chunk.role == 'assistant'

    def test_result_event(self):
        line = '{"type":"result","status":"success","stats":{}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'metadata'

    def test_init_event_returns_none(self):
        line = '{"type":"init","session_id":"abc"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_user_message_returns_none(self):
        line = '{"type":"message","role":"user","content":"hi"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match='Unparseable stream line'):
            self.client._parse_chunk('bad')


# ---------------------------------------------------------------------------
# Integration Tests — require real gemini binary
# ---------------------------------------------------------------------------

class TestGeminiIntegration:
    def setup_method(self):
        self.client = GeminiClient()

    @pytest.mark.asyncio
    async def test_successful_async_run(self):
        resp = await self.client.run('Reply with only the word YES')
        assert resp.status == 'success'
        assert 'YES' in resp.content

    def test_successful_sync_run(self):
        resp = self.client.run_sync('Reply with only the word YES')
        assert resp.status == 'success'
        assert 'YES' in resp.content

    @pytest.mark.asyncio
    async def test_timeout_fires(self):
        resp = await self.client.run('Write a 10000 word essay', RunOptions(timeout=0.001))
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.timed_out is True

    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        client = GeminiClient(ClientConfig(executable_path='/nonexistent/gemini'))
        resp = await client.run('hello')
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.error_code == 'ENOENT'

    @pytest.mark.asyncio
    async def test_async_stream_yields_chunks(self):
        chunks = []
        async for chunk in self.client.stream('say hello'):
            chunks.append(chunk)
        content_chunks = [c for c in chunks if c.type == 'content' and c.delta]
        assert len(content_chunks) > 0


# ---------------------------------------------------------------------------
# Session Integration Tests — require real gemini binary
# ---------------------------------------------------------------------------

class TestGeminiSessionIntegration:
    def setup_method(self):
        self.client = GeminiClient()

    @pytest.mark.asyncio
    async def test_persist_session_returns_session_id(self):
        resp = await self.client.run('Reply with only the word YES', RunOptions(persist_session=True))
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.session_id is not None
        assert len(resp.metadata.session_id) > 0

    @pytest.mark.asyncio
    async def test_resume_session_retains_context(self):
        n = random.randint(10000, 99999)
        resp1 = await self.client.run(
            f'Remember the number {n}. Do not use memory tools or write it down — keep it in context only. Reply SAVED.',
            RunOptions(persist_session=True),
        )
        assert resp1.status == 'success'
        assert resp1.metadata is not None
        assert resp1.metadata.session_id is not None

        resp2 = await self.client.run(
            'What number did I ask you to remember? Reply with only the number.',
            RunOptions(session_id=resp1.metadata.session_id),
        )
        assert resp2.status == 'success'
        assert str(n) in resp2.content
