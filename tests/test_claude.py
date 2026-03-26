from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from ai_harness_api.backends.claude import ClaudeClient, RateLimitError
from ai_harness_api.types import (
    AiChunk,
    AiResponse,
    AiResponseMetadata,
    AiUsage,
    ClientConfig,
    ResolvedOptions,
    RunOptions,
)


def _make_resolved(
    model: str = 'haiku',
    executable: str = 'claude',
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
        self.client = ClaudeClient()

    def test_minimal_run(self):
        args = self.client._build_args('hello', _make_resolved())
        assert args == ['-p', 'hello', '--no-session-persistence', '--output-format', 'json', '--model', 'haiku']

    def test_with_model(self):
        args = self.client._build_args('hello', _make_resolved(model='claude-opus-4-6'))
        assert '--model' in args
        idx = args.index('--model')
        assert args[idx + 1] == 'claude-opus-4-6'

    def test_with_additional_args(self):
        args = self.client._build_args('hello', _make_resolved(additional_args=['--verbose']))
        assert args[-1] == '--verbose'

    def test_no_session_flag_present_by_default(self):
        args = self.client._build_args('hello', _make_resolved())
        assert '--no-session-persistence' in args

    def test_persist_session_omits_no_persistence(self):
        args = self.client._build_args('hello', _make_resolved(persist_session=True))
        assert '--no-session-persistence' not in args
        assert '--resume' not in args

    def test_session_id_uses_resume_flag(self):
        args = self.client._build_args('hello', _make_resolved(session_id='abc-123'))
        assert '--resume' in args
        idx = args.index('--resume')
        assert args[idx + 1] == 'abc-123'
        assert '--no-session-persistence' not in args

    def test_arg_order(self):
        args = self.client._build_args('hello', _make_resolved(additional_args=['--extra']))
        assert args[0] == '-p'
        assert args[1] == 'hello'
        assert args[2] == '--no-session-persistence'
        assert args[3] == '--output-format'
        assert args[4] == 'json'
        assert args[5] == '--model'
        assert args[-1] == '--extra'


# ---------------------------------------------------------------------------
# Command Builder — _build_stream_args
# ---------------------------------------------------------------------------

class TestBuildStreamArgs:
    def setup_method(self):
        self.client = ClaudeClient()

    def test_stream_uses_stream_json(self):
        args = self.client._build_stream_args('hello', _make_resolved())
        assert '--output-format' in args
        idx = args.index('--output-format')
        assert args[idx + 1] == 'stream-json'

    def test_stream_includes_verbose(self):
        args = self.client._build_stream_args('hello', _make_resolved())
        assert '--verbose' in args

    def test_no_session_flag_in_stream_by_default(self):
        args = self.client._build_stream_args('hello', _make_resolved())
        assert '--no-session-persistence' in args

    def test_stream_persist_session_omits_no_persistence(self):
        args = self.client._build_stream_args('hello', _make_resolved(persist_session=True))
        assert '--no-session-persistence' not in args
        assert '--resume' not in args

    def test_stream_session_id_uses_resume_flag(self):
        args = self.client._build_stream_args('hello', _make_resolved(session_id='s-456'))
        assert '--resume' in args
        idx = args.index('--resume')
        assert args[idx + 1] == 's-456'
        assert '--no-session-persistence' not in args

    def test_stream_arg_order(self):
        args = self.client._build_stream_args('hello', _make_resolved(additional_args=['--extra']))
        assert args[0] == '-p'
        assert args[1] == 'hello'
        assert args[2] == '--no-session-persistence'
        assert args[3] == '--output-format'
        assert args[4] == 'stream-json'
        assert args[5] == '--verbose'
        assert args[6] == '--model'
        assert args[-1] == '--extra'


# ---------------------------------------------------------------------------
# Codec — _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.client = ClaudeClient()
        self.resolved = _make_resolved()

    def test_valid_json_with_result_field(self):
        stdout = '{"result":"Hello","model":"claude-3","usage":{"input_tokens":10,"output_tokens":5}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Hello'
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.backend == 'claude'
        assert resp.metadata.model == 'claude-3'
        assert resp.metadata.usage is not None
        assert resp.metadata.usage.input_tokens == 10
        assert resp.metadata.usage.output_tokens == 5

    def test_valid_json_with_content_field(self):
        stdout = '{"content":"Hi"}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Hi'
        assert resp.status == 'success'

    def test_valid_json_no_recognisable_field(self):
        stdout = '{"someOtherField":1}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == ''
        assert resp.status == 'success'

    def test_json_parse_fails_nonempty_stdout(self):
        stdout = 'This is plain text output'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'This is plain text output'
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.fallback is True

    def test_empty_stdout(self):
        resp = self.client._parse_response('', self.resolved)
        assert resp.content == ''
        assert resp.status == 'error'

    def test_whitespace_only_stdout(self):
        resp = self.client._parse_response('   ', self.resolved)
        assert resp.content == ''
        assert resp.status == 'error'

    def test_parse_response_extracts_session_id(self):
        stdout = '{"result":"Hi","session_id":"sess-001"}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id == 'sess-001'

    def test_parse_response_no_session_id_is_none(self):
        stdout = '{"result":"Hi"}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id is None

    def test_rate_limit_am_reset(self):
        stdout = "You've hit your limit · resets 5am (UTC)"
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.content == ''
        assert resp.metadata is not None
        assert resp.metadata.backend == 'claude'
        assert resp.metadata.rate_limited is True
        assert resp.metadata.message == 'rate_limited'
        assert resp.metadata.rate_limit_reset_at is not None
        assert isinstance(resp.metadata.rate_limit_reset_at, datetime)
        assert resp.metadata.rate_limit_reset_at.tzinfo is not None
        # Should be 5:01 AM UTC today or tomorrow
        reset = resp.metadata.rate_limit_reset_at
        assert reset.hour == 5
        assert reset.minute == 1
        assert reset.second == 0

    def test_rate_limit_pm_reset(self):
        stdout = "You've hit your limit · resets 11pm (UTC)"
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.content == ''
        assert resp.metadata is not None
        assert resp.metadata.rate_limited is True
        assert resp.metadata.rate_limit_reset_at is not None
        reset = resp.metadata.rate_limit_reset_at
        assert reset.hour == 23
        assert reset.minute == 1

    def test_rate_limit_past_advances_to_next_day(self):
        # Use a time that is almost certainly in the past today: 1am
        # If 1am already passed, it should be tomorrow
        stdout = "You've hit your limit · resets 1am (UTC)"
        now_utc = datetime.now(timezone.utc)
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.rate_limited is True
        reset = resp.metadata.rate_limit_reset_at
        assert reset is not None
        # reset_dt must be strictly in the future
        assert reset > now_utc

    def test_rate_limit_unparseable_token(self):
        stdout = "You've hit your limit · resets sometime soon"
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.rate_limited is True
        assert resp.metadata.rate_limit_reset_at is None

    def test_rate_limit_does_not_fall_through_to_fallback(self):
        stdout = "You've hit your limit · resets 5am (UTC)"
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.fallback is False


# ---------------------------------------------------------------------------
# Codec — _parse_chunk
# ---------------------------------------------------------------------------

class TestParseChunk:
    def setup_method(self):
        self.client = ClaudeClient()

    def test_assistant_with_text(self):
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"Hi there"}]}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == 'Hi there'
        assert chunk.type == 'content'

    def test_assistant_with_no_text_items(self):
        line = '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"..."}]}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'status'

    def test_system_event(self):
        line = '{"type":"system","subtype":"init"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'metadata'

    def test_result_event(self):
        line = '{"type":"result","subtype":"success"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'metadata'

    def test_unknown_type_returns_none(self):
        line = '{"type":"rate_limit_event"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match='Unparseable stream line'):
            self.client._parse_chunk('not json at all')

    def test_rate_limit_plain_text_raises_rate_limit_error(self):
        line = "You've hit your limit · resets 5am (UTC)"
        with pytest.raises(RateLimitError) as exc_info:
            self.client._parse_chunk(line)
        e = exc_info.value
        assert isinstance(e, ValueError)
        assert e.metadata.rate_limited is True
        assert e.metadata.rate_limit_reset_at is not None
        assert isinstance(e.metadata.rate_limit_reset_at, datetime)

    def test_rate_limit_unparseable_reset_raises_rate_limit_error(self):
        line = "You've hit your limit · resets sometime"
        with pytest.raises(RateLimitError) as exc_info:
            self.client._parse_chunk(line)
        e = exc_info.value
        assert isinstance(e, ValueError)
        assert e.metadata.rate_limited is True
        assert e.metadata.rate_limit_reset_at is None


# ---------------------------------------------------------------------------
# Integration Tests — require real claude binary
# ---------------------------------------------------------------------------

class TestClaudeIntegration:
    def setup_method(self):
        self.client = ClaudeClient()

    @pytest.mark.asyncio
    async def test_successful_async_run(self):
        resp = await self.client.run('Reply with the word OK only')
        assert resp.status == 'success'
        assert 'OK' in resp.content

    def test_successful_sync_run(self):
        resp = self.client.run_sync('Reply with the word OK only')
        assert resp.status == 'success'
        assert 'OK' in resp.content

    @pytest.mark.asyncio
    async def test_timeout_fires(self):
        resp = await self.client.run('Write a 10000 word essay', RunOptions(timeout=0.001))
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.timed_out is True

    def test_timeout_fires_sync(self):
        resp = self.client.run_sync('Write a 10000 word essay', RunOptions(timeout=0.001))
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.timed_out is True

    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        client = ClaudeClient(ClientConfig(executable_path='/nonexistent/claude'))
        resp = await client.run('hello')
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.error_code == 'ENOENT'

    @pytest.mark.asyncio
    async def test_async_stream_yields_chunks(self):
        chunks = []
        async for chunk in self.client.stream('Reply with OK'):
            chunks.append(chunk)
        content_chunks = [c for c in chunks if c.type == 'content' and c.delta]
        assert len(content_chunks) > 0

    def test_sync_stream_yields_chunks(self):
        chunks = list(self.client.stream_sync('Reply with OK'))
        content_chunks = [c for c in chunks if c.type == 'content' and c.delta]
        assert len(content_chunks) > 0


# ---------------------------------------------------------------------------
# Session Integration Tests — require real claude binary
# ---------------------------------------------------------------------------

class TestClaudeSessionIntegration:
    def setup_method(self):
        self.client = ClaudeClient()

    @pytest.mark.asyncio
    async def test_persist_session_returns_session_id(self):
        resp = await self.client.run('Reply with the word OK only', RunOptions(persist_session=True))
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
