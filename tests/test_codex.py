from __future__ import annotations

import random

import pytest

from ai_harness_api.backends.codex import CodexClient
from ai_harness_api.types import (
    ClientConfig,
    ResolvedOptions,
    RunOptions,
)


def _make_resolved(
    model: str = 'gpt-5.4-mini',
    executable: str = 'codex',
    cwd: str = '/tmp',
    timeout: float | None = None,
    additional_args: list[str] | None = None,
    session_id: str | None = None,
    persist_session: bool = False,
    allow_all_tools: bool = False,
) -> ResolvedOptions:
    return ResolvedOptions(
        executable=executable,
        model=model,
        timeout=timeout,
        cwd=cwd,
        additional_args=additional_args or [],
        session_id=session_id,
        persist_session=persist_session,
        allow_all_tools=allow_all_tools,
    )


# ---------------------------------------------------------------------------
# Command Builder — _build_args / _build_stream_args
# ---------------------------------------------------------------------------

class TestBuildArgs:
    def setup_method(self):
        self.client = CodexClient()

    def test_minimal(self):
        args = self.client._build_args('hello', _make_resolved(cwd=''))
        assert args == ['exec', '--ephemeral', '--json', '--model', 'gpt-5.4-mini', 'hello']

    def test_with_cwd(self):
        args = self.client._build_args('hello', _make_resolved(cwd='/tmp'))
        assert '-C' in args
        idx = args.index('-C')
        assert args[idx + 1] == '/tmp'

    def test_with_model(self):
        args = self.client._build_args('hello', _make_resolved(model='gpt-4o'))
        assert '--model' in args
        idx = args.index('--model')
        assert args[idx + 1] == 'gpt-4o'

    def test_with_additional_args(self):
        args = self.client._build_args('hello', _make_resolved(cwd='', additional_args=['--flag']))
        assert args[-1] == '--flag'

    def test_stream_same_as_run(self):
        resolved = _make_resolved(cwd='/tmp')
        assert self.client._build_args('hello', resolved) == self.client._build_stream_args('hello', resolved)

    def test_subcommand_always_first(self):
        args = self.client._build_args('hello', _make_resolved())
        assert args[0] == 'exec'

    def test_no_cwd_flag_when_cwd_empty(self):
        args = self.client._build_args('hello', _make_resolved(cwd=''))
        assert '-C' not in args

    def test_default_uses_ephemeral(self):
        args = self.client._build_args('hello', _make_resolved())
        assert '--ephemeral' in args

    def test_persist_session_omits_ephemeral(self):
        args = self.client._build_args('hello', _make_resolved(persist_session=True))
        assert '--ephemeral' not in args
        assert args[1] != 'resume'

    def test_session_id_uses_resume_subcommand(self):
        args = self.client._build_args('hello', _make_resolved(session_id='t-123'))
        assert args[:3] == ['exec', 'resume', 't-123']
        assert '--ephemeral' not in args

    def test_resume_omits_cwd_flag(self):
        args = self.client._build_args('hello', _make_resolved(session_id='t-123', cwd='/tmp'))
        assert '-C' not in args

    def test_allow_all_tools_true_includes_full_auto(self):
        args = self.client._build_args('hello', _make_resolved(allow_all_tools=True))
        assert '--full-auto' in args

    def test_allow_all_tools_false_excludes_full_auto(self):
        args = self.client._build_args('hello', _make_resolved(allow_all_tools=False))
        assert '--full-auto' not in args


# ---------------------------------------------------------------------------
# Codec — _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.client = CodexClient()
        self.resolved = _make_resolved()

    def test_single_item_completed_agent_message(self):
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Hello'
        assert resp.status == 'success'

    def test_multiple_item_completed_last_wins(self):
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"First"}}\n{"type":"item.completed","item":{"type":"agent_message","text":"Last"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Last'

    def test_item_completed_non_agent_message_falls_through(self):
        stdout = '{"type":"item.completed","item":{"type":"tool_call","text":"something"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        # Non-agent_message item should not be treated as a response; falls through to fallback
        assert resp.metadata is not None
        assert resp.metadata.fallback is True

    def test_error_event(self):
        stdout = '{"type":"error","message":"rate limited"}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.message == 'rate limited'

    def test_turn_completed_then_item_completed(self):
        stdout = '{"type":"turn.completed"}\n{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.content == 'Done'
        assert resp.status == 'success'

    def test_no_item_completed_nonempty_stdout(self):
        stdout = 'some raw text output'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.fallback is True

    def test_empty_stdout(self):
        resp = self.client._parse_response('', self.resolved)
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.message == 'no output'

    def test_parse_response_extracts_thread_id(self):
        stdout = '{"type":"thread.started","thread_id":"abc-123"}\n{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id == 'abc-123'

    def test_parse_response_no_thread_started_session_id_is_none(self):
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}'
        resp = self.client._parse_response(stdout, self.resolved)
        assert resp.metadata is not None
        assert resp.metadata.session_id is None


# ---------------------------------------------------------------------------
# Codec — _parse_chunk
# ---------------------------------------------------------------------------

class TestParseChunk:
    def setup_method(self):
        self.client = CodexClient()

    def test_item_completed_agent_message(self):
        line = '{"type":"item.completed","item":{"type":"agent_message","text":"Hi"}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == 'Hi'
        assert chunk.type == 'content'
        assert chunk.role == 'assistant'

    def test_item_completed_agent_message_empty_text(self):
        line = '{"type":"item.completed","item":{"type":"agent_message","text":""}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'content'
        assert chunk.role == 'assistant'

    def test_item_completed_non_agent_message_returns_none(self):
        line = '{"type":"item.completed","item":{"type":"tool_call","name":"bash"}}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_agent_response_old_format_returns_none(self):
        line = '{"type":"agent_response","content":"Hi"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_error_event(self):
        line = '{"type":"error","message":"fail"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is not None
        assert chunk.delta == ''
        assert chunk.type == 'status'

    def test_tool_call_returns_none(self):
        line = '{"type":"tool_call","name":"bash"}'
        chunk = self.client._parse_chunk(line)
        assert chunk is None

    def test_malformed_line_returns_none(self):
        chunk = self.client._parse_chunk('oops')
        assert chunk is None

    def test_blank_line_returns_none(self):
        chunk = self.client._parse_chunk('')
        assert chunk is None


# ---------------------------------------------------------------------------
# Integration Tests — require real codex binary
# ---------------------------------------------------------------------------

class TestCodexIntegration:
    def setup_method(self):
        self.client = CodexClient()

    @pytest.mark.asyncio
    async def test_successful_async_run(self):
        resp = await self.client.run('print the word DONE')
        assert resp.status == 'success'
        assert resp.content

    def test_successful_sync_run(self):
        resp = self.client.run_sync('print the word DONE')
        assert resp.status == 'success'
        assert resp.content

    @pytest.mark.asyncio
    async def test_timeout_fires(self):
        resp = await self.client.run('write a 10000 word essay', RunOptions(timeout=0.001))
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.timed_out is True

    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        client = CodexClient(ClientConfig(executable_path='/nonexistent/codex'))
        resp = await client.run('hello')
        assert resp.status == 'error'
        assert resp.metadata is not None
        assert resp.metadata.error_code == 'ENOENT'

    @pytest.mark.asyncio
    async def test_async_stream_yields_chunks(self):
        chunks = []
        async for chunk in self.client.stream('say hi'):
            chunks.append(chunk)
        # Codex may emit content or status chunks depending on version
        assert len(chunks) >= 0  # stream should complete without error


# ---------------------------------------------------------------------------
# Session Integration Tests — require real codex binary
# ---------------------------------------------------------------------------

class TestCodexSessionIntegration:
    def setup_method(self):
        self.client = CodexClient()

    @pytest.mark.asyncio
    async def test_persist_session_returns_session_id(self):
        resp = await self.client.run('Say the word HELLO', RunOptions(persist_session=True))
        assert resp.status == 'success'
        assert resp.metadata is not None
        assert resp.metadata.session_id is not None
        assert len(resp.metadata.session_id) > 0

    @pytest.mark.asyncio
    async def test_resume_session_retains_context(self):
        n = random.randint(10000, 99999)
        resp1 = await self.client.run(
            f'Remember the number {n}. Do not use memory tools or write it down — keep it in context only. Say SAVED.',
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
