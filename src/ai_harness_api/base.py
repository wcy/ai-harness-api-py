from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Iterator

from ai_harness_api.types import (
    AiChunk,
    AiResponse,
    AiResponseMetadata,
    ClientConfig,
    ResolvedOptions,
    RunOptions,
    SpawnOptions,
)


class AiCliClient(ABC):
    """Abstract base class for all AI CLI backend clients."""

    _default_executable: str
    _default_model: str

    def __init__(self, config: ClientConfig | None = None) -> None:
        self._default_config: ClientConfig = config if config is not None else ClientConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, prompt: str, options: RunOptions | None = None) -> AiResponse:
        """Execute a one-shot prompt and return the complete response. Never raises for runtime failures."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a non-empty string')

        resolved = self._merge_options(options)
        args = self._build_args(prompt, resolved)

        try:
            proc = await self._spawn_agent(SpawnOptions(
                executable=resolved.executable,
                args=args,
                cwd=resolved.cwd,
            ))
        except FileNotFoundError as exc:
            return AiResponse(
                content='',
                status='error',
                metadata=AiResponseMetadata(
                    backend=self._default_executable,
                    error_code='ENOENT',
                    message=str(exc),
                ),
            )

        try:
            async with asyncio.timeout(resolved.timeout):
                stdout_bytes, stderr_bytes = await proc.communicate()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return AiResponse(
                content='',
                status='error',
                metadata=AiResponseMetadata(
                    backend=self._default_executable,
                    timed_out=True,
                ),
            )

        stderr_str = stderr_bytes.decode()[:4096]

        if proc.returncode != 0:
            return AiResponse(
                content='',
                status='error',
                metadata=AiResponseMetadata(
                    backend=self._default_executable,
                    error_code=str(proc.returncode),
                    stderr=stderr_str,
                ),
            )

        return self._parse_response(stdout_bytes.decode(), resolved)

    def run_sync(self, prompt: str, options: RunOptions | None = None) -> AiResponse:
        """Synchronous wrapper for run(). Blocking; safe to call from non-async code."""
        return asyncio.run(self.run(prompt, options))

    async def stream(
        self, prompt: str, options: RunOptions | None = None
    ) -> AsyncGenerator[AiChunk, None]:
        """Stream the response as an async generator of chunks."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a non-empty string')

        resolved = self._merge_options(options)
        args = self._build_stream_args(prompt, resolved)

        try:
            proc = await self._spawn_agent(SpawnOptions(
                executable=resolved.executable,
                args=args,
                cwd=resolved.cwd,
            ))
        except FileNotFoundError:
            raise

        try:
            async with asyncio.timeout(resolved.timeout):
                async for line_bytes in proc.stdout:
                    line = line_bytes.decode().rstrip('\n')
                    if not line.strip():
                        continue

                    # prefix-split preprocessing
                    first_brace = line.find('{')
                    if first_brace > 0:
                        prefix = line[:first_brace].strip()
                        if prefix:
                            yield AiChunk(delta=prefix, type='error')
                        line = line[first_brace:]

                    chunk = self._parse_chunk(line)
                    if chunk is not None:
                        yield chunk

        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.communicate()

    def stream_sync(
        self, prompt: str, options: RunOptions | None = None
    ) -> Iterator[AiChunk]:
        """Synchronous wrapper for stream(). Yields chunks incrementally via a daemon thread."""
        chunk_queue: queue.Queue = queue.Queue()
        _DONE = object()

        async def _collect() -> None:
            try:
                async for chunk in self.stream(prompt, options):
                    chunk_queue.put(chunk)
            except Exception as exc:
                chunk_queue.put(exc)
            finally:
                chunk_queue.put(_DONE)

        thread = threading.Thread(
            target=lambda: asyncio.run(_collect()), daemon=True
        )
        thread.start()

        while True:
            item = chunk_queue.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

        thread.join()

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    async def _spawn_agent(self, options: SpawnOptions) -> asyncio.subprocess.Process:
        """Spawn the CLI process. Raises FileNotFoundError if executable not found."""
        proc = await asyncio.create_subprocess_exec(
            options.executable,
            *options.args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=options.cwd,
        )
        return proc

    def _merge_options(self, call_options: RunOptions | None) -> ResolvedOptions:
        """Merge call_options over _default_config, filling gaps with process-level defaults."""
        cfg = self._default_config

        # Resolve executable
        if call_options and call_options.executable_path:
            executable = call_options.executable_path
        elif cfg.executable_path:
            executable = cfg.executable_path
        else:
            executable = self._default_executable

        # Resolve model
        model: str
        if call_options and call_options.model is not None:
            model = call_options.model
        elif cfg.model is not None:
            model = cfg.model
        else:
            model = self._default_model

        # Resolve timeout
        timeout: float | None
        if call_options and call_options.timeout is not None:
            timeout = call_options.timeout
        elif cfg.timeout is not None:
            timeout = cfg.timeout
        else:
            timeout = None

        # Resolve cwd
        cwd: str
        if call_options and call_options.cwd is not None:
            cwd = call_options.cwd
        elif cfg.cwd is not None:
            cwd = cfg.cwd
        else:
            cwd = str(Path.cwd())

        # Concatenate additional_args
        additional_args: list[str] = list(cfg.additional_args)
        if call_options:
            additional_args = additional_args + list(call_options.additional_args)

        # Resolve session fields
        session_id = (
            call_options.session_id
            if call_options and call_options.session_id is not None
            else cfg.session_id
        )
        persist_session = (
            session_id is not None
            or (call_options.persist_session if call_options else False)
            or cfg.persist_session
        )

        return ResolvedOptions(
            executable=executable,
            model=model,
            timeout=timeout,
            cwd=cwd,
            additional_args=additional_args,
            session_id=session_id,
            persist_session=persist_session,
        )

    # ------------------------------------------------------------------
    # Abstract methods — implemented by each backend subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        """Return the CLI argument list for a one-shot run() call."""

    @abstractmethod
    def _build_stream_args(self, prompt: str, options: ResolvedOptions) -> list[str]:
        """Return the CLI argument list for a stream() call."""

    @abstractmethod
    def _parse_response(self, stdout: str, options: ResolvedOptions) -> AiResponse:
        """Parse complete buffered stdout into an AiResponse. Must never raise."""

    @abstractmethod
    def _parse_chunk(self, line: str) -> AiChunk | None:
        """Parse a single streamed line into an AiChunk, or None to skip."""
