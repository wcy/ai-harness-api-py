# ai-harness-api

`ai_harness_api` is a unified Python library that provides a consistent programmatic interface for interacting with AI-powered CLI agents: Claude Code (`claude`), Codex CLI (`codex`), and Gemini CLI (`gemini`). It abstracts each tool's unique invocation flags, output formats, and process lifecycle behind a single, coherent API that developers can import like any SDK.

---

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) package manager
- One or more AI CLI binaries available on `PATH`, each already authenticated. This library spawns
  them as subprocesses; it does not talk to any model API directly, and it performs no version or
  capability check on them.
  - `claude` — Claude Code
  - `codex` — OpenAI Codex CLI
  - `gemini` — Google Gemini CLI

---

## Installation

UV-first:

```bash
uv add ai-harness-api
```

Standard pip fallback:

```bash
pip install ai-harness-api
```

Development install (for contributors):

```bash
git clone <repo>
cd ai-harness-api
uv sync --extra dev
```

---

## API Reference

### Quick Start

```python
from ai_harness_api import ClaudeClient

client = ClaudeClient()
response = client.run_sync("What is 2 + 2?")
print(response.content)
```

### Clients

Each backend has its own client class. All three expose the same methods and accept the same
config — but they are **not** behaviourally identical. See [Backend Differences](#backend-differences).

```python
from ai_harness_api import ClaudeClient, CodexClient, GeminiClient
from ai_harness_api import ClientConfig

claude  = ClaudeClient(ClientConfig(model="haiku", timeout=60.0))
codex   = CodexClient(ClientConfig(model="gpt-5.4-mini", cwd="/my/project"))
gemini  = GeminiClient(ClientConfig(model="gemini-3-flash-preview"))
```

### ClientConfig

Construction-time settings, frozen after creation. Every field is optional.

| Field | Type | Default | Description |
|---|---|---|---|
| `executable_path` | `str \| None` | None | Binary name or absolute path; a bare name is resolved on `PATH` |
| `model` | `str \| None` | None | Model to pass to the backend |
| `timeout` | `float \| None` | None | Abort after N seconds; `None` means unbounded. Must be `> 0` |
| `cwd` | `str \| None` | None | Working directory; defaults to the process cwd at call time. Must be non-empty |
| `additional_args` | `list[str]` | `[]` | Raw backend-specific CLI flags, appended last |
| `session_id` | `str \| None` | None | Resume this session |
| `persist_session` | `bool` | `False` | Start a durable session |
| `allow_all_tools` | `bool` | `False` | Bypass all tool-approval prompts |

**Default models** — each client hardcodes one and sends `--model` on **every** call, so the CLI's
own configured default is never used:

| Client | Default model |
|---|---|
| `ClaudeClient` | `haiku` |
| `CodexClient` | `gpt-5.4-mini` |
| `GeminiClient` | `gemini-3-flash-preview` |

### One-shot Execution

`run_sync` (blocking) and `run` (async):

```python
# Blocking
response = client.run_sync("Explain asyncio")
print(response.content)           # assistant text
print(response.status)            # 'success' or 'error'
print(response.metadata.backend)  # 'claude'

# Async
import asyncio

async def main():
    response = await client.run("Explain asyncio")
    print(response.content)

asyncio.run(main())
```

### Per-call Options (RunOptions)

`RunOptions` has the same fields as `ClientConfig` and is merged over it per call. **Five of the
eight fields override; three do not** — see the Merge column below.

```python
from ai_harness_api import RunOptions

response = client.run_sync(
    "Summarise this file",
    options=RunOptions(cwd="/my/project", timeout=30.0, model="sonnet"),
)
```

| Field | Type | Default | Merge | Description |
|---|---|---|---|---|
| `cwd` | `str \| None` | None | call wins | Override working directory for this call |
| `executable_path` | `str \| None` | None | call wins | Override binary path for this call |
| `model` | `str \| None` | None | call wins | Override model for this call |
| `timeout` | `float \| None` | None | call wins | Override timeout (seconds). Cannot be disabled per call — `0` is rejected and `None` reads as "unset" |
| `additional_args` | `list[str]` | `[]` | **concatenated** | Extra flags, appended after the instance's. A call can add args but never remove them |
| `session_id` | `str \| None` | None | call wins | Resume this session. Cannot be reset to `None` if the instance has one |
| `persist_session` | `bool` | `False` | **OR** | Start a durable session. Forced `True` by any `session_id`; a call cannot turn it off |
| `allow_all_tools` | `bool` | `False` | **OR** | Bypass all tool-approval prompts. See the warning below |

> **`allow_all_tools` can be widened at call time but never narrowed.** It merges with a boolean OR,
> so a client built with `ClientConfig(allow_all_tools=True)` ignores a later
> `RunOptions(allow_all_tools=False)` and stays `True`. There is no `allowed_tools`,
> `disallowed_tools`, or `permission_mode` setting — narrowing tool grants is expressible only as raw
> backend flags via `additional_args`, and `cwd` is only the OS working directory, not a permission
> boundary. If you need bounded permissions, do not set `allow_all_tools` anywhere.

### Streaming

`stream_sync` (blocking iterator) and `stream` (async generator):

```python
# Blocking stream
for chunk in client.stream_sync("Write a haiku"):
    if chunk.type == "content":
        print(chunk.delta, end="", flush=True)

# Async stream
async def stream_it():
    async for chunk in client.stream("Write a haiku"):
        if chunk.type == "content":
            print(chunk.delta, end="", flush=True)

asyncio.run(stream_it())
```

### AiResponse and AiChunk Types

**AiResponse** — returned by `run()` and `run_sync()`:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Final text output; empty string on error |
| `status` | `'success' \| 'error'` | Outcome |
| `metadata` | `AiResponseMetadata \| None` | Backend, model, token usage, error info |

**AiChunk** — emitted by `stream()` and `stream_sync()`:

| Field | Type | Description |
|---|---|---|
| `delta` | `str` | New text in this chunk; empty for non-content events |
| `type` | `'content' \| 'metadata' \| 'status' \| 'error'` | Chunk kind. **No backend emits `'error'`** — it is produced only when a line has non-JSON noise before the payload |
| `role` | `str \| None` | Conversational role. Set to `'assistant'` by Codex and Gemini; **never set by Claude**, so do not filter a stream on this field |
| `raw` | `Any` | Original parsed JSON event, unvalidated |

### Sessions

Start a new persistent session and resume it later:

```python
# Start a new persistent session
response = client.run_sync("Hello", options=RunOptions(persist_session=True))
session_id = response.metadata.session_id
print(f"Session: {session_id}")

# Resume the session in a later call
response2 = client.run_sync("What did I just say?", options=RunOptions(session_id=session_id))
print(response2.content)
```

Note: `session_id` and `persist_session` are **coupled, not mutually exclusive** — supplying a
`session_id` forces `persist_session` to `True`. Passing both is accepted; the resume path wins.

`GeminiClient` ignores `persist_session` entirely (it emits identical arguments either way) and
relies on the `gemini` CLI's own default. Claude and Codex both honour it.

### Error Handling

The buffered and streaming paths use **opposite error models**.

`run` and `run_sync` convert **three** conditions into `AiResponse(status='error')`:

```python
response = client.run_sync("Do something")
if response.status == "error":
    print(response.metadata.message)
    print(response.metadata.error_code)   # 'ENOENT', or the process exit code as a string
    print(response.metadata.timed_out)
```

| Condition | Metadata |
|---|---|
| Executable not found | `error_code='ENOENT'`, `message` |
| Timeout expired | `timed_out=True` |
| Nonzero exit code | `error_code=str(returncode)`, `stderr` (first 4096 chars) |

Anything else **raises** — `PermissionError` for a non-executable binary, `UnicodeDecodeError` for
non-UTF-8 output, and `OSError`/`E2BIG` for a prompt too large to fit in one argv element (the prompt
is passed as a command-line argument, so the OS limit is on **bytes**, not characters).

`stream` and `stream_sync` raise for **every** failure rather than returning one:

| Exception | Cause |
|---|---|
| `FileNotFoundError` | Executable not found |
| `TimeoutError` | Timeout expired |
| `RateLimitError` | Claude only — carries `.metadata` with `rate_limit_reset_at` |
| `ValueError` | Malformed stream line (Claude and Gemini; Codex skips the line instead) |

Note that the streaming path does **not** check the process exit code, so a backend that fails after
emitting no output produces an empty stream rather than an error.

`ValueError('prompt must be a non-empty string')` is raised for an empty or non-`str` prompt —
eagerly by `run_sync`, and on first `await`/iteration by the other three entry points. Invalid config
values (`timeout <= 0`, empty `cwd`, empty `executable_path`) raise `pydantic.ValidationError` at
construction time.

### Backend Differences

The three clients share an interface but not a feature set. Where they diverge:

| Behaviour | `ClaudeClient` | `CodexClient` | `GeminiClient` |
|---|---|---|---|
| `persist_session` | honoured | honoured | **ignored** |
| Rate-limit detection | `RateLimitError` / `rate_limited` metadata | none | none |
| Token usage in metadata | yes | **never reported** | yes |
| `metadata.model` | reported | **always `None`** | reported |
| Backend-reported error | **not detected** (`status='success'`) | `status='error'` | **no error concept** (`fallback=True`) |
| Malformed stream line | raises `ValueError` | skipped silently | raises `ValueError` |
| `role` on content chunks | **not set** | `'assistant'` | `'assistant'` |
| `allow_all_tools` maps to | `--dangerously-skip-permissions` | `--full-auto` | `--yolo` |

When a backend cannot parse its own output, it returns the raw text as `content` with
`metadata.fallback = True` and `status='success'`. Check `fallback` to distinguish "the model said
this" from "we could not parse the CLI".

---

## CLI Reference

### Synopsis

```
aicli <backend> [options] <prompt>
```

Where `<backend>` is one of: `claude`, `codex`, `gemini`.

### Examples

```bash
aicli claude "What is the capital of France?"
aicli codex --cwd /my/project "Refactor this module"
aicli gemini --model gemini-3-flash-preview --timeout 30 "Summarise this PR"
aicli claude --new-session "Start a new conversation"
aicli claude --session-id <id> "Continue where we left off"
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model <name>` | string | the client's built-in default | Model passed to the backend |
| `--timeout <secs>` | float | none | Abort after N seconds; must be `> 0` |
| `--cwd <path>` | string | process working directory | Working directory for the spawned process |
| `--executable-path <path>` | string | PATH lookup | Override binary path |
| `--new-session` | flag | false | Create a new persistent session |
| `--session-id <id>` | string | none | Resume an existing session |
| `--allow-all-tools` | flag | false | Bypass all tool-approval prompts; maps to `--dangerously-skip-permissions` (claude), `--yolo` (gemini), `--full-auto` (codex) |
| `--output <format>` | `json\|jsonl\|text` | `json` | Output format |
| `-h` / `--help` | flag | false | Print help and exit |

All three subcommands accept exactly these flags — there are no backend-specific ones.

`--new-session` and `--session-id` are mutually exclusive.

**The CLI exposes a subset of the library.** There is no flag for `additional_args`, and per-call
`RunOptions` are not reachable. Because `additional_args` is the only way to express narrowed tool
grants, permission narrowing is **not available through `aicli`** — use the Python API for that.

The CLI always uses the streaming path, including for `--output json`, so it inherits the streaming
error model described above.

### Output Formats

**`--output json`** (default): single JSON object after stream completes.

```json
{
  "message": "The capital of France is Paris.",
  "session_id": null,
  "input_tokens": 12,
  "output_tokens": 8,
  "total_tokens": 20
}
```

`session_id` is populated **only** when `--new-session` was passed. Since `--new-session` cannot be
combined with `--session-id`, a resumed run always reports `"session_id": null`. The object is
written to stdout even when the run fails, alongside exit 1.

**`--output jsonl`**: one JSON line per `AiChunk` as it arrives, followed by a blank line.

```
{"delta": "The capital", "type": "content", "role": null, "raw": null}
{"delta": " of France is Paris.", "type": "content", "role": null, "raw": null}

```

Session id and token totals are not extracted in this mode — read them from `raw`.

**`--output text`**: raw delta text written incrementally to stdout; trailing newline after the
stream ends. With `--new-session`, `session_id: <id>` is written to **stderr** so stdout stays pure
model text.

Error messages go to stderr with an `error: ` prefix in all three modes. Chunks of type `'error'`
are handled inconsistently: `json` writes the text to stderr unprefixed, `jsonl` emits it as an
ordinary line, and `text` discards it.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Stream completed with no error; also `-h` / `--help` |
| `1` | Invalid flag value, backend error, or any exception during the run |
| `2` | Any argument-parsing failure: missing or unknown subcommand, missing prompt, unknown flag, non-numeric `--timeout`, or an `--output` value outside the three choices |

---

## License

```
MIT License

Copyright (c) <year> <author>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
