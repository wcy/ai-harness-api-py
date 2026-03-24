# ai-harness-api

`ai_harness_api` is a unified Python library that provides a consistent programmatic interface for interacting with AI-powered CLI agents: Claude Code (`claude`), Codex CLI (`codex`), and Gemini CLI (`gemini`). It abstracts each tool's unique invocation flags, output formats, and process lifecycle behind a single, coherent API that developers can import like any SDK.

---

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) package manager
- One or more AI CLI binaries available on `PATH`:
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

Each backend has its own client class. All three share the same interface.

```python
from ai_harness_api import ClaudeClient, CodexClient, GeminiClient
from ai_harness_api import ClientConfig

claude  = ClaudeClient(ClientConfig(model="claude-opus-4-6", timeout=60.0))
codex   = CodexClient(ClientConfig(model="o4-mini", cwd="/my/project"))
gemini  = GeminiClient(ClientConfig(model="gemini-2.0-flash"))
```

Default models: `ClaudeClient` → `haiku`, `CodexClient` → backend default, `GeminiClient` → backend default.

### ClientConfig Fields

Constructor-level configuration. All fields are optional.

| Field | Type | Default | Description |
|---|---|---|---|
| `executable_path` | `str \| None` | PATH lookup | Override binary path |
| `model` | `str \| None` | backend default | Model for all calls on this client |
| `timeout` | `float \| None` | None (no timeout) | Default timeout in seconds |
| `cwd` | `str \| None` | `os.getcwd()` | Default working directory |
| `additional_args` | `list[str]` | `[]` | Extra CLI flags appended to every call |
| `session_id` | `str \| None` | None | Session ID to resume on every call |
| `persist_session` | `bool` | `False` | Start a new persistent session |

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

`RunOptions` fields override constructor defaults for a single call:

```python
from ai_harness_api import RunOptions

response = client.run_sync(
    "Summarise this file",
    options=RunOptions(cwd="/my/project", timeout=30.0, model="claude-opus-4-6"),
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `cwd` | `str \| None` | None | Override working directory for this call only |
| `executable_path` | `str \| None` | None | Override binary path for this call only |
| `model` | `str \| None` | None | Override model for this call only |
| `timeout` | `float \| None` | None | Override timeout for this call only (seconds) |
| `additional_args` | `list[str]` | `[]` | Extra flags for this call only (concatenated with instance defaults) |
| `session_id` | `str \| None` | None | Resume this session for this call only |
| `persist_session` | `bool` | `False` | Start a new persistent session for this call only |

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
| `type` | `'content' \| 'metadata' \| 'status' \| 'error'` | Chunk kind |
| `role` | `str \| None` | Conversational role if reported by backend |
| `raw` | `Any \| None` | Original parsed JSON dict |

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

Note: `session_id` and `persist_session=True` are mutually exclusive.

### Error Handling

`run` and `run_sync` never raise for runtime errors — they return `AiResponse(status='error')`:

```python
response = client.run_sync("Do something")
if response.status == "error":
    print(response.metadata.message)
    print(response.metadata.error_code)
    print(response.metadata.timed_out)
```

`stream` and `stream_sync` may raise `asyncio.TimeoutError` if the timeout fires mid-stream.

`ValueError` is raised eagerly at construction or call time for invalid arguments (empty prompt, negative timeout, etc.).

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
aicli gemini --model gemini-2.0-flash --timeout 30 "Summarise this PR"
aicli claude --new-session "Start a new conversation"
aicli claude --session-id <id> "Continue where we left off"
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model <name>` | string | backend default | Model passed to the backend |
| `--timeout <secs>` | float | none | Abort after N seconds |
| `--cwd <path>` | string | current directory | Working directory for the spawned process |
| `--executable-path <path>` | string | PATH lookup | Override binary path |
| `--new-session` | flag | false | Create a new persistent session |
| `--session-id <id>` | string | none | Resume an existing session |
| `--output <format>` | `json\|jsonl\|text` | `json` | Output format |
| `-h` / `--help` | flag | false | Print help and exit |

`--new-session` and `--session-id` are mutually exclusive.

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

**`--output jsonl`**: one JSON line per `AiChunk` as it arrives.

```
{"delta": "The capital", "type": "content", "role": null, "raw": null}
{"delta": " of France is Paris.", "type": "content", "role": null, "raw": null}
```

**`--output text`**: raw delta text written incrementally; trailing newline after stream ends.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Stream completed successfully |
| `1` | Backend error, stream error, or invalid flags |
| `2` | Unknown subcommand or missing prompt |

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
