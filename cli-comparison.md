# CLI Comparison: claude vs gemini vs codex

## Shared / Similar Arguments

| Feature | claude | gemini | codex |
|---|---|---|---|
| **Help** | `-h, --help` | `-h, --help` | `-h, --help` |
| **Version** | `-v, --version` | `-v, --version` | `-V, --version` |
| **Model selection** | `--model <model>` | `-m, --model <model>` | `-m, --model <MODEL>` |
| **Non-interactive/print mode** | `-p, --print` | `-p, --prompt <prompt>` | `exec` subcommand (or prompt arg) |
| **Debug mode** | `-d, --debug [filter]` | `-d, --debug` | `debug` subcommand |
| **Resume session** | `-r, --resume [value]` | `-r, --resume <value>` | `resume` subcommand / `--last` |
| **Output format** | `--output-format <format>` (text/json/stream-json) | `-o, --output-format <format>` (text/json/stream-json) | _(not a top-level flag; subcommand-driven)_ |
| **MCP management** | `mcp` subcommand + `--mcp-config` | `mcp` subcommand + `--allowed-mcp-server-names` | `mcp` subcommand |
| **Allowed tools** | `--allowedTools / --allowed-tools` | `--allowed-tools` _(deprecated)_ | _(not supported; handled via approval policy)_ |
| **Additional directories** | `--add-dir <directories...>` | `--include-directories <dirs>` | `--add-dir <DIR>` |
| **Auto/bypass approvals** | `--dangerously-skip-permissions` / `--permission-mode bypassPermissions` | `-y, --yolo` / `--approval-mode yolo` | `--dangerously-bypass-approvals-and-sandbox` / `--full-auto` |
| **Permission/approval mode** | `--permission-mode <mode>` (acceptEdits/bypassPermissions/default/dontAsk/plan/auto) | `--approval-mode <mode>` (default/auto_edit/yolo/plan) | `-a, --ask-for-approval <policy>` (untrusted/on-failure/on-request/never) |
| **Sandbox mode** | `--permission-mode plan` | `-s, --sandbox` | `-s, --sandbox <mode>` (read-only/workspace-write/danger-full-access) |
| **System/initial prompt** | `--system-prompt <prompt>` | positional `query` arg | positional `[PROMPT]` arg |
| **Interactive-then-exit** | _(default is interactive)_ | `-i, --prompt-interactive` | _(default is interactive)_ |
| **Extensions/plugins** | `--plugin-dir` / `plugin` subcommand | `-e, --extensions` / `extensions` subcommand | _(not supported)_ |
| **Auth management** | `auth` subcommand | _(implicit via config)_ | `login` / `logout` subcommands |
| **Skills management** | `--disable-slash-commands` | `skills` subcommand | _(not supported)_ |
| **Fork session** | `--fork-session` | _(not supported)_ | `fork` subcommand |

---

## Arguments Unique to `claude`

| Argument | Description |
|---|---|
| `--agent <agent>` | Select a named agent for the session |
| `--agents <json>` | Define custom agents inline as JSON |
| `--bare` | Minimal mode: skips hooks, LSP, plugins, CLAUDE.md discovery, etc. |
| `--betas <betas...>` | Include beta headers in API requests (API key users only) |
| `--brief` | Enable `SendUserMessage` tool for agent-to-user communication |
| `--chrome` / `--no-chrome` | Enable/disable Claude in Chrome integration |
| `--debug-file <path>` | Write debug logs to a file |
| `--disallowedTools / --disallowed-tools` | Explicitly deny specific tools |
| `--effort <level>` | Set effort level (low/medium/high/max) |
| `--fallback-model <model>` | Fallback model when primary is overloaded (print mode only) |
| `--file <specs...>` | Download file resources at startup |
| `--from-pr [value]` | Resume a session linked to a GitHub PR |
| `--ide` | Auto-connect to an IDE on startup |
| `--include-hook-events` | Include hook lifecycle events in stream-json output |
| `--include-partial-messages` | Stream partial message chunks as they arrive |
| `--input-format <format>` | Input format: text or stream-json |
| `--json-schema <schema>` | JSON Schema for structured output validation |
| `--max-budget-usd <amount>` | Cap total API spend in dollars |
| `--mcp-debug` | _(deprecated)_ Enable MCP debug mode |
| `--mcp-config <configs...>` | Load MCP servers from JSON files or strings |
| `--strict-mcp-config` | Ignore all MCP configs except `--mcp-config` |
| `-n, --name <name>` | Set a display name for the session |
| `--no-session-persistence` | Disable saving sessions to disk |
| `--replay-user-messages` | Re-emit user messages on stdout for acknowledgment |
| `--session-id <uuid>` | Use a specific UUID for the session |
| `--setting-sources <sources>` | Choose which settings sources to load |
| `--settings <file-or-json>` | Load settings from a file or JSON string |
| `--append-system-prompt <prompt>` | Append to the default system prompt |
| `--system-prompt-file` | _(implied by bare docs)_ Load system prompt from a file |
| `--tmux` | Create a tmux session for the worktree |
| `--tools <tools...>` | Specify the exact set of available built-in tools |
| `-w, --worktree [name]` | Create a new git worktree for the session |
| `--verbose` | Override verbose mode from config |
| `auto-mode` subcommand | Inspect auto-mode classifier configuration |
| `doctor` subcommand | Check health of the auto-updater |
| `install` subcommand | Install a specific version of Claude Code |
| `setup-token` subcommand | Set up a long-lived auth token |
| `update/upgrade` subcommand | Check for and install updates |
| `agents` subcommand | List configured agents |

---

## Arguments Unique to `gemini`

| Argument | Description |
|---|---|
| `--policy <files...>` | Load additional policy files or directories |
| `--admin-policy <files...>` | Load additional admin policy files or directories |
| `--acp` | Start the agent in ACP (Agent Communication Protocol) mode |
| `--experimental-acp` | _(deprecated)_ ACP mode flag |
| `-l, --list-extensions` | List all available extensions and exit |
| `--list-sessions` | List available sessions for the current project and exit |
| `--delete-session <index>` | Delete a session by index number |
| `--screen-reader` | Enable screen reader mode for accessibility |
| `--raw-output` | Disable sanitization of model output (allow ANSI sequences) |
| `--accept-raw-output-risk` | Suppress security warning when using `--raw-output` |
| `hooks` subcommand | Manage Gemini CLI hooks |

---

## Arguments Unique to `codex`

| Argument | Description |
|---|---|
| `-c, --config <key=value>` | Override a config value (TOML dotted path, e.g. `model="o3"`) |
| `--enable <FEATURE>` | Enable a feature flag |
| `--disable <FEATURE>` | Disable a feature flag |
| `--remote <ADDR>` | Connect TUI to a remote app-server WebSocket endpoint |
| `-i, --image <FILE>...` | Attach image(s) to the initial prompt |
| `--oss` | Select the local open-source model provider (LM Studio / Ollama) |
| `--local-provider <provider>` | Specify which local provider to use (lmstudio or ollama) |
| `-p, --profile <CONFIG_PROFILE>` | Use a named configuration profile from `config.toml` |
| `-C, --cd <DIR>` | Set the agent's working root directory |
| `--search` | Enable live web search via the native `web_search` tool |
| `--no-alt-screen` | Run TUI inline (no alternate screen buffer) |
| `review` subcommand | Run a non-interactive code review |
| `mcp-server` subcommand | Start Codex as an MCP server (stdio) |
| `app-server` subcommand | Run the experimental app server |
| `completion` subcommand | Generate shell completion scripts |
| `sandbox` subcommand | Run commands within a Codex-managed sandbox |
| `apply` subcommand | Apply the latest agent diff as a `git apply` |
| `cloud` subcommand | Browse tasks from Codex Cloud and apply changes locally |
| `features` subcommand | Inspect feature flags |
