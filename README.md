# swcli

`swcli` is a lightweight Rust launcher that routes traffic through either
`codex` or `claude`/`claude-code` using locally managed provider credentials.

It can:

- Store and switch between multiple API credentials.
- Resolve provider model metadata from endpoints or bundled `models.dev` data.
- Automatically start local protocol routers when needed.
- Inject runtime environment and model catalog settings so the native tools can use the selected model.

## Install

### From source

```bash
cargo install --path .
```

### From GitHub releases

Download the matching asset for your platform from
[`swcli` releases](https://github.com/zhzy0077/swcli/releases).

## Requirements

- Rust stable toolchain
- A working `codex` binary (for Codex mode).
- A working `claude` binary (for Claude mode, also accepts `claude-code`).

## Usage

```bash
swcli <TOOL> [TOOL_ARGS...]
swcli <COMMAND>
```

`<TOOL>` is one of `codex`, `claude`, or `claude-code`.

Common launch options:

```bash
-k, --key <ID|NAME>   Use a specific saved key
-m, --model <MODEL>   Override model for launch
--dry-run             Print resolved command/env without launching
-e, --env KEY=VALUE   Add extra environment variables
```

Example:

```bash
swcli -k openai-main -m gpt-4.1 codex
swcli --dry-run codex -h
```

## Credential management

```bash
swcli keys                  # list keys
swcli keys add              # interactive add flow
swcli keys add <name> --provider openai --base-url https://api.openai.com/v1 --key <API_KEY>
swcli keys default <id|name> # set active key
swcli keys rm <id|name>
swcli keys cat <id|name>
swcli keys edit <id|name>
swcli keys ping             # test saved keys
swcli keys --ping           # list with ping status
swcli keys --json           # JSON output
```

### Presets

`--preset` supports several built-in aliases:

- `github-copilot`
- `minimax`
- `minimax-cn`
- `xiaomi-token-plan-cn`

Examples:

```bash
swcli keys add my-copilot --preset github-copilot
swcli keys add minimax-cn --preset minimax-cn
```

GitHub Copilot preset uses interactive device login when no OAuth token is supplied.

## List models

```bash
swcli models                # list models for active key
swcli models -k <id|name>   # for specific key
swcli models --search gpt   # substring filter
swcli models --refresh       # bypass cached model list
swcli models --json          # JSON output
```

## Config and cache files

`swcli` stores config under:

- `SWCLI_CONFIG_DIR` (if set)
- otherwise `$XDG_CONFIG_HOME/swcli`
- otherwise `~/.config/swcli`

Primary files:

- `config.json` (keys and active key)
- `models-cache.json`
- `models-dev-cache.json`

## Notes

- The tool validates whether the selected key/model is valid for `codex` or `claude` before launch.
- Some keys may require local router startup when protocol conversion is needed.
- Use `--dry-run` to inspect the exact command and environment that would be executed.

## Development

```bash
cargo check
cargo test
```

## Licensing

`swcli` is licensed under `MIT AND Apache-2.0` (see [`LICENSE`](./LICENSE)).
