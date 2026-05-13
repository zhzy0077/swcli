use crate::provider::{ProviderKind, WireProtocol};
use anyhow::{Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "swcli",
    about = "Unified launcher for Claude Code and Codex with local key management",
    override_usage = "swcli [OPTIONS] <TOOL> [TOOL_ARGS]...\n       swcli <COMMAND>",
    version
)]
pub(crate) struct Cli {
    /// Select saved key by id or name.
    #[arg(short = 'k', long, value_name = "ID|NAME")]
    pub(crate) key: Option<String>,
    /// Primary model override.
    #[arg(short, long, value_name = "MODEL")]
    pub(crate) model: Option<String>,
    /// Print command/env without launching.
    #[arg(long)]
    pub(crate) dry_run: bool,
    /// Extra environment variable, KEY=VALUE.
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    pub(crate) envs: Vec<String>,
    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
    /// Tool to run: codex, claude, or claude-code.
    #[arg(value_name = "TOOL")]
    pub(crate) tool: Option<String>,
    /// Arguments passed to the native tool.
    #[arg(
        value_name = "TOOL_ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub(crate) tool_args: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    /// Manage saved provider credentials.
    Keys(KeysArgs),
    /// List provider models with cache/search/json support.
    Models(ModelsArgs),
}

#[derive(Debug)]
pub(crate) struct LaunchArgs {
    pub(crate) tool: String,
    pub(crate) tool_args: Vec<String>,
    pub(crate) model: Option<String>,
    pub(crate) key: Option<String>,
    pub(crate) dry_run: bool,
    pub(crate) envs: Vec<String>,
}

#[derive(Args, Debug)]
pub(crate) struct KeysArgs {
    /// Action: add, default, rm, cat, edit, ping. No action lists keys.
    pub(crate) action: Option<String>,
    /// Action arguments.
    pub(crate) args: Vec<String>,
    /// Display name for add/edit.
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Provider base URL for add/edit.
    #[arg(long = "base-url")]
    pub(crate) base_url: Option<String>,
    /// API key for add/edit. Stored locally obfuscated, not printed in lists.
    #[arg(long)]
    pub(crate) key: Option<String>,
    /// Provider kind: openai, anthropic, codex-oauth, claude-oauth.
    #[arg(long)]
    pub(crate) provider: Option<ProviderKind>,
    /// Wire protocol: openai-responses, openai-completions, anthropic-messages.
    #[arg(long)]
    pub(crate) wire: Option<WireProtocol>,
    /// Builtin provider preset alias, such as github-copilot, minimax, minimax-cn, or xiaomi-token-plan-cn.
    #[arg(long)]
    pub(crate) preset: Option<String>,
    /// Refresh models.dev catalog before resolving a builtin preset.
    #[arg(short = 'r', long)]
    pub(crate) refresh: bool,
    /// OAuth access token for codex-oauth/claude-oauth entries.
    #[arg(long = "oauth-token")]
    pub(crate) oauth_token: Option<String>,
    /// Ping all keys.
    #[arg(long)]
    pub(crate) all: bool,
    /// List keys with ping status.
    #[arg(long)]
    pub(crate) ping: bool,
    /// JSON output.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args, Debug)]
pub(crate) struct ModelsArgs {
    /// Select saved key by id or name.
    #[arg(short = 'k', long, value_name = "ID|NAME")]
    pub(crate) key: Option<String>,
    /// Bypass cache.
    #[arg(short = 'r', long)]
    pub(crate) refresh: bool,
    /// Search by substring.
    #[arg(short = 's', long)]
    pub(crate) search: Option<String>,
    /// JSON output.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug)]
pub(crate) enum ParsedCommand {
    Launch(LaunchArgs),
    Keys(KeysArgs),
    Models(ModelsArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchToken {
    Tool(usize),
    Command,
}

pub(crate) fn parse_cli<I, T>(args: I) -> Result<Cli>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let raw_args: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    match find_dispatch_token(&raw_args) {
        Some(DispatchToken::Tool(index)) => {
            let mut cli = Cli::try_parse_from(raw_args[..=index].iter().cloned())?;
            cli.tool_args = raw_args[index + 1..]
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();
            Ok(cli)
        }
        _ => Ok(Cli::try_parse_from(raw_args)?),
    }
}

pub(crate) fn normalize_cli(cli: Cli) -> Result<ParsedCommand> {
    let command_name = match &cli.command {
        Some(Commands::Keys(_)) => Some("keys"),
        Some(Commands::Models(_)) => Some("models"),
        None => None,
    };
    if let Some(command_name) = command_name {
        reject_launch_options_for_management(&cli, command_name)?;
    }

    match cli.command {
        Some(Commands::Keys(args)) => Ok(ParsedCommand::Keys(args)),
        Some(Commands::Models(args)) => Ok(ParsedCommand::Models(args)),
        None => {
            let tool = cli
                .tool
                .ok_or_else(|| anyhow!("Missing tool. Use `swcli <tool> [args...]`."))?;

            Ok(ParsedCommand::Launch(LaunchArgs {
                tool,
                tool_args: cli.tool_args,
                model: cli.model,
                key: cli.key,
                dry_run: cli.dry_run,
                envs: cli.envs,
            }))
        }
    }
}

fn reject_launch_options_for_management(cli: &Cli, command_name: &str) -> Result<()> {
    if cli.key.is_some() || cli.model.is_some() || cli.dry_run || !cli.envs.is_empty() {
        bail!(
            "Launch options `-k`, `-m`, `--dry-run`, and `--env` are only valid when launching a tool, not with `swcli {command_name} ...`."
        );
    }
    if cli.tool.is_some() || !cli.tool_args.is_empty() {
        bail!("Unexpected launch arguments for `swcli {command_name} ...`.");
    }
    Ok(())
}

fn find_dispatch_token(args: &[std::ffi::OsString]) -> Option<DispatchToken> {
    let mut index = 1;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "-k" | "--key" | "-m" | "--model" | "-e" | "--env" => index += 2,
            "--dry-run" => index += 1,
            "keys" | "models" => return Some(DispatchToken::Command),
            "codex" | "claude" | "claude-code" => return Some(DispatchToken::Tool(index)),
            _ if arg.starts_with("--key=")
                || arg.starts_with("--model=")
                || arg.starts_with("--env=") =>
            {
                index += 1;
            }
            _ if arg.starts_with('-') => index += 1,
            _ => return None,
        }
    }
    None
}
