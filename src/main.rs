use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Password};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

mod copilot_proxy;
mod github_copilot;
mod responses_chat_bridge;
mod responses_router;
mod tui;

const CACHE_TTL_SECS: u64 = 3600;
const MODELS_DEV_CACHE_TTL_SECS: u64 = 24 * 3600;
const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const BUNDLED_MODELS_DEV_API_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/models_dev_api.json"));

#[derive(Parser, Debug)]
#[command(
    name = "swcli",
    about = "Unified launcher for Claude Code and Codex with local key management",
    override_usage = "swcli [OPTIONS] <TOOL> [TOOL_ARGS]...\n       swcli <COMMAND>",
    version
)]
struct Cli {
    /// Select saved key by id or name.
    #[arg(short = 'k', long, value_name = "ID|NAME")]
    key: Option<String>,
    /// Primary model override.
    #[arg(short, long, value_name = "MODEL")]
    model: Option<String>,
    /// Print command/env without launching.
    #[arg(long)]
    dry_run: bool,
    /// Extra environment variable, KEY=VALUE.
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    envs: Vec<String>,
    #[command(subcommand)]
    command: Option<Commands>,
    /// Tool to run: codex, claude, or claude-code.
    #[arg(value_name = "TOOL")]
    tool: Option<String>,
    /// Arguments passed to the native tool.
    #[arg(
        value_name = "TOOL_ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    tool_args: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Manage saved provider credentials.
    Keys(KeysArgs),
    /// List provider models with cache/search/json support.
    Models(ModelsArgs),
}

#[derive(Debug)]
struct LaunchArgs {
    tool: String,
    tool_args: Vec<String>,
    model: Option<String>,
    key: Option<String>,
    dry_run: bool,
    envs: Vec<String>,
}

#[derive(Args, Debug)]
struct KeysArgs {
    /// Action: add, default, rm, cat, edit, ping. No action lists keys.
    action: Option<String>,
    /// Action arguments.
    args: Vec<String>,
    /// Display name for add/edit.
    #[arg(long)]
    name: Option<String>,
    /// Provider base URL for add/edit.
    #[arg(long = "base-url")]
    base_url: Option<String>,
    /// API key for add/edit. Stored locally obfuscated, not printed in lists.
    #[arg(long)]
    key: Option<String>,
    /// Provider kind: openai, anthropic, codex-oauth, claude-oauth.
    #[arg(long)]
    provider: Option<ProviderKind>,
    /// Wire protocol: openai-responses, openai-completions, anthropic-messages.
    #[arg(long)]
    wire: Option<WireProtocol>,
    /// Builtin provider preset alias, such as github-copilot, minimax, minimax-cn, or xiaomi-token-plan-cn.
    #[arg(long)]
    preset: Option<String>,
    /// Refresh models.dev catalog before resolving a builtin preset.
    #[arg(short = 'r', long)]
    refresh: bool,
    /// OAuth access token for codex-oauth/claude-oauth entries.
    #[arg(long = "oauth-token")]
    oauth_token: Option<String>,
    /// Ping all keys.
    #[arg(long)]
    all: bool,
    /// List keys with ping status.
    #[arg(long)]
    ping: bool,
    /// JSON output.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ModelsArgs {
    /// Select saved key by id or name.
    #[arg(short = 'k', long, value_name = "ID|NAME")]
    key: Option<String>,
    /// Bypass cache.
    #[arg(short = 'r', long)]
    refresh: bool,
    /// Search by substring.
    #[arg(short = 's', long)]
    search: Option<String>,
    /// JSON output.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum ProviderKind {
    Openai,
    Anthropic,
    CodexOauth,
    ClaudeOauth,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ProviderKind::Openai => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::CodexOauth => "codex-oauth",
            ProviderKind::ClaudeOauth => "claude-oauth",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum WireProtocol {
    OpenaiResponses,
    OpenaiCompletions,
    AnthropicMessages,
}

impl std::fmt::Display for WireProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            WireProtocol::OpenaiResponses => "openai-responses",
            WireProtocol::OpenaiCompletions => "openai-completions",
            WireProtocol::AnthropicMessages => "anthropic-messages",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKey {
    id: String,
    name: String,
    #[serde(rename = "baseUrl")]
    base_url: String,
    provider: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    wire: Option<WireProtocol>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    oauth_token: Option<String>,
    #[serde(
        rename = "presetAlias",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    preset_alias: Option<String>,
    #[serde(
        rename = "modelsDevProviderName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    models_dev_provider_name: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Config {
    #[serde(default)]
    keys: Vec<ApiKey>,
    #[serde(default)]
    active_key_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    fetched_at: u64,
    models: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ModelsCache(HashMap<String, CacheEntry>);

#[derive(Debug, Default, Serialize, Deserialize)]
struct ModelsDevCache {
    fetched_at: u64,
    catalog: BTreeMap<String, ModelsDevProvider>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelsDevProvider {
    #[serde(default)]
    id: String,
    name: String,
    #[serde(default)]
    npm: Option<String>,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    doc: Option<String>,
    #[serde(default)]
    models: BTreeMap<String, ModelsDevModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelsDevModel {
    #[serde(default)]
    id: String,
    name: String,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Clone)]
struct ResolvedModel {
    id: String,
    name: String,
    context_window: Option<u64>,
    supports_reasoning: bool,
    wire: WireProtocol,
}

#[derive(Debug, Clone, Copy)]
enum AddProviderChoice {
    Builtin(&'static BuiltinPreset),
    Custom(ProviderKind),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = parse_cli(std::env::args_os())?;
    let mut store = Store::new()?;
    store.ensure_key_wires().await?;

    match normalize_cli(cli)? {
        ParsedCommand::Launch(args) => run_tool(&mut store, args).await,
        ParsedCommand::Keys(args) => handle_keys(&mut store, args).await,
        ParsedCommand::Models(args) => handle_models(&mut store, args).await,
    }
}

fn parse_cli<I, T>(args: I) -> Result<Cli>
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchToken {
    Tool(usize),
    Command,
}

#[derive(Debug)]
enum ParsedCommand {
    Launch(LaunchArgs),
    Keys(KeysArgs),
    Models(ModelsArgs),
}

fn normalize_cli(cli: Cli) -> Result<ParsedCommand> {
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

#[derive(Debug)]
struct Store {
    config_path: PathBuf,
    cache_path: PathBuf,
    models_dev_cache_path: PathBuf,
    config: Config,
}

impl Store {
    fn new() -> Result<Self> {
        let dir = config_dir()?;
        let config_path = dir.join("config.json");
        let cache_path = dir.join("models-cache.json");
        let models_dev_cache_path = dir.join("models-dev-cache.json");
        let config = read_json(&config_path)?.unwrap_or_default();
        Ok(Self {
            config_path,
            cache_path,
            models_dev_cache_path,
            config,
        })
    }

    async fn save(&self) -> Result<()> {
        write_json(&self.config_path, &self.config).await
    }

    fn all_keys(&self) -> &[ApiKey] {
        &self.config.keys
    }

    fn active_key(&self) -> Result<ApiKey> {
        let id = self.config.active_key_id.as_deref().ok_or_else(|| {
            anyhow!("No default key. Run `swcli keys add ...` or `swcli keys default <name>`.")
        })?;
        self.resolve_key(id)
    }

    fn resolve_key(&self, query: &str) -> Result<ApiKey> {
        let matches: Vec<_> = self
            .config
            .keys
            .iter()
            .filter(|k| k.id == query || k.name == query || k.id.starts_with(query))
            .cloned()
            .collect();
        match matches.len() {
            0 => bail!("No key matches `{query}`."),
            1 => Ok(matches[0].clone()),
            _ => bail!("Multiple keys match `{query}`; use a longer id."),
        }
    }

    async fn set_active(&mut self, id: String) -> Result<()> {
        self.config.active_key_id = Some(id);
        self.save().await
    }

    async fn read_cache(&self) -> Result<ModelsCache> {
        Ok(read_json(&self.cache_path)?.unwrap_or_default())
    }

    async fn write_cache(&self, cache: &ModelsCache) -> Result<()> {
        write_json(&self.cache_path, cache).await
    }

    async fn read_models_dev_cache(&self) -> Result<Option<ModelsDevCache>> {
        read_json(&self.models_dev_cache_path)
    }

    async fn write_models_dev_cache(&self, cache: &ModelsDevCache) -> Result<()> {
        write_json(&self.models_dev_cache_path, cache).await
    }

    async fn ensure_key_wires(&mut self) -> Result<()> {
        let mut changed = false;
        for key in &mut self.config.keys {
            if key.wire.is_none() {
                key.wire = Some(infer_key_wire(key));
                changed = true;
            }
        }
        if changed {
            self.save().await?;
        }
        Ok(())
    }
}

fn config_dir() -> Result<PathBuf> {
    if let Ok(dir) = env::var("AIVO_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(dir) = env::var("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(dir).join("switchcli"));
    }
    let home = env::var("HOME").context("HOME is not set; set AIVO_CONFIG_DIR explicitly")?;
    Ok(PathBuf::from(home).join(".config").join("switchcli"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(Some(serde_json::from_str(&data).with_context(|| {
        format!("Invalid JSON in {}", path.display())
    })?))
}

async fn write_json<T: Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let data = serde_json::to_string_pretty(value)?;
    tokio::fs::write(path, data).await?;
    Ok(())
}

async fn handle_keys(store: &mut Store, args: KeysArgs) -> Result<()> {
    if args.ping {
        return list_keys(store, true, args.json).await;
    }
    match args.action.as_deref() {
        None => list_keys(store, false, args.json).await,
        Some("add") => add_key(store, args).await,
        Some("default") => default_key(store, args).await,
        Some("rm") | Some("remove") => remove_key(store, args).await,
        Some("cat") | Some("show") => cat_key(store, args).await,
        Some("edit") => edit_key(store, args).await,
        Some("ping") => ping_keys(store, args).await,
        Some(other) => bail!("Unknown keys action `{other}`."),
    }
}

async fn add_key(store: &mut Store, args: KeysArgs) -> Result<()> {
    let mut add = if should_prompt_for_key_add(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        prompt_add_key(args)?
    } else {
        args
    };

    let preset = match add.preset.as_deref() {
        Some(alias) => Some(resolve_builtin_preset(alias)?),
        None => None,
    };
    let catalog_provider = if let Some(preset) = preset {
        match preset.models_dev_provider_name {
            Some(provider_name) => {
                let catalog = load_models_dev_catalog(store, add.refresh).await?;
                Some(resolve_models_dev_provider(&catalog, provider_name)?.clone())
            }
            None => None,
        }
    } else {
        None
    };

    let name = add
        .name
        .take()
        .or_else(|| add.args.first().cloned())
        .or_else(|| preset.map(|p| p.alias.to_string()))
        .ok_or_else(|| anyhow!("Missing key name. Use `swcli keys add <name> ...` or run `swcli keys add` interactively."))?;
    let provider = add
        .provider
        .or_else(|| preset.map(|p| p.provider))
        .or_else(|| catalog_provider.as_ref().and_then(infer_provider_kind))
        .unwrap_or(ProviderKind::Openai);
    let base_url = add
        .base_url
        .take()
        .or_else(|| preset.map(|p| p.endpoint.to_string()))
        .unwrap_or_else(|| default_base_url(provider).to_string());
    let secret = add.key.take().map(|s| encode_secret(&s));
    let mut oauth_token = add.oauth_token.take().map(|s| encode_secret(&s));
    if preset.map(|p| p.alias) == Some("github-copilot") && oauth_token.is_none() {
        if !(io::stdin().is_terminal() && io::stdout().is_terminal()) {
            bail!(
                "Missing GitHub Copilot login. Run `swcli keys add {name} --preset github-copilot` interactively to complete device login."
            );
        }
        let token = github_copilot::device_flow_login().await?;
        oauth_token = Some(encode_secret(&token));
    }
    if secret.is_none() && oauth_token.is_none() {
        bail!(
            "Missing credential. Pass --key/--oauth-token or run `swcli keys add` interactively."
        );
    }
    let token = secret
        .as_ref()
        .or(oauth_token.as_ref())
        .map(|s| decode_secret(s))
        .transpose()?;
    let wire = match add.wire {
        Some(wire) => wire,
        None if let Some(preset) = preset => preset.wire,
        None if catalog_provider.is_some() => catalog_provider
            .as_ref()
            .and_then(infer_wire_protocol)
            .unwrap_or_else(|| default_wire_for_provider(provider)),
        None => probe_wire_protocol(&base_url, token.as_deref().unwrap_or_default()).await?,
    };

    let key = ApiKey {
        id: random_id(),
        name,
        base_url,
        provider,
        wire: Some(wire),
        secret,
        oauth_token,
        preset_alias: preset.map(|p| p.alias.to_string()),
        models_dev_provider_name: catalog_provider.map(|p| p.name.clone()),
        created_at: Utc::now(),
    };
    let id = key.id.clone();
    let name = key.name.clone();
    store.config.keys.push(key);
    if store.config.active_key_id.is_none() {
        store.config.active_key_id = Some(id.clone());
    }
    store.save().await?;
    println!("Added key {name} ({id}).");
    Ok(())
}

async fn default_key(store: &mut Store, args: KeysArgs) -> Result<()> {
    let query = args
        .args
        .first()
        .ok_or_else(|| anyhow!("Missing key id or name."))?;
    let key = store.resolve_key(query)?;
    store.set_active(key.id.clone()).await?;
    println!("Default key: {} ({})", key.name, key.id);
    Ok(())
}

fn should_prompt_for_key_add(args: &KeysArgs) -> bool {
    args.args.is_empty()
        && args.name.is_none()
        && args.base_url.is_none()
        && args.key.is_none()
        && args.provider.is_none()
        && args.wire.is_none()
        && args.preset.is_none()
        && args.oauth_token.is_none()
}

fn is_github_copilot_key(key: &ApiKey) -> bool {
    key.preset_alias.as_deref() == Some("github-copilot")
}

fn prompt_add_key(mut args: KeysArgs) -> Result<KeysArgs> {
    eprintln!("Add key");
    eprintln!();

    let choice = prompt_provider_choice()?;
    match choice {
        AddProviderChoice::Builtin(preset) => {
            args.preset = Some(preset.alias.to_string());
            if args.provider.is_none() {
                args.provider = Some(preset.provider);
            }
        }
        AddProviderChoice::Custom(provider) => {
            args.provider = Some(provider);
        }
    }

    let default_name = args
        .preset
        .clone()
        .or_else(|| args.provider.map(default_key_name_for_provider));
    let name = prompt_line("Name", default_name.as_deref(), false)?;
    if !name.is_empty() {
        args.name = Some(name);
    }

    if args.preset.is_none() {
        let default_url = args.provider.map(default_base_url);
        let base_url = prompt_line("Base URL", default_url, true)?;
        if base_url.is_empty() {
            bail!("Base URL is required.");
        }
        args.base_url = Some(base_url);
    }

    if let Some(alias) = args.preset.as_deref()
        && alias == "github-copilot"
    {
        return Ok(args);
    }

    match args.provider.unwrap_or(ProviderKind::Openai) {
        ProviderKind::CodexOauth | ProviderKind::ClaudeOauth => {
            let token = prompt_secret_line("OAuth Token")?;
            if token.is_empty() {
                bail!("OAuth token is required.");
            }
            args.oauth_token = Some(token);
        }
        _ => {
            let key = prompt_secret_line("API Key")?;
            if key.is_empty() {
                bail!("API key is required.");
            }
            args.key = Some(key);
        }
    }

    Ok(args)
}

fn default_key_name_for_provider(provider: ProviderKind) -> String {
    match provider {
        ProviderKind::Openai => "openai".to_string(),
        ProviderKind::Anthropic => "anthropic".to_string(),
        ProviderKind::CodexOauth => "codex-oauth".to_string(),
        ProviderKind::ClaudeOauth => "claude-oauth".to_string(),
    }
}

fn prompt_provider_choice() -> Result<AddProviderChoice> {
    let mut items = Vec::new();
    for preset in BUILTIN_PRESETS {
        items.push((
            AddProviderChoice::Builtin(preset),
            tui::PickerItem {
                label: preset.alias.to_string(),
                detail: Some(preset.endpoint.to_string()),
            },
        ));
    }
    for provider in [ProviderKind::Openai, ProviderKind::Anthropic] {
        items.push((
            AddProviderChoice::Custom(provider),
            tui::PickerItem {
                label: format!("custom-{}", provider),
                detail: Some(default_base_url(provider).to_string()),
            },
        ));
    }
    let labels: Vec<tui::PickerItem> = items.iter().map(|(_, item)| item.clone()).collect();
    let index = select_interactive("Provider", &labels)?
        .ok_or_else(|| anyhow!("Key creation cancelled."))?;
    Ok(items[index].0)
}

fn prompt_line(label: &str, default: Option<&str>, allow_empty_default: bool) -> Result<String> {
    let theme = ColorfulTheme::default();
    let mut input = Input::<String>::with_theme(&theme).with_prompt(label);
    if let Some(default) = default {
        input = input.default(default.to_string());
    } else if allow_empty_default {
        input = input.allow_empty(true);
    }
    input.interact_text().map_err(Into::into)
}

fn prompt_secret_line(label: &str) -> Result<String> {
    Password::with_theme(&ColorfulTheme::default())
        .with_prompt(label)
        .interact()
        .map_err(Into::into)
}

async fn remove_key(store: &mut Store, args: KeysArgs) -> Result<()> {
    let query = args
        .args
        .first()
        .ok_or_else(|| anyhow!("Missing key id or name."))?;
    let key = store.resolve_key(query)?;
    store.config.keys.retain(|k| k.id != key.id);
    if store.config.active_key_id.as_deref() == Some(&key.id) {
        store.config.active_key_id = store.config.keys.first().map(|k| k.id.clone());
    }
    store.save().await?;
    println!("Removed key {} ({}).", key.name, key.id);
    Ok(())
}

async fn edit_key(store: &mut Store, args: KeysArgs) -> Result<()> {
    let should_reprobe_wire =
        args.wire.is_none() && (args.base_url.is_some() || args.provider.is_some());
    let query = args
        .args
        .first()
        .ok_or_else(|| anyhow!("Missing key id or name."))?;
    let target = store.resolve_key(query)?;
    let key = store
        .config
        .keys
        .iter_mut()
        .find(|k| k.id == target.id)
        .expect("resolved key exists");
    if let Some(name) = args.name {
        key.name = name;
    }
    if let Some(base_url) = args.base_url {
        key.base_url = base_url;
    }
    if let Some(secret) = args.key {
        key.secret = Some(encode_secret(&secret));
    }
    if let Some(token) = args.oauth_token {
        key.oauth_token = Some(encode_secret(&token));
    }
    if let Some(provider) = args.provider {
        key.provider = provider;
    }
    if let Some(wire) = args.wire {
        key.wire = Some(wire);
    } else if should_reprobe_wire {
        let token = key.plain_secret()?;
        key.wire = Some(probe_wire_protocol(&key.base_url, &token).await?);
    }
    let name = key.name.clone();
    let id = key.id.clone();
    store.save().await?;
    println!("Updated key {name} ({id}).");
    Ok(())
}

async fn cat_key(store: &Store, args: KeysArgs) -> Result<()> {
    let query = args.args.first().map(String::as_str);
    let key = match query {
        Some(q) => store.resolve_key(q)?,
        None => store.active_key()?,
    };
    let output = json!({
        "id": key.id,
        "name": key.name,
        "baseUrl": key.base_url,
        "provider": key.provider,
        "wire": key.wire_protocol(),
        "presetAlias": key.preset_alias,
        "modelsDevProviderName": key.models_dev_provider_name,
        "active": store.config.active_key_id.as_deref() == Some(&key.id),
        "hasSecret": key.secret.is_some(),
        "hasOAuthToken": key.oauth_token.is_some(),
        "createdAt": key.created_at,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

async fn list_keys(store: &Store, with_ping: bool, json_output: bool) -> Result<()> {
    let mut rows = Vec::new();
    for key in store.all_keys() {
        let ping = if with_ping {
            Some(ping_key(key).await.unwrap_or_else(|e| PingResult {
                ok: false,
                message: e.to_string(),
            }))
        } else {
            None
        };
        rows.push(json!({
            "id": key.id,
            "name": key.name,
            "provider": key.provider,
            "wire": key.wire_protocol(),
            "baseUrl": key.base_url,
            "presetAlias": key.preset_alias,
            "modelsDevProviderName": key.models_dev_provider_name,
            "active": store.config.active_key_id.as_deref() == Some(&key.id),
            "ping": ping,
        }));
    }
    if json_output {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("No keys configured.");
        return Ok(());
    }
    for row in rows {
        let active = if row["active"].as_bool() == Some(true) {
            "*"
        } else {
            " "
        };
        let ping = row
            .get("ping")
            .and_then(|p| p.get("ok"))
            .and_then(Value::as_bool)
            .map(|ok| if ok { " ok" } else { " fail" })
            .unwrap_or("");
        println!(
            "{active} {:12} {:14} {:20} {}{}",
            row["id"].as_str().unwrap_or(""),
            row["provider"].as_str().unwrap_or(""),
            row["wire"].as_str().unwrap_or(""),
            row["name"].as_str().unwrap_or(""),
            ping
        );
    }
    Ok(())
}

async fn ping_keys(store: &Store, args: KeysArgs) -> Result<()> {
    let keys = if args.all || args.args.is_empty() {
        store.all_keys().to_vec()
    } else {
        vec![store.resolve_key(&args.args[0])?]
    };
    let mut results = Vec::new();
    for key in keys {
        let ping = ping_key(&key).await.unwrap_or_else(|e| PingResult {
            ok: false,
            message: e.to_string(),
        });
        results.push(json!({
            "id": key.id,
            "name": key.name,
            "wire": key.wire_protocol(),
            "ok": ping.ok,
            "message": ping.message,
        }));
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        for row in results {
            let status = if row["ok"].as_bool() == Some(true) {
                "ok"
            } else {
                "fail"
            };
            println!(
                "{status:4} {:12} {}",
                row["id"].as_str().unwrap_or(""),
                row["message"].as_str().unwrap_or("")
            );
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct PingResult {
    ok: bool,
    message: String,
}

async fn ping_key(key: &ApiKey) -> Result<PingResult> {
    let models = fetch_models(key).await?;
    Ok(PingResult {
        ok: true,
        message: format!("{} models", models.len()),
    })
}

async fn handle_models(store: &mut Store, args: ModelsArgs) -> Result<()> {
    let key = match args.key.as_deref() {
        Some(q) => store.resolve_key(q)?,
        None => store.active_key()?,
    };
    if is_github_copilot_key(&key) {
        return handle_models_from_github_copilot(&key, &args).await;
    }
    if let Some(provider_name) = key.models_dev_provider_name.as_deref() {
        return handle_models_from_models_dev(store, &key, provider_name, &args).await;
    }
    let cache_key = format!("{}:{}", key.provider, key.base_url);
    let mut cache = store.read_cache().await?;
    let mut models = None;
    if !args.refresh {
        models = cache.0.get(&cache_key).and_then(|entry| {
            let now = now_secs();
            (now.saturating_sub(entry.fetched_at) < CACHE_TTL_SECS).then(|| entry.models.clone())
        });
    }
    let models = match models {
        Some(m) => m,
        None => {
            let fresh = fetch_models(&key).await?;
            cache.0.insert(
                cache_key,
                CacheEntry {
                    fetched_at: now_secs(),
                    models: fresh.clone(),
                },
            );
            store.write_cache(&cache).await?;
            fresh
        }
    };
    let mut models: Vec<_> = models
        .into_iter()
        .filter(|m| {
            args.search
                .as_ref()
                .map(|s| m.to_lowercase().contains(&s.to_lowercase()))
                .unwrap_or(true)
        })
        .collect();
    models.sort();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "provider": key.provider,
                "baseUrl": key.base_url,
                "models": models,
            }))?
        );
    } else {
        for model in models {
            println!("{model}");
        }
    }
    Ok(())
}

async fn fetch_models(key: &ApiKey) -> Result<Vec<String>> {
    if is_github_copilot_key(key) {
        return Ok(github_copilot::fetch_models(
            &key.plain_oauth_token()?,
            Some(&key.base_url),
            None,
        )
        .await?
        .into_iter()
        .map(|model| model.id)
        .collect());
    }
    let url = models_url(key);
    let token = key.plain_secret()?;
    let response = reqwest::Client::new()
        .get(&url)
        .bearer_auth(token)
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .with_context(|| format!("Failed to fetch {url}"))?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        bail!("models request failed with {status}: {body}");
    }
    let data = body
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("models response did not contain data[]"))?;
    let models = data
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    Ok(models)
}

async fn handle_models_from_github_copilot(key: &ApiKey, args: &ModelsArgs) -> Result<()> {
    let mut models =
        github_copilot::fetch_models(&key.plain_oauth_token()?, Some(&key.base_url), None).await?;
    let query = args.search.as_ref().map(|s| s.to_ascii_lowercase());
    models.retain(|model| {
        query
            .as_ref()
            .map(|needle| {
                model.id.to_ascii_lowercase().contains(needle)
                    || model.name.to_ascii_lowercase().contains(needle)
            })
            .unwrap_or(true)
    });

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "provider": key.provider,
                "baseUrl": key.base_url,
                "models": models.iter().map(|model| json!({
                    "id": model.id,
                    "name": model.name,
                    "wire": model.wire,
                    "contextWindow": model.context_window,
                    "supportsReasoning": model.supports_reasoning,
                })).collect::<Vec<_>>(),
            }))?
        );
    } else {
        for model in models {
            println!("{}", model.id);
        }
    }
    Ok(())
}

async fn probe_wire_protocol(base_url: &str, token: &str) -> Result<WireProtocol> {
    let client = reqwest::Client::new();
    for (wire, path, body) in [
        (
            WireProtocol::OpenaiResponses,
            "/v1/responses",
            json!({
                "model": "swcli-probe",
                "input": "ping",
                "max_output_tokens": 1,
                "stream": false
            }),
        ),
        (
            WireProtocol::OpenaiCompletions,
            "/v1/chat/completions",
            json!({
                "model": "swcli-probe",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "stream": false
            }),
        ),
        (
            WireProtocol::AnthropicMessages,
            "/v1/messages",
            json!({
                "model": "swcli-probe",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "stream": false
            }),
        ),
    ] {
        let url = build_endpoint_url(base_url, path);
        let response = client
            .post(&url)
            .bearer_auth(token)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await;
        let Ok(response) = response else {
            continue;
        };
        let status = response.status();
        if status.as_u16() != 404 {
            return Ok(wire);
        }
    }
    bail!("Could not determine wire protocol for `{base_url}` by probing endpoints.")
}

fn build_endpoint_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    if base.ends_with("/v1") && path.starts_with("v1/") {
        format!("{base}/{}", path.trim_start_matches("v1/"))
    } else {
        format!("{base}/{path}")
    }
}

async fn handle_models_from_models_dev(
    store: &Store,
    key: &ApiKey,
    provider_name: &str,
    args: &ModelsArgs,
) -> Result<()> {
    let catalog = load_models_dev_catalog(store, args.refresh).await?;
    let provider = resolve_models_dev_provider(&catalog, provider_name)?;
    let query = args.search.as_ref().map(|s| s.to_lowercase());
    let mut models: Vec<_> = provider
        .models
        .values()
        .filter(|model| {
            query
                .as_ref()
                .map(|q| {
                    model.id.to_lowercase().contains(q) || model.name.to_lowercase().contains(q)
                })
                .unwrap_or(true)
        })
        .cloned()
        .collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "provider": key.provider,
                "baseUrl": key.base_url,
                "modelsDevProviderName": provider.name.clone(),
                "modelsDevProviderId": provider.id.clone(),
                "models": models.iter().map(model_json).collect::<Vec<_>>(),
            }))?
        );
    } else {
        for model in models {
            println!("{}", model.id);
        }
    }
    Ok(())
}

fn model_json(model: &ModelsDevModel) -> Value {
    let mut value = model.extra.clone();
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object set above");
    object.insert("id".to_string(), Value::String(model.id.clone()));
    object.insert("name".to_string(), Value::String(model.name.clone()));
    value
}

async fn load_models_dev_catalog(store: &Store, refresh: bool) -> Result<ModelsDevCache> {
    if !refresh {
        if let Some(cache) = store.read_models_dev_cache().await? {
            if now_secs().saturating_sub(cache.fetched_at) < MODELS_DEV_CACHE_TTL_SECS {
                return Ok(cache);
            }
        }
        return parse_bundled_models_dev_catalog();
    }

    let response = reqwest::Client::new()
        .get(MODELS_DEV_URL)
        .send()
        .await
        .with_context(|| format!("Failed to fetch {MODELS_DEV_URL}"))?;
    let status = response.status();
    if !status.is_success() {
        bail!("models.dev request failed with {status}");
    }
    let value: Value = response.json().await?;
    let cache = parse_models_dev_catalog(value)?;
    store.write_models_dev_cache(&cache).await?;
    Ok(cache)
}

fn parse_bundled_models_dev_catalog() -> Result<ModelsDevCache> {
    let value: Value = serde_json::from_str(BUNDLED_MODELS_DEV_API_JSON)
        .context("Bundled models.dev catalog is invalid JSON")?;
    parse_models_dev_catalog(value)
}

fn parse_models_dev_catalog(value: Value) -> Result<ModelsDevCache> {
    let providers = value
        .as_object()
        .ok_or_else(|| anyhow!("models.dev catalog root is not an object"))?;
    let mut catalog = BTreeMap::new();
    for (provider_id, provider_value) in providers {
        let mut provider: ModelsDevProvider = serde_json::from_value(provider_value.clone())
            .with_context(|| format!("Invalid models.dev provider `{provider_id}`"))?;
        if provider.id.is_empty() {
            provider.id = provider_id.clone();
        }
        for (model_id, model) in &mut provider.models {
            if model.id.is_empty() {
                model.id = model_id.clone();
            }
        }
        catalog.insert(provider_id.clone(), provider);
    }
    Ok(ModelsDevCache {
        fetched_at: now_secs(),
        catalog,
    })
}

fn resolve_models_dev_provider<'a>(
    catalog: &'a ModelsDevCache,
    provider_name: &str,
) -> Result<&'a ModelsDevProvider> {
    catalog
        .catalog
        .values()
        .find(|provider| provider.name == provider_name)
        .ok_or_else(|| {
            anyhow!(
                "models.dev provider named `{provider_name}` was not found. Run with --refresh after the catalog updates."
            )
        })
}

fn infer_provider_kind(provider: &ModelsDevProvider) -> Option<ProviderKind> {
    match provider.npm.as_deref() {
        Some("@ai-sdk/anthropic") => Some(ProviderKind::Anthropic),
        Some("@ai-sdk/openai") | Some("@ai-sdk/openai-compatible") => Some(ProviderKind::Openai),
        _ => None,
    }
}

fn infer_wire_protocol(provider: &ModelsDevProvider) -> Option<WireProtocol> {
    match provider.npm.as_deref() {
        Some("@ai-sdk/anthropic") => Some(WireProtocol::AnthropicMessages),
        Some("@ai-sdk/openai") => Some(WireProtocol::OpenaiResponses),
        Some("@ai-sdk/openai-compatible") => Some(WireProtocol::OpenaiCompletions),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct BuiltinPreset {
    alias: &'static str,
    provider: ProviderKind,
    models_dev_provider_name: Option<&'static str>,
    endpoint: &'static str,
    wire: WireProtocol,
}

const BUILTIN_PRESETS: &[BuiltinPreset] = &[
    BuiltinPreset {
        alias: "github-copilot",
        provider: ProviderKind::Openai,
        models_dev_provider_name: None,
        endpoint: "https://api.githubcopilot.com",
        wire: WireProtocol::OpenaiCompletions,
    },
    BuiltinPreset {
        alias: "minimax",
        provider: ProviderKind::Anthropic,
        models_dev_provider_name: Some("MiniMax (minimax.io)"),
        endpoint: "https://api.minimax.io/anthropic/v1",
        wire: WireProtocol::AnthropicMessages,
    },
    BuiltinPreset {
        alias: "minimax-cn",
        provider: ProviderKind::Anthropic,
        models_dev_provider_name: Some("MiniMax (minimaxi.com)"),
        endpoint: "https://api.minimaxi.com/anthropic/v1",
        wire: WireProtocol::AnthropicMessages,
    },
    BuiltinPreset {
        alias: "xiaomi-token-plan-cn",
        provider: ProviderKind::Openai,
        models_dev_provider_name: Some("Xiaomi Token Plan (China)"),
        endpoint: "https://token-plan-cn.xiaomimimo.com/v1",
        wire: WireProtocol::OpenaiCompletions,
    },
];

fn resolve_builtin_preset(alias: &str) -> Result<&'static BuiltinPreset> {
    BUILTIN_PRESETS
        .iter()
        .find(|preset| preset.alias == alias)
        .ok_or_else(|| {
            let names = BUILTIN_PRESETS
                .iter()
                .map(|preset| preset.alias)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("Unknown builtin preset `{alias}`. Available presets: {names}")
        })
}

async fn resolve_launch_model(
    store: &Store,
    key: &ApiKey,
    args: &mut LaunchArgs,
) -> Result<Option<ResolvedModel>> {
    if is_github_copilot_key(key) {
        return resolve_github_copilot_launch_model(key, args)
            .await
            .map(Some);
    }
    let Some(provider_name) = key.models_dev_provider_name.as_deref() else {
        return Ok(None);
    };
    let catalog = load_models_dev_catalog(store, false).await?;
    let provider = resolve_models_dev_provider(&catalog, provider_name)?;

    let primary = match args.model.as_deref() {
        Some(model) => {
            let resolved = resolve_model(provider, model)?;
            args.model = Some(resolved.id.clone());
            Some(resolved)
        }
        None if io::stdin().is_terminal() => {
            let resolved = pick_model_interactive(provider)?;
            args.model = Some(resolved.id.clone());
            Some(resolved)
        }
        None => {
            bail!(
                "No model selected for builtin provider `{provider_name}`. Pass `-m <model>` when stdin is not interactive."
            );
        }
    };

    Ok(primary)
}

fn resolve_model(provider: &ModelsDevProvider, query: &str) -> Result<ResolvedModel> {
    let provider_wire = infer_wire_protocol(provider)
        .unwrap_or_else(|| default_wire_for_provider(ProviderKind::Openai));
    if let Some(model) = provider
        .models
        .values()
        .find(|model| model.id == query || model.name == query)
    {
        return Ok(resolved_model_from(model, provider_wire));
    }
    let needle = query.to_ascii_lowercase();
    if let Some(model) = provider.models.values().find(|model| {
        model.id.to_ascii_lowercase() == needle || model.name.to_ascii_lowercase() == needle
    }) {
        return Ok(resolved_model_from(model, provider_wire));
    }
    bail!(
        "Model `{query}` was not found in models.dev provider `{}`.",
        provider.name
    )
}

fn resolved_model_from(model: &ModelsDevModel, wire: WireProtocol) -> ResolvedModel {
    ResolvedModel {
        id: model.id.clone(),
        name: model.name.clone(),
        context_window: model_context_window(model),
        supports_reasoning: model_supports_reasoning(model),
        wire,
    }
}

async fn resolve_github_copilot_launch_model(
    key: &ApiKey,
    args: &mut LaunchArgs,
) -> Result<ResolvedModel> {
    let models =
        github_copilot::fetch_models(&key.plain_oauth_token()?, Some(&key.base_url), None).await?;
    let resolved = match args.model.as_deref() {
        Some(model) => resolve_github_copilot_model(&models, model)?,
        None if io::stdin().is_terminal() => {
            pick_resolved_model_interactive("Model (github-copilot)", &models)?
        }
        None => {
            bail!(
                "No model selected for builtin provider `github-copilot`. Pass `-m <model>` when stdin is not interactive."
            );
        }
    };
    args.model = Some(resolved.id.clone());
    Ok(resolved)
}

fn resolve_github_copilot_model(
    models: &[github_copilot::GithubCopilotModel],
    query: &str,
) -> Result<ResolvedModel> {
    let needle = query.to_ascii_lowercase();
    let model = models
        .iter()
        .find(|model| model.id == query || model.name == query)
        .or_else(|| {
            models.iter().find(|model| {
                model.id.to_ascii_lowercase() == needle || model.name.to_ascii_lowercase() == needle
            })
        })
        .ok_or_else(|| {
            anyhow!("Model `{query}` was not found in builtin provider `github-copilot`.")
        })?;
    Ok(ResolvedModel {
        id: model.id.clone(),
        name: model.name.clone(),
        context_window: model.context_window,
        supports_reasoning: model.supports_reasoning,
        wire: model.wire,
    })
}

fn model_supports_reasoning(model: &ModelsDevModel) -> bool {
    model
        .extra
        .get("reasoning")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn model_context_window(model: &ModelsDevModel) -> Option<u64> {
    let value = model.extra.get("limit")?.get("context")?;
    value
        .as_u64()
        .or_else(|| value.as_f64().map(|n| n as u64))
        .or_else(|| value.as_str()?.parse::<u64>().ok())
}

fn context_tag_for_window(context_window: Option<u64>) -> Option<String> {
    match context_window {
        Some(n) if n >= 2_000_000 => Some("2m".to_string()),
        Some(n) if n >= 1_000_000 => Some("1m".to_string()),
        _ => None,
    }
}

fn has_model_context_window_config(args: &[String]) -> bool {
    args.iter().any(|arg| arg.contains("model_context_window"))
}

fn has_model_catalog_config(args: &[String]) -> bool {
    args.iter().any(|arg| arg.contains("model_catalog_json"))
}

fn pick_model_interactive(provider: &ModelsDevProvider) -> Result<ResolvedModel> {
    let mut models: Vec<&ModelsDevModel> = provider.models.values().collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    if models.is_empty() {
        bail!("models.dev provider `{}` has no models.", provider.name);
    }

    let items: Vec<tui::PickerItem> = models
        .iter()
        .map(|model| tui::PickerItem {
            label: model.id.clone(),
            detail: model_context_window(model).map(|n| format!("ctx={n}")),
        })
        .collect();
    let index = select_interactive(&format!("Model ({})", provider.name), &items)?
        .ok_or_else(|| anyhow!("Model selection cancelled."))?;
    let wire = infer_wire_protocol(provider)
        .unwrap_or_else(|| default_wire_for_provider(ProviderKind::Openai));
    Ok(resolved_model_from(models[index], wire))
}

fn pick_resolved_model_interactive(
    prompt: &str,
    models: &[github_copilot::GithubCopilotModel],
) -> Result<ResolvedModel> {
    if models.is_empty() {
        bail!("github-copilot did not return any models.");
    }
    let items: Vec<tui::PickerItem> = models
        .iter()
        .map(|model| tui::PickerItem {
            label: model.id.clone(),
            detail: model.context_window.map(|n| format!("ctx={n}")),
        })
        .collect();
    let index =
        select_interactive(prompt, &items)?.ok_or_else(|| anyhow!("Model selection cancelled."))?;
    resolve_github_copilot_model(models, &models[index].id)
}

fn select_interactive(prompt: &str, items: &[tui::PickerItem]) -> Result<Option<usize>> {
    tui::pick(prompt, items).map_err(Into::into)
}

async fn run_tool(store: &mut Store, mut args: LaunchArgs) -> Result<()> {
    let tool = Tool::parse(&args.tool)?;
    let key = match args.key.as_deref() {
        Some(q) => store.resolve_key(q)?,
        None => store.active_key()?,
    };
    let resolved_model = resolve_launch_model(store, &key, &mut args).await?;
    let effective_wire = resolved_model
        .as_ref()
        .map(|model| model.wire)
        .unwrap_or_else(|| key.wire_protocol());
    validate_tool_provider(tool, &key, effective_wire)?;
    let router_model = resolved_model
        .as_ref()
        .map(|model| responses_router::RouterModelMetadata {
            id: model.id.clone(),
            name: model.name.clone(),
            context_window: model.context_window,
            supports_reasoning: model.supports_reasoning,
        });
    let mut command = match tool {
        Tool::Codex => Command::new("codex"),
        Tool::Claude => Command::new("claude"),
    };
    command.stdin(Stdio::inherit());
    command.stdout(Stdio::inherit());
    command.stderr(Stdio::inherit());

    let mut tool_args = Vec::new();
    let mut launch_base_url = key.base_url.clone();
    let launch_token = if is_github_copilot_key(&key) {
        "github-copilot".to_string()
    } else {
        key.plain_secret()?
    };
    let mut anthropic_model_override = None;
    if let Some(model) = args.model.take() {
        match tool {
            Tool::Codex => {
                tool_args.push("-m".to_string());
                tool_args.push(model);
            }
            Tool::Claude => {
                let context_tag = resolved_model
                    .as_ref()
                    .and_then(|model| context_tag_for_window(model.context_window));
                anthropic_model_override =
                    Some(maybe_context_model(&model, context_tag.as_deref()));
            }
        }
    }
    let mut _copilot_proxy = None;
    let _router = if is_github_copilot_key(&key) {
        if args.dry_run {
            launch_base_url = "http://127.0.0.1:<swcli-github-copilot-router>/v1".to_string();
            None
        } else {
            let refresh_token = key.plain_oauth_token()?;
            let router = match tool {
                Tool::Codex => match effective_wire {
                    WireProtocol::OpenaiResponses => {
                        let proxy = copilot_proxy::start_openai_responses_proxy(
                            refresh_token.clone(),
                            key.base_url.clone(),
                            router_model.clone(),
                        )
                        .await?;
                        launch_base_url = format!("http://127.0.0.1:{}/v1", proxy.port);
                        _copilot_proxy = Some(proxy);
                        None
                    }
                    WireProtocol::OpenaiCompletions => {
                        let proxy = copilot_proxy::start_openai_chat_proxy(
                            refresh_token.clone(),
                            key.base_url.clone(),
                        )
                        .await?;
                        let proxy_base_url = format!("http://127.0.0.1:{}/v1", proxy.port);
                        let router = responses_router::start_openai_chat_responses_router(
                            proxy_base_url,
                            "github-copilot".to_string(),
                            router_model.clone(),
                        )
                        .await?;
                        _copilot_proxy = Some(proxy);
                        Some(router)
                    }
                    WireProtocol::AnthropicMessages => {
                        let proxy = copilot_proxy::start_anthropic_messages_proxy(
                            refresh_token.clone(),
                            key.base_url.clone(),
                        )
                        .await?;
                        let proxy_base_url = format!("http://127.0.0.1:{}/v1", proxy.port);
                        let router = responses_router::start_anthropic_responses_router(
                            proxy_base_url,
                            "github-copilot".to_string(),
                            router_model.clone(),
                        )
                        .await?;
                        _copilot_proxy = Some(proxy);
                        Some(router)
                    }
                },
                Tool::Claude => {
                    let proxy = copilot_proxy::start_anthropic_messages_proxy(
                        refresh_token,
                        key.base_url.clone(),
                    )
                    .await?;
                    launch_base_url = format!("http://127.0.0.1:{}/v1", proxy.port);
                    _copilot_proxy = Some(proxy);
                    None
                }
            };
            if let Some(ref router) = router {
                launch_base_url = format!("http://127.0.0.1:{}/v1", router.port);
            }
            router
        }
    } else if tool == Tool::Codex && wire_needs_codex_router(effective_wire) {
        if args.dry_run {
            launch_base_url = "http://127.0.0.1:<swcli-responses-router>/v1".to_string();
            None
        } else {
            let api_key = key.plain_secret()?;
            let router = match effective_wire {
                WireProtocol::AnthropicMessages => {
                    responses_router::start_anthropic_responses_router(
                        key.base_url.clone(),
                        api_key,
                        router_model.clone(),
                    )
                    .await?
                }
                WireProtocol::OpenaiCompletions => {
                    responses_router::start_openai_chat_responses_router(
                        key.base_url.clone(),
                        api_key,
                        router_model.clone(),
                    )
                    .await?
                }
                WireProtocol::OpenaiResponses => unreachable!("openai-responses is direct"),
            };
            launch_base_url = format!("http://127.0.0.1:{}/v1", router.port);
            Some(router)
        }
    } else {
        None
    };

    let mut envs = tool_env(tool, &launch_base_url, &launch_token);
    for pair in &args.envs {
        let (k, v) = pair
            .split_once('=')
            .ok_or_else(|| anyhow!("--env expects KEY=VALUE, got `{pair}`"))?;
        envs.insert(k.to_string(), v.to_string());
    }
    if let Some(model) = anthropic_model_override {
        envs.insert("ANTHROPIC_MODEL".to_string(), model);
    }
    let _codex_model_catalog = if tool == Tool::Codex
        && router_model.is_some()
        && !has_model_catalog_config(&args.tool_args)
    {
        if args.dry_run {
            tool_args.push("--config".to_string());
            tool_args.push(
                "model_catalog_json=\"/tmp/swcli-codex-model-catalog-<pid>.json\"".to_string(),
            );
            None
        } else {
            let catalog = write_codex_model_catalog(router_model.as_ref()).await?;
            tool_args.push("--config".to_string());
            tool_args.push(format!(
                "model_catalog_json=\"{}\"",
                catalog.path.display().to_string().replace('"', "\\\"")
            ));
            Some(catalog)
        }
    } else {
        None
    };

    if tool == Tool::Codex {
        inject_codex_provider_config(&mut envs, &mut tool_args, &launch_base_url);
        let context_tokens = resolved_model
            .as_ref()
            .and_then(|model| model.context_window);
        if let Some(tokens) = context_tokens
            && !has_model_context_window_config(&tool_args)
            && !has_model_context_window_config(&args.tool_args)
        {
            tool_args.push("--config".to_string());
            tool_args.push(format!("model_context_window={tokens}"));
        }
    }
    tool_args.extend(args.tool_args);

    if args.dry_run {
        println!(
            "command: {} {}",
            tool.command_name(),
            shell_join(&tool_args)
        );
        if let Some(model) = &resolved_model {
            println!(
                "resolved_model: {} ({}) context_window={}",
                model.id,
                model.name,
                model
                    .context_window
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
        }
        println!("env:");
        let mut pairs: Vec<_> = envs.into_iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, value) in pairs {
            let value = if key.contains("KEY") || key.contains("TOKEN") {
                redact(&value)
            } else {
                value
            };
            println!("  {key}={value}");
        }
        return Ok(());
    }

    command.args(tool_args);
    command.envs(envs);
    let status = command.spawn()?.wait().await?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Codex,
    Claude,
}

impl Tool {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "codex" => Ok(Self::Codex),
            "claude" | "claude-code" => Ok(Self::Claude),
            _ => bail!("Unsupported tool `{s}`. First step supports codex and claude-code only."),
        }
    }

    fn command_name(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
        }
    }
}

fn validate_tool_provider(tool: Tool, key: &ApiKey, wire: WireProtocol) -> Result<()> {
    if tool_supports_wire(tool, wire) {
        return Ok(());
    }

    match tool {
        Tool::Claude => {
            if is_github_copilot_key(key) {
                bail!(
                    "Key `{}` uses wire `{}` ({}). Claude Code requires `anthropic-messages`. For `github-copilot`, choose a Claude model that exposes `/v1/messages`.",
                    key.name,
                    wire,
                    key.base_url,
                )
            }
            bail!(
                "Key `{}` uses wire `{}` ({}). Claude Code requires `anthropic-messages`. Use `swcli -k {} codex` or choose an Anthropic-compatible key such as `minimax-cn`/`custom-anthropic`.",
                key.name,
                wire,
                key.base_url,
                key.name
            )
        }
        Tool::Codex => Ok(()),
    }
}

fn tool_supports_wire(tool: Tool, wire: WireProtocol) -> bool {
    matches!(
        (tool, wire),
        (
            Tool::Codex,
            WireProtocol::OpenaiResponses
                | WireProtocol::OpenaiCompletions
                | WireProtocol::AnthropicMessages
        ) | (Tool::Claude, WireProtocol::AnthropicMessages)
    )
}

fn wire_needs_codex_router(wire: WireProtocol) -> bool {
    matches!(
        wire,
        WireProtocol::OpenaiCompletions | WireProtocol::AnthropicMessages
    )
}

fn tool_env(tool: Tool, base_url: &str, token: &str) -> HashMap<String, String> {
    let mut envs = HashMap::new();
    match tool {
        Tool::Codex => {
            envs.insert("OPENAI_API_KEY".to_string(), token.to_string());
            envs.insert("OPENAI_BASE_URL".to_string(), base_url.to_string());
        }
        Tool::Claude => {
            envs.insert("ANTHROPIC_API_KEY".to_string(), token.to_string());
            envs.insert("ANTHROPIC_BASE_URL".to_string(), base_url.to_string());
        }
    }
    envs
}

fn inject_codex_provider_config(
    envs: &mut HashMap<String, String>,
    args: &mut Vec<String>,
    base_url: &str,
) {
    if args.iter().any(|a| a.contains("model_provider")) {
        return;
    }
    if let Some(api_key) = envs.remove("OPENAI_API_KEY") {
        envs.remove("OPENAI_BASE_URL");
        envs.insert("AIVO_CODEX_API_KEY".to_string(), api_key);
        args.extend([
            "--config".to_string(),
            "model_provider=\"aivo\"".to_string(),
            "--config".to_string(),
            "model_providers.aivo.name=\"aivo\"".to_string(),
            "--config".to_string(),
            format!(
                "model_providers.aivo.base_url=\"{}\"",
                base_url.replace('"', "\\\"")
            ),
            "--config".to_string(),
            "model_providers.aivo.env_key=\"AIVO_CODEX_API_KEY\"".to_string(),
        ]);
    }
}

struct TempCodexModelCatalog {
    path: PathBuf,
}

impl Drop for TempCodexModelCatalog {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn write_codex_model_catalog(
    model: Option<&responses_router::RouterModelMetadata>,
) -> Result<TempCodexModelCatalog> {
    let path = env::temp_dir().join(format!(
        "swcli-codex-model-catalog-{}-{}.json",
        std::process::id(),
        random_id()
    ));
    let body = serde_json::to_vec_pretty(&responses_router::codex_models_response(model))?;
    tokio::fs::write(&path, body).await?;
    Ok(TempCodexModelCatalog { path })
}

fn maybe_context_model(model: &str, ctx: Option<&str>) -> String {
    match ctx {
        Some(ctx) if !model.ends_with(']') => format!("{model}[{ctx}]"),
        _ => model.to_string(),
    }
}

fn default_base_url(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Openai | ProviderKind::CodexOauth => "https://api.openai.com/v1",
        ProviderKind::Anthropic | ProviderKind::ClaudeOauth => "https://api.anthropic.com/v1",
    }
}

fn default_wire_for_provider(provider: ProviderKind) -> WireProtocol {
    match provider {
        ProviderKind::Openai | ProviderKind::CodexOauth => WireProtocol::OpenaiResponses,
        ProviderKind::Anthropic | ProviderKind::ClaudeOauth => WireProtocol::AnthropicMessages,
    }
}

fn infer_key_wire(key: &ApiKey) -> WireProtocol {
    if let Some(alias) = key.preset_alias.as_deref()
        && let Ok(preset) = resolve_builtin_preset(alias)
    {
        return preset.wire;
    }
    default_wire_for_provider(key.provider)
}

fn models_url(key: &ApiKey) -> String {
    let base = key.base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

impl ApiKey {
    fn wire_protocol(&self) -> WireProtocol {
        self.wire.unwrap_or_else(|| infer_key_wire(self))
    }

    fn plain_oauth_token(&self) -> Result<String> {
        self.oauth_token
            .as_ref()
            .ok_or_else(|| anyhow!("Key `{}` has no OAuth token.", self.name))
            .and_then(|s| decode_secret(s))
    }

    fn plain_secret(&self) -> Result<String> {
        self.secret
            .as_ref()
            .or(self.oauth_token.as_ref())
            .ok_or_else(|| anyhow!("Key `{}` has no usable secret.", self.name))
            .and_then(|s| decode_secret(s))
    }
}

fn encode_secret(secret: &str) -> String {
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let cipher = Aes256Gcm::new_from_slice(&local_secret_key()).expect("valid AES-256 key");
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce_bytes), secret.as_bytes())
        .expect("encryption should succeed");
    format!(
        "enc:v1:{}:{}",
        BASE64.encode(nonce_bytes),
        BASE64.encode(ciphertext)
    )
}

fn decode_secret(secret: &str) -> Result<String> {
    if let Some(rest) = secret.strip_prefix("enc:v1:") {
        let (nonce, ciphertext) = rest
            .split_once(':')
            .ok_or_else(|| anyhow!("Invalid encrypted secret format"))?;
        let nonce = BASE64.decode(nonce)?;
        let ciphertext = BASE64.decode(ciphertext)?;
        let cipher = Aes256Gcm::new_from_slice(&local_secret_key()).expect("valid AES-256 key");
        let plaintext = cipher
            .decrypt(Nonce::from_slice(&nonce), ciphertext.as_ref())
            .map_err(|_| anyhow!("Failed to decrypt saved secret on this machine"))?;
        return Ok(String::from_utf8(plaintext)?);
    }
    let encoded = secret.strip_prefix("b64:").unwrap_or(secret);
    let bytes = BASE64.decode(encoded)?;
    Ok(String::from_utf8(bytes)?)
}

fn local_secret_key() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"switchcli-local-key-v1");
    hasher.update(env::var("HOME").unwrap_or_default().as_bytes());
    hasher.update(env::var("USER").unwrap_or_default().as_bytes());
    hasher.finalize().into()
}

fn random_id() -> String {
    let mut bytes = [0u8; 6];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|s| {
            if s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./:=@".contains(c))
            {
                s.clone()
            } else {
                format!("{:?}", s)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact(value: &str) -> String {
    let _ = value;
    "***".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_top_level_launch_and_passthrough_args() {
        let cli = parse_cli([
            "swcli",
            "-k",
            "minimax",
            "-m",
            "minimax-m2.7",
            "codex",
            "--auto",
        ])
        .unwrap();

        let ParsedCommand::Launch(args) = normalize_cli(cli).unwrap() else {
            panic!("expected launch command");
        };
        assert_eq!(args.key.as_deref(), Some("minimax"));
        assert_eq!(args.model.as_deref(), Some("minimax-m2.7"));
        assert_eq!(args.tool, "codex");
        assert_eq!(args.tool_args, vec!["--auto"]);
    }

    #[test]
    fn parses_management_subcommand_without_launch_mode() {
        let cli = parse_cli(["swcli", "models", "--json"]).unwrap();

        match normalize_cli(cli).unwrap() {
            ParsedCommand::Models(args) => assert!(args.json),
            _ => panic!("expected models command"),
        }
    }

    #[test]
    fn rejects_launch_flags_for_management_subcommands() {
        let cli = parse_cli(["swcli", "-k", "minimax", "models"]).unwrap();
        assert!(normalize_cli(cli).is_err());
    }

    #[test]
    fn passes_through_swcli_like_flags_after_tool_name() {
        let cli = parse_cli(["swcli", "codex", "-k", "mimo"]).unwrap();
        let ParsedCommand::Launch(args) = normalize_cli(cli).unwrap() else {
            panic!("expected launch command");
        };
        assert_eq!(args.tool, "codex");
        assert_eq!(args.tool_args, vec!["-k", "mimo"]);
    }

    #[test]
    fn appends_claude_context_suffix_once() {
        assert_eq!(
            maybe_context_model("claude-sonnet-4-5", Some("1m")),
            "claude-sonnet-4-5[1m]"
        );
        assert_eq!(
            maybe_context_model("claude-sonnet-4-5[1m]", Some("2m")),
            "claude-sonnet-4-5[1m]"
        );
    }

    #[test]
    fn builds_models_url_without_double_v1() {
        let key = ApiKey {
            id: "id".to_string(),
            name: "name".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            provider: ProviderKind::Openai,
            wire: Some(WireProtocol::OpenaiResponses),
            secret: Some(encode_secret("sk-test")),
            oauth_token: None,
            preset_alias: None,
            models_dev_provider_name: None,
            created_at: Utc::now(),
        };
        assert_eq!(models_url(&key), "https://api.example.com/v1/models");
    }

    #[test]
    fn resolves_builtin_minimax_to_exact_models_dev_name() {
        let preset = resolve_builtin_preset("minimax-cn").unwrap();
        assert_eq!(
            preset.models_dev_provider_name,
            Some("MiniMax (minimaxi.com)")
        );
        assert_eq!(preset.endpoint, "https://api.minimaxi.com/anthropic/v1");
    }

    #[test]
    fn resolves_builtin_xiaomi_token_plan_china() {
        let preset = resolve_builtin_preset("xiaomi-token-plan-cn").unwrap();
        assert_eq!(
            preset.models_dev_provider_name,
            Some("Xiaomi Token Plan (China)")
        );
        assert_eq!(preset.endpoint, "https://token-plan-cn.xiaomimimo.com/v1");
    }

    #[test]
    fn resolves_builtin_github_copilot_preset() {
        let preset = resolve_builtin_preset("github-copilot").unwrap();
        assert_eq!(preset.models_dev_provider_name, None);
        assert_eq!(preset.endpoint, "https://api.githubcopilot.com");
    }

    #[test]
    fn parses_models_dev_catalog_and_matches_exact_provider_name() {
        let catalog = parse_models_dev_catalog(json!({
            "minimax-cn": {
                "name": "MiniMax (minimaxi.com)",
                "npm": "@ai-sdk/anthropic",
                "api": "https://api.minimaxi.com/anthropic/v1",
                "env": ["MINIMAX_API_KEY"],
                "models": {
                    "MiniMax-M2.7": {
                        "name": "MiniMax M2.7",
                        "reasoning": true,
                        "tool_call": true,
                        "limit": { "context": 1000000, "output": 131072 },
                        "cost": { "input": 0.3, "output": 1.2 }
                    }
                }
            }
        }))
        .unwrap();

        let provider = resolve_models_dev_provider(&catalog, "MiniMax (minimaxi.com)").unwrap();
        assert_eq!(provider.id, "minimax-cn");
        assert_eq!(infer_provider_kind(provider), Some(ProviderKind::Anthropic));
        let model = provider.models.get("MiniMax-M2.7").unwrap();
        assert_eq!(model.id, "MiniMax-M2.7");
        assert_eq!(model.name, "MiniMax M2.7");
        assert_eq!(model.extra["reasoning"], true);
        assert!(resolved_model_from(model, WireProtocol::AnthropicMessages).supports_reasoning);
    }

    #[test]
    fn resolves_model_by_exact_and_casefolded_id_or_name() {
        let catalog = parse_models_dev_catalog(json!({
            "minimax-cn": {
                "name": "MiniMax (minimaxi.com)",
                "models": {
                    "MiniMax-M2.7": {
                        "name": "MiniMax M2.7",
                        "limit": { "context": 204800 }
                    }
                }
            }
        }))
        .unwrap();
        let provider = resolve_models_dev_provider(&catalog, "MiniMax (minimaxi.com)").unwrap();

        assert_eq!(
            resolve_model(provider, "MiniMax-M2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert_eq!(
            resolve_model(provider, "minimax-m2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert_eq!(
            resolve_model(provider, "MiniMax M2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert!(
            !resolve_model(provider, "MiniMax M2.7")
                .unwrap()
                .supports_reasoning
        );
        assert!(resolve_model(provider, "MiniMax-M2").is_err());
    }

    #[test]
    fn derives_context_tags_from_models_dev_limit() {
        assert_eq!(context_tag_for_window(Some(999_999)), None);
        assert_eq!(
            context_tag_for_window(Some(1_000_000)).as_deref(),
            Some("1m")
        );
        assert_eq!(
            context_tag_for_window(Some(2_000_000)).as_deref(),
            Some("2m")
        );
    }

    #[test]
    fn detects_existing_codex_context_config() {
        assert!(has_model_context_window_config(&[
            "--config".to_string(),
            "model_context_window=204800".to_string()
        ]));
        assert!(!has_model_context_window_config(&[
            "--config".to_string(),
            "approval_policy=never".to_string()
        ]));
    }

    #[test]
    fn validates_native_tool_provider_protocols() {
        assert!(tool_supports_wire(
            Tool::Codex,
            WireProtocol::OpenaiResponses
        ));
        assert!(tool_supports_wire(
            Tool::Codex,
            WireProtocol::OpenaiCompletions
        ));
        assert!(tool_supports_wire(
            Tool::Codex,
            WireProtocol::AnthropicMessages
        ));
        assert!(!wire_needs_codex_router(WireProtocol::OpenaiResponses));
        assert!(wire_needs_codex_router(WireProtocol::OpenaiCompletions));
        assert!(wire_needs_codex_router(WireProtocol::AnthropicMessages));

        assert!(tool_supports_wire(
            Tool::Claude,
            WireProtocol::AnthropicMessages
        ));
        assert!(!tool_supports_wire(
            Tool::Claude,
            WireProtocol::OpenaiResponses
        ));
        assert!(!tool_supports_wire(
            Tool::Claude,
            WireProtocol::OpenaiCompletions
        ));
    }
}
