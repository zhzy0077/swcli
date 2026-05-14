use anyhow::{Result, anyhow, bail};
use chrono::Utc;
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Password};
use serde::Serialize;
use serde_json::{Value, json};
use std::io::{self, IsTerminal};

mod cli;
mod config;
mod db;
mod launcher;
mod models;
#[path = "nyro_protocol/mod.rs"]
mod protocol;
mod provider;
mod router;
mod tui;

use cli::{KeysArgs, ParsedCommand, normalize_cli, parse_cli};
use config::{ApiKey, Store, random_id};
use provider::github_copilot;
use provider::{
    BUILTIN_PRESETS, BuiltinPreset, ProviderKind, default_base_url, default_wire_for_provider,
    resolve_builtin_preset,
};

#[cfg(test)]
use provider::WireProtocol;

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
        ParsedCommand::Launch(args) => launcher::run_tool(&mut store, args).await,
        ParsedCommand::Keys(args) => handle_keys(&mut store, args).await,
        ParsedCommand::Models(args) => models::handle_models(&mut store, args).await,
    }
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
                let catalog = models::load_models_dev_catalog(store, add.refresh).await?;
                Some(models::resolve_models_dev_provider(&catalog, provider_name)?.clone())
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
        .or_else(|| {
            catalog_provider
                .as_ref()
                .and_then(models::infer_provider_kind)
        })
        .unwrap_or(ProviderKind::Openai);
    let base_url = add
        .base_url
        .take()
        .or_else(|| preset.map(|p| p.endpoint.to_string()))
        .unwrap_or_else(|| default_base_url(provider).to_string());
    let secret = add.key.take();
    let mut oauth_token = add.oauth_token.take();
    if preset.map(|p| p.alias) == Some("github-copilot") && oauth_token.is_none() {
        if !(io::stdin().is_terminal() && io::stdout().is_terminal()) {
            bail!(
                "Missing GitHub Copilot login. Run `swcli keys add {name} --preset github-copilot` interactively to complete device login."
            );
        }
        let token = github_copilot::device_flow_login().await?;
        oauth_token = Some(token);
    }
    if secret.is_none() && oauth_token.is_none() {
        bail!(
            "Missing credential. Pass --key/--oauth-token or run `swcli keys add` interactively."
        );
    }
    let token = secret.as_ref().or(oauth_token.as_ref()).cloned();
    let wire = match add.wire {
        Some(wire) => wire,
        None if let Some(preset) = preset => preset.wire,
        None if catalog_provider.is_some() => catalog_provider
            .as_ref()
            .and_then(models::infer_wire_protocol)
            .unwrap_or_else(|| default_wire_for_provider(provider)),
        None => {
            models::probe_wire_protocol(&base_url, token.as_deref().unwrap_or_default()).await?
        }
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

fn select_interactive(prompt: &str, items: &[tui::PickerItem]) -> Result<Option<usize>> {
    tui::pick(prompt, items).map_err(Into::into)
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
        key.secret = Some(secret);
    }
    if let Some(token) = args.oauth_token {
        key.oauth_token = Some(token);
    }
    if let Some(provider) = args.provider {
        key.provider = provider;
    }
    if let Some(wire) = args.wire {
        key.wire = Some(wire);
    } else if should_reprobe_wire {
        let token = key.plain_secret()?;
        key.wire = Some(models::probe_wire_protocol(&key.base_url, &token).await?);
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
    let models = models::fetch_models(key).await?;
    Ok(PingResult {
        ok: true,
        message: format!("{} models", models.len()),
    })
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
            launcher::maybe_context_model("claude-sonnet-4-5", Some("1m")),
            "claude-sonnet-4-5[1m]"
        );
        assert_eq!(
            launcher::maybe_context_model("claude-sonnet-4-5[1m]", Some("2m")),
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
            secret: Some("sk-test".to_string()),
            oauth_token: None,
            preset_alias: None,
            models_dev_provider_name: None,
            created_at: Utc::now(),
        };
        assert_eq!(
            models::models_url(&key),
            "https://api.example.com/v1/models"
        );
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
        let catalog = models::parse_models_dev_catalog(json!({
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

        let provider =
            models::resolve_models_dev_provider(&catalog, "MiniMax (minimaxi.com)").unwrap();
        assert_eq!(provider.id, "minimax-cn");
        assert_eq!(
            models::infer_provider_kind(provider),
            Some(ProviderKind::Anthropic)
        );
        let model = provider.models.get("MiniMax-M2.7").unwrap();
        assert_eq!(model.id, "MiniMax-M2.7");
        assert_eq!(model.name, "MiniMax M2.7");
        assert_eq!(model.extra["reasoning"], true);
        assert!(
            models::resolved_model_from(model, WireProtocol::AnthropicMessages).supports_reasoning
        );
    }

    #[test]
    fn resolves_model_by_exact_and_casefolded_id_or_name() {
        let catalog = models::parse_models_dev_catalog(json!({
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
        let provider =
            models::resolve_models_dev_provider(&catalog, "MiniMax (minimaxi.com)").unwrap();

        assert_eq!(
            models::resolve_model(provider, "MiniMax-M2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert_eq!(
            models::resolve_model(provider, "minimax-m2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert_eq!(
            models::resolve_model(provider, "MiniMax M2.7").unwrap().id,
            "MiniMax-M2.7"
        );
        assert!(
            !models::resolve_model(provider, "MiniMax M2.7")
                .unwrap()
                .supports_reasoning
        );
        assert!(models::resolve_model(provider, "MiniMax-M2").is_err());
    }

    #[test]
    fn derives_context_tags_from_models_dev_limit() {
        assert_eq!(models::context_tag_for_window(Some(999_999)), None);
        assert_eq!(
            models::context_tag_for_window(Some(1_000_000)).as_deref(),
            Some("1m")
        );
        assert_eq!(
            models::context_tag_for_window(Some(2_000_000)).as_deref(),
            Some("2m")
        );
    }

    #[test]
    fn detects_existing_codex_context_config() {
        assert!(models::has_model_context_window_config(&[
            "--config".to_string(),
            "model_context_window=204800".to_string()
        ]));
        assert!(!models::has_model_context_window_config(&[
            "--config".to_string(),
            "approval_policy=never".to_string()
        ]));
    }

    #[test]
    fn validates_native_tool_provider_protocols() {
        assert!(launcher::tool_supports_wire(
            launcher::Tool::Codex,
            WireProtocol::OpenaiResponses
        ));
        assert!(launcher::tool_supports_wire(
            launcher::Tool::Codex,
            WireProtocol::OpenaiCompletions
        ));
        assert!(launcher::tool_supports_wire(
            launcher::Tool::Codex,
            WireProtocol::AnthropicMessages
        ));
        assert!(!launcher::wire_needs_codex_router(
            WireProtocol::OpenaiResponses
        ));
        assert!(launcher::wire_needs_codex_router(
            WireProtocol::OpenaiCompletions
        ));
        assert!(launcher::wire_needs_codex_router(
            WireProtocol::AnthropicMessages
        ));

        assert!(launcher::tool_supports_wire(
            launcher::Tool::Claude,
            WireProtocol::AnthropicMessages
        ));
        assert!(!launcher::tool_supports_wire(
            launcher::Tool::Claude,
            WireProtocol::OpenaiResponses
        ));
        assert!(!launcher::tool_supports_wire(
            launcher::Tool::Claude,
            WireProtocol::OpenaiCompletions
        ));
    }
}
