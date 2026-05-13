use crate::cli::LaunchArgs;
use crate::copilot_proxy;
use crate::models;
use crate::provider::WireProtocol;
use crate::responses_router;
use crate::store::{ApiKey, Store, random_id};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

pub(crate) async fn run_tool(store: &mut Store, mut args: LaunchArgs) -> Result<()> {
    let tool = Tool::parse(&args.tool)?;
    let key = match args.key.as_deref() {
        Some(q) => store.resolve_key(q)?,
        None => store.active_key()?,
    };
    let resolved_model = models::resolve_launch_model(store, &key, &mut args).await?;
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
                    .and_then(|model| models::context_tag_for_window(model.context_window));
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
        && !models::has_model_catalog_config(&args.tool_args)
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
            && !models::has_model_context_window_config(&tool_args)
            && !models::has_model_context_window_config(&args.tool_args)
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
pub(crate) enum Tool {
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

fn is_github_copilot_key(key: &ApiKey) -> bool {
    key.preset_alias.as_deref() == Some("github-copilot")
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

pub(crate) fn tool_supports_wire(tool: Tool, wire: WireProtocol) -> bool {
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

pub(crate) fn wire_needs_codex_router(wire: WireProtocol) -> bool {
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

pub(crate) fn maybe_context_model(model: &str, ctx: Option<&str>) -> String {
    match ctx {
        Some(ctx) if !model.ends_with(']') => format!("{model}[{ctx}]"),
        _ => model.to_string(),
    }
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
