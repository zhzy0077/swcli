use crate::cli::{LaunchArgs, ModelsArgs};
use crate::config::{
    ApiKey, CacheEntry, ModelsDevCache, ModelsDevModel, ModelsDevProvider, Store, now_secs,
};
use crate::provider::github_copilot;
use crate::provider::{ProviderKind, WireProtocol, default_wire_for_provider};
use crate::tui;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{self, IsTerminal};

const CACHE_TTL_SECS: u64 = 3600;
const MODELS_DEV_CACHE_TTL_SECS: u64 = 24 * 3600;
const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const BUNDLED_MODELS_DEV_API_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/models_dev_api.json"));

#[derive(Debug, Clone)]
pub(crate) struct ResolvedModel {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) context_window: Option<u64>,
    pub(crate) supports_reasoning: bool,
    pub(crate) wire: WireProtocol,
}

pub(crate) async fn handle_models(store: &mut Store, args: ModelsArgs) -> Result<()> {
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

pub(crate) async fn fetch_models(key: &ApiKey) -> Result<Vec<String>> {
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

fn is_github_copilot_key(key: &ApiKey) -> bool {
    key.preset_alias.as_deref() == Some("github-copilot")
}

pub(crate) fn models_url(key: &ApiKey) -> String {
    let base = key.base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

pub(crate) async fn handle_models_from_github_copilot(
    key: &ApiKey,
    args: &ModelsArgs,
) -> Result<()> {
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

pub(crate) async fn probe_wire_protocol(base_url: &str, token: &str) -> Result<WireProtocol> {
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

pub(crate) async fn handle_models_from_models_dev(
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

pub(crate) async fn load_models_dev_catalog(
    store: &Store,
    refresh: bool,
) -> Result<ModelsDevCache> {
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

pub(crate) fn parse_models_dev_catalog(value: Value) -> Result<ModelsDevCache> {
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

pub(crate) fn resolve_models_dev_provider<'a>(
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

pub(crate) fn infer_provider_kind(provider: &ModelsDevProvider) -> Option<ProviderKind> {
    match provider.npm.as_deref() {
        Some("@ai-sdk/anthropic") => Some(ProviderKind::Anthropic),
        Some("@ai-sdk/openai") | Some("@ai-sdk/openai-compatible") => Some(ProviderKind::Openai),
        _ => None,
    }
}

pub(crate) fn infer_wire_protocol(provider: &ModelsDevProvider) -> Option<WireProtocol> {
    match provider.npm.as_deref() {
        Some("@ai-sdk/anthropic") => Some(WireProtocol::AnthropicMessages),
        Some("@ai-sdk/openai") => Some(WireProtocol::OpenaiResponses),
        Some("@ai-sdk/openai-compatible") => Some(WireProtocol::OpenaiCompletions),
        _ => None,
    }
}

pub(crate) async fn resolve_launch_model(
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

pub(crate) fn resolve_model(provider: &ModelsDevProvider, query: &str) -> Result<ResolvedModel> {
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

pub(crate) fn resolved_model_from(model: &ModelsDevModel, wire: WireProtocol) -> ResolvedModel {
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

pub(crate) fn context_tag_for_window(context_window: Option<u64>) -> Option<String> {
    match context_window {
        Some(n) if n >= 2_000_000 => Some("2m".to_string()),
        Some(n) if n >= 1_000_000 => Some("1m".to_string()),
        _ => None,
    }
}

pub(crate) fn has_model_context_window_config(args: &[String]) -> bool {
    args.iter().any(|arg| arg.contains("model_context_window"))
}

pub(crate) fn has_model_catalog_config(args: &[String]) -> bool {
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
