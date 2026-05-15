use crate::provider::{
    ProviderKind, WireProtocol, default_wire_for_provider, resolve_builtin_preset,
};
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApiKey {
    pub(crate) id: String,
    pub(crate) name: String,
    #[serde(rename = "baseUrl")]
    pub(crate) base_url: String,
    pub(crate) provider: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) wire: Option<WireProtocol>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) oauth_token: Option<String>,
    #[serde(
        rename = "presetAlias",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) preset_alias: Option<String>,
    #[serde(
        rename = "modelsDevProviderName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) models_dev_provider_name: Option<String>,
    #[serde(rename = "createdAt")]
    pub(crate) created_at: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Config {
    #[serde(default)]
    pub(crate) keys: Vec<ApiKey>,
    #[serde(default)]
    pub(crate) active_key_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CacheEntry {
    pub(crate) fetched_at: u64,
    pub(crate) models: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ModelsCache(pub(crate) HashMap<String, CacheEntry>);

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ModelsDevCache {
    pub(crate) fetched_at: u64,
    pub(crate) catalog: BTreeMap<String, ModelsDevProvider>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModelsDevProvider {
    #[serde(default)]
    pub(crate) id: String,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) npm: Option<String>,
    #[serde(default)]
    pub(crate) api: Option<String>,
    #[serde(default)]
    pub(crate) env: Vec<String>,
    #[serde(default)]
    pub(crate) doc: Option<String>,
    #[serde(default)]
    pub(crate) models: BTreeMap<String, ModelsDevModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModelsDevModel {
    #[serde(default)]
    pub(crate) id: String,
    pub(crate) name: String,
    #[serde(flatten)]
    pub(crate) extra: Value,
}

#[derive(Debug)]
pub(crate) struct Store {
    config_path: PathBuf,
    cache_path: PathBuf,
    models_dev_cache_path: PathBuf,
    pub(crate) config: Config,
}

impl Store {
    pub(crate) fn new() -> Result<Self> {
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

    pub(crate) async fn save(&self) -> Result<()> {
        write_json(&self.config_path, &self.config).await
    }

    pub(crate) fn all_keys(&self) -> &[ApiKey] {
        &self.config.keys
    }

    pub(crate) fn active_key(&self) -> Result<ApiKey> {
        let id = self.config.active_key_id.as_deref().ok_or_else(|| {
            anyhow!("No default key. Run `swcli keys add ...` or `swcli keys default <name>`.")
        })?;
        self.resolve_key(id)
    }

    pub(crate) fn resolve_key(&self, query: &str) -> Result<ApiKey> {
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

    pub(crate) async fn set_active(&mut self, id: String) -> Result<()> {
        self.config.active_key_id = Some(id);
        self.save().await
    }

    pub(crate) async fn read_cache(&self) -> Result<ModelsCache> {
        Ok(read_json(&self.cache_path)?.unwrap_or_default())
    }

    pub(crate) async fn write_cache(&self, cache: &ModelsCache) -> Result<()> {
        write_json(&self.cache_path, cache).await
    }

    pub(crate) async fn read_models_dev_cache(&self) -> Result<Option<ModelsDevCache>> {
        read_json(&self.models_dev_cache_path)
    }

    pub(crate) async fn write_models_dev_cache(&self, cache: &ModelsDevCache) -> Result<()> {
        write_json(&self.models_dev_cache_path, cache).await
    }

    pub(crate) async fn ensure_key_wires(&mut self) -> Result<()> {
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

impl ApiKey {
    pub(crate) fn wire_protocol(&self) -> WireProtocol {
        self.wire.unwrap_or_else(|| infer_key_wire(self))
    }

    pub(crate) fn plain_oauth_token(&self) -> Result<String> {
        self.oauth_token
            .as_ref()
            .ok_or_else(|| anyhow!("Key `{}` has no OAuth token.", self.name))
            .cloned()
    }

    pub(crate) fn plain_secret(&self) -> Result<String> {
        self.secret
            .as_ref()
            .or(self.oauth_token.as_ref())
            .ok_or_else(|| anyhow!("Key `{}` has no usable secret.", self.name))
            .cloned()
    }
}

pub(crate) fn random_id() -> String {
    let mut bytes = [0u8; 6];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn infer_key_wire(key: &ApiKey) -> WireProtocol {
    if let Some(alias) = key.preset_alias.as_deref()
        && let Ok(preset) = resolve_builtin_preset(alias)
    {
        return preset.wire;
    }
    default_wire_for_provider(key.provider)
}

fn config_dir() -> Result<PathBuf> {
    if let Ok(dir) = env::var("SWCLI_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(dir) = env::var("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(dir).join("swcli"));
    }
    let home = env::var("HOME").context("HOME is not set; set SWCLI_CONFIG_DIR explicitly")?;
    Ok(PathBuf::from(home).join(".config").join("swcli"))
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
