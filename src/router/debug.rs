use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;

static DEBUG_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct DebugDump {
    dir: Option<PathBuf>,
}

impl DebugDump {
    pub(crate) async fn start(route: &str) -> Self {
        if !debug_enabled() {
            return Self { dir: None };
        }

        let id = DEBUG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from("/tmp/swcli").join(format!(
            "{}-{}-{}",
            unix_millis(),
            sanitize_name(route),
            id
        ));
        if let Err(err) = fs::create_dir_all(&dir).await {
            eprintln!(
                "swcli: failed to create debug dump dir {}: {err}",
                dir.display()
            );
            return Self { dir: None };
        }

        Self { dir: Some(dir) }
    }

    pub(crate) async fn json(&self, name: &str, value: &Value) {
        let Some(dir) = &self.dir else {
            return;
        };
        let path = dir.join(format!("{name}.json"));
        let Ok(bytes) = serde_json::to_vec_pretty(value) else {
            return;
        };
        if let Err(err) = fs::write(&path, bytes).await {
            eprintln!(
                "swcli: failed to write debug dump {}: {err}",
                path.display()
            );
        }
    }

    pub(crate) async fn text(&self, name: &str, content: &str) {
        let Some(dir) = &self.dir else {
            return;
        };
        let path = dir.join(name);
        if let Err(err) = fs::write(&path, content).await {
            eprintln!(
                "swcli: failed to write debug dump {}: {err}",
                path.display()
            );
        }
    }
}

fn debug_enabled() -> bool {
    matches!(
        env::var("SWCLI_DEBUG").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn sanitize_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('-').to_string()
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
