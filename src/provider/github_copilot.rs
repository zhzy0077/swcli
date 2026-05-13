use crate::provider::WireProtocol;
use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

const API_VERSION: &str = "2025-05-01";
const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const OAUTH_SCOPES: &str = "read:user";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

#[derive(Debug, Clone)]
pub struct GithubCopilotSession {
    pub base_url: String,
    pub access_token: String,
}

#[derive(Debug, Clone)]
pub struct GithubCopilotModel {
    pub id: String,
    pub name: String,
    pub context_window: Option<u64>,
    pub supports_reasoning: bool,
    pub wire: WireProtocol,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    token: String,
    #[serde(rename = "expires_at")]
    _expires_at: i64,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default = "default_poll_interval_secs")]
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct DeviceAccessTokenResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<LiveModel>,
}

#[derive(Debug, Deserialize)]
struct LiveModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    model_picker_enabled: Option<bool>,
    #[serde(default)]
    supported_endpoints: Vec<String>,
    #[serde(default)]
    capabilities: Option<LiveCapabilities>,
}

#[derive(Debug, Default, Deserialize)]
struct LiveCapabilities {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    limits: LiveLimits,
    #[serde(default)]
    supports: LiveSupports,
}

#[derive(Debug, Default, Deserialize)]
struct LiveLimits {
    #[serde(default)]
    max_context_window_tokens: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct LiveSupports {
    #[serde(default)]
    adaptive_thinking: Option<bool>,
    #[serde(default)]
    max_thinking_budget: Option<u64>,
    #[serde(default)]
    reasoning_effort: Option<Vec<String>>,
}

pub async fn resolve_session(
    refresh_token: &str,
    base_url: Option<&str>,
    enterprise_url: Option<&str>,
) -> Result<GithubCopilotSession> {
    let domain = normalize_domain(enterprise_url.unwrap_or("github.com"));
    let base_url = base_url
        .map(str::to_string)
        .unwrap_or_else(|| "https://api.githubcopilot.com".to_string());
    exchange_copilot_token(refresh_token, &base_url, Some(&domain)).await
}

pub async fn fetch_models(
    refresh_token: &str,
    base_url: Option<&str>,
    enterprise_url: Option<&str>,
) -> Result<Vec<GithubCopilotModel>> {
    let session = resolve_session(refresh_token, base_url, enterprise_url).await?;
    let response = Client::new()
        .get(endpoint_url(&session.base_url, "/models"))
        .headers(model_headers(&session.access_token))
        .send()
        .await
        .context("GitHub Copilot models request failed")?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or_else(|_| Value::Null);
    if !status.is_success() {
        bail!("GitHub Copilot models request failed with {status}: {body}");
    }
    Ok(parse_models_response(body)?)
}

pub async fn device_flow_login() -> Result<String> {
    let client = Client::new();
    let response = client
        .post(DEVICE_CODE_URL)
        .header(reqwest::header::ACCEPT, "application/json")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .headers(vscode_headers())
        .json(&serde_json::json!({
            "client_id": CLIENT_ID,
            "scope": OAUTH_SCOPES,
        }))
        .send()
        .await
        .context("Failed to request GitHub device code")?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or_else(|_| Value::Null);
    if !status.is_success() {
        bail!("GitHub device code request failed with {status}: {body}");
    }
    let device: DeviceCodeResponse =
        serde_json::from_value(body).context("Invalid GitHub device code response")?;

    eprintln!(
        "GitHub Copilot login: visit {} and enter code {}",
        device.verification_uri, device.user_code
    );

    poll_for_device_access_token(&client, &device).await
}

pub fn conversation_headers(access_token: &str, body: Option<&Value>) -> Vec<(String, String)> {
    let (is_vision, is_agent) = conversation_metadata(body);
    let mut headers = vec![
        (
            "Authorization".to_string(),
            format!("Bearer {access_token}"),
        ),
        (
            "User-Agent".to_string(),
            "GitHubCopilotChat/0.38.0".to_string(),
        ),
        ("Editor-Version".to_string(), "vscode/1.110.1".to_string()),
        (
            "Editor-Plugin-Version".to_string(),
            "copilot-chat/0.38.0".to_string(),
        ),
        (
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        ),
        (
            "Openai-Intent".to_string(),
            "conversation-agent".to_string(),
        ),
        ("X-GitHub-Api-Version".to_string(), API_VERSION.to_string()),
        (
            "X-Initiator".to_string(),
            if is_agent { "agent" } else { "user" }.to_string(),
        ),
        ("X-Interaction-Id".to_string(), random_request_id()),
        (
            "X-Interaction-Type".to_string(),
            "conversation-agent".to_string(),
        ),
        ("X-Request-Id".to_string(), random_request_id()),
    ];
    if is_vision {
        headers.push(("Copilot-Vision-Request".to_string(), "true".to_string()));
    }
    headers
}

fn parse_models_response(body: Value) -> Result<Vec<GithubCopilotModel>> {
    let parsed: ModelsResponse =
        serde_json::from_value(body).context("Invalid GitHub Copilot models response")?;
    let mut models = parsed
        .data
        .into_iter()
        .filter(is_picker_model)
        .map(|model| {
            let supports_messages = model
                .supported_endpoints
                .iter()
                .any(|endpoint| endpoint == "/v1/messages");
            let supports_responses = model
                .supported_endpoints
                .iter()
                .any(|endpoint| endpoint == "/responses");
            let capabilities = model.capabilities.unwrap_or_default();
            GithubCopilotModel {
                id: model.id.clone(),
                name: model.name.unwrap_or_else(|| model.id.clone()),
                context_window: if supports_messages {
                    Some(1_000_000)
                } else {
                    capabilities.limits.max_context_window_tokens
                },
                supports_reasoning: model_supports_reasoning(&capabilities),
                wire: if supports_messages {
                    WireProtocol::AnthropicMessages
                } else if supports_responses {
                    WireProtocol::OpenaiResponses
                } else {
                    WireProtocol::OpenaiCompletions
                },
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

async fn poll_for_device_access_token(
    client: &Client,
    device: &DeviceCodeResponse,
) -> Result<String> {
    let mut interval = device.interval.max(1);
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        let response = client
            .post(ACCESS_TOKEN_URL)
            .header(reqwest::header::ACCEPT, "application/json")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .headers(vscode_headers())
            .json(&serde_json::json!({
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }))
            .send()
            .await
            .context("Failed while polling GitHub device login")?;
        let token_response: DeviceAccessTokenResponse = response
            .json()
            .await
            .context("Invalid GitHub device access token response")?;
        if let Some(access_token) = token_response.access_token {
            return Ok(access_token);
        }
        match token_response.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                interval = token_response
                    .interval
                    .unwrap_or(interval + 5)
                    .max(interval + 1);
            }
            Some("expired_token") => bail!("GitHub device code expired. Please try again."),
            Some("access_denied") => bail!("GitHub device login was denied."),
            Some(error) => bail!("GitHub device login failed: {error}"),
            None => continue,
        }
    }
}

async fn exchange_copilot_token(
    refresh_token: &str,
    base_url: &str,
    domain: Option<&str>,
) -> Result<GithubCopilotSession> {
    let domain = domain.unwrap_or("github.com");
    let response = Client::new()
        .get(format!(
            "https://{}/copilot_internal/v2/token",
            api_domain(domain)
        ))
        .headers(base_headers(refresh_token))
        .send()
        .await
        .context("GitHub Copilot token exchange failed")?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or_else(|_| Value::Null);
    if !status.is_success() {
        bail!("GitHub Copilot token exchange failed with {status}: {body}");
    }
    let parsed: TokenResponse =
        serde_json::from_value(body).context("Invalid GitHub Copilot token response")?;
    Ok(GithubCopilotSession {
        base_url: base_url.to_string(),
        access_token: parsed.token,
    })
}

fn is_picker_model(model: &LiveModel) -> bool {
    model.model_picker_enabled != Some(false)
        && model
            .capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.kind.as_deref())
            == Some("chat")
}

fn model_supports_reasoning(capabilities: &LiveCapabilities) -> bool {
    capabilities.supports.adaptive_thinking == Some(true)
        || capabilities.supports.max_thinking_budget.is_some()
        || capabilities
            .supports
            .reasoning_effort
            .as_ref()
            .map(|values| !values.is_empty())
            .unwrap_or(false)
}

fn model_headers(access_token: &str) -> reqwest::header::HeaderMap {
    let mut headers = base_headers(access_token);
    headers.insert(
        "Openai-Intent",
        reqwest::header::HeaderValue::from_static("model-access"),
    );
    headers.insert(
        "X-GitHub-Api-Version",
        reqwest::header::HeaderValue::from_static(API_VERSION),
    );
    headers.insert(
        "X-Interaction-Type",
        reqwest::header::HeaderValue::from_static("model-access"),
    );
    headers.insert(
        "X-Request-Id",
        reqwest::header::HeaderValue::from_str(&random_request_id())
            .expect("request id header should be valid"),
    );
    headers
}

fn base_headers(token: &str) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .expect("authorization header should be valid"),
    );
    headers.extend(vscode_headers());
    headers
}

fn vscode_headers() -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static("GitHubCopilotChat/0.38.0"),
    );
    headers.insert(
        "Editor-Version",
        reqwest::header::HeaderValue::from_static("vscode/1.110.1"),
    );
    headers.insert(
        "Editor-Plugin-Version",
        reqwest::header::HeaderValue::from_static("copilot-chat/0.38.0"),
    );
    headers.insert(
        "Copilot-Integration-Id",
        reqwest::header::HeaderValue::from_static("vscode-chat"),
    );
    headers
}

fn default_poll_interval_secs() -> u64 {
    5
}

fn normalize_domain(url: &str) -> String {
    url.trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

fn api_domain(domain: &str) -> String {
    if domain == "github.com" {
        "api.github.com".to_string()
    } else {
        format!("api.{domain}")
    }
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    if base.ends_with("/v1") && path.starts_with("v1/") {
        format!("{base}/{}", path.trim_start_matches("v1/"))
    } else {
        format!("{base}/{path}")
    }
}

fn random_request_id() -> String {
    let bytes = rand::random::<[u8; 16]>();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn conversation_metadata(body: Option<&Value>) -> (bool, bool) {
    let Some(body) = body else {
        return (false, false);
    };

    if let Some(messages) = body.get("messages").and_then(|value| value.as_array()) {
        let is_vision = messages.iter().any(|message| {
            openai_message_has_image(message) || anthropic_message_has_image(message)
        });
        let is_agent = messages
            .last()
            .and_then(|message| message.get("role"))
            .and_then(|value| value.as_str())
            .map(|role| matches!(role, "assistant" | "tool"))
            .unwrap_or(false);
        return (is_vision, is_agent);
    }

    if let Some(items) = body.get("input").and_then(|value| value.as_array()) {
        let is_vision = items.iter().any(|item| responses_item_has_image(item));
        let is_agent = items.last().map(item_is_agent).unwrap_or(false);
        return (is_vision, is_agent);
    }

    (false, false)
}

fn openai_message_has_image(message: &Value) -> bool {
    message
        .get("content")
        .and_then(|value| value.as_array())
        .map(|parts| {
            parts
                .iter()
                .any(|part| part.get("type").and_then(|value| value.as_str()) == Some("image_url"))
        })
        .unwrap_or(false)
}

fn responses_item_has_image(item: &Value) -> bool {
    item.get("content")
        .and_then(|value| value.as_array())
        .map(|parts| {
            parts.iter().any(|part| {
                part.get("type").and_then(|value| value.as_str()) == Some("input_image")
            })
        })
        .unwrap_or(false)
}

fn anthropic_message_has_image(message: &Value) -> bool {
    message
        .get("content")
        .and_then(|value| value.as_array())
        .map(|parts| {
            parts
                .iter()
                .any(|part| part.get("type").and_then(|value| value.as_str()) == Some("image"))
        })
        .unwrap_or(false)
}

fn item_is_agent(item: &Value) -> bool {
    item.get("role")
        .and_then(|value| value.as_str())
        .map(|role| role == "assistant")
        .unwrap_or(false)
        || item
            .get("type")
            .and_then(|value| value.as_str())
            .map(|kind| {
                matches!(
                    kind,
                    "file_search_call"
                        | "computer_call"
                        | "computer_call_output"
                        | "web_search_call"
                        | "function_call"
                        | "function_call_output"
                        | "image_generation_call"
                        | "code_interpreter_call"
                        | "local_shell_call"
                        | "local_shell_call_output"
                        | "mcp_list_tools"
                        | "mcp_approval_request"
                        | "mcp_approval_response"
                        | "mcp_call"
                        | "reasoning"
                )
            })
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_picker_models_and_promotes_messages_models_to_1m() {
        let models = parse_models_response(json!({
            "data": [
                {
                    "id": "claude-sonnet-4.6",
                    "name": "Claude Sonnet 4.6",
                    "model_picker_enabled": true,
                    "supported_endpoints": ["/chat/completions", "/v1/messages"],
                    "capabilities": {
                        "type": "chat",
                        "limits": {
                            "max_context_window_tokens": 200000
                        },
                        "supports": {
                            "adaptive_thinking": true
                        }
                    }
                },
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "model_picker_enabled": true,
                    "supported_endpoints": ["/chat/completions"],
                    "capabilities": {
                        "type": "chat",
                        "limits": {
                            "max_context_window_tokens": 128000
                        },
                        "supports": {}
                    }
                },
                {
                    "id": "gpt-5.5",
                    "name": "GPT-5.5",
                    "model_picker_enabled": true,
                    "supported_endpoints": ["/responses", "ws:/responses"],
                    "capabilities": {
                        "type": "chat",
                        "limits": {
                            "max_context_window_tokens": 400000
                        },
                        "supports": {}
                    }
                }
            ]
        }))
        .unwrap();

        assert_eq!(models.len(), 3);
        assert_eq!(models[0].wire, WireProtocol::AnthropicMessages);
        assert_eq!(models[0].context_window, Some(1_000_000));
        assert!(models[0].supports_reasoning);
        assert_eq!(models[1].wire, WireProtocol::OpenaiCompletions);
        assert_eq!(models[1].context_window, Some(128000));
        assert_eq!(models[2].wire, WireProtocol::OpenaiResponses);
        assert_eq!(models[2].context_window, Some(400000));
    }

    #[test]
    fn parses_device_code_response() {
        let response: DeviceCodeResponse = serde_json::from_value(json!({
            "device_code": "abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 8
        }))
        .unwrap();
        assert_eq!(response.device_code, "abc");
        assert_eq!(response.user_code, "ABCD-1234");
        assert_eq!(response.verification_uri, "https://github.com/login/device");
        assert_eq!(response.interval, 8);
    }

    #[test]
    fn device_code_response_defaults_interval() {
        let response: DeviceCodeResponse = serde_json::from_value(json!({
            "device_code": "abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device"
        }))
        .unwrap();
        assert_eq!(response.interval, 5);
    }
}
