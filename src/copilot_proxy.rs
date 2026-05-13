use crate::github_copilot;
use crate::responses_router;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

pub struct ProxyHandle {
    pub port: u16,
    _task: JoinHandle<()>,
}

#[derive(Clone)]
struct ProxyState {
    provider_id: String,
    mode: ProxyMode,
    model: Option<responses_router::RouterModelMetadata>,
    client: reqwest::Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProxyMode {
    OpenaiResponses,
    OpenaiChat,
    AnthropicMessages,
}

pub async fn start_openai_responses_proxy(
    provider_id: String,
    model: Option<responses_router::RouterModelMetadata>,
) -> Result<ProxyHandle> {
    start_proxy(provider_id, ProxyMode::OpenaiResponses, model).await
}

pub async fn start_openai_chat_proxy(provider_id: String) -> Result<ProxyHandle> {
    start_proxy(provider_id, ProxyMode::OpenaiChat, None).await
}

pub async fn start_anthropic_messages_proxy(provider_id: String) -> Result<ProxyHandle> {
    start_proxy(provider_id, ProxyMode::AnthropicMessages, None).await
}

async fn start_proxy(
    provider_id: String,
    mode: ProxyMode,
    model: Option<responses_router::RouterModelMetadata>,
) -> Result<ProxyHandle> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let state = Arc::new(ProxyState {
        provider_id,
        mode,
        model,
        client: reqwest::Client::new(),
    });

    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let state = state.clone();
            tokio::spawn(async move {
                if let Err(err) = handle_connection(stream, state).await {
                    eprintln!("swcli: github copilot proxy request failed: {err:#}");
                }
            });
        }
    });

    Ok(ProxyHandle { port, _task: task })
}

async fn handle_connection(mut stream: TcpStream, state: Arc<ProxyState>) -> Result<()> {
    let request = read_http_request(&mut stream).await?;
    let response = route_request(&request, &state).await;
    let bytes = match response {
        Ok(response) => response,
        Err(err) => http_json(
            500,
            &json!({
                "error": {
                    "message": err.to_string(),
                    "type": "swcli_github_copilot_proxy_error"
                }
            }),
        ),
    };
    stream.write_all(&bytes).await?;
    Ok(())
}

async fn route_request(request: &HttpRequest, state: &ProxyState) -> Result<Vec<u8>> {
    match state.mode {
        ProxyMode::OpenaiResponses => match request.path.as_str() {
            "/responses" | "/v1/responses" => {
                forward_json(state, "/responses", request, false).await
            }
            "/models" | "/v1/models" => Ok(http_json(
                200,
                &responses_router::codex_models_response(state.model.as_ref()),
            )),
            _ => Ok(http_json(
                404,
                &json!({"error": {"message": format!("Unsupported proxy path {}", request.path)}}),
            )),
        },
        ProxyMode::OpenaiChat => match request.path.as_str() {
            "/chat/completions" | "/v1/chat/completions" => {
                forward_json(state, "/chat/completions", request, false).await
            }
            _ => Ok(http_json(
                404,
                &json!({"error": {"message": format!("Unsupported proxy path {}", request.path)}}),
            )),
        },
        ProxyMode::AnthropicMessages => match request.path.as_str() {
            "/messages" | "/v1/messages" => {
                forward_json(state, "/v1/messages", request, true).await
            }
            _ => Ok(http_json(
                404,
                &json!({"error": {"message": format!("Unsupported proxy path {}", request.path)}}),
            )),
        },
    }
}

async fn forward_json(
    state: &ProxyState,
    upstream_path: &str,
    request: &HttpRequest,
    anthropic: bool,
) -> Result<Vec<u8>> {
    let body = if request.body.is_empty() {
        None
    } else {
        Some(serde_json::from_slice::<Value>(&request.body).context("Invalid proxy request JSON")?)
    };
    let session = github_copilot::resolve_session(&state.provider_id)
        .await
        .context("GitHub Copilot auth resolution failed")?;
    let mut upstream = state
        .client
        .post(endpoint_url(&session.base_url, upstream_path));
    for (name, value) in github_copilot::conversation_headers(&session.access_token, body.as_ref())
    {
        upstream = upstream.header(name, value);
    }
    if anthropic {
        upstream = upstream.header("anthropic-version", "2023-06-01");
    }
    if let Some(body) = &body {
        upstream = upstream.json(body);
    }

    let response = upstream
        .send()
        .await
        .with_context(|| format!("Failed to proxy GitHub Copilot request to {upstream_path}"))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let body = response.text().await.unwrap_or_default();
    Ok(http_text(status, &content_type, body))
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

struct HttpRequest {
    path: String,
    body: Vec<u8>,
}

async fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Err(anyhow!("connection closed before headers"));
        }
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_header_end(&buffer) {
            break pos;
        }
        if buffer.len() > 1024 * 1024 {
            return Err(anyhow!("request headers too large"));
        }
    };

    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let path = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string();
    let content_length = content_length(&headers)?;
    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let mut chunk = vec![0u8; body_start + content_length - buffer.len()];
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
    let body = buffer
        .get(body_start..body_start + content_length)
        .unwrap_or_default()
        .to_vec();
    Ok(HttpRequest { path, body })
}

fn content_length(headers: &str) -> Result<usize> {
    Ok(headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>())
        })
        .transpose()?
        .unwrap_or_default())
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn http_json(status: u16, body: &Value) -> Vec<u8> {
    http_text(
        status,
        "application/json",
        serde_json::to_string(body).unwrap_or_else(|_| "{}".to_string()),
    )
}

fn http_text(status: u16, content_type: &str, body: impl Into<String>) -> Vec<u8> {
    let body = body.into();
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Upstream",
    };
    format!(
        "HTTP/1.1 {status} {status_text}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_copilot_endpoint_urls_without_double_v1() {
        assert_eq!(
            endpoint_url("https://api.githubcopilot.com", "/responses"),
            "https://api.githubcopilot.com/responses"
        );
        assert_eq!(
            endpoint_url("http://127.0.0.1:1234/v1", "/v1/chat/completions"),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
    }
}
