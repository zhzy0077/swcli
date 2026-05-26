use super::debug::DebugDump;
use crate::provider::{WireProtocol, github_copilot};
use anyhow::{Context, Result, anyhow};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};
use std::io::ErrorKind;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use crate::protocol::ids::{
    ANTHROPIC_MESSAGES_2023_06_01, OPENAI_CHAT_COMPLETIONS_V1, OPENAI_RESPONSES_V1,
    ProtocolEndpoint,
};
use crate::protocol::ir::{AiRequest, AiResponse, AiStreamDelta, ReasoningEffort, Role};
use crate::protocol::{self as nyro_protocol, StreamResponseDecoder};

const CODEX_BASE_INSTRUCTIONS: &str = "You are Codex, a coding agent. You collaborate with the user on software engineering tasks. Use the provided tools when needed, keep changes scoped, and communicate clearly.";
const THINKING_SIGNATURE_CACHE_TTL: Duration = Duration::from_secs(10 * 60);

#[derive(Clone)]
struct ThinkingSignatureCache {
    ttl: Duration,
    entries: Arc<Mutex<Vec<ThinkingSignatureEntry>>>,
}

#[derive(Clone)]
struct ThinkingSignatureEntry {
    reasoning: String,
    signature: String,
    inserted_at: Instant,
}

impl ThinkingSignatureCache {
    fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn insert(&self, reasoning: &str, signature: &str) {
        let reasoning = normalize_cached_reasoning(reasoning);
        let signature = signature.trim();
        if reasoning.is_empty() || signature.is_empty() {
            return;
        }

        let now = Instant::now();
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        entries.retain(|entry| now.duration_since(entry.inserted_at) <= self.ttl);
        if let Some(entry) = entries
            .iter_mut()
            .find(|entry| entry.reasoning == reasoning)
        {
            entry.signature = signature.to_string();
            entry.inserted_at = now;
            return;
        }
        entries.push(ThinkingSignatureEntry {
            reasoning,
            signature: signature.to_string(),
            inserted_at: now,
        });
    }

    fn lookup(&self, reasoning: &str) -> Option<String> {
        let reasoning = normalize_cached_reasoning(reasoning);
        if reasoning.is_empty() {
            return None;
        }

        let now = Instant::now();
        let Ok(mut entries) = self.entries.lock() else {
            return None;
        };
        entries.retain(|entry| now.duration_since(entry.inserted_at) <= self.ttl);
        entries
            .iter()
            .rev()
            .find(|entry| entry.reasoning == reasoning)
            .map(|entry| entry.signature.clone())
    }
}

fn normalize_cached_reasoning(reasoning: &str) -> String {
    reasoning.trim().to_string()
}

pub struct RouterHandle {
    pub port: u16,
    _task: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub struct RouterModelMetadata {
    pub id: String,
    pub name: String,
    pub context_window: Option<u64>,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub supports_search: bool,
}

#[derive(Clone)]
struct RouterState {
    target_base_url: String,
    auth: RouterAuth,
    model: Option<RouterModelMetadata>,
    target_protocol: ProtocolEndpoint,
    client: reqwest::Client,
    thinking_signature_cache: ThinkingSignatureCache,
    copilot_token_cache: github_copilot::CopilotTokenCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolMode {
    Native,
    Transform,
}

#[derive(Clone)]
enum RouterAuth {
    Bearer(String),
    GithubCopilot { refresh_token: String },
}

pub async fn start_anthropic_responses_router(
    target_base_url: String,
    api_key: String,
    model: Option<RouterModelMetadata>,
) -> Result<RouterHandle> {
    start_responses_router(
        target_base_url,
        RouterAuth::Bearer(api_key),
        model,
        ANTHROPIC_MESSAGES_2023_06_01,
    )
    .await
}

pub async fn start_openai_chat_responses_router(
    target_base_url: String,
    api_key: String,
    model: Option<RouterModelMetadata>,
) -> Result<RouterHandle> {
    start_responses_router(
        target_base_url,
        RouterAuth::Bearer(api_key),
        model,
        OPENAI_CHAT_COMPLETIONS_V1,
    )
    .await
}

pub async fn start_github_copilot_codex_router(
    target_base_url: String,
    refresh_token: String,
    target_wire: WireProtocol,
    model: Option<RouterModelMetadata>,
) -> Result<RouterHandle> {
    start_responses_router(
        target_base_url,
        RouterAuth::GithubCopilot { refresh_token },
        model,
        router_target_wire(target_wire)?,
    )
    .await
}

pub async fn start_github_copilot_anthropic_messages_router(
    target_base_url: String,
    refresh_token: String,
    model: Option<RouterModelMetadata>,
) -> Result<RouterHandle> {
    start_responses_router(
        target_base_url,
        RouterAuth::GithubCopilot { refresh_token },
        model,
        ANTHROPIC_MESSAGES_2023_06_01,
    )
    .await
}

async fn start_responses_router(
    target_base_url: String,
    auth: RouterAuth,
    model: Option<RouterModelMetadata>,
    target_protocol: ProtocolEndpoint,
) -> Result<RouterHandle> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let state = Arc::new(RouterState {
        target_base_url,
        auth,
        model,
        target_protocol,
        client: reqwest::Client::new(),
        thinking_signature_cache: ThinkingSignatureCache::new(THINKING_SIGNATURE_CACHE_TTL),
        copilot_token_cache: github_copilot::CopilotTokenCache::new(),
    });

    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let state = state.clone();
            tokio::spawn(async move {
                if let Err(err) = handle_connection(stream, state).await {
                    eprintln!("swcli: codex responses router request failed: {err:#}");
                }
            });
        }
    });

    Ok(RouterHandle { port, _task: task })
}

async fn handle_connection(mut stream: TcpStream, state: Arc<RouterState>) -> Result<()> {
    let request = read_http_request(&mut stream).await?;
    let response = route_request(&request, &state).await;
    match response {
        Ok(RouterResponse::Buffered(bytes)) => stream.write_all(&bytes).await?,
        Ok(RouterResponse::ResponsesStream {
            upstream,
            ingress_protocol,
            target_protocol,
            debug,
        }) => {
            stream
                .write_all(&http_chunked_head(200, "text/event-stream"))
                .await?;
            if let Err(err) = stream_target_to_responses(
                upstream,
                ingress_protocol,
                target_protocol,
                &debug,
                &state.thinking_signature_cache,
                &mut stream,
            )
            .await
            {
                if !is_broken_pipe(&err) {
                    return Err(err);
                }
            }
        }
        Ok(RouterResponse::NativeStream {
            upstream,
            content_type,
            debug,
        }) => {
            stream
                .write_all(&http_chunked_head(200, &content_type))
                .await?;
            if let Err(err) = stream_upstream_passthrough(upstream, &debug, &mut stream).await
                && !is_broken_pipe(&err)
            {
                return Err(err);
            }
        }
        Err(err) => {
            let bytes = http_json(
                500,
                &json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "swcli_router_error"
                    }
                }),
            );
            stream.write_all(&bytes).await?;
        }
    };
    Ok(())
}

fn is_broken_pipe(err: &anyhow::Error) -> bool {
    err.downcast_ref::<std::io::Error>()
        .is_some_and(|err| err.kind() == ErrorKind::BrokenPipe)
}

enum RouterResponse {
    Buffered(Vec<u8>),
    NativeStream {
        upstream: reqwest::Response,
        content_type: String,
        debug: DebugDump,
    },
    ResponsesStream {
        upstream: reqwest::Response,
        ingress_protocol: ProtocolEndpoint,
        target_protocol: ProtocolEndpoint,
        debug: DebugDump,
    },
}

async fn route_request(request: &HttpRequest, state: &RouterState) -> Result<RouterResponse> {
    match request.path.as_str() {
        "/responses" | "/v1/responses" => {
            handle_protocol_request(request, state, OPENAI_RESPONSES_V1).await
        }
        "/messages" | "/v1/messages" => {
            handle_protocol_request(request, state, ANTHROPIC_MESSAGES_2023_06_01).await
        }
        "/models" | "/v1/models" => Ok(RouterResponse::Buffered(http_json(
            200,
            &codex_models_response(state.model.as_ref()),
        ))),
        _ => Ok(RouterResponse::Buffered(http_json(
            404,
            &json!({"error": {"message": format!("Unsupported router path {}", request.path)}}),
        ))),
    }
}

async fn handle_protocol_request(
    request: &HttpRequest,
    state: &RouterState,
    ingress_protocol: ProtocolEndpoint,
) -> Result<RouterResponse> {
    let body: Value = serde_json::from_slice(&request.body).context("Invalid request JSON")?;
    let debug = DebugDump::start(debug_name(ingress_protocol)).await;
    debug
        .text(
            "00-client-request.http",
            &format_http_message(&redact_http_head(&request.head), &request.body),
        )
        .await;
    debug.json("01-request-before-convert", &body).await;

    let decoded = decode_ingress_request(ingress_protocol, &body)?;
    let stream = decoded.stream.enabled;
    let original_model = decoded.model.clone();
    let mode = protocol_mode(ingress_protocol, state.target_protocol);

    let (upstream_request, upstream_headers, upstream_path) = encode_request_for_target(
        body,
        decoded,
        ingress_protocol,
        state.target_protocol,
        state.model.as_ref(),
        Some(&state.thinking_signature_cache),
    )?;
    debug
        .json("02-request-after-convert-upstream", &upstream_request)
        .await;
    debug
        .text(
            "02-upstream-request.http",
            &format_upstream_request_http(
                state,
                &upstream_path,
                &upstream_request,
                &upstream_headers,
            ),
        )
        .await;

    let upstream = post_target(state, &upstream_path, &upstream_request, &upstream_headers).await?;
    if !upstream.status().is_success() {
        let status = upstream.status().as_u16();
        let status_line = upstream_status_line(&upstream);
        let headers = format_response_headers(upstream.headers());
        let text = upstream.text().await.unwrap_or_default();
        debug
            .text("03-response-before-convert-error.txt", &text)
            .await;
        debug
            .text(
                "03-response-before-convert-error.http",
                &format_http_message(&format!("{status_line}\r\n{headers}"), text.as_bytes()),
            )
            .await;
        return Ok(RouterResponse::Buffered(http_text(
            status,
            "application/json",
            text,
        )));
    }

    let content_type = response_content_type(&upstream);
    if stream && content_type.contains("text/event-stream") {
        if mode == ProtocolMode::Native {
            return Ok(RouterResponse::NativeStream {
                upstream,
                content_type,
                debug,
            });
        }
        return Ok(RouterResponse::ResponsesStream {
            upstream,
            ingress_protocol,
            target_protocol: state.target_protocol,
            debug,
        });
    }

    let status = upstream.status().as_u16();
    let status_line = upstream_status_line(&upstream);
    let headers = format_response_headers(upstream.headers());
    let upstream_text = upstream.text().await?;
    debug
        .text(
            "03-response-before-convert-upstream.http",
            &format_http_message(
                &format!("{status_line}\r\n{headers}"),
                upstream_text.as_bytes(),
            ),
        )
        .await;
    if mode == ProtocolMode::Native {
        debug
            .text("04-response-after-convert", &upstream_text)
            .await;
        return Ok(RouterResponse::Buffered(http_text(
            status,
            &content_type,
            upstream_text,
        )));
    }
    let upstream_response: Value = serde_json::from_str(&upstream_text)?;
    debug
        .json("03-response-before-convert-upstream", &upstream_response)
        .await;
    let response =
        parse_target_response(upstream_response, state.target_protocol, &original_model)?;
    cache_response_thinking_signature(
        &state.thinking_signature_cache,
        state.target_protocol,
        &response,
    );
    let response_body = nyro_protocol::format_response(ingress_protocol, &response);

    if stream {
        let sse = events_to_sse(nyro_protocol::format_response_stream(
            ingress_protocol,
            &response,
        ));
        debug.text("04-response-after-convert.sse", &sse).await;
        Ok(RouterResponse::Buffered(http_text(
            200,
            "text/event-stream",
            sse,
        )))
    } else {
        debug
            .json("04-response-after-convert", &response_body)
            .await;
        Ok(RouterResponse::Buffered(http_json(200, &response_body)))
    }
}

fn encode_request_for_target(
    body: Value,
    mut request: AiRequest,
    ingress_protocol: ProtocolEndpoint,
    target_protocol: ProtocolEndpoint,
    model: Option<&RouterModelMetadata>,
    thinking_signature_cache: Option<&ThinkingSignatureCache>,
) -> Result<(Value, HeaderMap, String)> {
    if protocol_mode(ingress_protocol, target_protocol) == ProtocolMode::Native {
        let mut body = body;
        let model_id = model
            .map(|model| model.id.as_str())
            .unwrap_or(request.model.as_str());
        set_body_model(&mut body, model_id);
        return Ok((
            body,
            native_headers(target_protocol),
            target_path(target_protocol, request.stream.enabled),
        ));
    }

    if let Some(model) = model {
        request.model = model.id.clone();
    }
    if ingress_protocol == OPENAI_RESPONSES_V1 && target_protocol == OPENAI_CHAT_COMPLETIONS_V1 {
        mirror_responses_assistant_text_as_reasoning(&mut request);
    }
    if target_protocol == ANTHROPIC_MESSAGES_2023_06_01
        && let Some(cache) = thinking_signature_cache
    {
        restore_cached_thinking_signatures(&mut request, cache);
    }
    if ingress_protocol == OPENAI_RESPONSES_V1 && target_protocol == ANTHROPIC_MESSAGES_2023_06_01 {
        sanitize_responses_reasoning_for_anthropic(&mut request);
    }

    let (mut encoded, headers, path) = nyro_protocol::encode_request(target_protocol, &request)?;
    if ingress_protocol == OPENAI_RESPONSES_V1 && target_protocol == ANTHROPIC_MESSAGES_2023_06_01 {
        apply_anthropic_reasoning_config(&request, &mut encoded);
    }
    Ok((encoded, headers, path))
}

#[cfg(test)]
fn encode_responses_for_target(
    body: Value,
    target_protocol: ProtocolEndpoint,
    model: Option<&RouterModelMetadata>,
    thinking_signature_cache: Option<&ThinkingSignatureCache>,
) -> Result<(Value, HeaderMap, String)> {
    let request = decode_ingress_request(OPENAI_RESPONSES_V1, &body)?;
    encode_request_for_target(
        body,
        request,
        OPENAI_RESPONSES_V1,
        target_protocol,
        model,
        thinking_signature_cache,
    )
}

fn protocol_mode(
    ingress_protocol: ProtocolEndpoint,
    target_protocol: ProtocolEndpoint,
) -> ProtocolMode {
    if ingress_protocol == target_protocol {
        ProtocolMode::Native
    } else {
        ProtocolMode::Transform
    }
}

fn set_body_model(body: &mut Value, model: &str) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
}

fn target_path(target_protocol: ProtocolEndpoint, stream: bool) -> String {
    nyro_protocol::endpoint_path(target_protocol, stream)
}

fn native_headers(target_protocol: ProtocolEndpoint) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if target_protocol == ANTHROPIC_MESSAGES_2023_06_01 {
        headers.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static("2023-06-01"),
        );
    }
    headers
}

fn decode_ingress_request(protocol: ProtocolEndpoint, body: &Value) -> Result<AiRequest> {
    let decode_body = if protocol == OPENAI_RESPONSES_V1 {
        normalize_responses_body_for_nyro_decode(body)
    } else {
        body.clone()
    };
    nyro_protocol::decode_request(protocol, decode_body)
}

fn normalize_responses_body_for_nyro_decode(body: &Value) -> Value {
    let mut normalized = body.clone();
    let Some(items) = normalized
        .get_mut("input")
        .and_then(|input| input.as_array_mut())
    else {
        return normalized;
    };

    for item in items {
        if let Some(text) = item.as_str().map(str::to_string) {
            *item = json!({
                "type": "message",
                "role": "user",
                "content": text,
            });
            continue;
        }

        let Some(obj) = item.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(|value| value.as_str()).is_none() {
            obj.insert("type".to_string(), json!("message"));
        }
        if obj.get("type").and_then(|value| value.as_str()) != Some("message") {
            continue;
        }
        let Some(content) = obj.get_mut("content") else {
            continue;
        };
        if let Some(content_obj) = content.as_object() {
            let text = content_obj
                .get("text")
                .and_then(|value| value.as_str())
                .or_else(|| content_obj.get("content").and_then(|value| value.as_str()));
            if let Some(text) = text {
                *content = Value::String(text.to_string());
            }
        }
    }

    normalized
}

async fn post_target(
    state: &RouterState,
    path: &str,
    body: &Value,
    headers: &HeaderMap,
) -> Result<reqwest::Response> {
    let url = endpoint_url(&state.target_base_url, path);
    let mut request = state.client.post(url).json(body);
    match &state.auth {
        RouterAuth::Bearer(api_key) => {
            request = request.bearer_auth(api_key);
        }
        RouterAuth::GithubCopilot { refresh_token } => {
            let session = state
                .copilot_token_cache
                .resolve_session(refresh_token, Some(&state.target_base_url), None)
                .await
                .context("GitHub Copilot auth resolution failed")?;
            for (name, value) in
                github_copilot::conversation_headers(&session.access_token, Some(body))
            {
                request = request.header(name, value);
            }
        }
    }
    for (name, value) in headers {
        request = request.header(name, value);
    }
    if state.target_protocol == ANTHROPIC_MESSAGES_2023_06_01 {
        request = request.header("anthropic-version", "2023-06-01");
    }
    request.send().await.context("target request failed")
}

fn parse_target_response(
    body: Value,
    target_protocol: ProtocolEndpoint,
    original_model: &str,
) -> Result<AiResponse> {
    let body = if target_protocol == OPENAI_CHAT_COMPLETIONS_V1 {
        normalize_openai_chat_response_for_nyro_parse(body)
    } else {
        body
    };
    let mut response = nyro_protocol::parse_response(target_protocol, body)?;
    if response.model.is_empty() {
        response.model = original_model.to_string();
    }
    Ok(response)
}

fn normalize_openai_chat_response_for_nyro_parse(mut body: Value) -> Value {
    let Some(choices) = body
        .get_mut("choices")
        .and_then(|value| value.as_array_mut())
    else {
        return body;
    };
    for choice in choices {
        let Some(content) = choice
            .get_mut("message")
            .and_then(|message| message.get_mut("content"))
        else {
            continue;
        };
        if !content.is_string() {
            *content = Value::String(openai_chat_content_text(content));
        }
    }
    body
}

fn openai_chat_content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                if let Some(text) = part.as_str() {
                    return Some(text.to_string());
                }
                part.get("text")
                    .and_then(|value| value.as_str())
                    .or_else(|| part.get("content").and_then(|value| value.as_str()))
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(obj) => obj
            .get("text")
            .and_then(|value| value.as_str())
            .or_else(|| obj.get("content").and_then(|value| value.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

fn mirror_responses_assistant_text_as_reasoning(request: &mut AiRequest) {
    if !request.reasoning.enabled {
        return;
    }
    for msg in &mut request.messages {
        if msg.role != Role::Assistant || msg.tool_calls.is_some() {
            continue;
        }
        let text = msg.content.to_text();
        if text.trim().is_empty() {
            continue;
        }
        let extra = msg.meta.get_or_insert_with(|| json!({}));
        let Some(extra_obj) = extra.as_object_mut() else {
            continue;
        };
        extra_obj
            .entry("reasoning_content".to_string())
            .or_insert_with(|| Value::String(text));
    }
}

fn sanitize_responses_reasoning_for_anthropic(request: &mut AiRequest) {
    let attached_reasoning = request
        .messages
        .iter()
        .filter(|msg| msg.role == Role::Assistant && msg.tool_calls.is_some())
        .filter_map(message_reasoning_pair)
        .collect::<Vec<_>>();

    request.messages.retain(|msg| {
        if msg.role != Role::Assistant
            || msg.tool_calls.is_some()
            || !msg.content.to_text().is_empty()
        {
            return true;
        }
        let Some(pair) = message_reasoning_pair(msg) else {
            return true;
        };
        !attached_reasoning.iter().any(|seen| seen == &pair)
    });

    for msg in &mut request.messages {
        if msg.role != Role::Assistant {
            continue;
        }
        let Some(extra_obj) = msg.meta.as_mut().and_then(|meta| meta.as_object_mut()) else {
            continue;
        };
        let has_signature = extra_obj
            .get("reasoning_signature")
            .and_then(|value| value.as_str())
            .is_some_and(|signature| !signature.trim().is_empty());
        if !has_signature {
            extra_obj.remove("reasoning_content");
        }
    }
}

fn message_reasoning_pair(msg: &crate::protocol::ir::request::Message) -> Option<(String, String)> {
    let obj = msg.meta.as_ref()?.as_object()?;
    let reasoning = obj
        .get("reasoning_content")
        .and_then(|value| value.as_str())?
        .to_string();
    let signature = obj
        .get("reasoning_signature")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    Some((reasoning, signature))
}

fn apply_anthropic_reasoning_config(request: &AiRequest, body: &mut Value) {
    let Some(budget_tokens) = anthropic_budget_tokens(&request.reasoning.effort) else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    let max_tokens = obj
        .get("max_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(4096);
    if max_tokens <= budget_tokens as u64 {
        obj.insert(
            "max_tokens".to_string(),
            Value::Number(serde_json::Number::from(budget_tokens.saturating_add(1024))),
        );
    }
    obj.insert(
        "thinking".to_string(),
        json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }),
    );
    if let Some(effort) = anthropic_effort(&request.reasoning.effort) {
        obj.insert("output_config".to_string(), json!({ "effort": effort }));
    }
}

fn anthropic_budget_tokens(effort: &Option<ReasoningEffort>) -> Option<u32> {
    match effort.as_ref()? {
        ReasoningEffort::None => None,
        ReasoningEffort::Minimal | ReasoningEffort::Low => Some(1024),
        ReasoningEffort::Medium => Some(4096),
        ReasoningEffort::High => Some(16384),
        ReasoningEffort::Xhigh => Some(32000),
        ReasoningEffort::Budget(tokens) => Some(*tokens),
    }
}

fn anthropic_effort(effort: &Option<ReasoningEffort>) -> Option<&'static str> {
    match effort.as_ref()? {
        ReasoningEffort::None | ReasoningEffort::Minimal | ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium => Some("medium"),
        ReasoningEffort::High => Some("high"),
        ReasoningEffort::Xhigh | ReasoningEffort::Budget(_) => Some("max"),
    }
}

fn restore_cached_thinking_signatures(request: &mut AiRequest, cache: &ThinkingSignatureCache) {
    for msg in &mut request.messages {
        if msg.role != Role::Assistant {
            continue;
        }
        let extra = msg.meta.get_or_insert_with(|| json!({}));
        let Some(extra_obj) = extra.as_object_mut() else {
            continue;
        };
        if extra_obj
            .get("reasoning_signature")
            .and_then(|value| value.as_str())
            .is_some_and(|signature| !signature.trim().is_empty())
        {
            continue;
        }
        let Some(reasoning) = extra_obj
            .get("reasoning_content")
            .and_then(|value| value.as_str())
            .filter(|reasoning| !reasoning.trim().is_empty())
            .map(str::to_string)
        else {
            continue;
        };
        if let Some(signature) = cache.lookup(&reasoning) {
            extra_obj.insert("reasoning_signature".to_string(), json!(signature));
        }
    }
}

fn cache_response_thinking_signature(
    cache: &ThinkingSignatureCache,
    target_protocol: ProtocolEndpoint,
    response: &AiResponse,
) {
    if target_protocol != ANTHROPIC_MESSAGES_2023_06_01 {
        return;
    }
    if let (Some(reasoning), Some(signature)) = (
        response.reasoning_content.as_deref(),
        response.reasoning_signature.as_deref(),
    ) {
        cache.insert(reasoning, signature);
    }
}

fn capture_thinking_signature_from_deltas(
    target_protocol: ProtocolEndpoint,
    deltas: &[AiStreamDelta],
    reasoning: &mut String,
    signature: &mut String,
) {
    if target_protocol != ANTHROPIC_MESSAGES_2023_06_01 {
        return;
    }
    for delta in deltas {
        match delta {
            AiStreamDelta::ThinkingDelta(text) => reasoning.push_str(text),
            AiStreamDelta::ThinkingSignature(text) => signature.push_str(text),
            _ => {}
        }
    }
}

async fn stream_target_to_responses(
    mut upstream: reqwest::Response,
    ingress_protocol: ProtocolEndpoint,
    target_protocol: ProtocolEndpoint,
    debug: &DebugDump,
    thinking_signature_cache: &ThinkingSignatureCache,
    stream: &mut TcpStream,
) -> Result<()> {
    let upstream_status_line = upstream_status_line(&upstream);
    let upstream_headers = format_response_headers(upstream.headers());
    let mut parser: Box<dyn StreamResponseDecoder> =
        nyro_protocol::stream_response_decoder(target_protocol);
    let mut formatter = nyro_protocol::stream_response_encoder(ingress_protocol);
    let mut debug_sse = String::new();
    let mut upstream_sse = String::new();
    let mut captured_reasoning = String::new();
    let mut captured_signature = String::new();

    while let Some(chunk) = upstream.chunk().await? {
        let text = String::from_utf8_lossy(&chunk);
        upstream_sse.push_str(&text);
        let deltas = parser.parse_chunk(&text)?;
        capture_thinking_signature_from_deltas(
            target_protocol,
            &deltas,
            &mut captured_reasoning,
            &mut captured_signature,
        );
        let events = formatter.format_deltas(&deltas);
        write_sse_events(stream, &events, &mut debug_sse).await?;
    }

    let deltas = parser.finish()?;
    capture_thinking_signature_from_deltas(
        target_protocol,
        &deltas,
        &mut captured_reasoning,
        &mut captured_signature,
    );
    let events = formatter.format_deltas(&deltas);
    write_sse_events(stream, &events, &mut debug_sse).await?;

    let done_events = formatter.format_done();
    write_sse_events(stream, &done_events, &mut debug_sse).await?;

    thinking_signature_cache.insert(&captured_reasoning, &captured_signature);

    debug
        .text(
            "03-response-before-convert-upstream.http",
            &format_http_message(
                &format!("{upstream_status_line}\r\n{upstream_headers}"),
                upstream_sse.as_bytes(),
            ),
        )
        .await;
    debug
        .text("03-response-before-convert-upstream.sse", &upstream_sse)
        .await;
    debug
        .text("04-response-after-convert.sse", &debug_sse)
        .await;
    stream.write_all(b"0\r\n\r\n").await?;
    Ok(())
}

async fn stream_upstream_passthrough(
    mut upstream: reqwest::Response,
    debug: &DebugDump,
    stream: &mut TcpStream,
) -> Result<()> {
    let upstream_status_line = upstream_status_line(&upstream);
    let upstream_headers = format_response_headers(upstream.headers());
    let mut upstream_body = Vec::new();

    while let Some(chunk) = upstream.chunk().await? {
        upstream_body.extend_from_slice(&chunk);
        write_chunk(stream, &chunk).await?;
    }

    debug
        .text(
            "03-response-before-convert-upstream.http",
            &format_http_message(
                &format!("{upstream_status_line}\r\n{upstream_headers}"),
                &upstream_body,
            ),
        )
        .await;
    debug
        .text(
            "04-response-after-convert.passthrough",
            &String::from_utf8_lossy(&upstream_body),
        )
        .await;
    stream.write_all(b"0\r\n\r\n").await?;
    Ok(())
}

async fn write_sse_events(
    stream: &mut TcpStream,
    events: &[crate::protocol::SseEvent],
    debug_sse: &mut String,
) -> Result<()> {
    for event in events {
        let sse = event.to_sse_string();
        debug_sse.push_str(&sse);
        write_chunk(stream, sse.as_bytes()).await?;
    }
    Ok(())
}

fn codex_model_info(model: &RouterModelMetadata) -> Value {
    let context_window = model.context_window.unwrap_or(272_000);
    let default_reasoning_level = if model.supports_reasoning {
        json!("medium")
    } else {
        Value::Null
    };
    let supported_reasoning_levels = if model.supports_reasoning {
        json!([
            {"effort": "low", "description": "low"},
            {"effort": "medium", "description": "medium"},
            {"effort": "high", "description": "high"},
            {"effort": "xhigh", "description": "xhigh"}
        ])
    } else {
        json!([])
    };
    let input_modalities = if model.supports_vision {
        json!(["text", "image"])
    } else {
        json!(["text"])
    };
    let web_search_tool_type = if model.supports_search && model.supports_vision {
        "text_and_image"
    } else {
        "text"
    };
    json!({
        "slug": model.id,
        "display_name": model.name,
        "description": null,
        "default_reasoning_level": default_reasoning_level,
        "supported_reasoning_levels": supported_reasoning_levels,
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "availability_nux": null,
        "upgrade": null,
        "base_instructions": CODEX_BASE_INSTRUCTIONS,
        "supports_reasoning_summaries": model.supports_reasoning,
        "default_reasoning_summary": "none",
        "support_verbosity": false,
        "default_verbosity": null,
        "apply_patch_tool_type": null,
        "web_search_tool_type": web_search_tool_type,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": true,
        "supports_image_detail_original": model.supports_vision,
        "context_window": context_window,
        "max_context_window": context_window,
        "auto_compact_token_limit": null,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": input_modalities,
        "supports_search_tool": model.supports_search
    })
}

pub fn codex_models_response(model: Option<&RouterModelMetadata>) -> Value {
    json!({
        "models": model.map(|model| vec![codex_model_info(model)]).unwrap_or_default()
    })
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

fn events_to_sse(events: Vec<crate::protocol::SseEvent>) -> String {
    events
        .into_iter()
        .map(|event| event.to_sse_string())
        .collect::<String>()
}

struct HttpRequest {
    path: String,
    head: String,
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

    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
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
    Ok(HttpRequest {
        path,
        head: headers,
        body,
    })
}

fn format_upstream_request_http(
    state: &RouterState,
    path: &str,
    body: &Value,
    headers: &HeaderMap,
) -> String {
    let url = endpoint_url(&state.target_base_url, path);
    let body = serde_json::to_vec_pretty(body).unwrap_or_else(|_| b"{}".to_vec());
    let mut head = format!("POST {url} HTTP/1.1\r\n");
    head.push_str("authorization: <redacted>\r\n");
    head.push_str("content-type: application/json\r\n");
    for (name, value) in headers {
        if is_sensitive_header(name.as_str()) {
            head.push_str(&format!("{}: <redacted>\r\n", name.as_str()));
        } else if let Ok(value) = value.to_str() {
            head.push_str(&format!("{}: {value}\r\n", name.as_str()));
        }
    }
    head.push_str(&format!("content-length: {}\r\n", body.len()));
    format_http_message(&head, &body)
}

fn upstream_status_line(response: &reqwest::Response) -> String {
    let status = response.status();
    format!(
        "HTTP/1.1 {} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or("Upstream")
    )
}

fn format_response_headers(headers: &HeaderMap) -> String {
    let mut out = String::new();
    for (name, value) in headers {
        if is_sensitive_header(name.as_str()) {
            out.push_str(&format!("{}: <redacted>\r\n", name.as_str()));
        } else if let Ok(value) = value.to_str() {
            out.push_str(&format!("{}: {value}\r\n", name.as_str()));
        }
    }
    out
}

fn format_http_message(head: &str, body: &[u8]) -> String {
    let mut out = head.trim_end_matches("\r\n").to_string();
    out.push_str("\r\n\r\n");
    out.push_str(&String::from_utf8_lossy(body));
    out
}

fn redact_http_head(head: &str) -> String {
    head.lines()
        .map(|line| {
            let Some((name, _)) = line.split_once(':') else {
                return line.to_string();
            };
            if is_sensitive_header(name) {
                format!("{}: <redacted>", name.trim())
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\r\n")
}

fn is_sensitive_header(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "x-api-key" | "api-key" | "anthropic-api-key" | "cookie" | "set-cookie"
    )
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

fn http_chunked_head(status: u16, content_type: &str) -> Vec<u8> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Upstream",
    };
    format!(
        "HTTP/1.1 {status} {status_text}\r\ncontent-type: {content_type}\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n"
    )
    .into_bytes()
}

async fn write_chunk(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    stream
        .write_all(format!("{:x}\r\n", bytes.len()).as_bytes())
        .await?;
    stream.write_all(bytes).await?;
    stream.write_all(b"\r\n").await?;
    Ok(())
}

fn response_content_type(response: &reqwest::Response) -> String {
    response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn router_target_wire(wire: WireProtocol) -> Result<ProtocolEndpoint> {
    match wire {
        WireProtocol::OpenaiResponses => Ok(OPENAI_RESPONSES_V1),
        WireProtocol::OpenaiCompletions => Ok(OPENAI_CHAT_COMPLETIONS_V1),
        WireProtocol::AnthropicMessages => Ok(ANTHROPIC_MESSAGES_2023_06_01),
    }
}

fn debug_name(ingress_protocol: ProtocolEndpoint) -> &'static str {
    if ingress_protocol == OPENAI_RESPONSES_V1 {
        "responses"
    } else if ingress_protocol == ANTHROPIC_MESSAGES_2023_06_01 {
        "messages"
    } else {
        "protocol"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RouterModelMetadata {
        RouterModelMetadata {
            id: "actual-model".to_string(),
            name: "Actual Model".to_string(),
            context_window: None,
            supports_reasoning: true,
            supports_vision: false,
            supports_search: false,
        }
    }

    #[test]
    fn codex_model_catalog_advertises_vision_and_search() {
        let model = RouterModelMetadata {
            id: "gpt-5.5".to_string(),
            name: "GPT-5.5".to_string(),
            context_window: Some(1_050_000),
            supports_reasoning: true,
            supports_vision: true,
            supports_search: true,
        };

        let response = codex_models_response(Some(&model));
        let info = &response["models"][0];

        assert_eq!(info["input_modalities"], json!(["text", "image"]));
        assert_eq!(info["supports_image_detail_original"], json!(true));
        assert_eq!(info["supports_search_tool"], json!(true));
        assert_eq!(info["web_search_tool_type"], json!("text_and_image"));
        assert_eq!(info["context_window"], json!(1_050_000));
    }

    #[test]
    fn responses_to_anthropic_uses_nyro_reasoning_effort_mapping() {
        let (request, _, path) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": "hello",
                "stream": true,
                "max_output_tokens": 1024,
                "reasoning": {"effort": "high"}
            }),
            ANTHROPIC_MESSAGES_2023_06_01,
            Some(&model()),
            None,
        )
        .unwrap();

        assert_eq!(path, "/v1/messages");
        assert_eq!(request["model"], "actual-model");
        assert_eq!(request["stream"], true);
        assert_eq!(request["thinking"]["type"], "enabled");
        assert_eq!(request["thinking"]["budget_tokens"], 16384);
        assert_eq!(request["max_tokens"], 17408);
        assert_eq!(request["output_config"]["effort"], "high");
    }

    #[test]
    fn responses_to_openai_chat_uses_nyro_and_preserves_stream() {
        let (request, _, path) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "prior"}]},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]}
                ],
                "stream": true,
                "reasoning": {"effort": "high"}
            }),
            OPENAI_CHAT_COMPLETIONS_V1,
            Some(&model()),
            None,
        )
        .unwrap();

        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(request["model"], "actual-model");
        assert_eq!(request["stream"], true);
        assert_eq!(request["stream_options"]["include_usage"], true);
        assert_eq!(request["reasoning"]["effort"], "high");
        assert_eq!(request["messages"][0]["reasoning_content"], "prior");
    }

    #[test]
    fn native_responses_passthrough_preserves_body_and_overrides_model() {
        let body = json!({
            "model": "codex-model",
            "input": "hello",
            "stream": true,
            "metadata": {"keep": true}
        });
        let decoded = nyro_protocol::decode_request(OPENAI_RESPONSES_V1, body.clone()).unwrap();
        let (request, headers, path) = encode_request_for_target(
            body,
            decoded,
            OPENAI_RESPONSES_V1,
            OPENAI_RESPONSES_V1,
            Some(&model()),
            None,
        )
        .unwrap();

        assert_eq!(path, "/v1/responses");
        assert!(headers.is_empty());
        assert_eq!(request["model"], "actual-model");
        assert_eq!(request["input"], "hello");
        assert_eq!(request["metadata"], json!({"keep": true}));
        assert_eq!(request["stream"], true);
    }

    #[test]
    fn native_anthropic_passthrough_preserves_body_and_version_header() {
        let body = json!({
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "stream": true
        });
        let decoded =
            nyro_protocol::decode_request(ANTHROPIC_MESSAGES_2023_06_01, body.clone()).unwrap();
        let (request, headers, path) = encode_request_for_target(
            body,
            decoded,
            ANTHROPIC_MESSAGES_2023_06_01,
            ANTHROPIC_MESSAGES_2023_06_01,
            Some(&model()),
            None,
        )
        .unwrap();

        assert_eq!(path, "/v1/messages");
        assert_eq!(request["model"], "actual-model");
        assert_eq!(request["messages"][0]["content"], "hello");
        assert_eq!(request["max_tokens"], 1024);
        assert_eq!(
            headers
                .get("anthropic-version")
                .and_then(|value| value.to_str().ok()),
            Some("2023-06-01")
        );
    }

    #[test]
    fn responses_to_anthropic_drops_unsigned_reasoning_summary() {
        let (request, _, _) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "show me last bump reason"}]},
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "private summary"}]},
                    {"type": "function_call", "call_id": "toolu_1", "name": "exec_command", "arguments": "{\"cmd\":\"git log --oneline | head -20\"}"},
                    {"type": "function_call_output", "call_id": "toolu_1", "output": "ok"}
                ],
                "stream": true,
                "reasoning": {"effort": "medium"}
            }),
            ANTHROPIC_MESSAGES_2023_06_01,
            Some(&model()),
            None,
        )
        .unwrap();

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        let assistant_content = messages[1]["content"].as_array().unwrap();
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert!(
            assistant_content
                .iter()
                .all(|part| part.get("type").and_then(|value| value.as_str()) != Some("thinking")),
            "Responses reasoning summaries are not valid Anthropic signed thinking blocks"
        );
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn responses_to_anthropic_restores_cached_thinking_signature() {
        let cache = ThinkingSignatureCache::new(THINKING_SIGNATURE_CACHE_TTL);
        cache.insert("private summary", "SIG_42");

        let (request, _, _) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "show me last bump reason"}]},
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "private summary"}]},
                    {"type": "function_call", "call_id": "toolu_1", "name": "exec_command", "arguments": "{\"cmd\":\"git log --oneline | head -20\"}"},
                    {"type": "function_call_output", "call_id": "toolu_1", "output": "ok"}
                ],
                "stream": true,
                "reasoning": {"effort": "medium"}
            }),
            ANTHROPIC_MESSAGES_2023_06_01,
            Some(&model()),
            Some(&cache),
        )
        .unwrap();

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        let assistant_content = messages[1]["content"].as_array().unwrap();
        assert_eq!(assistant_content[0]["type"], "thinking");
        assert_eq!(assistant_content[0]["thinking"], "private summary");
        assert_eq!(assistant_content[0]["signature"], "SIG_42");
        assert_eq!(assistant_content[1]["type"], "tool_use");
    }

    #[test]
    fn nyro_handles_legacy_responses_input_shapes() {
        let (request, _, _) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": [
                    "plain string item",
                    {"type": "message", "role": "user", "content": {"text": "object content"}}
                ]
            }),
            OPENAI_CHAT_COMPLETIONS_V1,
            Some(&model()),
            None,
        )
        .unwrap();

        assert_eq!(request["messages"][0]["content"], "plain string item");
        assert_eq!(request["messages"][1]["content"], "object content");
    }

    #[test]
    fn target_response_is_formatted_as_responses_by_nyro() {
        let response = parse_target_response(
            json!({
                "id": "msg_1",
                "model": "upstream-model",
                "content": [
                    {"type": "thinking", "thinking": "think"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
                "stop_reason": "tool_use"
            }),
            ANTHROPIC_MESSAGES_2023_06_01,
            "fallback-model",
        )
        .unwrap();
        let body = nyro_protocol::format_response(OPENAI_RESPONSES_V1, &response);

        assert_eq!(body["model"], "upstream-model");
        assert_eq!(body["output"][0]["type"], "reasoning");
        assert_eq!(body["output"][1]["type"], "function_call");
        assert_eq!(body["output"][2]["content"][0]["text"], "answer");
        assert_eq!(body["usage"]["input_tokens"], 3);
    }

    #[test]
    fn nyro_handles_non_string_openai_chat_content() {
        let response = parse_target_response(
            json!({
                "id": "chatcmpl_1",
                "model": "openai-compatible",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
                    },
                    "finish_reason": "stop"
                }]
            }),
            OPENAI_CHAT_COMPLETIONS_V1,
            "fallback-model",
        )
        .unwrap();
        let body = nyro_protocol::format_response(OPENAI_RESPONSES_V1, &response);

        assert_eq!(body["output"][0]["content"][0]["text"], "hello\n world");
    }

    #[test]
    fn buffered_stream_fallback_uses_responses_stream_formatter() {
        let mut response = AiResponse::new("resp_1", "model");
        response.content = "hello".to_string();
        response.reasoning_content = Some("think".to_string());
        response.stop_reason = Some("stop".to_string());
        let sse = events_to_sse(nyro_protocol::format_response_stream(
            OPENAI_RESPONSES_V1,
            &response,
        ));

        assert!(sse.contains("response.reasoning_summary_text.delta"));
        assert!(sse.contains("response.output_text.delta"));
        assert!(sse.contains("data: [DONE]"));
    }
}
