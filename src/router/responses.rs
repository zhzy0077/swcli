use super::responses_chat_bridge;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

const CODEX_BASE_INSTRUCTIONS: &str = "You are Codex, a coding agent. You collaborate with the user on software engineering tasks. Use the provided tools when needed, keep changes scoped, and communicate clearly.";

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
}

#[derive(Clone)]
struct RouterState {
    target_base_url: String,
    api_key: String,
    model: Option<RouterModelMetadata>,
    target_wire: RouterTargetWire,
    client: reqwest::Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterTargetWire {
    OpenaiCompletions,
    AnthropicMessages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanonicalEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Max,
}

impl CanonicalEffort {
    fn from_openai_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "max" => Some(Self::Max),
            _ => None,
        }
    }

    fn to_anthropic_effort(self) -> &'static str {
        match self {
            Self::None | Self::Minimal | Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    fn to_anthropic_budget_tokens(self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::Minimal | Self::Low => Some(1024),
            Self::Medium => Some(4096),
            Self::High => Some(16384),
            Self::Max => Some(32000),
        }
    }
}

pub async fn start_anthropic_responses_router(
    target_base_url: String,
    api_key: String,
    model: Option<RouterModelMetadata>,
) -> Result<RouterHandle> {
    start_responses_router(
        target_base_url,
        api_key,
        model,
        RouterTargetWire::AnthropicMessages,
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
        api_key,
        model,
        RouterTargetWire::OpenaiCompletions,
    )
    .await
}

async fn start_responses_router(
    target_base_url: String,
    api_key: String,
    model: Option<RouterModelMetadata>,
    target_wire: RouterTargetWire,
) -> Result<RouterHandle> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let state = Arc::new(RouterState {
        target_base_url,
        api_key,
        model,
        target_wire,
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
    let bytes = match response {
        Ok(response) => response,
        Err(err) => http_json(
            500,
            &json!({
                "error": {
                    "message": err.to_string(),
                    "type": "swcli_router_error"
                }
            }),
        ),
    };
    stream.write_all(&bytes).await?;
    Ok(())
}

async fn route_request(request: &HttpRequest, state: &RouterState) -> Result<Vec<u8>> {
    match request.path.as_str() {
        "/responses" | "/v1/responses" => handle_responses(request, state).await,
        "/models" | "/v1/models" => {
            Ok(http_json(200, &codex_models_response(state.model.as_ref())))
        }
        _ => Ok(http_json(
            404,
            &json!({"error": {"message": format!("Unsupported router path {}", request.path)}}),
        )),
    }
}

async fn handle_responses(request: &HttpRequest, state: &RouterState) -> Result<Vec<u8>> {
    let body: Value = serde_json::from_slice(&request.body).context("Invalid Responses JSON")?;
    let stream = body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let original_model = body
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| {
            state
                .model
                .as_ref()
                .map(|model| model.id.as_str())
                .unwrap_or("model")
        })
        .to_string();
    let upstream_request = match state.target_wire {
        RouterTargetWire::AnthropicMessages => responses_to_anthropic_request(
            &body,
            state.model.as_ref().map(|model| model.id.as_str()),
        ),
        RouterTargetWire::OpenaiCompletions => responses_to_openai_chat_request(
            &body,
            state.model.as_ref().map(|model| model.id.as_str()),
            target_requires_reasoning_content(&state.target_base_url),
        ),
    };
    let upstream = match state.target_wire {
        RouterTargetWire::AnthropicMessages => {
            post_anthropic_messages(state, &upstream_request).await?
        }
        RouterTargetWire::OpenaiCompletions => post_openai_chat(state, &upstream_request).await?,
    };
    if !upstream.status().is_success() {
        let status = upstream.status().as_u16();
        let text = upstream.text().await.unwrap_or_default();
        return Ok(http_text(status, "application/json", text));
    }
    let upstream_response: Value = upstream.json().await?;
    if stream {
        let sse = match state.target_wire {
            RouterTargetWire::AnthropicMessages => {
                let response = anthropic_to_responses_response(&upstream_response, &original_model);
                responses_response_to_sse(&response)
            }
            RouterTargetWire::OpenaiCompletions => openai_chat_to_responses_sse(
                &upstream_response,
                &original_model,
                target_requires_reasoning_content(&state.target_base_url),
            ),
        };
        Ok(http_text(200, "text/event-stream", sse))
    } else {
        let response = match state.target_wire {
            RouterTargetWire::AnthropicMessages => {
                anthropic_to_responses_response(&upstream_response, &original_model)
            }
            RouterTargetWire::OpenaiCompletions => {
                openai_chat_to_responses_response(&upstream_response, &original_model)
            }
        };
        Ok(http_json(200, &response))
    }
}

async fn post_anthropic_messages(state: &RouterState, body: &Value) -> Result<reqwest::Response> {
    let url = endpoint_url(&state.target_base_url, "/v1/messages");
    state
        .client
        .post(url)
        .bearer_auth(&state.api_key)
        .header("anthropic-version", "2023-06-01")
        .json(body)
        .send()
        .await
        .context("Anthropic messages request failed")
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
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": true,
        "supports_image_detail_original": false,
        "context_window": context_window,
        "max_context_window": context_window,
        "auto_compact_token_limit": null,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": false
    })
}

pub fn codex_models_response(model: Option<&RouterModelMetadata>) -> Value {
    json!({
        "models": model.map(|model| vec![codex_model_info(model)]).unwrap_or_default()
    })
}

async fn post_openai_chat(state: &RouterState, body: &Value) -> Result<reqwest::Response> {
    let url = endpoint_url(&state.target_base_url, "/v1/chat/completions");
    state
        .client
        .post(url)
        .bearer_auth(&state.api_key)
        .json(body)
        .send()
        .await
        .context("OpenAI chat completions request failed")
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

fn responses_to_openai_chat_request(
    body: &Value,
    actual_model: Option<&str>,
    requires_reasoning_content: bool,
) -> Value {
    responses_chat_bridge::convert_responses_to_openai_chat_request(
        body,
        actual_model,
        requires_reasoning_content,
    )
}

fn extract_responses_reasoning_effort(body: &Value) -> Option<CanonicalEffort> {
    body.get("reasoning")
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(|value| value.as_str())
        .and_then(CanonicalEffort::from_openai_str)
}

fn responses_to_anthropic_request(body: &Value, actual_model: Option<&str>) -> Value {
    let mut system = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(|value| value.as_str())
        && !instructions.is_empty()
    {
        system.push(json!({"type": "text", "text": instructions}));
    }

    let messages = responses_input_to_anthropic_messages(body.get("input"));

    let mut req = json!({
        "model": actual_model
            .or_else(|| body.get("model").and_then(|value| value.as_str()))
            .unwrap_or("model"),
        "messages": messages,
        "max_tokens": response_max_tokens(body),
        "stream": false
    });
    if !system.is_empty() {
        req["system"] = Value::Array(system);
    }
    if let Some(tools) = responses_tools_to_anthropic_tools(body.get("tools")) {
        req["tools"] = Value::Array(tools);
    }
    for field in ["temperature", "top_p", "stop_sequences"] {
        if let Some(value) = body.get(field) {
            req[field] = value.clone();
        }
    }
    if let Some(effort) = extract_responses_reasoning_effort(body) {
        if let Some(budget_tokens) = effort.to_anthropic_budget_tokens() {
            req["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget_tokens
            });
            ensure_anthropic_max_tokens_exceeds_thinking_budget(&mut req, budget_tokens);
        }
        req["output_config"] = json!({
            "effort": effort.to_anthropic_effort()
        });
    }
    req
}

fn ensure_anthropic_max_tokens_exceeds_thinking_budget(req: &mut Value, budget_tokens: u64) {
    let max_tokens = req.get("max_tokens").and_then(json_u64).unwrap_or(0);
    if max_tokens <= budget_tokens {
        req["max_tokens"] = json!(budget_tokens.saturating_add(1024));
    }
}

fn response_max_tokens(body: &Value) -> u64 {
    body.get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(json_u64)
        .filter(|tokens| *tokens > 0)
        .unwrap_or(4096)
}

fn responses_input_to_anthropic_messages(input: Option<&Value>) -> Vec<Value> {
    let Some(items) = input.and_then(|value| value.as_array()) else {
        return Vec::new();
    };
    let mut messages = Vec::new();
    let mut pending_thinking = Vec::new();
    let mut index = 0;
    while index < items.len() {
        let item = &items[index];
        if item.get("type").and_then(|value| value.as_str()) == Some("reasoning") {
            pending_thinking.extend(responses_reasoning_item_to_anthropic_blocks(item));
            index += 1;
            continue;
        }
        if item.get("type").and_then(|value| value.as_str()) == Some("function_call") {
            let mut content = Vec::new();
            while index < items.len()
                && items[index].get("type").and_then(|value| value.as_str())
                    == Some("function_call")
            {
                content.push(responses_function_call_to_anthropic_tool_use(&items[index]));
                index += 1;
            }
            let mut content = Value::Array(content);
            prepend_pending_thinking(&mut content, &mut pending_thinking);
            messages.push(json!({
                "role": "assistant",
                "content": content
            }));
            continue;
        }
        if item.get("type").and_then(|value| value.as_str()) == Some("function_call_output") {
            let mut content = Vec::new();
            while index < items.len()
                && items[index].get("type").and_then(|value| value.as_str())
                    == Some("function_call_output")
            {
                content.push(responses_function_call_output_to_anthropic_tool_result(
                    &items[index],
                ));
                index += 1;
            }
            messages.push(json!({
                "role": "user",
                "content": content
            }));
            continue;
        }
        if let Some(message) =
            responses_input_item_to_anthropic_message(item, &mut pending_thinking)
        {
            messages.push(message);
        }
        index += 1;
    }
    if !pending_thinking.is_empty() {
        messages.push(json!({
            "role": "assistant",
            "content": pending_thinking
        }));
    }
    messages
}

fn responses_function_call_to_anthropic_tool_use(item: &Value) -> Value {
    let call_id = item
        .get("call_id")
        .and_then(|value| value.as_str())
        .or_else(|| item.get("id").and_then(|value| value.as_str()))
        .unwrap_or("call_0");
    let name = item
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let input = item
        .get("arguments")
        .and_then(|value| value.as_str())
        .and_then(|args| serde_json::from_str::<Value>(args).ok())
        .unwrap_or_else(|| json!({}));
    json!({
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": input
    })
}

fn responses_function_call_output_to_anthropic_tool_result(item: &Value) -> Value {
    let call_id = item
        .get("call_id")
        .and_then(|value| value.as_str())
        .unwrap_or("call_0");
    let output = item
        .get("output")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    json!({
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": output
    })
}

fn responses_input_item_to_anthropic_message(
    item: &Value,
    pending_thinking: &mut Vec<Value>,
) -> Option<Value> {
    match item.get("type").and_then(|value| value.as_str()) {
        Some("message") => {
            let role = item
                .get("role")
                .and_then(|value| value.as_str())
                .filter(|role| matches!(*role, "user" | "assistant"))
                .unwrap_or("user");
            let mut content = responses_content_to_anthropic_content(item.get("content"));
            if role == "assistant" {
                prepend_pending_thinking(&mut content, pending_thinking);
            }
            Some(json!({
                "role": role,
                "content": content
            }))
        }
        Some("function_call") => {
            let call_id = item
                .get("call_id")
                .and_then(|value| value.as_str())
                .or_else(|| item.get("id").and_then(|value| value.as_str()))
                .unwrap_or("call_0");
            let name = item
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let input = item
                .get("arguments")
                .and_then(|value| value.as_str())
                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                .unwrap_or_else(|| json!({}));
            let mut content = json!([{
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input
            }]);
            prepend_pending_thinking(&mut content, pending_thinking);
            Some(json!({
                "role": "assistant",
                "content": content
            }))
        }
        Some("function_call_output") => {
            let call_id = item
                .get("call_id")
                .and_then(|value| value.as_str())
                .unwrap_or("call_0");
            let output = item
                .get("output")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            Some(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output
                }]
            }))
        }
        None => item.as_str().map(|text| {
            json!({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
        }),
        _ => None,
    }
}

fn prepend_pending_thinking(content: &mut Value, pending_thinking: &mut Vec<Value>) {
    if pending_thinking.is_empty() {
        return;
    }
    let mut merged = std::mem::take(pending_thinking);
    if let Some(blocks) = content.as_array_mut() {
        merged.append(blocks);
        *blocks = merged;
    }
}

fn responses_reasoning_item_to_anthropic_blocks(item: &Value) -> Vec<Value> {
    let mut blocks = responses_item_reasoning_content(item)
        .into_iter()
        .map(|text| json!({"type": "thinking", "thinking": text}))
        .collect::<Vec<_>>();
    if blocks.is_empty() {
        blocks = item
            .get("summary")
            .and_then(|value| value.as_array())
            .map(|summary| {
                summary
                    .iter()
                    .filter_map(|part| {
                        part.get("text")
                            .and_then(|value| value.as_str())
                            .filter(|text| !text.is_empty())
                            .map(|text| json!({"type": "thinking", "thinking": text}))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
    }
    blocks
}

fn responses_item_reasoning_content(item: &Value) -> Vec<String> {
    if let Some(reasoning_content) = item
        .get("reasoning_content")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
    {
        return vec![reasoning_content.to_string()];
    }
    if let Some(encrypted_content) = item
        .get("encrypted_content")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
    {
        return vec![encrypted_content.to_string()];
    }
    item.get("content")
        .and_then(|value| value.as_array())
        .map(|content| {
            content
                .iter()
                .filter_map(|part| {
                    part.get("reasoning")
                        .and_then(|value| value.as_str())
                        .or_else(|| {
                            (part.get("type").and_then(|value| value.as_str())
                                == Some("reasoning_text"))
                            .then(|| part.get("text").and_then(|value| value.as_str()))
                            .flatten()
                        })
                        .filter(|text| !text.is_empty())
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn target_requires_reasoning_content(_base_url: &str) -> bool {
    true
}

fn responses_content_to_anthropic_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => json!([{"type": "text", "text": text}]),
        Some(Value::Array(parts)) => {
            let blocks = parts
                .iter()
                .filter_map(responses_content_part_to_anthropic_block)
                .collect::<Vec<_>>();
            Value::Array(blocks)
        }
        Some(Value::Object(object)) => json!([{
            "type": "text",
            "text": object
                .get("text")
                .and_then(|value| value.as_str())
                .or_else(|| object.get("content").and_then(|value| value.as_str()))
                .unwrap_or("")
        }]),
        _ => json!([{"type": "text", "text": ""}]),
    }
}

fn responses_content_part_to_anthropic_block(part: &Value) -> Option<Value> {
    if let Some(text) = part.as_str() {
        return Some(json!({"type": "text", "text": text}));
    }
    match part.get("type").and_then(|value| value.as_str()) {
        Some("input_text") | Some("output_text") | Some("text") | None => part
            .get("text")
            .and_then(|value| value.as_str())
            .or_else(|| part.get("content").and_then(|value| value.as_str()))
            .map(|text| json!({"type": "text", "text": text})),
        Some("reasoning") => part
            .get("reasoning")
            .and_then(|value| value.as_str())
            .filter(|text| !text.is_empty())
            .map(|text| json!({"type": "thinking", "thinking": text})),
        _ => None,
    }
}

fn responses_tools_to_anthropic_tools(tools: Option<&Value>) -> Option<Vec<Value>> {
    let tools = tools?.as_array()?;
    let converted = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(|value| value.as_str()) == Some("function"))
        .filter_map(|tool| {
            let name = tool.get("name").and_then(|value| value.as_str())?;
            Some(json!({
                "name": name,
                "description": tool
                    .get("description")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
                "input_schema": tool
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
            }))
        })
        .collect::<Vec<_>>();
    (!converted.is_empty()).then_some(converted)
}

fn anthropic_to_responses_response(body: &Value, model: &str) -> Value {
    let response_id = gen_id("resp");
    let output = anthropic_content_to_responses_output(body.get("content"));
    let mut response = json!({
        "id": response_id,
        "object": "response",
        "created_at": unix_ts(),
        "model": model,
        "status": "completed",
        "output": output
    });
    if let Some(usage) = anthropic_usage_to_responses_usage(body.get("usage")) {
        response["usage"] = usage;
    }
    response
}

fn openai_chat_to_responses_response(body: &Value, model: &str) -> Value {
    let response_id = gen_id("resp");
    let output = openai_chat_choices_to_responses_output(body.get("choices"));
    let mut response = json!({
        "id": response_id,
        "object": "response",
        "created_at": unix_ts(),
        "model": model,
        "status": "completed",
        "output": output
    });
    if let Some(usage) = openai_usage_to_responses_usage(body.get("usage")) {
        response["usage"] = usage;
    }
    response
}

fn openai_chat_choices_to_responses_output(choices: Option<&Value>) -> Vec<Value> {
    let Some(choices) = choices.and_then(|value| value.as_array()) else {
        return vec![message_output_item("")];
    };
    let mut output = Vec::new();
    for choice in choices {
        let Some(message) = choice.get("message") else {
            continue;
        };
        let reasoning = extract_openai_message_reasoning_content(message);
        let text = extract_openai_message_text(message.get("content"));
        if !text.is_empty() {
            output.push(message_output_item_with_reasoning(
                &text,
                reasoning.as_deref(),
            ));
        }
        if let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
            for tool_call in tool_calls {
                let call_id = tool_call
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("call_0");
                let name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let arguments = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("{}");
                let mut item = json!({
                    "id": gen_id("fc"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments
                });
                if let Some(reasoning) = reasoning
                    .as_deref()
                    .filter(|reasoning| !reasoning.is_empty())
                {
                    item["reasoning_content"] = json!(reasoning);
                }
                output.push(item);
            }
        }
    }
    if output.is_empty() {
        output.push(message_output_item(""));
    }
    output
}

fn extract_openai_message_reasoning_content(message: &Value) -> Option<String> {
    message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn extract_openai_message_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|value| value.as_str())
                    .or_else(|| part.get("content").and_then(|value| value.as_str()))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Object(object)) => object
            .get("text")
            .and_then(|value| value.as_str())
            .or_else(|| object.get("content").and_then(|value| value.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

fn anthropic_content_to_responses_output(content: Option<&Value>) -> Vec<Value> {
    let Some(blocks) = content.and_then(|value| value.as_array()) else {
        return vec![message_output_item("")];
    };

    let mut text = String::new();
    let mut reasoning_parts = Vec::new();
    let mut output = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|value| value.as_str()) {
            Some("thinking") | Some("reasoning") => {
                if let Some(thinking) = block
                    .get("thinking")
                    .or_else(|| block.get("reasoning"))
                    .or_else(|| block.get("text"))
                    .and_then(|value| value.as_str())
                    .filter(|thinking| !thinking.is_empty())
                {
                    reasoning_parts.push(thinking.to_string());
                }
            }
            Some("text") => {
                if let Some(part) = block.get("text").and_then(|value| value.as_str()) {
                    text.push_str(part);
                }
            }
            Some("tool_use") => {
                if !text.is_empty() {
                    output.push(message_output_item_with_reasoning(
                        &text,
                        joined_reasoning(&reasoning_parts).as_deref(),
                    ));
                    text.clear();
                    reasoning_parts.clear();
                }
                let call_id = block
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("call_0");
                let name = block
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let arguments = block
                    .get("input")
                    .map(|value| serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()))
                    .unwrap_or_else(|| "{}".to_string());
                let mut item = json!({
                    "id": gen_id("fc"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments
                });
                attach_reasoning_content_to_item(&mut item, &reasoning_parts);
                reasoning_parts.clear();
                output.push(item);
            }
            _ => {}
        }
    }
    if !text.is_empty() || output.is_empty() {
        output.push(message_output_item_with_reasoning(
            &text,
            joined_reasoning(&reasoning_parts).as_deref(),
        ));
    }
    output
}

fn joined_reasoning(reasoning_parts: &[String]) -> Option<String> {
    (!reasoning_parts.is_empty()).then(|| reasoning_parts.join("\n"))
}

fn attach_reasoning_content_to_item(item: &mut Value, reasoning_parts: &[String]) {
    if let Some(reasoning) = joined_reasoning(reasoning_parts) {
        item["reasoning_content"] = json!(reasoning);
    }
}

fn message_output_item(text: &str) -> Value {
    message_output_item_with_reasoning(text, None)
}

fn message_output_item_with_reasoning(text: &str, reasoning: Option<&str>) -> Value {
    let mut content = vec![json!({
        "type": "output_text",
        "text": text,
        "annotations": []
    })];
    let mut item = json!({
        "id": gen_id("msg"),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": content
    });
    if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.is_empty()) {
        content.push(json!({
            "type": "reasoning",
            "reasoning": reasoning
        }));
        item["reasoning_content"] = json!(reasoning);
    }
    item["content"] = json!(content);
    item
}

fn openai_usage_to_responses_usage(usage: Option<&Value>) -> Option<Value> {
    let usage = usage?;
    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(json_u64)
        .unwrap_or_default();
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(json_u64)
        .unwrap_or_default();
    let total_tokens = usage
        .get("total_tokens")
        .and_then(json_u64)
        .unwrap_or(input_tokens + output_tokens);
    Some(json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage
                .get("prompt_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
                .and_then(json_u64)
                .unwrap_or_default()
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": usage
                .get("completion_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(json_u64)
                .unwrap_or_default()
        },
        "total_tokens": total_tokens
    }))
}

fn openai_chat_to_responses_sse(
    chat: &Value,
    original_model: &str,
    requires_reasoning_content: bool,
) -> String {
    responses_chat_bridge::convert_openai_chat_response_to_responses_sse(
        chat,
        original_model,
        requires_reasoning_content,
    )
}

fn anthropic_usage_to_responses_usage(usage: Option<&Value>) -> Option<Value> {
    let usage = usage?;
    let input_tokens = usage
        .get("input_tokens")
        .and_then(json_u64)
        .unwrap_or_default();
    let output_tokens = usage
        .get("output_tokens")
        .and_then(json_u64)
        .unwrap_or_default();
    Some(json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage
                .get("cache_read_input_tokens")
                .and_then(json_u64)
                .unwrap_or_default()
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": 0
        },
        "total_tokens": input_tokens + output_tokens
    }))
}

fn responses_response_to_sse(response: &Value) -> String {
    let response_id = response
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or("resp_swcli");
    let mut sse = String::new();
    sse.push_str(&sse_event(
        "response.created",
        &json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "model": response.get("model").cloned().unwrap_or_else(|| json!("model")),
                "created_at": response.get("created_at").cloned().unwrap_or_else(|| json!(unix_ts())),
                "status": "in_progress",
                "output": []
            }
        }),
    ));

    if let Some(output) = response.get("output").and_then(|value| value.as_array()) {
        for (index, item) in output.iter().enumerate() {
            push_output_item_sse(&mut sse, response_id, index, item);
        }
    }

    sse.push_str(&sse_event(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": response
        }),
    ));
    sse.push_str("data: [DONE]\n\n");
    sse
}

fn push_output_item_sse(sse: &mut String, response_id: &str, index: usize, item: &Value) {
    match item.get("type").and_then(|value| value.as_str()) {
        Some("function_call") => {
            let item_id = item
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("fc_swcli");
            let arguments = item
                .get("arguments")
                .and_then(|value| value.as_str())
                .unwrap_or("{}");
            let mut added = item.clone();
            added["status"] = json!("in_progress");
            added["arguments"] = json!("");
            if let Some(object) = added.as_object_mut() {
                object.remove("reasoning_content");
            }
            sse.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": index,
                    "item": added
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": response_id,
                    "output_index": index,
                    "item_id": item_id,
                    "delta": arguments
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.done",
                &json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "output_index": index,
                    "item_id": item_id,
                    "arguments": arguments
                }),
            ));
            sse.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": index,
                    "item": item
                }),
            ));
        }
        _ => {
            let item_id = item
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("msg_swcli");
            let text = item
                .get("content")
                .and_then(|value| value.as_array())
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("text"))
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let reasoning = item
                .get("content")
                .and_then(|value| value.as_array())
                .and_then(|parts| {
                    parts.iter().find_map(|part| {
                        match part.get("type").and_then(|value| value.as_str()) {
                            Some("reasoning") => part
                                .get("reasoning")
                                .or_else(|| part.get("text"))
                                .and_then(|value| value.as_str()),
                            _ => None,
                        }
                    })
                })
                .filter(|value| !value.is_empty());
            let mut added = item.clone();
            added["status"] = json!("in_progress");
            added["content"] = json!([]);
            sse.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": index,
                    "item": added
                }),
            ));
            sse.push_str(&sse_event(
                "response.content_part.added",
                &json!({
                    "type": "response.content_part.added",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""}
                }),
            ));
            if !text.is_empty() {
                sse.push_str(&sse_event(
                    "response.output_text.delta",
                    &json!({
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "delta": text
                    }),
                ));
            }
            sse.push_str(&sse_event(
                "response.output_text.done",
                &json!({
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "text": text
                }),
            ));
            sse.push_str(&sse_event(
                "response.content_part.done",
                &json!({
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text}
                }),
            ));
            if let Some(reasoning) = reasoning {
                sse.push_str(&sse_event(
                    "response.content_part.added",
                    &json!({
                        "type": "response.content_part.added",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 1,
                        "part": {"type": "reasoning", "reasoning": ""}
                    }),
                ));
                sse.push_str(&sse_event(
                    "response.content_part.done",
                    &json!({
                        "type": "response.content_part.done",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 1,
                        "part": {"type": "reasoning", "reasoning": reasoning}
                    }),
                ));
            }
            sse.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": index,
                    "item": item
                }),
            ));
        }
    }
}

fn sse_event(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
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

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_f64().map(|number| number as u64))
        .or_else(|| value.as_str()?.parse().ok())
}

fn gen_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{}_{:06}", unix_ts(), n % 1_000_000)
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_responses_text_request_to_anthropic_messages() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "MiniMax-M2.7-highspeed",
                "instructions": "be concise",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}]
                }],
                "max_output_tokens": 123
            }),
            None,
        );

        assert_eq!(request["model"], "MiniMax-M2.7-highspeed");
        assert_eq!(request["max_tokens"], 123);
        assert_eq!(request["system"][0]["text"], "be concise");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn maps_responses_reasoning_effort_to_anthropic_thinking() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "claude-opus-4-7",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
                "reasoning": {"effort": "high"}
            }),
            None,
        );

        assert_eq!(request["thinking"]["type"], "enabled");
        assert_eq!(request["thinking"]["budget_tokens"], 16384);
        assert_eq!(request["max_tokens"], 17408);
        assert_eq!(request["output_config"]["effort"], "high");
    }

    #[test]
    fn maps_responses_xhigh_to_anthropic_max_effort() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "claude-opus-4-7",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
                "reasoning": {"effort": "xhigh"}
            }),
            None,
        );

        assert_eq!(request["thinking"]["budget_tokens"], 32000);
        assert_eq!(request["output_config"]["effort"], "max");
    }

    #[test]
    fn reasoning_none_does_not_enable_anthropic_thinking() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "claude-opus-4-7",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
                "reasoning": {"effort": "none"}
            }),
            None,
        );

        assert!(request.get("thinking").is_none());
        assert_eq!(request["output_config"]["effort"], "low");
    }

    #[test]
    fn preserves_anthropic_max_tokens_when_above_thinking_budget() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "claude-opus-4-7",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
                "max_output_tokens": 20000,
                "reasoning": {"effort": "high"}
            }),
            None,
        );

        assert_eq!(request["thinking"]["budget_tokens"], 16384);
        assert_eq!(request["max_tokens"], 20000);
    }

    #[test]
    fn converts_responses_tool_items_to_anthropic_tool_blocks() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "placeholder",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"pwd\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_abc",
                        "output": "/tmp"
                    }
                ],
                "tools": [{
                    "type": "function",
                    "name": "shell",
                    "description": "run a command",
                    "parameters": {"type": "object"}
                }]
            }),
            Some("MiniMax-M2.7-highspeed"),
        );

        assert_eq!(request["model"], "MiniMax-M2.7-highspeed");
        assert_eq!(request["messages"][0]["content"][0]["type"], "tool_use");
        assert_eq!(request["messages"][0]["content"][0]["input"]["cmd"], "pwd");
        assert_eq!(request["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(request["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn groups_parallel_responses_tool_calls_for_anthropic_messages() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "placeholder",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"git diff\"}"
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"git status --short\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "diff"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "status"
                    }
                ]
            }),
            Some("MiniMax-M2.7-highspeed"),
        );

        assert_eq!(request["messages"].as_array().unwrap().len(), 2);
        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(request["messages"][0]["content"][0]["type"], "tool_use");
        assert_eq!(request["messages"][0]["content"][0]["id"], "call_1");
        assert_eq!(request["messages"][0]["content"][1]["type"], "tool_use");
        assert_eq!(request["messages"][0]["content"][1]["id"], "call_2");
        assert_eq!(request["messages"][1]["role"], "user");
        assert_eq!(request["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(
            request["messages"][1]["content"][0]["tool_use_id"],
            "call_1"
        );
        assert_eq!(request["messages"][1]["content"][1]["type"], "tool_result");
        assert_eq!(
            request["messages"][1]["content"][1]["tool_use_id"],
            "call_2"
        );
    }

    #[test]
    fn groups_parallel_responses_tool_calls_for_openai_chat() {
        let request = responses_to_openai_chat_request(
            &json!({
                "model": "mimo-test",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"git diff\"}"
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"git status --short\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "diff"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "status"
                    }
                ]
            }),
            None,
            false,
        );

        assert_eq!(request["messages"].as_array().unwrap().len(), 3);
        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(request["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(request["messages"][0]["tool_calls"][1]["id"], "call_2");
        assert_eq!(request["messages"][1]["role"], "tool");
        assert_eq!(request["messages"][1]["tool_call_id"], "call_1");
        assert_eq!(request["messages"][2]["role"], "tool");
        assert_eq!(request["messages"][2]["tool_call_id"], "call_2");
    }

    #[test]
    fn replays_responses_reasoning_as_anthropic_thinking_before_assistant_text() {
        let request = responses_to_anthropic_request(
            &json!({
                "model": "MiniMax-M2.7",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "prior thinking"}],
                        "encrypted_content": null
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "prior answer"}]
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "next"}]
                    }
                ]
            }),
            None,
        );

        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(request["messages"][0]["content"][0]["type"], "thinking");
        assert_eq!(
            request["messages"][0]["content"][0]["thinking"],
            "prior thinking"
        );
        assert_eq!(request["messages"][0]["content"][1]["type"], "text");
        assert_eq!(request["messages"][0]["content"][1]["text"], "prior answer");
        assert_eq!(request["messages"][1]["role"], "user");
    }

    #[test]
    fn converts_anthropic_tool_use_to_responses_sse() {
        let response = anthropic_to_responses_response(
            &json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "shell",
                    "input": {"cmd": "pwd"}
                }],
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }),
            "MiniMax-M2.7-highspeed",
        );
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "toolu_1");

        let sse = responses_response_to_sse(&response);
        assert!(sse.contains("response.function_call_arguments.delta"));
        assert!(sse.contains("response.completed"));
    }

    #[test]
    fn omits_reasoning_content_from_in_progress_function_call_sse_item() {
        let response = openai_chat_to_responses_response(
            &json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Need to inspect files.",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}
                        }]
                    }
                }]
            }),
            "mimo-test",
        );

        let sse = responses_response_to_sse(&response);
        let added_start = sse.find("event: response.output_item.added").unwrap();
        let done_start = sse.find("event: response.output_item.done").unwrap();
        let added_slice = &sse[added_start..done_start];
        assert!(!added_slice.contains("reasoning_content"));
        assert!(sse.contains("Need to inspect files."));
    }

    #[test]
    fn emits_reasoning_content_parts_in_message_sse() {
        let response = openai_chat_to_responses_response(
            &json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Final answer.",
                        "reasoning_content": "Private reasoning."
                    }
                }]
            }),
            "mimo-test",
        );

        let sse = responses_response_to_sse(&response);
        assert!(sse.contains("Private reasoning."));
    }

    #[test]
    fn converts_anthropic_thinking_to_responses_reasoning_item() {
        let response = anthropic_to_responses_response(
            &json!({
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect files."},
                    {"type": "text", "text": "Done."}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }),
            "MiniMax-M2.7",
        );

        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][1]["reasoning"],
            "Need to inspect files."
        );
        assert_eq!(response["output"][0]["content"][0]["text"], "Done.");
    }

    #[test]
    fn converts_responses_request_to_openai_chat_and_back() {
        let request = responses_to_openai_chat_request(
            &json!({
                "model": "gpt-test",
                "instructions": "system text",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}]
                }],
                "tools": [{
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object"}
                }],
                "reasoning": {"effort": "xhigh"}
            }),
            Some("actual-model"),
            false,
        );
        assert_eq!(request["model"], "actual-model");
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["content"], "hello");
        assert_eq!(request["tools"][0]["function"]["name"], "shell");
        assert_eq!(request["reasoning_effort"], "high");

        let response = openai_chat_to_responses_response(
            &json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}
                        }]
                    }
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            }),
            "gpt-test",
        );
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "call_1");
        assert_eq!(response["usage"]["total_tokens"], 7);
    }

    #[test]
    fn replays_responses_reasoning_as_openai_chat_reasoning_content() {
        let request = responses_to_openai_chat_request(
            &json!({
                "model": "mimo-test",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "prior thinking"}],
                        "encrypted_content": null
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "prior answer"}]
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "next"}]
                    }
                ]
            }),
            None,
            false,
        );

        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(
            request["messages"][0]["reasoning_content"],
            "prior thinking"
        );
        assert_eq!(request["messages"][0]["content"], "prior answer");
        assert_eq!(request["messages"][1]["role"], "user");
    }

    #[test]
    fn fills_missing_reasoning_content_for_mimo_assistant_messages() {
        let request = responses_to_openai_chat_request(
            &json!({
                "model": "mimo-test",
                "input": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}]
                }]
            }),
            None,
            true,
        );

        assert_eq!(request["messages"][0]["reasoning_content"], "prior answer");
    }

    #[test]
    fn fills_missing_reasoning_content_for_mimo_tool_calls() {
        let request = responses_to_openai_chat_request(
            &json!({
                "model": "mimo-test",
                "input": [{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{}"
                }]
            }),
            None,
            true,
        );

        assert_eq!(request["messages"][0]["reasoning_content"], " ");
    }

    #[test]
    fn converts_openai_chat_reasoning_content_to_responses_message_reasoning_part() {
        let response = openai_chat_to_responses_response(
            &json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Need to inspect files.",
                        "content": "Done."
                    }
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            }),
            "mimo-test",
        );

        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][1]["reasoning"],
            "Need to inspect files."
        );
        assert_eq!(response["output"][0]["content"][0]["text"], "Done.");
    }

    #[test]
    fn copies_openai_chat_reasoning_content_to_function_call_item() {
        let response = openai_chat_to_responses_response(
            &json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Need to call a tool.",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}
                        }]
                    }
                }]
            }),
            "mimo-test",
        );

        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(
            response["output"][0]["reasoning_content"],
            "Need to call a tool."
        );
    }

    #[test]
    fn builds_codex_models_response_metadata() {
        let model = RouterModelMetadata {
            id: "MiniMax-M2.7-highspeed".to_string(),
            name: "MiniMax-M2.7-highspeed".to_string(),
            context_window: Some(204_800),
            supports_reasoning: true,
        };
        let info = codex_model_info(&model);

        assert_eq!(info["slug"], "MiniMax-M2.7-highspeed");
        assert_eq!(info["display_name"], "MiniMax-M2.7-highspeed");
        assert_eq!(info["context_window"], 204_800);
        assert_eq!(info["max_context_window"], 204_800);
        assert_eq!(info["shell_type"], "shell_command");
        assert_eq!(info["truncation_policy"]["mode"], "tokens");
        assert_eq!(info["supports_reasoning_summaries"], true);
        assert_eq!(info["default_reasoning_level"], "medium");
        assert_eq!(info["supported_reasoning_levels"][3]["effort"], "xhigh");
    }

    #[test]
    fn builds_codex_models_response_without_reasoning_for_plain_models() {
        let model = RouterModelMetadata {
            id: "plain-model".to_string(),
            name: "Plain Model".to_string(),
            context_window: None,
            supports_reasoning: false,
        };
        let info = codex_model_info(&model);

        assert_eq!(info["supports_reasoning_summaries"], false);
        assert_eq!(info["default_reasoning_level"], Value::Null);
        assert!(
            info["supported_reasoning_levels"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
