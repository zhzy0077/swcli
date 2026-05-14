use super::debug::DebugDump;
use anyhow::{Context, Result, anyhow};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};
use std::io::ErrorKind;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use crate::protocol::codec::anthropic_messages::{
    encoder::AnthropicEncoder,
    stream::{AnthropicResponseParser, AnthropicStreamParser},
};
use crate::protocol::codec::openai_compatible::{
    encoder::OpenAIEncoder,
    stream::{OpenAIResponseParser, OpenAIStreamParser},
};
use crate::protocol::codec::openai_responses::{
    decoder::ResponsesDecoder, formatter::ResponsesResponseFormatter,
    stream::ResponsesStreamFormatter,
};
use crate::protocol::types::InternalResponse;
use crate::protocol::{
    EgressEncoder, IngressDecoder, ResponseFormatter, ResponseParser, StreamFormatter, StreamParser,
};

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
    match response {
        Ok(RouterResponse::Buffered(bytes)) => stream.write_all(&bytes).await?,
        Ok(RouterResponse::ResponsesStream {
            upstream,
            target_wire,
            original_model,
            debug,
        }) => {
            stream
                .write_all(&http_chunked_head(200, "text/event-stream"))
                .await?;
            if let Err(err) = stream_target_to_responses(
                upstream,
                target_wire,
                &original_model,
                &debug,
                &mut stream,
            )
            .await
            {
                if !is_broken_pipe(&err) {
                    return Err(err);
                }
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
    ResponsesStream {
        upstream: reqwest::Response,
        target_wire: RouterTargetWire,
        original_model: String,
        debug: DebugDump,
    },
}

async fn route_request(request: &HttpRequest, state: &RouterState) -> Result<RouterResponse> {
    match request.path.as_str() {
        "/responses" | "/v1/responses" => handle_responses(request, state).await,
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

async fn handle_responses(request: &HttpRequest, state: &RouterState) -> Result<RouterResponse> {
    let body: Value = serde_json::from_slice(&request.body).context("Invalid Responses JSON")?;
    let debug = DebugDump::start("responses").await;
    debug.json("01-request-before-convert", &body).await;

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

    let (upstream_request, upstream_headers, upstream_path) =
        encode_responses_for_target(body, state.target_wire, state.model.as_ref())?;
    debug
        .json("02-request-after-convert-upstream", &upstream_request)
        .await;

    let upstream = post_target(state, &upstream_path, &upstream_request, &upstream_headers).await?;
    if !upstream.status().is_success() {
        let status = upstream.status().as_u16();
        let text = upstream.text().await.unwrap_or_default();
        debug
            .text("03-response-before-convert-error.txt", &text)
            .await;
        return Ok(RouterResponse::Buffered(http_text(
            status,
            "application/json",
            text,
        )));
    }

    if stream && response_content_type(&upstream).contains("text/event-stream") {
        return Ok(RouterResponse::ResponsesStream {
            upstream,
            target_wire: state.target_wire,
            original_model,
            debug,
        });
    }

    let upstream_response: Value = upstream.json().await?;
    debug
        .json("03-response-before-convert-upstream", &upstream_response)
        .await;
    let response = parse_target_response(upstream_response, state.target_wire, &original_model)?;
    let responses_body = ResponsesResponseFormatter.format_response(&response);

    if stream {
        let sse = events_to_sse(ResponsesStreamFormatter::format_response(&response));
        debug.text("04-response-after-convert.sse", &sse).await;
        Ok(RouterResponse::Buffered(http_text(
            200,
            "text/event-stream",
            sse,
        )))
    } else {
        debug
            .json("04-response-after-convert", &responses_body)
            .await;
        Ok(RouterResponse::Buffered(http_json(200, &responses_body)))
    }
}

fn encode_responses_for_target(
    body: Value,
    target_wire: RouterTargetWire,
    model: Option<&RouterModelMetadata>,
) -> Result<(Value, HeaderMap, String)> {
    let mut request = ResponsesDecoder.decode_request(body)?;
    if let Some(model) = model {
        request.model = model.id.clone();
    }

    match target_wire {
        RouterTargetWire::AnthropicMessages => {
            let encoder = AnthropicEncoder;
            let path = encoder.egress_path(&request.model, request.stream);
            let (body, headers) = encoder.encode_request(&request)?;
            Ok((body, headers, path))
        }
        RouterTargetWire::OpenaiCompletions => {
            let encoder = OpenAIEncoder;
            let path = encoder.egress_path(&request.model, request.stream);
            let (body, headers) = encoder.encode_request(&request)?;
            Ok((body, headers, path))
        }
    }
}

async fn post_target(
    state: &RouterState,
    path: &str,
    body: &Value,
    headers: &HeaderMap,
) -> Result<reqwest::Response> {
    let url = endpoint_url(&state.target_base_url, path);
    let mut request = state
        .client
        .post(url)
        .bearer_auth(&state.api_key)
        .json(body);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    request.send().await.context("target request failed")
}

fn parse_target_response(
    body: Value,
    target_wire: RouterTargetWire,
    original_model: &str,
) -> Result<InternalResponse> {
    let mut response = match target_wire {
        RouterTargetWire::AnthropicMessages => AnthropicResponseParser.parse_response(body)?,
        RouterTargetWire::OpenaiCompletions => OpenAIResponseParser.parse_response(body)?,
    };
    if response.model.is_empty() {
        response.model = original_model.to_string();
    }
    Ok(response)
}

async fn stream_target_to_responses(
    mut upstream: reqwest::Response,
    target_wire: RouterTargetWire,
    original_model: &str,
    debug: &DebugDump,
    stream: &mut TcpStream,
) -> Result<()> {
    let mut parser: Box<dyn StreamParser> = match target_wire {
        RouterTargetWire::AnthropicMessages => Box::new(AnthropicStreamParser::new()),
        RouterTargetWire::OpenaiCompletions => Box::new(OpenAIStreamParser::new()),
    };
    let mut formatter = ResponsesStreamFormatter::new();
    let mut debug_sse = String::new();

    while let Some(chunk) = upstream.chunk().await? {
        let text = String::from_utf8_lossy(&chunk);
        let deltas = parser.parse_chunk(&text)?;
        let events = formatter.format_deltas(&deltas);
        write_sse_events(stream, &events, &mut debug_sse).await?;
    }

    let deltas = parser.finish()?;
    let events = formatter.format_deltas(&deltas);
    write_sse_events(stream, &events, &mut debug_sse).await?;

    let done_events = formatter.format_done();
    write_sse_events(stream, &done_events, &mut debug_sse).await?;

    debug
        .text("04-response-after-convert.sse", &debug_sse)
        .await;
    let _ = original_model;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RouterModelMetadata {
        RouterModelMetadata {
            id: "actual-model".to_string(),
            name: "Actual Model".to_string(),
            context_window: None,
            supports_reasoning: true,
        }
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
            RouterTargetWire::AnthropicMessages,
            Some(&model()),
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
            RouterTargetWire::OpenaiCompletions,
            Some(&model()),
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
    fn nyro_handles_legacy_responses_input_shapes() {
        let (request, _, _) = encode_responses_for_target(
            json!({
                "model": "codex-model",
                "input": [
                    "plain string item",
                    {"type": "message", "role": "user", "content": {"text": "object content"}}
                ]
            }),
            RouterTargetWire::OpenaiCompletions,
            Some(&model()),
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
            RouterTargetWire::AnthropicMessages,
            "fallback-model",
        )
        .unwrap();
        let body = ResponsesResponseFormatter.format_response(&response);

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
            RouterTargetWire::OpenaiCompletions,
            "fallback-model",
        )
        .unwrap();
        let body = ResponsesResponseFormatter.format_response(&response);

        assert_eq!(body["output"][0]["content"][0]["text"], "hello\n world");
    }

    #[test]
    fn buffered_stream_fallback_uses_responses_stream_formatter() {
        let response = InternalResponse {
            id: "resp_1".to_string(),
            model: "model".to_string(),
            content: "hello".to_string(),
            reasoning_content: Some("think".to_string()),
            reasoning_signature: None,
            tool_calls: vec![],
            response_items: None,
            stop_reason: Some("stop".to_string()),
            usage: Default::default(),
        };
        let sse = events_to_sse(ResponsesStreamFormatter::format_response(&response));

        assert!(sse.contains("response.reasoning_summary_text.delta"));
        assert!(sse.contains("response.output_text.delta"));
        assert!(sse.contains("data: [DONE]"));
    }
}
