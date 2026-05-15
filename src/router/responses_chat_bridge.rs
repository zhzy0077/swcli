// SPDX-License-Identifier: Apache-2.0
// Adapted from Nyro: https://github.com/nyroway/nyro
// Local modifications for swcli.

use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Adapted from Nyro's Responses↔Chat bridge (<https://github.com/nyroway/nyro>),
/// licensed under the Apache License 2.0; see `THIRD_PARTY_NOTICES.md`.
/// Local changes keep Responses `reasoning` item replay and Codex snapshot
/// compatibility.
pub fn convert_responses_to_openai_chat_request(
    body: &Value,
    actual_model: Option<&str>,
    requires_reasoning_content: bool,
) -> Value {
    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(|value| value.as_str())
        && !instructions.is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }

    if let Some(items) = body.get("input").and_then(|value| value.as_array()) {
        let mut pending_reasoning = Vec::new();
        let mut index = 0;
        while index < items.len() {
            let item = &items[index];
            match item.get("type").and_then(|value| value.as_str()) {
                Some("reasoning") => {
                    pending_reasoning.extend(responses_reasoning_item_texts(item));
                }
                Some("message") => {
                    let role = item
                        .get("role")
                        .and_then(|value| value.as_str())
                        .filter(|role| matches!(*role, "system" | "user" | "assistant" | "tool"))
                        .unwrap_or("user");
                    let mut message = json!({
                        "role": role,
                        "content": responses_content_to_openai_chat_content(item.get("content"))
                    });
                    if role == "assistant" {
                        pending_reasoning.extend(responses_item_reasoning_content(item));
                        attach_pending_reasoning_content(
                            &mut message,
                            &mut pending_reasoning,
                            requires_reasoning_content,
                        );
                    }
                    messages.push(message);
                }
                Some("function_call") => {
                    let mut tool_calls = Vec::new();
                    while index < items.len()
                        && items[index].get("type").and_then(|value| value.as_str())
                            == Some("function_call")
                    {
                        let item = &items[index];
                        pending_reasoning.extend(responses_item_reasoning_content(item));
                        tool_calls.push(openai_chat_tool_call_from_responses_item(item));
                        index += 1;
                    }
                    let mut message = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": tool_calls
                    });
                    attach_pending_reasoning_content(
                        &mut message,
                        &mut pending_reasoning,
                        requires_reasoning_content,
                    );
                    messages.push(message);
                    continue;
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
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output
                    }));
                }
                None => {
                    if let Some(text) = item.as_str() {
                        messages.push(json!({"role": "user", "content": text}));
                    }
                }
                _ => {}
            }
            index += 1;
        }
        if !pending_reasoning.is_empty() {
            messages.push(json!({
                "role": "assistant",
                "content": null,
                "reasoning_content": pending_reasoning.join("\n")
            }));
        }
    }

    let mut req = json!({
        "model": actual_model
            .or_else(|| body.get("model").and_then(|value| value.as_str()))
            .unwrap_or("model"),
        "messages": messages,
        "stream": false
    });
    if let Some(tools) = responses_tools_to_openai_chat_tools(body.get("tools")) {
        req["tools"] = Value::Array(tools);
    }
    if let Some(tokens) = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .cloned()
    {
        req["max_tokens"] = tokens;
    }
    for field in ["temperature", "top_p"] {
        if let Some(value) = body.get(field) {
            req[field] = value.clone();
        }
    }
    if let Some(effort) = body
        .get("reasoning")
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(|value| value.as_str())
        .map(openai_chat_reasoning_effort)
    {
        req["reasoning_effort"] = json!(effort);
    }
    req
}

fn openai_chat_tool_call_from_responses_item(item: &Value) -> Value {
    let call_id = item
        .get("call_id")
        .and_then(|value| value.as_str())
        .or_else(|| item.get("id").and_then(|value| value.as_str()))
        .unwrap_or("call_0");
    let name = item
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let arguments = item
        .get("arguments")
        .and_then(|value| value.as_str())
        .unwrap_or("{}");
    json!({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    })
}

pub fn convert_openai_chat_response_to_responses_sse(
    chat: &Value,
    original_model: &str,
    requires_reasoning_content: bool,
) -> String {
    let response_id = gen_id("resp");
    let created_at = unix_ts();
    let mut sse = String::new();
    let mut output_items = Vec::new();

    sse.push_str(&sse_event(
        "response.created",
        &json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "model": original_model,
                "created_at": created_at,
                "status": "in_progress",
                "output": []
            }
        }),
    ));

    let (content, tool_calls, reasoning_content) = extract_chat_response_payload(chat);
    let reasoning_for_tool = if !reasoning_content.is_empty() {
        reasoning_content.clone()
    } else if requires_reasoning_content {
        if !content.is_empty() {
            content.clone()
        } else {
            " ".to_string()
        }
    } else {
        String::new()
    };

    if !tool_calls.is_empty() {
        for (index, tool_call) in tool_calls.iter().enumerate() {
            let call_id = tool_call
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or("call_0");
            let item_id = gen_id("fc");
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

            sse.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": index,
                    "item": {
                        "id": item_id,
                        "call_id": call_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": name,
                        "arguments": ""
                    }
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

            let mut done_item = json!({
                "id": item_id,
                "call_id": call_id,
                "type": "function_call",
                "status": "completed",
                "name": name,
                "arguments": arguments
            });
            if !reasoning_for_tool.is_empty() {
                done_item["reasoning_content"] = json!(reasoning_for_tool.clone());
            }
            sse.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": index,
                    "item": done_item
                }),
            ));
            output_items.push(done_item);
        }
    } else {
        let item_id = gen_id("msg");
        sse.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        if !content.is_empty() {
            sse.push_str(&sse_event(
                "response.output_text.delta",
                &json!({
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content
                }),
            ));
        }
        sse.push_str(&sse_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": content
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": content}
            }),
        ));

        let mut done_item = json!({
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": []
            }]
        });
        if !reasoning_content.is_empty() {
            done_item["reasoning_content"] = json!(reasoning_content);
        }
        sse.push_str(&sse_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": done_item
            }),
        ));
        output_items.push(done_item);
    }

    let mut response = json!({
        "id": response_id,
        "object": "response",
        "model": original_model,
        "created_at": created_at,
        "status": "completed",
        "output": output_items
    });
    if let Some(usage) = openai_usage_to_responses_usage(chat.get("usage")) {
        response["usage"] = usage;
    }
    sse.push_str(&sse_event(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": response
        }),
    ));
    sse
}

fn attach_pending_reasoning_content(
    message: &mut Value,
    pending_reasoning: &mut Vec<String>,
    requires_reasoning_content: bool,
) {
    if !pending_reasoning.is_empty() {
        message["reasoning_content"] = json!(pending_reasoning.join("\n"));
        pending_reasoning.clear();
        return;
    }
    if !requires_reasoning_content {
        return;
    }
    message["reasoning_content"] = json!(assistant_reasoning_fallback(message));
}

fn assistant_reasoning_fallback(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) if !text.is_empty() => text.clone(),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .or_else(|| part.get("content"))
                        .and_then(|value| value.as_str())
                })
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                " ".to_string()
            } else {
                text
            }
        }
        Some(Value::Object(object)) => object
            .get("text")
            .or_else(|| object.get("content"))
            .and_then(|value| value.as_str())
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| " ".to_string()),
        _ => " ".to_string(),
    }
}

fn responses_reasoning_item_texts(item: &Value) -> Vec<String> {
    responses_item_reasoning_content(item)
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

fn responses_content_to_openai_chat_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => Value::String(text.clone()),
        Some(Value::Array(parts)) => {
            let converted = parts
                .iter()
                .filter_map(responses_content_part_to_openai_chat_part)
                .collect::<Vec<_>>();
            if converted
                .iter()
                .all(|part| part.get("type").and_then(|value| value.as_str()) == Some("text"))
            {
                Value::String(
                    converted
                        .iter()
                        .filter_map(|part| part.get("text").and_then(|value| value.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            } else {
                Value::Array(converted)
            }
        }
        Some(Value::Object(object)) => Value::String(
            object
                .get("text")
                .and_then(|value| value.as_str())
                .or_else(|| object.get("content").and_then(|value| value.as_str()))
                .unwrap_or("")
                .to_string(),
        ),
        _ => Value::String(String::new()),
    }
}

fn responses_content_part_to_openai_chat_part(part: &Value) -> Option<Value> {
    if let Some(text) = part.as_str() {
        return Some(json!({"type": "text", "text": text}));
    }
    match part.get("type").and_then(|value| value.as_str()) {
        Some("input_text") | Some("output_text") | Some("text") | None => part
            .get("text")
            .and_then(|value| value.as_str())
            .or_else(|| part.get("content").and_then(|value| value.as_str()))
            .map(|text| json!({"type": "text", "text": text})),
        Some("input_image") => part
            .get("image_url")
            .and_then(|value| match value {
                Value::String(url) => Some(json!({"url": url})),
                Value::Object(_) => Some(value.clone()),
                _ => None,
            })
            .map(|image_url| json!({"type": "image_url", "image_url": image_url})),
        _ => None,
    }
}

fn responses_tools_to_openai_chat_tools(tools: Option<&Value>) -> Option<Vec<Value>> {
    let tools = tools?.as_array()?;
    let converted = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(|value| value.as_str()) == Some("function"))
        .map(|tool| {
            if tool.get("function").is_some() {
                tool.clone()
            } else {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or_else(|| json!("")),
                        "description": tool.get("description").cloned().unwrap_or_else(|| json!("")),
                        "parameters": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}))
                    }
                })
            }
        })
        .collect::<Vec<_>>();
    (!converted.is_empty()).then_some(converted)
}

fn openai_chat_reasoning_effort(value: &str) -> &'static str {
    match value.to_ascii_lowercase().as_str() {
        "none" => "none",
        "minimal" => "minimal",
        "low" => "low",
        "medium" => "medium",
        "high" | "xhigh" | "max" => "high",
        _ => "medium",
    }
}

fn openai_usage_to_responses_usage(usage: Option<&Value>) -> Option<Value> {
    let usage = usage?;
    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let total_tokens = usage.get("total_tokens").cloned().unwrap_or_else(|| {
        let input = input_tokens.as_u64().unwrap_or(0);
        let output = output_tokens.as_u64().unwrap_or(0);
        json!(input + output)
    });
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    Some(json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens
    }))
}

fn extract_chat_response_payload(chat: &Value) -> (String, Vec<Value>, String) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_parts = Vec::new();

    if let Some(choices) = chat.get("choices").and_then(|choices| choices.as_array()) {
        for choice in choices {
            let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
            let text = extract_openai_message_text(message.get("content"));
            if !text.is_empty() {
                text_parts.push(text);
            }
            if let Some(reasoning) = message
                .get("reasoning_content")
                .and_then(|reasoning| reasoning.as_str())
                .filter(|reasoning| !reasoning.is_empty())
            {
                reasoning_parts.push(reasoning.to_string());
            }
            if let Some(message_tool_calls) =
                message.get("tool_calls").and_then(|calls| calls.as_array())
            {
                tool_calls.extend(message_tool_calls.iter().cloned());
            }
        }
    }

    if text_parts.is_empty() && tool_calls.is_empty() {
        if let Some(output_items) = chat
            .get("output")
            .or_else(|| {
                chat.get("response")
                    .and_then(|response| response.get("output"))
            })
            .and_then(|value| value.as_array())
        {
            for item in output_items {
                match item.get("type").and_then(|value| value.as_str()) {
                    Some("message") => {
                        let text = extract_openai_message_text(item.get("content"));
                        if !text.is_empty() {
                            text_parts.push(text);
                        }
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
                        let arguments = item
                            .get("arguments")
                            .and_then(|value| value.as_str())
                            .unwrap_or("{}");
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments}
                        }));
                    }
                    Some("output_text") => {
                        if let Some(text) = item
                            .get("text")
                            .and_then(|value| value.as_str())
                            .filter(|text| !text.is_empty())
                        {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if text_parts.is_empty() {
        if let Some(text) = chat
            .get("result")
            .and_then(|result| result.get("response"))
            .and_then(|value| value.as_str())
        {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("response").and_then(|value| value.as_str()) {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("output_text").and_then(|value| value.as_str()) {
            text_parts.push(text.to_string());
        }
    }

    (
        text_parts.join("\n"),
        tool_calls,
        reasoning_parts.join("\n"),
    )
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

fn sse_event(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
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
