// SPDX-License-Identifier: Apache-2.0
// Adapted from Nyro: https://github.com/nyroway/nyro
// Local modifications for swcli.

use serde_json::Value;
use uuid::Uuid;

use crate::protocol::ResponseFormatter;
use crate::protocol::types::{InternalResponse, ResponseItem, TokenUsage};

pub struct ResponsesResponseFormatter;

impl ResponseFormatter for ResponsesResponseFormatter {
    fn format_response(&self, resp: &InternalResponse) -> Value {
        let resp_id = if resp.id.is_empty() {
            format!("resp_{}", Uuid::new_v4().simple())
        } else {
            resp.id.clone()
        };
        let msg_id = format!("msg_{}", Uuid::new_v4().simple());

        let mut output: Vec<Value> = Vec::new();
        let mut output_text = String::new();

        if let Some(items) = &resp.response_items {
            for item in items {
                match item {
                    ResponseItem::Reasoning { text } => {
                        output.push(serde_json::json!({
                            "type": "reasoning",
                            "id": format!("rs_{}", Uuid::new_v4().simple()),
                            "summary": [{
                                "type": "summary_text",
                                "text": text
                            }]
                        }));
                    }
                    ResponseItem::FunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        output.push(serde_json::json!({
                            "type": "function_call",
                            "id": format!("fc_{}", Uuid::new_v4().simple()),
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                            "status": "completed"
                        }));
                    }
                    ResponseItem::Message { text } => {
                        output_text.push_str(text);
                    }
                }
            }
        } else {
            if let Some(reasoning) = &resp.reasoning_content {
                output.push(serde_json::json!({
                    "type": "reasoning",
                    "id": format!("rs_{}", Uuid::new_v4().simple()),
                    "summary": [{
                        "type": "summary_text",
                        "text": reasoning
                    }]
                }));
            }
            for tc in &resp.tool_calls {
                output.push(serde_json::json!({
                    "type": "function_call",
                    "id": format!("fc_{}", Uuid::new_v4().simple()),
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": "completed"
                }));
            }
            output_text.push_str(&resp.content);
        }

        if !output_text.is_empty() {
            output.push(serde_json::json!({
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": output_text,
                    "annotations": []
                }]
            }));
        }

        serde_json::json!({
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": resp.model,
            "output": output,
            "output_text": output_text,
            "usage": responses_usage_json(&resp.usage)
        })
    }
}

fn responses_usage_json(usage: &TokenUsage) -> Value {
    let mut value = serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    });

    if let Some(obj) = value.as_object_mut() {
        if usage.cache_read_input_tokens.is_some() || usage.cache_creation_input_tokens.is_some() {
            obj.insert(
                "input_tokens_details".to_string(),
                serde_json::json!({
                    "cached_tokens": usage.cache_read_input_tokens.unwrap_or(0),
                    "cache_creation_tokens": usage.cache_creation_input_tokens.unwrap_or(0),
                }),
            );
        }
        if let Some(v) = usage.cache_read_input_tokens {
            obj.insert("cache_read_input_tokens".to_string(), serde_json::json!(v));
        }
        if let Some(v) = usage.cache_creation_input_tokens {
            obj.insert(
                "cache_creation_input_tokens".to_string(),
                serde_json::json!(v),
            );
        }
        if let Some(server_tool_use) = &usage.server_tool_use {
            obj.insert(
                "server_tool_use".to_string(),
                serde_json::json!({
                    "web_search_requests": server_tool_use.web_search_requests,
                    "web_fetch_requests": server_tool_use.web_fetch_requests,
                }),
            );
        }
    }

    value
}
