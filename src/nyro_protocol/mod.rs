// SPDX-License-Identifier: Apache-2.0
// Adapted from Nyro: https://github.com/nyroway/nyro
// Local modifications for swcli.

//! Protocol layer.
//!
//! Portions adapted from Nyro <https://github.com/nyroway/nyro>,
//! licensed under the Apache License 2.0. See `THIRD_PARTY_NOTICES.md`
//! and `LICENSES/Apache-2.0.txt`.
#![allow(dead_code, unused_imports)]
//!
//! # Three-layer identity
//!
//! Canonical form: `{protocol}/{name}/{version}`.
//!
//! - `protocol`: closed `Protocol` enum (`openai-compat` / `openai-resps` / `anthropic-msgs` / `google-genai`).
//! - `name`: wire-format endpoint name (`chat-completions`, `responses`, `messages`, `generate-content`).
//! - `version`: schema version as the vendor labels it (`v1`, `2023-06-01`, `v1beta`).
//!
//! See [`ids`], [`traits`], [`registry`], and [`codec`] for the model.
//!
//! ## Codec layout
//!
//! Each `codec/<protocol>/` directory co-locates the wire codecs **and** the
//! thin `EndpointHandler` registration shell for every endpoint:
//!
//! - `codec/openai_compatible/chat_completions.rs` — `OpenAIChatCompletionsV1`
//! - `codec/openai_compatible/embeddings.rs` — `OpenAIEmbeddingsV1`
//! - `codec/openai_responses/responses.rs` — `OpenAIResponsesV1`
//! - `codec/anthropic_messages/messages.rs` — `AnthropicMessages2023`
//! - `codec/google_generative/generate_content.rs` — `GoogleGenerateContentV1Beta`
//!
//! Shared semantic utilities live in `codec/reasoning.rs` and
//! `codec/tool_correlation.rs`.
//!
//! ## Alias table
//!
//! See [`registry::ProtocolRegistry`] for three-tier resolution of endpoint aliases
//! and [`registry::ProtocolRegistry::parse_protocol`] for Protocol-level resolution.

pub mod codec;

pub mod ids;
pub mod ir;
pub mod registry;
pub mod traits;

use reqwest::header::HeaderMap;

use crate::db::models::{ProtocolEndpointEntry, Provider};
use crate::protocol::ids::{OPENAI_CHAT_COMPLETIONS_V1, ProtocolEndpoint};

// ── Client → Gateway ──

pub trait RequestDecoder {
    fn decode_request(&self, body: serde_json::Value) -> anyhow::Result<ir::AiRequest>;
}

// ── Gateway → Provider ──

pub trait RequestEncoder {
    fn encode_request(&self, req: &ir::AiRequest)
    -> anyhow::Result<(serde_json::Value, HeaderMap)>;

    fn egress_path(&self, model: &str, stream: bool) -> String;
}

// ── Provider response → internal ──

pub trait ResponseDecoder: Send {
    fn parse_response(&self, resp: serde_json::Value) -> anyhow::Result<ir::AiResponse>;
}

// ── Internal → client response ──

pub trait ResponseEncoder: Send {
    fn format_response(&self, resp: &ir::AiResponse) -> serde_json::Value;
}

// ── Streaming: provider → internal deltas ──

pub trait StreamResponseDecoder: Send {
    fn parse_chunk(&mut self, raw: &str) -> anyhow::Result<Vec<ir::AiStreamDelta>>;
    fn finish(&mut self) -> anyhow::Result<Vec<ir::AiStreamDelta>>;
}

// ── Streaming: internal deltas → client SSE ──

pub trait StreamResponseEncoder: Send {
    fn format_deltas(&mut self, deltas: &[ir::AiStreamDelta]) -> Vec<SseEvent>;
    fn format_done(&mut self) -> Vec<SseEvent>;
    fn usage(&self) -> ir::Usage;
}

// ── SSE helper ──

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl SseEvent {
    pub fn new(event: Option<&str>, data: impl Into<String>) -> Self {
        Self {
            event: event.map(|e| e.to_string()),
            data: data.into(),
        }
    }

    pub fn to_sse_string(&self) -> String {
        let mut s = String::new();
        if let Some(ref event) = self.event {
            s.push_str(&format!("event: {event}\n"));
        }
        s.push_str(&format!("data: {}\n\n", self.data));
        s
    }
}

pub fn decode_request(
    endpoint: ProtocolEndpoint,
    body: serde_json::Value,
) -> anyhow::Result<ir::AiRequest> {
    endpoint
        .handler()
        .make_request_decoder()
        .decode_request(body)
}

pub fn encode_request(
    endpoint: ProtocolEndpoint,
    request: &ir::AiRequest,
) -> anyhow::Result<(serde_json::Value, HeaderMap, String)> {
    let encoder = endpoint.handler().make_request_encoder();
    let path = encoder.egress_path(&request.model, request.stream.enabled);
    let (body, headers) = encoder.encode_request(request)?;
    Ok((body, headers, path))
}

pub fn endpoint_path(endpoint: ProtocolEndpoint, stream: bool) -> String {
    endpoint
        .handler()
        .make_request_encoder()
        .egress_path("", stream)
}

pub fn parse_response(
    endpoint: ProtocolEndpoint,
    body: serde_json::Value,
) -> anyhow::Result<ir::AiResponse> {
    endpoint
        .handler()
        .make_response_decoder()
        .parse_response(body)
}

pub fn format_response(endpoint: ProtocolEndpoint, response: &ir::AiResponse) -> serde_json::Value {
    endpoint
        .handler()
        .make_response_encoder()
        .format_response(response)
}

pub fn stream_response_decoder(endpoint: ProtocolEndpoint) -> Box<dyn StreamResponseDecoder> {
    endpoint.handler().make_stream_response_decoder()
}

pub fn stream_response_encoder(endpoint: ProtocolEndpoint) -> Box<dyn StreamResponseEncoder> {
    endpoint.handler().make_stream_response_encoder()
}

pub fn format_response_stream(
    endpoint: ProtocolEndpoint,
    response: &ir::AiResponse,
) -> Vec<SseEvent> {
    let mut formatter = endpoint.handler().make_stream_response_encoder();
    let deltas = response_to_deltas(response);
    let mut events = formatter.format_deltas(&deltas);
    events.extend(formatter.format_done());
    events
}

fn response_to_deltas(response: &ir::AiResponse) -> Vec<ir::AiStreamDelta> {
    let id = if response.id.is_empty() {
        "resp_buffered".to_string()
    } else {
        response.id.clone()
    };
    let model = if response.model.is_empty() {
        "model".to_string()
    } else {
        response.model.clone()
    };
    let mut deltas = vec![ir::AiStreamDelta::MessageStart { id, model }];

    if let Some(reasoning) = response
        .reasoning_content
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        deltas.push(ir::AiStreamDelta::ThinkingDelta(reasoning.to_string()));
    }
    if !response.content.is_empty() {
        deltas.push(ir::AiStreamDelta::TextDelta(response.content.clone()));
    }
    for (index, tool_call) in response.tool_calls.iter().enumerate() {
        deltas.push(ir::AiStreamDelta::ToolCallStart {
            index,
            id: tool_call.id.clone(),
            name: tool_call.name.clone(),
        });
        if !tool_call.arguments.is_empty() {
            deltas.push(ir::AiStreamDelta::ToolCallDelta {
                index,
                arguments: tool_call.arguments.clone(),
            });
        }
    }
    deltas.push(ir::AiStreamDelta::Usage(response.usage.clone()));
    deltas.push(ir::AiStreamDelta::Done {
        stop_reason: response.stop_reason.clone().unwrap_or_else(|| {
            if response.tool_calls.is_empty() {
                "stop".to_string()
            } else {
                "tool_calls".to_string()
            }
        }),
    });
    deltas
}

// ── Provider multi-protocol negotiation ──

/// Declared protocol capabilities of a single provider, built from the DB row.
///
/// **`endpoints` is a `Vec` (ordered, not `HashMap`) so that fallback
/// resolution is deterministic.**  The order matches the JSON key order of the
/// stored `protocol_endpoints` column after normalization; later entries have
/// lower priority.
#[derive(Debug, Clone)]
pub struct ProviderProtocols {
    pub default: ProtocolEndpoint,
    /// Ordered list of supported endpoints.  First match wins in fallback logic.
    pub endpoints: Vec<(ProtocolEndpoint, ProtocolEndpointEntry)>,
}

#[derive(Debug, Clone)]
pub struct ResolvedEgress {
    pub protocol: ProtocolEndpoint,
    pub base_url: String,
    pub needs_conversion: bool,
}

impl ProviderProtocols {
    /// Best-effort string → [`ProtocolEndpoint`] resolver.
    pub fn parse_protocol_key(s: &str) -> Option<ProtocolEndpoint> {
        registry::ProtocolRegistry::global().resolve_alias(s)
    }

    /// Build from a provider DB row.
    ///
    /// Handles both old endpoint-keyed format and new protocol-keyed format:
    /// - **Old** `{"openai-compat/chat-completions/v1": {"base_url": "..."}}` —
    ///   each key resolves directly to a `ProtocolEndpoint`.
    /// - **New** `{"openai-compat": {"base_url": "..."}}` —
    ///   key resolves to a `Protocol`; expands to all its endpoints.
    ///
    /// The `endpoints` Vec preserves the iteration order of the JSON map
    /// (serde_json preserves insertion order).
    pub fn from_provider(provider: &Provider) -> Self {
        let reg = registry::ProtocolRegistry::global();
        let raw_map = provider.parsed_protocol_endpoints();
        let mut seen = std::collections::HashSet::new();
        let mut endpoints: Vec<(ProtocolEndpoint, ProtocolEndpointEntry)> = Vec::new();

        for (key, entry) in &raw_map {
            // First try protocol-keyed format (new)
            if let Some(protocol) = reg.parse_protocol(key) {
                for handler in reg.list_by_protocol(protocol) {
                    let id = handler.id();
                    if seen.insert(id) {
                        endpoints.push((
                            id,
                            ProtocolEndpointEntry {
                                base_url: entry.base_url.clone(),
                            },
                        ));
                    }
                }
                continue;
            }

            // Fall back to endpoint-keyed format (old / normalized)
            if let Some(id) = Self::parse_protocol_key(key)
                && seen.insert(id)
            {
                endpoints.push((
                    id,
                    ProtocolEndpointEntry {
                        base_url: entry.base_url.clone(),
                    },
                ));
            }
        }

        let declared_default = Self::parse_protocol_key(provider.effective_default_protocol());
        let default = declared_default
            .filter(|id| endpoints.iter().any(|(eid, _)| eid == id))
            .or_else(|| endpoints.first().map(|(id, _)| *id))
            .or(declared_default)
            .unwrap_or(OPENAI_CHAT_COMPLETIONS_V1);

        Self { default, endpoints }
    }

    /// Returns `true` if the provider declares support for `protocol`.
    pub fn supports(&self, protocol: ProtocolEndpoint) -> bool {
        self.endpoints.iter().any(|(id, _)| *id == protocol)
    }

    /// Look up the endpoint entry for a specific protocol endpoint.
    pub fn get(&self, protocol: ProtocolEndpoint) -> Option<&ProtocolEndpointEntry> {
        self.endpoints
            .iter()
            .find_map(|(id, ep)| if *id == protocol { Some(ep) } else { None })
    }

    /// Deterministic three-tier egress resolution:
    ///
    /// 1. **Exact match** — ingress endpoint declared by the provider.
    /// 2. **Same-protocol, first declared** — iterates `endpoints` in Vec order,
    ///    which is JSON insertion order after normalization.  Deterministic.
    /// 3. **Provider default** — last resort.
    pub fn resolve_egress(&self, ingress: ProtocolEndpoint) -> ResolvedEgress {
        // Tier 1: exact match.
        if let Some(ep) = self.get(ingress) {
            return ResolvedEgress {
                protocol: ingress,
                base_url: ep.base_url.clone(),
                needs_conversion: false,
            };
        }

        // Tier 2: same protocol suite, first in declared order.
        if let Some((id, ep)) = self
            .endpoints
            .iter()
            .find(|(id, _)| id.protocol == ingress.protocol)
        {
            return ResolvedEgress {
                protocol: *id,
                base_url: ep.base_url.clone(),
                needs_conversion: true,
            };
        }

        // Tier 3: provider default.
        if let Some(ep) = self.get(self.default) {
            ResolvedEgress {
                protocol: self.default,
                base_url: ep.base_url.clone(),
                needs_conversion: true,
            }
        } else {
            ResolvedEgress {
                protocol: self.default,
                base_url: String::new(),
                needs_conversion: true,
            }
        }
    }
}
