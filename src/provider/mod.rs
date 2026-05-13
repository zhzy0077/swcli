use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub(crate) mod github_copilot;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ProviderKind {
    Openai,
    Anthropic,
    CodexOauth,
    ClaudeOauth,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ProviderKind::Openai => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::CodexOauth => "codex-oauth",
            ProviderKind::ClaudeOauth => "claude-oauth",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireProtocol {
    OpenaiResponses,
    OpenaiCompletions,
    AnthropicMessages,
}

impl std::fmt::Display for WireProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            WireProtocol::OpenaiResponses => "openai-responses",
            WireProtocol::OpenaiCompletions => "openai-completions",
            WireProtocol::AnthropicMessages => "anthropic-messages",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuiltinPreset {
    pub(crate) alias: &'static str,
    pub(crate) provider: ProviderKind,
    pub(crate) models_dev_provider_name: Option<&'static str>,
    pub(crate) endpoint: &'static str,
    pub(crate) wire: WireProtocol,
}

pub(crate) const BUILTIN_PRESETS: &[BuiltinPreset] = &[
    BuiltinPreset {
        alias: "github-copilot",
        provider: ProviderKind::Openai,
        models_dev_provider_name: None,
        endpoint: "https://api.githubcopilot.com",
        wire: WireProtocol::OpenaiCompletions,
    },
    BuiltinPreset {
        alias: "minimax",
        provider: ProviderKind::Anthropic,
        models_dev_provider_name: Some("MiniMax (minimax.io)"),
        endpoint: "https://api.minimax.io/anthropic/v1",
        wire: WireProtocol::AnthropicMessages,
    },
    BuiltinPreset {
        alias: "minimax-cn",
        provider: ProviderKind::Anthropic,
        models_dev_provider_name: Some("MiniMax (minimaxi.com)"),
        endpoint: "https://api.minimaxi.com/anthropic/v1",
        wire: WireProtocol::AnthropicMessages,
    },
    BuiltinPreset {
        alias: "xiaomi-token-plan-cn",
        provider: ProviderKind::Openai,
        models_dev_provider_name: Some("Xiaomi Token Plan (China)"),
        endpoint: "https://token-plan-cn.xiaomimimo.com/v1",
        wire: WireProtocol::OpenaiCompletions,
    },
];

pub(crate) fn resolve_builtin_preset(alias: &str) -> Result<&'static BuiltinPreset> {
    BUILTIN_PRESETS
        .iter()
        .find(|preset| preset.alias == alias)
        .ok_or_else(|| {
            let names = BUILTIN_PRESETS
                .iter()
                .map(|preset| preset.alias)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("Unknown builtin preset `{alias}`. Available presets: {names}")
        })
}

pub(crate) fn default_base_url(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Openai | ProviderKind::CodexOauth => "https://api.openai.com/v1",
        ProviderKind::Anthropic | ProviderKind::ClaudeOauth => "https://api.anthropic.com/v1",
    }
}

pub(crate) fn default_wire_for_provider(provider: ProviderKind) -> WireProtocol {
    match provider {
        ProviderKind::Openai | ProviderKind::CodexOauth => WireProtocol::OpenaiResponses,
        ProviderKind::Anthropic | ProviderKind::ClaudeOauth => WireProtocol::AnthropicMessages,
    }
}
