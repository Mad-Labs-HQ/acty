//! Loads Acty's explicit TOML configuration into typed runtime settings.
//!
//! Assumes every user-facing value appears in the config file. Missing files are
//! reported as absent input rather than replaced by embedded defaults.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub workspace: WorkspaceConfig,
    pub runtime: RuntimeConfig,
    pub llm: LlmConfig,
    pub agent: AgentConfig,
    pub tools: ToolsConfig,
}

impl Config {
    /// Reads and parses the user-supplied TOML file.
    ///
    /// The loader keeps configuration authority in the file by requiring the
    /// path from the CLI and deserializing directly into required fields.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;

        toml::from_str(&text).with_context(|| format!("parsing config file {}", path.display()))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceConfig {
    pub root: std::path::PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    pub log_filter: String,
    pub mailbox_capacity: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub provider_label: String,
    pub protocol: LlmProtocol,
    pub base_url: String,
    pub auth: AuthConfig,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Temperature,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProtocol {
    Openai,
    Anthropic,
}

impl LlmProtocol {
    /// Describes the wire protocol for prompt-visible runtime identity.
    ///
    /// The label is derived from the enum variant so config controls protocol
    /// choice while display text stays consistent across README examples and
    /// prompt injection.
    pub fn label(self) -> &'static str {
        match self {
            Self::Openai => "OpenAI-compatible Chat Completions",
            Self::Anthropic => "Anthropic Messages API",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AuthConfig {
    None,
    ApiKeyEnv { name: String },
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Temperature {
    ProviderDefault,
    Fixed { value: f32 },
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub max_turns: u32,
    pub system_prompt: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolsConfig {
    pub allowed: Vec<String>,
    pub bash_timeout_secs: u64,
    pub output_byte_limit: usize,
}

impl ToolsConfig {
    /// Checks whether a model-requested tool is enabled in explicit config.
    ///
    /// The tool registry calls this before dispatch so the allow-list remains
    /// the single source of truth for model-facing capabilities.
    pub fn is_allowed(&self, name: &str) -> bool {
        self.allowed.iter().any(|allowed| allowed == name)
    }
}
