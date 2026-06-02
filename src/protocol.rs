//! Defines the small internal protocol shared by the agent, LLM, and tools.
//!
//! Assumes provider-specific JSON quirks stay in `llm.rs`; these types describe
//! Acty's own conversation and tool-call vocabulary.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub enum ConversationItem {
    User(String),
    Assistant(AssistantTurn),
    Tool(ToolTurn),
}

#[derive(Debug, Clone)]
pub struct AssistantTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone)]
pub struct ToolTurn {
    pub call_id: ToolCallId,
    pub output: ToolOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(pub String);

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub enum ToolOutput {
    Text(String),
    Error(String),
}

impl ToolOutput {
    /// Renders tool output for model feedback and terminal display.
    ///
    /// Text results pass through unchanged, while errors get a visible prefix so
    /// the next model turn can distinguish failed tool calls from normal output.
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Error(message) => format!("Error: {message}"),
        }
    }

    /// Reports whether this output represents a failed tool call.
    ///
    /// The UI uses this to present tool status without inspecting the formatted
    /// text returned to the model.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Known(String),
    Missing,
}

impl FinishReason {
    /// Builds a typed finish reason from an optional provider field.
    ///
    /// Providers may omit finish metadata, so the internal protocol preserves
    /// that absence instead of fabricating a default string.
    pub fn from_provider(value: Option<String>) -> Self {
        match value {
            Some(reason) => Self::Known(reason),
            None => Self::Missing,
        }
    }

    /// Renders the finish reason for terminal output.
    ///
    /// The display boundary is where missing provider metadata becomes a label;
    /// actor state still keeps missing distinct from a provider-supplied value.
    pub fn as_label(&self) -> &str {
        match self {
            Self::Known(reason) => reason,
            Self::Missing => "missing",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: String,
    pub conversation: Vec<ConversationItem>,
    pub tools: Vec<ToolSchema>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}
