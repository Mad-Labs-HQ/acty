//! Defines the messages exchanged by Acty's system, LLM, tool, agent, and UI
//! actors.
//!
//! Assumes each message enum is consumed by exactly one actor. Keeping the
//! mailboxes explicit makes ownership boundaries visible in the teaching repo.

use crate::{
    actor::Address,
    protocol::{LlmRequest, LlmResponse, ToolCall, ToolCallId, ToolOutput},
};

#[derive(Debug)]
pub enum SystemMsg {
    UserMessage { text: String },
    Shutdown,
}

#[derive(Debug)]
pub enum LlmMsg {
    Generate {
        request: LlmRequest,
        reply_to: Address<AgentMsg>,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum ToolRegistryMsg {
    DispatchBatch {
        calls: Vec<ToolCall>,
        reply_to: Address<AgentMsg>,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum AgentMsg {
    UserMessage {
        text: String,
    },
    Shutdown,
    LlmFinished(Result<LlmResponse, String>),
    ToolFinished {
        call_id: ToolCallId,
        name: String,
        output: ToolOutput,
    },
}

#[derive(Debug, Clone)]
pub enum UiEvent {
    UserMessage(String),
    TurnStart(u32),
    AssistantMessage {
        text: String,
        tool_calls: usize,
        finish_reason: String,
    },
    ToolStart {
        name: String,
        arguments: String,
    },
    ToolEnd {
        name: String,
        ok: bool,
        output: String,
    },
    PromptDone,
    Error(String),
}

impl UiEvent {
    /// Reports whether the current user prompt has reached a terminal event.
    ///
    /// One-shot mode uses this to stop reading events after the agent finishes
    /// the prompt-local loop.
    pub fn is_prompt_done(&self) -> bool {
        matches!(self, Self::PromptDone)
    }
}
