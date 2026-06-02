//! Owns the goal-directed coding loop as one stateful actor.
//!
//! Assumes LLM calls and tool calls report back through messages rather than
//! direct callbacks. Gotcha: the agent never reaches into the LLM or tool actors;
//! it records intent, sends a bounded mailbox message, and continues when a
//! result returns.

use std::collections::HashMap;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::{
    actor::Address,
    config::{AgentConfig, LlmConfig, ToolsConfig},
    messages::{AgentMsg, LlmMsg, ToolRegistryMsg, UiEvent},
    protocol::{
        AssistantTurn, ConversationItem, LlmRequest, LlmResponse, ToolCall, ToolCallId, ToolOutput,
        ToolTurn,
    },
    tools,
};

#[derive(Debug, Clone)]
pub struct RuntimeIdentity {
    provider_label: String,
    protocol_label: String,
    model: String,
}

impl RuntimeIdentity {
    /// Builds prompt-visible runtime identity from explicit LLM configuration.
    ///
    /// The provider label is user-authored config rather than inferred from the
    /// endpoint, so hosted and local OpenAI-compatible servers can identify
    /// themselves accurately.
    pub fn from_llm_config(config: &LlmConfig) -> Self {
        Self {
            provider_label: config.provider_label.clone(),
            protocol_label: config.protocol.label().to_string(),
            model: config.model.clone(),
        }
    }

    /// Appends runtime identity facts to the configured agent system prompt.
    ///
    /// This keeps runtime truth owned by config while making it visible to the
    /// model for questions about provider or model identity.
    fn append_to_system_prompt(&self, system_prompt: &str) -> String {
        format!(
            "{system_prompt}\n\nRuntime identity:\n- Provider: {}\n- Protocol: {}\n- Model: {}\n\nIf the user asks what model or provider you are using, answer from these runtime identity facts.",
            self.provider_label, self.protocol_label, self.model
        )
    }
}

#[derive(Debug)]
enum AgentPhase {
    Idle,
    WaitingForLlm,
    WaitingForTools {
        remaining: HashMap<ToolCallId, String>,
    },
}

struct AgentActor {
    config: AgentConfig,
    runtime_identity: RuntimeIdentity,
    tools_config: ToolsConfig,
    llm: Address<LlmMsg>,
    tools: Address<ToolRegistryMsg>,
    ui: mpsc::Sender<UiEvent>,
    myself: Address<AgentMsg>,
    inbox: mpsc::Receiver<AgentMsg>,
    conversation: Vec<ConversationItem>,
    phase: AgentPhase,
    turn: u32,
}

impl AgentActor {
    /// Runs the agent mailbox until all senders are dropped.
    ///
    /// Each message is interpreted against the actor's private phase and
    /// conversation state, which is the central actor-model lesson of the repo.
    async fn run(mut self) {
        while let Some(message) = self.inbox.recv().await {
            if matches!(message, AgentMsg::Shutdown) {
                break;
            }

            match self.handle(message).await {
                Ok(()) => {}
                Err(error) => {
                    let _ = self.ui.send(UiEvent::Error(error.to_string())).await;
                    let _ = self.ui.send(UiEvent::PromptDone).await;
                    self.phase = AgentPhase::Idle;
                }
            }
        }
    }

    /// Applies one agent message to the current state.
    ///
    /// User messages start a model turn, LLM responses either finish or dispatch
    /// tools, and tool results resume the loop when the pending batch completes.
    async fn handle(&mut self, message: AgentMsg) -> Result<()> {
        match message {
            AgentMsg::UserMessage { text } => self.handle_user_message(text).await,
            AgentMsg::Shutdown => Ok(()),
            AgentMsg::LlmFinished(result) => self.handle_llm_finished(result).await,
            AgentMsg::ToolFinished {
                call_id,
                name,
                output,
            } => self.handle_tool_finished(call_id, name, output).await,
        }
    }

    /// Records a user message and starts a fresh model turn.
    async fn handle_user_message(&mut self, text: String) -> Result<()> {
        self.ui.send(UiEvent::UserMessage(text.clone())).await?;
        self.conversation.push(ConversationItem::User(text));
        self.turn = 0;
        self.start_llm_turn().await
    }

    /// Records an LLM response and either finishes or dispatches tool work.
    async fn handle_llm_finished(&mut self, result: Result<LlmResponse, String>) -> Result<()> {
        let response = match result {
            Ok(response) => response,
            Err(message) => {
                self.ui.send(UiEvent::Error(message)).await?;
                self.ui.send(UiEvent::PromptDone).await?;
                self.phase = AgentPhase::Idle;
                return Ok(());
            }
        };

        let tool_calls = response.tool_calls.clone();
        self.conversation
            .push(ConversationItem::Assistant(AssistantTurn {
                text: response.text.clone(),
                tool_calls: response.tool_calls,
            }));
        self.ui
            .send(UiEvent::AssistantMessage {
                text: response.text,
                tool_calls: tool_calls.len(),
                finish_reason: response.finish_reason.as_label().to_string(),
            })
            .await?;

        if tool_calls.is_empty() {
            self.ui.send(UiEvent::PromptDone).await?;
            self.phase = AgentPhase::Idle;
            return Ok(());
        }

        self.dispatch_tools(tool_calls).await
    }

    /// Records one tool result and resumes the LLM after the batch completes.
    async fn handle_tool_finished(
        &mut self,
        call_id: ToolCallId,
        name: String,
        output: ToolOutput,
    ) -> Result<()> {
        self.ui
            .send(UiEvent::ToolEnd {
                name: name.clone(),
                ok: !output.is_error(),
                output: output.as_text(),
            })
            .await?;
        self.conversation.push(ConversationItem::Tool(ToolTurn {
            call_id: call_id.clone(),
            output,
        }));

        match &mut self.phase {
            AgentPhase::WaitingForTools { remaining } => {
                remaining.remove(&call_id);

                if remaining.is_empty() {
                    self.start_llm_turn().await?;
                }
            }
            AgentPhase::Idle | AgentPhase::WaitingForLlm => {
                self.ui
                    .send(UiEvent::Error(format!(
                        "received stale tool result for {name}"
                    )))
                    .await?;
            }
        }

        Ok(())
    }

    /// Sends the current transcript to the LLM actor.
    ///
    /// The agent records that it is waiting before sending the message, so a
    /// provider response always has an explicit state to return to.
    async fn start_llm_turn(&mut self) -> Result<()> {
        self.turn += 1;
        self.ui.send(UiEvent::TurnStart(self.turn)).await?;

        if self.turn > self.config.max_turns {
            self.ui
                .send(UiEvent::Error(format!(
                    "turn cap reached at {}",
                    self.config.max_turns
                )))
                .await?;
            self.ui.send(UiEvent::PromptDone).await?;
            self.phase = AgentPhase::Idle;
            return Ok(());
        }

        self.phase = AgentPhase::WaitingForLlm;
        self.llm
            .send(LlmMsg::Generate {
                request: LlmRequest {
                    system_prompt: self
                        .runtime_identity
                        .append_to_system_prompt(&self.config.system_prompt),
                    conversation: self.conversation.clone(),
                    tools: tools::schemas(&self.tools_config),
                },
                reply_to: self.myself.clone(),
            })
            .await?;

        Ok(())
    }

    /// Sends a batch of tool calls to the registry actor.
    ///
    /// Pending call IDs become continuation state owned by the agent actor until
    /// the registry sends the matching `ToolFinished` messages back.
    async fn dispatch_tools(&mut self, calls: Vec<ToolCall>) -> Result<()> {
        let mut remaining = HashMap::new();

        for call in &calls {
            remaining.insert(call.id.clone(), call.name.clone());
            self.ui
                .send(UiEvent::ToolStart {
                    name: call.name.clone(),
                    arguments: call.arguments.to_string(),
                })
                .await?;
        }

        self.phase = AgentPhase::WaitingForTools { remaining };
        self.tools
            .send(ToolRegistryMsg::DispatchBatch {
                calls,
                reply_to: self.myself.clone(),
            })
            .await?;

        Ok(())
    }
}

/// Starts the agent actor and returns its mailbox address.
///
/// The actor owns the conversation transcript and phase state; all external work
/// is delegated through LLM and tool actor addresses.
pub fn spawn(
    config: AgentConfig,
    runtime_identity: RuntimeIdentity,
    tools_config: ToolsConfig,
    llm: Address<LlmMsg>,
    tools: Address<ToolRegistryMsg>,
    ui: mpsc::Sender<UiEvent>,
    mailbox_capacity: usize,
) -> Result<Address<AgentMsg>> {
    let (tx, rx) = mpsc::channel(mailbox_capacity);
    let address = Address::new(tx);
    let actor = AgentActor {
        config,
        runtime_identity,
        tools_config,
        llm,
        tools,
        ui,
        myself: address.clone(),
        inbox: rx,
        conversation: Vec::new(),
        phase: AgentPhase::Idle,
        turn: 0,
    };

    tokio::spawn(actor.run());
    Ok(address)
}
