//! Supervises Acty's top-level runtime actors.
//!
//! Assumes configuration has already been loaded from `acty.toml`. The
//! supervisor owns child actor addresses for provider transport, tool dispatch,
//! and the agent loop so lifecycle wiring lives in one place. Gotcha: every
//! actor mailbox uses the same configured bounded capacity for teaching clarity.

use anyhow::Result;
use tokio::sync::mpsc;

use crate::{
    actor::Address,
    agent,
    config::Config,
    llm,
    messages::{AgentMsg, LlmMsg, SystemMsg, ToolRegistryMsg, UiEvent},
    tools,
};

struct SystemSupervisor {
    agent: Address<AgentMsg>,
    llm: Address<LlmMsg>,
    tools: Address<ToolRegistryMsg>,
    inbox: mpsc::Receiver<SystemMsg>,
}

impl SystemSupervisor {
    /// Runs the system supervisor mailbox until shutdown.
    ///
    /// User messages are routed to the agent actor, while shutdown is fan-out
    /// lifecycle control for every child actor the supervisor started.
    async fn run(mut self) {
        while let Some(message) = self.inbox.recv().await {
            match message {
                SystemMsg::UserMessage { text } => {
                    let _ = self.agent.send(AgentMsg::UserMessage { text }).await;
                }
                SystemMsg::Shutdown => {
                    self.shutdown_children().await;
                    break;
                }
            }
        }
    }

    /// Sends shutdown to all child actors owned by the supervisor.
    async fn shutdown_children(&self) {
        let _ = self.agent.send(AgentMsg::Shutdown).await;
        let _ = self.llm.send(LlmMsg::Shutdown).await;
        let _ = self.tools.send(ToolRegistryMsg::Shutdown).await;
    }
}

/// Starts the system supervisor and its child runtime actors.
///
/// The returned address is the frontend's handle into the runtime; the frontend
/// does not need direct access to the agent, LLM, or tool registry actors.
pub fn spawn(config: Config, ui: mpsc::Sender<UiEvent>) -> Result<Address<SystemMsg>> {
    let mailbox_capacity = config.runtime.mailbox_capacity;
    let runtime_identity = agent::RuntimeIdentity::from_llm_config(&config.llm);
    let llm = llm::spawn(config.llm, mailbox_capacity)?;
    let tools = tools::spawn(
        config.workspace.root,
        config.tools.clone(),
        mailbox_capacity,
    )?;

    let agent = agent::spawn(
        config.agent,
        runtime_identity,
        config.tools,
        llm.clone(),
        tools.clone(),
        ui,
        mailbox_capacity,
    )?;

    let (tx, rx) = mpsc::channel(mailbox_capacity);
    let supervisor = SystemSupervisor {
        agent,
        llm,
        tools,
        inbox: rx,
    };

    tokio::spawn(supervisor.run());

    Ok(Address::new(tx))
}
