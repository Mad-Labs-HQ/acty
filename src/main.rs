//! Starts Acty's teaching runtime, wires the actor mailboxes, and runs the
//! terminal chat loop.
//!
//! Assumes `acty.toml` names every runtime setting instead of relying on code
//! defaults. Gotcha: the actor mailboxes are bounded by config, so sends can
//! naturally wait when another actor falls behind.

mod actor;
mod agent;
mod config;
mod llm;
mod messages;
mod protocol;
mod system;
mod tools;
mod tui;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use config::Config;
use messages::{SystemMsg, UiEvent};
use tokio::sync::mpsc;

#[derive(Debug, Parser)]
#[command(name = "acty", version)]
#[command(about = "A tiny actor-style coding agent for teaching")]
struct Cli {
    /// Path to the explicit Acty config file.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,

    /// Send one prompt and exit after the agent completes it.
    #[arg(long, value_name = "PROMPT")]
    prompt: Option<String>,
}

/// Boots the actor runtime and starts either one-shot or interactive chat.
///
/// The function loads configuration from the user-supplied file, starts the
/// system supervisor, then lets the frontend send user messages to that root
/// runtime address.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;
    let mailbox_capacity = config.runtime.mailbox_capacity;

    tracing_subscriber::fmt()
        .with_env_filter(config.runtime.log_filter.clone())
        .with_writer(std::io::stderr)
        .init();

    let (ui_tx, ui_rx) = mpsc::channel(mailbox_capacity);
    let system = system::spawn(config, ui_tx)?;

    match cli.prompt {
        Some(prompt) => run_one_prompt(system, ui_rx, prompt).await,
        None => tui::run(system, ui_rx).await,
    }
}

/// Sends one user prompt into the agent and prints events until completion.
///
/// This mirrors the interactive frontend but keeps conference demos scriptable
/// when a presenter wants to show the actor loop without typing live.
async fn run_one_prompt(
    system: actor::Address<SystemMsg>,
    mut ui_rx: mpsc::Receiver<UiEvent>,
    prompt: String,
) -> Result<()> {
    system
        .send(SystemMsg::UserMessage { text: prompt })
        .await
        .context("sending prompt to system supervisor")?;

    while let Some(event) = ui_rx.recv().await {
        let complete = event.is_prompt_done();
        tui::print_event(event);

        if complete {
            let _ = system.send(SystemMsg::Shutdown).await;
            return Ok(());
        }
    }

    Ok(())
}
