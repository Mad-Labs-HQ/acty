//! Provides Acty's minimal terminal chat interface.
//!
//! Assumes plain line-oriented input is enough for the teaching repository.
//! Gotcha: this is intentionally a tiny chat TUI, so rendering stays transparent
//! instead of introducing a large terminal widget framework.

use anyhow::{Context, Result};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::mpsc,
};

use crate::{
    actor::Address,
    messages::{SystemMsg, UiEvent},
};

/// Runs the interactive line-oriented chat loop.
///
/// The UI owns stdin/stdout interaction while the actor runtime owns
/// conversation state; user input and runtime updates cross that boundary as
/// messages.
pub async fn run(system: Address<SystemMsg>, mut events: mpsc::Receiver<UiEvent>) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    stdout
        .write_all(b"acty teaching chat. Type a prompt, or /quit.\n> ")
        .await
        .context("writing prompt")?;
    stdout.flush().await.context("flushing prompt")?;

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Some(line) = line.context("reading stdin")? else {
                    return Ok(());
                };
                let trimmed = line.trim();

                if trimmed == "/quit" {
                    let _ = system.send(SystemMsg::Shutdown).await;
                    return Ok(());
                }

                if !trimmed.is_empty() {
                    system
                        .send(SystemMsg::UserMessage { text: line })
                        .await
                        .context("sending user message to system supervisor")?;
                }
            }
            event = events.recv() => {
                let Some(event) = event else {
                    return Ok(());
                };
                print_event(event);
                stdout.write_all(b"> ").await.context("writing prompt")?;
                stdout.flush().await.context("flushing prompt")?;
            }
        }
    }
}

/// Prints one runtime event in a compact teaching-friendly format.
pub fn print_event(event: UiEvent) {
    match event {
        UiEvent::UserMessage(text) => println!("user: {text}"),
        UiEvent::TurnStart(n) => println!("\n[turn {n}]"),
        UiEvent::AssistantMessage {
            text,
            tool_calls,
            finish_reason,
        } => {
            if !text.is_empty() {
                println!("assistant: {text}");
            }

            if tool_calls > 0 {
                println!("[assistant requested {tool_calls} tool call(s)]");
            }

            println!("[finish_reason={finish_reason}]");
        }
        UiEvent::ToolStart { name, arguments } => println!("[tool start] {name} {arguments}"),
        UiEvent::ToolEnd { name, ok, output } => {
            let status = if ok { "ok" } else { "error" };
            println!("[tool end] {name} {status}\n{output}");
        }
        UiEvent::PromptDone => println!("[done]"),
        UiEvent::Error(message) => println!("[error] {message}"),
    }
}
