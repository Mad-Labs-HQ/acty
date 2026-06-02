//! Owns Acty's coding tools and the tool registry actor.
//!
//! Assumes file tools stay inside the configured workspace while Bash runs as a
//! local command in that workspace. Gotcha: Bash is intentionally not a security
//! boundary, and tool results return through bounded actor mailboxes.

use std::{
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{process::Command, sync::mpsc, time};

use crate::{
    actor::Address,
    config::ToolsConfig,
    messages::{AgentMsg, ToolRegistryMsg},
    protocol::{ToolCall, ToolOutput, ToolSchema},
};

struct ToolRegistryActor {
    workspace: PathBuf,
    config: ToolsConfig,
    tools: Vec<Box<dyn ToolHandler>>,
    inbox: mpsc::Receiver<ToolRegistryMsg>,
}

impl ToolRegistryActor {
    /// Runs the tool registry mailbox until all senders are dropped.
    ///
    /// Tool calls are executed in model order so mutation effects remain easy to
    /// explain in a teaching setting.
    async fn run(mut self) {
        while let Some(message) = self.inbox.recv().await {
            match message {
                ToolRegistryMsg::DispatchBatch { calls, reply_to } => {
                    self.dispatch_batch(calls, reply_to).await;
                }
                ToolRegistryMsg::Shutdown => break,
            }
        }
    }

    /// Executes a model-requested tool batch and replies to the agent actor.
    ///
    /// The registry checks the explicit allow-list before lookup, then converts
    /// every handler result into an ordinary `AgentMsg::ToolFinished` message.
    async fn dispatch_batch(&self, calls: Vec<ToolCall>, reply_to: Address<AgentMsg>) {
        for call in calls {
            let output = self.dispatch_one(&call).await.unwrap_or_else(|error| {
                ToolOutput::Error(format!("tool '{}' failed: {error}", call.name))
            });

            let _ = reply_to
                .send(AgentMsg::ToolFinished {
                    call_id: call.id,
                    name: call.name,
                    output,
                })
                .await;
        }
    }

    /// Runs one tool call after allow-list and name checks.
    async fn dispatch_one(&self, call: &ToolCall) -> Result<ToolOutput> {
        if !self.config.is_allowed(&call.name) {
            bail!("tool is not allowed by config");
        }

        let Some(tool) = self
            .tools
            .iter()
            .find(|tool| tool.schema().name == call.name)
        else {
            bail!("tool is not registered");
        };

        let context = ToolContext {
            workspace: self.workspace.clone(),
            config: self.config.clone(),
        };

        tool.execute(call.arguments.clone(), context).await
    }
}

/// Starts the tool registry actor and returns its mailbox address.
///
/// The registry owns the handler list and workspace path, while the agent only
/// sees schemas and tool-result messages.
pub fn spawn(
    workspace: PathBuf,
    config: ToolsConfig,
    mailbox_capacity: usize,
) -> Result<Address<ToolRegistryMsg>> {
    let workspace = std::env::current_dir()
        .context("reading current directory")?
        .join(workspace);

    let tools = builtin_tools();
    let (tx, rx) = mpsc::channel(mailbox_capacity);
    let actor = ToolRegistryActor {
        workspace,
        config,
        tools,
        inbox: rx,
    };

    tokio::spawn(actor.run());
    Ok(Address::new(tx))
}

/// Returns schemas for the builtin tools allowed by the active config.
pub fn schemas(config: &ToolsConfig) -> Vec<ToolSchema> {
    builtin_tools()
        .into_iter()
        .map(|tool| tool.schema())
        .filter(|schema| config.is_allowed(&schema.name))
        .collect()
}

fn builtin_tools() -> Vec<Box<dyn ToolHandler>> {
    vec![
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(EditTool),
        Box::new(BashTool),
    ]
}

#[derive(Debug, Clone)]
struct ToolContext {
    workspace: PathBuf,
    config: ToolsConfig,
}

impl ToolContext {
    /// Resolves a model path and rejects paths outside the configured workspace.
    ///
    /// The function uses lexical normalization for files that may not exist yet,
    /// which lets `Write` create new files without requiring parent canonicality.
    fn resolve_workspace_path(&self, raw: &str) -> Result<PathBuf> {
        let path = Path::new(raw);
        let relative = workspace_relative_path(path)?;

        let normalized = normalize_relative(relative)?;
        Ok(self.workspace.join(normalized))
    }
}

/// Converts model-facing paths into workspace-relative paths.
///
/// Absolute `/workspace/...` paths are accepted because the system prompt may
/// teach that vocabulary; other absolute paths are rejected so file tools cannot
/// escape the configured workspace by accident.
fn workspace_relative_path(path: &Path) -> Result<&Path> {
    if !path.is_absolute() {
        return Ok(path);
    }

    match path.strip_prefix("/workspace") {
        Ok(relative) => Ok(relative),
        Err(_) => bail!("absolute paths must be under /workspace"),
    }
}

#[async_trait]
trait ToolHandler: Send + Sync {
    /// Describes the model-facing tool name, purpose, and JSON input shape.
    fn schema(&self) -> ToolSchema;

    /// Executes one decoded model tool call in the configured workspace context.
    async fn execute(&self, input: Value, context: ToolContext) -> Result<ToolOutput>;
}

struct ReadTool;

#[async_trait]
impl ToolHandler for ReadTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Read".to_string(),
            description: "Read a UTF-8 file from the workspace.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        }
    }

    /// Reads a workspace file and returns byte-limited text to the model.
    ///
    /// The handler resolves the path through `ToolContext`, reads it as UTF-8,
    /// then truncates the end of the output when the configured cap is exceeded.
    async fn execute(&self, input: Value, context: ToolContext) -> Result<ToolOutput> {
        let args: PathArgs = serde_json::from_value(input).context("decoding Read args")?;
        let path = context.resolve_workspace_path(&args.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("reading {}", path.display()))?;

        Ok(ToolOutput::Text(truncate_text(
            text,
            context.config.output_byte_limit,
        )))
    }
}

struct WriteTool;

#[async_trait]
impl ToolHandler for WriteTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Write".to_string(),
            description: "Create or replace a UTF-8 file in the workspace.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    /// Writes model-supplied UTF-8 content into a workspace file.
    ///
    /// The handler creates parent directories before writing so coding tasks can
    /// introduce new files without separate setup commands.
    async fn execute(&self, input: Value, context: ToolContext) -> Result<ToolOutput> {
        let args: WriteArgs = serde_json::from_value(input).context("decoding Write args")?;
        let path = context.resolve_workspace_path(&args.path)?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating {}", parent.display()))?;
        }

        tokio::fs::write(&path, args.content)
            .await
            .with_context(|| format!("writing {}", path.display()))?;

        Ok(ToolOutput::Text(format!("wrote {}", path.display())))
    }
}

struct EditTool;

#[async_trait]
impl ToolHandler for EditTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Edit".to_string(),
            description: "Replace one exact text span in a UTF-8 workspace file.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old": { "type": "string" },
                    "new": { "type": "string" }
                },
                "required": ["path", "old", "new"]
            }),
        }
    }

    /// Applies one exact replacement to a workspace file.
    ///
    /// The handler refuses ambiguous replacements so the model must provide a
    /// span that identifies exactly one edit site.
    async fn execute(&self, input: Value, context: ToolContext) -> Result<ToolOutput> {
        let args: EditArgs = serde_json::from_value(input).context("decoding Edit args")?;
        let path = context.resolve_workspace_path(&args.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("reading {}", path.display()))?;
        let matches = text.matches(&args.old).count();

        match matches {
            0 => bail!("old text was not found"),
            1 => {
                let edited = text.replacen(&args.old, &args.new, 1);
                tokio::fs::write(&path, edited)
                    .await
                    .with_context(|| format!("writing {}", path.display()))?;

                Ok(ToolOutput::Text(format!("edited {}", path.display())))
            }
            _ => bail!("old text matched {matches} times"),
        }
    }
}

struct BashTool;

#[async_trait]
impl ToolHandler for BashTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Bash".to_string(),
            description: "Run a shell command in the workspace using soft local execution."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }
    }

    /// Runs one shell command in the workspace and captures combined output.
    ///
    /// The handler enforces the configured timeout and appends a compact status
    /// footer so the model can see the exit code and working directory.
    async fn execute(&self, input: Value, context: ToolContext) -> Result<ToolOutput> {
        let args: BashArgs = serde_json::from_value(input).context("decoding Bash args")?;
        let mut child = Command::new("sh");
        child
            .arg("-lc")
            .arg(&args.command)
            .current_dir(&context.workspace)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = time::timeout(
            Duration::from_secs(context.config.bash_timeout_secs),
            child.output(),
        )
        .await;

        match output {
            Ok(result) => format_command_output(result?, &context, false),
            Err(_) => Ok(ToolOutput::Error(format!(
                "timed out after {}s\n[exit=? cwd={} timed_out=true]",
                context.config.bash_timeout_secs,
                context.workspace.display()
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
}

/// Normalizes a relative path without touching the filesystem.
fn normalize_relative(path: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("path escapes the workspace");
                }
            }
            Component::RootDir | Component::Prefix(_) => {}
        }
    }

    Ok(normalized)
}

/// Formats command output into Acty's tool-output shape.
fn format_command_output(
    output: std::process::Output,
    context: &ToolContext,
    timed_out: bool,
) -> Result<ToolOutput> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let code = output
        .status
        .code()
        .map_or("?".to_string(), |code| code.to_string());
    let mut body = if stderr.is_empty() {
        stdout.to_string()
    } else {
        format!("--- stdout ---\n{stdout}--- stderr ---\n{stderr}")
    };

    body = truncate_text(body, context.config.output_byte_limit);
    body.push_str(&format!(
        "\n[exit={code} cwd={} timed_out={timed_out}]",
        context.workspace.display()
    ));

    if output.status.success() {
        Ok(ToolOutput::Text(body))
    } else {
        Ok(ToolOutput::Error(body))
    }
}

/// Truncates text on a character boundary and marks the truncation.
fn truncate_text(mut text: String, limit: usize) -> String {
    if text.len() <= limit {
        return text;
    }

    let mut end = limit;

    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    text.truncate(end);
    text.push_str("\n[truncated]");
    text
}
