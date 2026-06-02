//! Owns the provider actor and translates Acty's protocol into LLM APIs.
//!
//! Assumes one configured provider is enough for the teaching repo. Gotcha:
//! OpenAI and Anthropic expose similar tool semantics with different JSON
//! envelopes, so this module keeps those translations behind one bounded
//! mailbox.

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    actor::Address,
    config::{AuthConfig, LlmConfig, LlmProtocol, Temperature},
    messages::{AgentMsg, LlmMsg},
    protocol::{
        AssistantTurn, ConversationItem, FinishReason, LlmRequest, LlmResponse, ToolCall,
        ToolCallId, ToolSchema, ToolTurn,
    },
};

struct LlmActor {
    config: LlmConfig,
    client: Client,
    auth: Auth,
    inbox: mpsc::Receiver<LlmMsg>,
}

impl LlmActor {
    /// Runs the provider mailbox until all senders are dropped.
    ///
    /// Each `Generate` message performs one HTTP request, normalizes the response
    /// into Acty's internal protocol, and replies to the requesting agent actor.
    async fn run(mut self) {
        while let Some(message) = self.inbox.recv().await {
            match message {
                LlmMsg::Generate { request, reply_to } => {
                    let result = self.generate(request).await.map_err(format_error_chain);

                    let _ = reply_to.send(AgentMsg::LlmFinished(result)).await;
                }
                LlmMsg::Shutdown => break,
            }
        }
    }

    /// Routes one generation request through the configured provider protocol.
    ///
    /// The provider-specific request builders share the same internal
    /// conversation and tool schema types so the agent actor stays protocol-free.
    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse> {
        match self.config.protocol {
            LlmProtocol::Openai => self.generate_openai(request).await,
            LlmProtocol::Anthropic => self.generate_anthropic(request).await,
        }
    }

    /// Sends one OpenAI-compatible chat completion request.
    ///
    /// The function maps Acty's conversation items into chat messages, preserving
    /// tool call IDs so tool results can be associated with assistant requests.
    async fn generate_openai(&self, request: LlmRequest) -> Result<LlmResponse> {
        let url = join_url(&self.config.base_url, "chat/completions");
        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": openai_messages(&request),
            "tools": openai_tools(&request.tools),
            "tool_choice": "auto"
        });

        apply_temperature(&mut body, self.config.temperature);

        let response = self
            .client
            .post(url)
            .with_auth(&self.auth, ProviderHeader::OpenAi)
            .json(&body)
            .send()
            .await
            .context("sending OpenAI-compatible request")?
            .error_for_status()
            .context("OpenAI-compatible provider returned an error")?
            .json::<OpenAiResponse>()
            .await
            .context("decoding OpenAI-compatible response")?;

        parse_openai_response(response)
    }

    /// Sends one Anthropic Messages API request.
    ///
    /// The function emits Anthropic `tool_use` and `tool_result` content blocks
    /// while keeping Acty's agent transcript unchanged.
    async fn generate_anthropic(&self, request: LlmRequest) -> Result<LlmResponse> {
        let url = join_url(&self.config.base_url, "messages");
        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": request.system_prompt,
            "messages": anthropic_messages(&request.conversation),
            "tools": anthropic_tools(&request.tools)
        });

        apply_temperature(&mut body, self.config.temperature);

        let response = self
            .client
            .post(url)
            .with_auth(&self.auth, ProviderHeader::Anthropic)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .context("sending Anthropic request")?
            .error_for_status()
            .context("Anthropic provider returned an error")?
            .json::<AnthropicResponse>()
            .await
            .context("decoding Anthropic response")?;

        Ok(parse_anthropic_response(response))
    }
}

/// Formats an anyhow error with its causal chain for frontend display.
fn format_error_chain(error: anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// Starts the LLM actor and returns its mailbox address.
///
/// Authentication is resolved during startup so missing configured credentials
/// fail before a user prompt enters the agent loop.
pub fn spawn(config: LlmConfig, mailbox_capacity: usize) -> Result<Address<LlmMsg>> {
    let auth = Auth::from_config(&config.auth)?;
    let (tx, rx) = mpsc::channel(mailbox_capacity);
    let actor = LlmActor {
        config,
        client: Client::new(),
        auth,
        inbox: rx,
    };

    tokio::spawn(actor.run());
    Ok(Address::new(tx))
}

#[derive(Debug, Clone)]
enum Auth {
    None,
    ApiKey(String),
}

impl Auth {
    /// Resolves provider authentication from explicit config.
    ///
    /// `None` is a named choice for local servers, while `ApiKeyEnv` fails at
    /// startup if the configured variable is missing.
    fn from_config(config: &AuthConfig) -> Result<Self> {
        match config {
            AuthConfig::None => Ok(Self::None),
            AuthConfig::ApiKeyEnv { name } => {
                let value = std::env::var(name)
                    .with_context(|| format!("reading API key env var {name}"))?;

                Ok(Self::ApiKey(value))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ProviderHeader {
    OpenAi,
    Anthropic,
}

trait AuthenticatedRequest {
    /// Applies the provider-specific auth header when auth is configured.
    fn with_auth(self, auth: &Auth, header: ProviderHeader) -> Self;
}

impl AuthenticatedRequest for reqwest::RequestBuilder {
    fn with_auth(self, auth: &Auth, header: ProviderHeader) -> Self {
        match (auth, header) {
            (Auth::None, _) => self,
            (Auth::ApiKey(value), ProviderHeader::OpenAi) => self.bearer_auth(value),
            (Auth::ApiKey(value), ProviderHeader::Anthropic) => self.header("x-api-key", value),
        }
    }
}

/// Joins a provider base URL and endpoint without assuming either slash shape.
fn join_url(base: &str, endpoint: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), endpoint)
}

/// Adds temperature only when the config names a fixed value.
fn apply_temperature(body: &mut Value, temperature: Temperature) {
    match temperature {
        Temperature::ProviderDefault => {}
        Temperature::Fixed { value } => {
            body["temperature"] = json!(value);
        }
    }
}

/// Converts Acty's request into OpenAI chat messages.
fn openai_messages(request: &LlmRequest) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": request.system_prompt
    })];

    for item in &request.conversation {
        match item {
            ConversationItem::User(text) => messages.push(json!({
                "role": "user",
                "content": text
            })),
            ConversationItem::Assistant(turn) => messages.push(openai_assistant_message(turn)),
            ConversationItem::Tool(turn) => messages.push(json!({
                "role": "tool",
                "tool_call_id": turn.call_id.0,
                "content": turn.output.as_text()
            })),
        }
    }

    messages
}

/// Converts one assistant turn into an OpenAI assistant message.
fn openai_assistant_message(turn: &AssistantTurn) -> Value {
    let tool_calls = turn
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id.0,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments.to_string()
                }
            })
        })
        .collect::<Vec<_>>();

    if tool_calls.is_empty() {
        json!({
            "role": "assistant",
            "content": turn.text
        })
    } else {
        json!({
            "role": "assistant",
            "content": turn.text,
            "tool_calls": tool_calls
        })
    }
}

/// Converts Acty's tool schemas into OpenAI function tool declarations.
fn openai_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        })
        .collect()
}

/// Converts Acty's transcript into Anthropic Messages API messages.
fn anthropic_messages(conversation: &[ConversationItem]) -> Vec<Value> {
    conversation
        .iter()
        .map(|item| match item {
            ConversationItem::User(text) => json!({
                "role": "user",
                "content": [{ "type": "text", "text": text }]
            }),
            ConversationItem::Assistant(turn) => anthropic_assistant_message(turn),
            ConversationItem::Tool(turn) => anthropic_tool_message(turn),
        })
        .collect()
}

/// Converts one assistant turn into Anthropic text and tool-use blocks.
fn anthropic_assistant_message(turn: &AssistantTurn) -> Value {
    let mut content = Vec::new();

    if !turn.text.is_empty() {
        content.push(json!({
            "type": "text",
            "text": turn.text
        }));
    }

    for call in &turn.tool_calls {
        content.push(json!({
            "type": "tool_use",
            "id": call.id.0,
            "name": call.name,
            "input": call.arguments
        }));
    }

    json!({
        "role": "assistant",
        "content": content
    })
}

/// Converts one tool result into an Anthropic user-side tool result message.
fn anthropic_tool_message(turn: &ToolTurn) -> Value {
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": turn.call_id.0,
            "content": turn.output.as_text(),
            "is_error": turn.output.is_error()
        }]
    })
}

/// Converts Acty's tool schemas into Anthropic tool declarations.
fn anthropic_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            })
        })
        .collect()
}

/// Normalizes an OpenAI-compatible response into Acty's LLM response.
fn parse_openai_response(response: OpenAiResponse) -> Result<LlmResponse> {
    let Some(choice) = response.choices.into_iter().next() else {
        bail!("provider returned no choices");
    };

    let tool_calls = choice
        .message
        .tool_calls
        .into_iter()
        .flatten()
        .map(parse_openai_tool_call)
        .collect::<Result<Vec<_>>>()?;

    Ok(LlmResponse {
        text: openai_visible_text(choice.message.content),
        tool_calls,
        finish_reason: FinishReason::from_provider(choice.finish_reason),
    })
}

/// Converts OpenAI's optional assistant content into visible assistant text.
///
/// OpenAI-compatible providers may omit `content` when the assistant only emits
/// tool calls. Acty's transcript stores visible assistant text as a string, so
/// absent content becomes empty visible text at the provider boundary.
fn openai_visible_text(content: Option<String>) -> String {
    content.unwrap_or_default()
}

/// Decodes one OpenAI function tool call.
fn parse_openai_tool_call(call: OpenAiToolCall) -> Result<ToolCall> {
    let arguments = serde_json::from_str(&call.function.arguments).with_context(|| {
        format!(
            "decoding arguments for OpenAI tool call {}",
            call.function.name
        )
    })?;

    Ok(ToolCall {
        id: ToolCallId(call.id),
        name: call.function.name,
        arguments,
    })
}

/// Normalizes an Anthropic response into Acty's LLM response.
fn parse_anthropic_response(response: AnthropicResponse) -> LlmResponse {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in response.content {
        match block {
            AnthropicContentBlock::Text { text: block_text } => {
                text.push_str(&block_text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: ToolCallId(id),
                    name,
                    arguments: input,
                });
            }
            AnthropicContentBlock::Other => {}
        }
    }

    LlmResponse {
        text,
        tool_calls,
        finish_reason: FinishReason::from_provider(response.stop_reason),
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}
