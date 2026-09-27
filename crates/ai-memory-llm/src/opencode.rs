//! OpenCode Zen/Go provider.
//!
//! Thin wrapper around [`OpenAiCompatProvider`] that bakes in the OpenCode
//! Zen API base URL (`https://opencode.ai/zen/go/v1`) and names the provider
//! `"opencode"`. Accepts an `sk-...` API key from `OPENCODE_API_KEY`.

use async_trait::async_trait;
use secrecy::SecretString;

use crate::error::LlmResult;
use crate::openai_compat::OpenAiCompatProvider;
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};

/// Public OpenCode Zen/Go OpenAI-compatible base URL.
pub const OPENCODE_ZEN_BASE_URL: &str = "https://opencode.ai/zen/go/v1";

/// Default model when `AI_MEMORY_LLM_MODEL` is not set.
pub const OPENCODE_DEFAULT_MODEL: &str = "claude-sonnet-4-6";

/// OpenCode Zen/Go LLM provider.
///
/// Routes through `https://opencode.ai/zen/go/v1` using the OpenAI chat
/// completions wire format. Authenticate with the `sk-...` key obtained
/// from <https://opencode.ai/auth>.
pub struct OpenCodeProvider {
    inner: OpenAiCompatProvider,
}

impl OpenCodeProvider {
    /// Construct an OpenCode Zen/Go provider.
    ///
    /// # Errors
    /// Returns a `reqwest::Error` if the HTTP client cannot be built.
    pub fn new(api_key: SecretString, model: impl Into<String>) -> LlmResult<Self> {
        Self::new_with_base_url(api_key, model, OPENCODE_ZEN_BASE_URL)
    }

    fn new_with_base_url(
        api_key: SecretString,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> LlmResult<Self> {
        let inner = OpenAiCompatProvider::new(base_url, Some(api_key), model.into())?
            .with_client_headers(
                concat!("ai-memory/", env!("CARGO_PKG_VERSION")),
                "x-opencode-session",
            );
        Ok(Self { inner })
    }

    #[cfg(test)]
    fn with_strict(mut self, strict: bool) -> Self {
        self.inner = self.inner.with_strict(strict);
        self
    }

    /// Override the per-request timeout on the wrapped
    /// [`OpenAiCompatProvider`]. The factory calls this with
    /// `ProviderConfig::request_timeout_secs`.
    #[must_use]
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.inner = self.inner.with_timeout_secs(secs);
        self
    }

    /// Forward reasoning effort to the OpenAI-compatible Zen/Go client.
    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: Option<crate::ReasoningEffort>) -> Self {
        self.inner = self.inner.with_reasoning_effort(effort);
        self
    }
}

#[async_trait]
impl LlmProvider for OpenCodeProvider {
    fn name(&self) -> &'static str {
        "opencode"
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        self.complete_with_operation_id(request, LlmOperationId::new())
            .await
    }

    async fn complete_with_operation_id(
        &self,
        request: ChatRequest,
        operation_id: LlmOperationId,
    ) -> LlmResult<ChatResponse> {
        self.inner
            .complete_with_operation_id(request, operation_id)
            .await
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        self.complete_structured_raw_with_operation_id(request, schema, LlmOperationId::new())
            .await
    }

    async fn complete_structured_raw_with_operation_id(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
        operation_id: LlmOperationId,
    ) -> LlmResult<serde_json::Value> {
        self.inner
            .complete_structured_raw_with_operation_id(request, schema, operation_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn response_with_content(content: &str) -> serde_json::Value {
        json!({
            "model": "model-x",
            "choices": [{
                "message": { "content": content },
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 },
        })
    }

    fn header_value<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
        request
            .headers
            .get(name)
            .and_then(|value| value.to_str().ok())
    }

    #[test]
    fn provider_reports_opencode_name_and_configured_model() {
        let provider = OpenCodeProvider::new(SecretString::from("sk-test"), "model-x").unwrap();
        assert_eq!(provider.name(), "opencode");
        assert_eq!(provider.model(), "model-x");
    }

    #[test]
    fn public_constants_point_at_zen_base_url() {
        assert_eq!(OPENCODE_ZEN_BASE_URL, "https://opencode.ai/zen/go/v1");
        assert!(!OPENCODE_DEFAULT_MODEL.is_empty());
    }

    #[tokio::test]
    async fn completion_identifies_ai_memory_and_its_logical_operation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response_with_content("ok")))
            .mount(&server)
            .await;

        let provider = OpenCodeProvider::new_with_base_url(
            SecretString::from("sk-test"),
            "model-x",
            server.uri(),
        )
        .unwrap();
        let operation_id = LlmOperationId::new();
        provider
            .complete_with_operation_id(ChatRequest::user_prompt("hello"), operation_id)
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            header_value(&requests[0], "user-agent"),
            Some(concat!("ai-memory/", env!("CARGO_PKG_VERSION")))
        );
        assert_eq!(
            header_value(&requests[0], "x-opencode-session"),
            Some(operation_id.to_string().as_str())
        );
    }

    #[tokio::test]
    async fn structured_fallback_reuses_its_logical_operation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(|request: &Request| {
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).expect("request body is JSON");
                if body.get("response_format").is_some() {
                    ResponseTemplate::new(400)
                        .set_body_string("unsupported parameter: response_format")
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(response_with_content(r#"{"ok":true}"#))
                }
            })
            .mount(&server)
            .await;

        let provider = OpenCodeProvider::new_with_base_url(
            SecretString::from("sk-test"),
            "model-x",
            server.uri(),
        )
        .unwrap()
        .with_strict(true);
        let operation_id = LlmOperationId::new();
        let value = provider
            .complete_structured_raw_with_operation_id(
                ChatRequest::user_prompt("emit JSON"),
                json!({
                    "type": "object",
                    "properties": { "ok": { "type": "boolean" } },
                    "required": ["ok"],
                }),
                operation_id,
            )
            .await
            .unwrap();

        assert_eq!(value, json!({"ok": true}));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 2);
        let expected_id = operation_id.to_string();
        assert_eq!(
            header_value(&requests[0], "x-opencode-session"),
            Some(expected_id.as_str())
        );
        assert_eq!(
            header_value(&requests[1], "x-opencode-session"),
            Some(expected_id.as_str())
        );
    }
}
