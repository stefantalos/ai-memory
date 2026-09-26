//! LLM error type.

use thiserror::Error;

/// Result alias used throughout the LLM crate.
pub type LlmResult<T> = Result<T, LlmError>;

/// Errors raised by LLM providers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlmError {
    /// Underlying HTTP failure.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    /// Provider returned a non-2xx status.
    #[error("provider error {status}: {body}")]
    Provider {
        /// HTTP status code.
        status: u16,
        /// Response body (truncated).
        body: String,
    },

    /// JSON (de)serialization failure.
    #[error("serde: {0}")]
    Serde(String),

    /// Provider gave a response with unexpected shape (e.g. no tool
    /// use block where structured output was requested).
    #[error("unexpected response shape: {0}")]
    UnexpectedShape(String),

    /// Configured provider lacks the env var we need.
    #[error("provider not configured: {0}")]
    NotConfigured(String),

    /// Provider authentication failed or expired.
    #[error("auth: {0}")]
    Auth(String),

    /// JSON schema for structured output could not be derived.
    #[error("schema: {0}")]
    Schema(String),

    /// The provider answered successfully but produced no usable text
    /// (e.g. a Gemini candidate with no parts). Like [`Self::Truncated`] it
    /// is a zero-yield outcome: paid for, nothing delivered.
    #[error("empty response: {0}")]
    EmptyResponse(String),

    /// A flat-rate lane's per-day allocation is used up and no lane before it
    /// answered. Nothing was sent; the request waits for the allocation to
    /// reset. Not a session failure: callers defer the work to `resets_at_unix`
    /// without spending a retry attempt.
    #[error("flat lane {lane} daily allocation used ({used}/{cap}); resets at unix {resets_at_unix}")]
    AllocationExhausted {
        /// Lane label (`codex-oauth`).
        lane: String,
        /// Requests admitted today.
        used: u32,
        /// Requests allowed per UTC day.
        cap: u32,
        /// Unix second of the next reset (00:00 UTC).
        resets_at_unix: i64,
    },

    /// Every configured LLM lane is paused by its circuit breaker. No
    /// request was sent; the lane, not the request, is unavailable.
    #[error("all LLM lanes paused: {0}")]
    LanesPaused(String),

    /// Provider cut the completion off at its output-token ceiling
    /// (`finish_reason == "length"`) and the truncated text could not be
    /// parsed as the requested structured output.
    ///
    /// Deliberately distinct from [`Self::Serde`]/[`Self::UnexpectedShape`]:
    /// both of those are treated as "shape mismatch, worth a tolerant
    /// retry" by `openai_compat::is_parse_shape_error`. A retry after
    /// truncation would resend the same (or larger) token budget with NO
    /// schema at all and, empirically (FN8-8120, Poolside/Laguna S 2.1,
    /// measured 2026-09-22), truncates again — doubling metered spend for
    /// a call that could not have succeeded. Keeping this variant out of
    /// `is_parse_shape_error`'s match arms is what suppresses that retry.
    #[error("truncated: provider stopped early (finish_reason={finish_reason})")]
    Truncated {
        /// The provider's own `finish_reason` value (normally `"length"`).
        finish_reason: String,
        /// The text the provider did emit before it was cut, when the
        /// provider exposes it. Callers may salvage complete items from it
        /// (auto_improve recovers whole proposals from a valid prefix);
        /// `Debug` prints only its length so an error log never carries the
        /// session-derived body.
        partial: Option<PartialText>,
    },
}

/// Text a provider emitted before an output-limit cut.
///
/// A newtype so `Debug` (and therefore `{err:?}` in logs) reports only the
/// size, never the body.
#[derive(Clone, PartialEq, Eq)]
pub struct PartialText(pub String);

impl std::fmt::Debug for PartialText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PartialText({} bytes)", self.0.len())
    }
}

impl LlmError {
    /// Whether this failure is worth a short, bounded retry.
    ///
    /// True only for errors that a subsequent identical request could plausibly
    /// succeed on: a server-side `Provider` status (`429` or any `5xx`,
    /// including Cloudflare's `52x`), or an `Http` transport timeout / connect
    /// failure. Everything else — auth, schema, a malformed-request `4xx`, a
    /// deserialization or unexpected-shape error — is deterministic: retrying
    /// only burns another expensive call. Callers must keep the retry *short
    /// and bounded* (a few attempts, seconds apart); this is not a license for
    /// tenacity-style 8–128s backoff (see the cognee #2840 lesson in `lib.rs`).
    #[must_use]
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Provider { status, .. } => *status == 429 || (500..=599).contains(status),
            Self::Http(e) => e.is_timeout() || e.is_connect(),
            _ => false,
        }
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_covers_429_and_5xx_only() {
        assert!(
            LlmError::Provider {
                status: 429,
                body: String::new()
            }
            .is_transient()
        );
        for status in [500, 502, 503, 504, 520, 524] {
            assert!(
                LlmError::Provider {
                    status,
                    body: String::new()
                }
                .is_transient(),
                "status {status} should be transient"
            );
        }
        // A malformed-request 4xx (not 429) is deterministic — do not retry.
        for status in [400, 401, 403, 404, 422] {
            assert!(
                !LlmError::Provider {
                    status,
                    body: String::new()
                }
                .is_transient(),
                "status {status} must not be treated as transient"
            );
        }
    }

    #[test]
    fn deterministic_errors_are_not_transient() {
        assert!(!LlmError::Auth("expired".into()).is_transient());
        assert!(!LlmError::Schema("bad".into()).is_transient());
        assert!(!LlmError::Serde("nope".into()).is_transient());
        assert!(!LlmError::UnexpectedShape("no tool block".into()).is_transient());
        assert!(!LlmError::NotConfigured("no key".into()).is_transient());
    }
}
