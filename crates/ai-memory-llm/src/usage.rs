//! Per-request usage and caller attribution, carried task-locally.
//!
//! Structured calls return a bare `serde_json::Value`, so token counts never
//! reach the caller through the return type. Providers instead [`report`]
//! the usage their response carried into a task-local slot, and the lane
//! chain ([`crate::FallbackProvider`]) reads it back per lane call to write
//! the call ledger. Outside a capture scope a report is a no-op.
//!
//! The caller label ("auto_improve", "consolidate", …) is set by the code
//! that issues the request via [`with_caller`], so the ledger can count
//! calls per operation kind without threading a parameter through every
//! provider signature.

use std::cell::{Cell, RefCell};
use std::future::Future;

/// Token usage a provider reported for one HTTP response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReportedUsage {
    /// Prompt / input tokens.
    pub input_tokens: u32,
    /// Output / completion tokens.
    pub output_tokens: u32,
}

tokio::task_local! {
    static USAGE: Cell<Option<ReportedUsage>>;
    static CALLER: RefCell<&'static str>;
}

/// Record the usage of the response just received. Providers call this.
pub fn report(input_tokens: u32, output_tokens: u32) {
    let _ = USAGE.try_with(|slot| {
        slot.set(Some(ReportedUsage {
            input_tokens,
            output_tokens,
        }));
    });
}

/// Run `fut` and return the last usage a provider reported inside it.
pub async fn capture_usage<F: Future>(fut: F) -> (F::Output, Option<ReportedUsage>) {
    USAGE
        .scope(Cell::new(None), async move {
            let out = fut.await;
            let usage = USAGE.with(Cell::take);
            (out, usage)
        })
        .await
}

/// Label every LLM request issued inside `fut` with `caller`.
pub async fn with_caller<F: Future>(caller: &'static str, fut: F) -> F::Output {
    CALLER.scope(RefCell::new(caller), fut).await
}

/// The caller label of the current task, or `"unlabelled"`.
#[must_use]
pub fn current_caller() -> &'static str {
    CALLER.try_with(|c| *c.borrow()).unwrap_or("unlabelled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn usage_is_captured_inside_a_scope_and_ignored_outside() {
        report(1, 2); // no scope: must not panic
        let ((), usage) = capture_usage(async { report(10, 20) }).await;
        assert_eq!(
            usage,
            Some(ReportedUsage {
                input_tokens: 10,
                output_tokens: 20
            })
        );
        let ((), none) = capture_usage(async {}).await;
        assert_eq!(none, None);
    }

    #[tokio::test]
    async fn caller_label_defaults_and_scopes() {
        assert_eq!(current_caller(), "unlabelled");
        let inner = with_caller("auto_improve", async { current_caller() }).await;
        assert_eq!(inner, "auto_improve");
    }
}
