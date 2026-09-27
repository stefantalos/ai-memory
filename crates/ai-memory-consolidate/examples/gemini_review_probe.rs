//! One real auto_improve review against a read-only DB copy, with the raw
//! provider output kept for diagnosis.
//!
//! ```text
//! cargo run -p ai-memory-consolidate --example gemini_review_probe -- \
//!     <db.sqlite> <workspace-uuid> <project-uuid> <session-uuid> <gemini-key-file> <out-dir> [model]
//! ```
//!
//! Makes exactly ONE metered Gemini call. The key is read from a file and
//! never printed. The review path is read-only by contract
//! (`run_auto_improve_review` writes no rows or wiki files), and the reader
//! pool opens SQLite with `SQLITE_OPEN_READ_ONLY`; still, point it at a copy.
//! Writes `<out-dir>/raw.txt` (the text Gemini emitted, complete or cut) and
//! `<out-dir>/report.json`.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use ai_memory_consolidate::{AutoImproveReviewConfig, auto_improve::run_auto_improve_review};
use ai_memory_core::{ProjectId, SessionId, WorkspaceId};
use ai_memory_llm::{
    ChatRequest, ChatResponse, GeminiProvider, LlmError, LlmProvider, LlmResult, PartialText,
};
use ai_memory_store::ReaderPool;
use async_trait::async_trait;
use secrecy::SecretString;

/// Keeps what the inner provider emitted, success or truncation.
struct Tap {
    inner: GeminiProvider,
    raw: Mutex<Option<String>>,
}

#[async_trait]
impl LlmProvider for Tap {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn model(&self) -> &str {
        self.inner.model()
    }
    async fn complete(&self, r: ChatRequest) -> LlmResult<ChatResponse> {
        self.inner.complete(r).await
    }
    async fn complete_structured_raw(
        &self,
        r: ChatRequest,
        s: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        eprintln!(
            "probe: request max_tokens={} system_chars={} user_chars={}",
            r.max_tokens,
            r.system.as_deref().map_or(0, str::len),
            r.messages.iter().map(|m| m.content.len()).sum::<usize>()
        );
        let out = self.inner.complete_structured_raw(r, s).await;
        let raw = match &out {
            Ok(v) => Some(serde_json::to_string_pretty(v).unwrap_or_default()),
            Err(LlmError::Truncated {
                partial: Some(PartialText(t)),
                ..
            }) => Some(t.clone()),
            Err(_) => None,
        };
        *self.raw.lock().unwrap() = raw;
        out
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 7,
        "usage: <db> <ws> <proj> <session> <key-file> <out-dir> [model]"
    );
    let db = PathBuf::from(&args[1]);
    let ws = WorkspaceId::from_str(&args[2])?;
    let proj = ProjectId::from_str(&args[3])?;
    let session = SessionId::from_str(&args[4])?;
    let key = std::fs::read_to_string(&args[5])?.trim().to_string();
    let out_dir = PathBuf::from(&args[6]);
    let model = args
        .get(7)
        .cloned()
        .unwrap_or_else(|| "gemini-2.5-flash".into());
    std::fs::create_dir_all(&out_dir)?;

    let reader = ReaderPool::new(&db, 2)?;
    let tap = Tap {
        inner: GeminiProvider::new(SecretString::from(key), model)?.with_timeout_secs(300),
        raw: Mutex::new(None),
    };
    let tap = Arc::new(tap);
    let started = std::time::Instant::now();
    let report = run_auto_improve_review(
        &reader,
        tap.as_ref(),
        ws,
        proj,
        session,
        AutoImproveReviewConfig::default(),
    )
    .await;
    let elapsed = started.elapsed();
    if let Some(raw) = tap.raw.lock().unwrap().as_ref() {
        std::fs::write(out_dir.join("raw.txt"), raw)?;
        eprintln!("probe: raw output {} bytes", raw.len());
    }
    match report {
        Ok(r) => {
            std::fs::write(
                out_dir.join("report.json"),
                serde_json::to_string_pretty(&r)?,
            )?;
            eprintln!(
                "probe: elapsed={:.1}s est_input_tokens={} proposals={} rejected={} summary={:?} warnings={:?}",
                elapsed.as_secs_f64(),
                r.estimated_input_tokens,
                r.proposals.len(),
                r.rejected_candidates.len(),
                r.summary,
                r.warnings
            );
        }
        Err(e) => eprintln!("probe: review error after {elapsed:?}: {e}"),
    }
    Ok(())
}
