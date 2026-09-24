//! Runtime primary→secondary LLM fallback with a circuit breaker.
//!
//! The lane is chosen once, at service start, by whatever launches the
//! server. Without this wrapper a quota wall (`429 usage limit exceeded`) or a
//! provider outage on that lane stalls every LLM-backed pass until a restart,
//! even when an approved second lane is configured.
//!
//! [`FallbackProvider`] wraps a primary and a secondary:
//!
//! * a **transient / provider-side** failure of the primary (`429`, `5xx`,
//!   transport timeout / connect failure, output truncation) is retried once
//!   on the secondary for that same request;
//! * a **deterministic** failure (any other `4xx`, auth, schema, serde,
//!   unexpected shape) is returned unchanged — another provider would only
//!   burn a second call on a request that cannot succeed as written;
//! * a `429` opens the circuit immediately, and [`CIRCUIT_5XX_THRESHOLD`]
//!   consecutive server/transport failures open it too. While open, the
//!   primary is skipped entirely; once the cooldown elapses the next request
//!   probes the primary again (half-open) and a success closes the circuit.
//!
//! Attribution. `name()` / `model()` are constant and always report the
//! primary: they describe the *configured* lane. Which provider actually
//! answered a given request is recorded per task through
//! [`capture_responder`], which callers that persist a provider label (the
//! auto_improve report) wrap around their call. It is task-local, so
//! concurrent callers sharing one `Arc` never see each other's answer.
//!
//! `Retry-After` is not honoured: [`LlmError::Provider`] carries only the
//! status and body, so the header never reaches this layer. The cooldown is
//! the fixed [`DEFAULT_COOLDOWN`] (or whatever the caller configures).

use std::cell::RefCell;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::error::{LlmError, LlmResult};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};

/// Consecutive `5xx` / transport failures that open the circuit.
pub const CIRCUIT_5XX_THRESHOLD: u32 = 3;

/// How long the primary is skipped once the circuit opens.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// The provider that actually produced (or failed) a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Responder {
    /// Provider wire name (`"gemini"`, `"openai-compat"`, …).
    pub provider: &'static str,
    /// Model id that was hit.
    pub model: String,
}

tokio::task_local! {
    static RESPONDER: RefCell<Option<Responder>>;
}

/// Run `fut` and report which provider a [`FallbackProvider`] used inside it.
///
/// Returns `None` when no fallback-aware provider ran (a plain provider), so
/// callers keep their existing `llm.name()` / `llm.model()` default. When a
/// fallback provider ran several requests, the last one wins.
pub async fn capture_responder<F: Future>(fut: F) -> (F::Output, Option<Responder>) {
    RESPONDER
        .scope(RefCell::new(None), async move {
            let out = fut.await;
            let responder = RESPONDER.with(|cell| cell.borrow_mut().take());
            (out, responder)
        })
        .await
}

fn record_responder(provider: &dyn LlmProvider) {
    // Outside a capture scope there is nobody to tell; that is not an error.
    let _ = RESPONDER.try_with(|cell| {
        *cell.borrow_mut() = Some(Responder {
            provider: provider.name(),
            model: provider.model().to_string(),
        });
    });
}

/// Monotonic clock, injectable so tests can move time.
pub trait Clock: Send + Sync {
    /// Current instant.
    fn now(&self) -> Instant;
}

/// The real clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// How a primary failure is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    /// `429`: quota / rate wall. Fall back and open the circuit now.
    Quota,
    /// `5xx` or transport timeout / connect: fall back, count toward the circuit.
    Server,
    /// Output truncated by the provider: fall back, circuit untouched
    /// (the lane is healthy; this request was too big for it).
    Truncated,
    /// Deterministic: return as-is, no fallback, circuit untouched.
    Fatal,
}

fn classify(err: &LlmError) -> FailureClass {
    match err {
        LlmError::Provider { status: 429, .. } => FailureClass::Quota,
        LlmError::Provider { status, .. } if (500..=599).contains(status) => FailureClass::Server,
        LlmError::Http(e) if e.is_timeout() || e.is_connect() => FailureClass::Server,
        LlmError::Truncated { .. } => FailureClass::Truncated,
        _ => FailureClass::Fatal,
    }
}

#[derive(Debug, Default)]
struct Circuit {
    open_until: Option<Instant>,
    consecutive_server_failures: u32,
}

/// Primary provider with a runtime fallback to a secondary. See module docs.
pub struct FallbackProvider {
    primary: Arc<dyn LlmProvider>,
    secondary: Arc<dyn LlmProvider>,
    cooldown: Duration,
    clock: Arc<dyn Clock>,
    circuit: Mutex<Circuit>,
}

impl FallbackProvider {
    /// Wrap `primary` with `secondary` as its runtime fallback.
    #[must_use]
    pub fn new(primary: Arc<dyn LlmProvider>, secondary: Arc<dyn LlmProvider>) -> Self {
        Self {
            primary,
            secondary,
            cooldown: DEFAULT_COOLDOWN,
            clock: Arc::new(SystemClock),
            circuit: Mutex::new(Circuit::default()),
        }
    }

    /// Override the circuit cooldown.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Inject a clock (tests).
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Circuit> {
        // A poisoned lock only means another request panicked mid-update;
        // the circuit state is still a valid (if stale) value.
        self.circuit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether the primary should be skipped for this request.
    fn primary_skipped(&self) -> bool {
        let now = self.clock.now();
        let mut c = self.lock();
        match c.open_until {
            Some(until) if now < until => true,
            Some(_) => {
                // Cooldown elapsed: half-open. Let this request probe the
                // primary; its outcome re-opens or closes the circuit.
                c.open_until = None;
                false
            }
            None => false,
        }
    }

    fn on_primary_success(&self) {
        let mut c = self.lock();
        c.open_until = None;
        c.consecutive_server_failures = 0;
    }

    fn on_primary_failure(&self, class: FailureClass) {
        let now = self.clock.now();
        let mut c = self.lock();
        match class {
            FailureClass::Quota => {
                c.open_until = Some(now + self.cooldown);
                c.consecutive_server_failures = 0;
            }
            FailureClass::Server => {
                c.consecutive_server_failures += 1;
                if c.consecutive_server_failures >= CIRCUIT_5XX_THRESHOLD {
                    c.open_until = Some(now + self.cooldown);
                    c.consecutive_server_failures = 0;
                }
            }
            FailureClass::Truncated | FailureClass::Fatal => {}
        }
    }

    /// Core routing, shared by all four trait entry points.
    async fn route<T, P, S, PF, SF>(&self, call_primary: P, call_secondary: S) -> LlmResult<T>
    where
        P: FnOnce(Arc<dyn LlmProvider>) -> PF,
        S: FnOnce(Arc<dyn LlmProvider>) -> SF,
        PF: Future<Output = LlmResult<T>>,
        SF: Future<Output = LlmResult<T>>,
    {
        if self.primary_skipped() {
            record_responder(self.secondary.as_ref());
            return call_secondary(self.secondary.clone()).await;
        }
        record_responder(self.primary.as_ref());
        match call_primary(self.primary.clone()).await {
            Ok(value) => {
                self.on_primary_success();
                Ok(value)
            }
            Err(err) => {
                let class = classify(&err);
                self.on_primary_failure(class);
                if class == FailureClass::Fatal {
                    return Err(err);
                }
                tracing::warn!(
                    primary = self.primary.name(),
                    primary_model = self.primary.model(),
                    secondary = self.secondary.name(),
                    secondary_model = self.secondary.model(),
                    error = %err,
                    "primary LLM failed transiently; retrying this request on the fallback provider",
                );
                record_responder(self.secondary.as_ref());
                call_secondary(self.secondary.clone()).await
            }
        }
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        self.primary.name()
    }

    fn model(&self) -> &str {
        self.primary.model()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        let again = request.clone();
        self.route(
            |p| async move { p.complete(request).await },
            |s| async move { s.complete(again).await },
        )
        .await
    }

    async fn complete_with_operation_id(
        &self,
        request: ChatRequest,
        operation_id: LlmOperationId,
    ) -> LlmResult<ChatResponse> {
        let again = request.clone();
        self.route(
            |p| async move { p.complete_with_operation_id(request, operation_id).await },
            |s| async move { s.complete_with_operation_id(again, operation_id).await },
        )
        .await
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        let (again, schema_again) = (request.clone(), schema.clone());
        self.route(
            |p| async move { p.complete_structured_raw(request, schema).await },
            |s| async move { s.complete_structured_raw(again, schema_again).await },
        )
        .await
    }

    async fn complete_structured_raw_with_operation_id(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
        operation_id: LlmOperationId,
    ) -> LlmResult<serde_json::Value> {
        let (again, schema_again) = (request.clone(), schema.clone());
        self.route(
            |p| async move {
                p.complete_structured_raw_with_operation_id(request, schema, operation_id)
                    .await
            },
            |s| async move {
                s.complete_structured_raw_with_operation_id(again, schema_again, operation_id)
                    .await
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake whose structured answer is scripted per call; `None` = success.
    struct Scripted {
        name: &'static str,
        model: &'static str,
        errors: Mutex<Vec<Option<LlmError>>>,
        calls: AtomicUsize,
    }

    impl Scripted {
        fn new(
            name: &'static str,
            model: &'static str,
            errors: Vec<Option<LlmError>>,
        ) -> Arc<Self> {
            Arc::new(Self {
                name,
                model,
                errors: Mutex::new(errors),
                calls: AtomicUsize::new(0),
            })
        }
        fn ok(name: &'static str, model: &'static str) -> Arc<Self> {
            Self::new(name, model, Vec::new())
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn next(&self) -> LlmResult<serde_json::Value> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut errs = self.errors.lock().unwrap();
            let e = if errs.is_empty() {
                None
            } else {
                errs.remove(0)
            };
            match e {
                Some(e) => Err(e),
                None => Ok(serde_json::json!({ "answered_by": self.name })),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for Scripted {
        fn name(&self) -> &'static str {
            self.name
        }
        fn model(&self) -> &str {
            self.model
        }
        async fn complete(&self, _r: ChatRequest) -> LlmResult<ChatResponse> {
            self.next().map(|v| ChatResponse {
                text: v.to_string(),
                usage: None,
                model: self.model.into(),
            })
        }
        async fn complete_structured_raw(
            &self,
            _r: ChatRequest,
            _s: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            self.next()
        }
    }

    struct FakeClock(Mutex<Instant>);
    impl FakeClock {
        fn advance(&self, d: Duration) {
            *self.0.lock().unwrap() += d;
        }
    }
    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }
    }

    fn status(code: u16) -> LlmError {
        LlmError::Provider {
            status: code,
            body: format!("status {code}"),
        }
    }

    fn req() -> ChatRequest {
        ChatRequest {
            system: None,
            messages: Vec::new(),
            max_tokens: 16,
            temperature: None,
        }
    }

    fn build(
        primary: &Arc<Scripted>,
        secondary: &Arc<Scripted>,
    ) -> (FallbackProvider, Arc<FakeClock>) {
        let clock = Arc::new(FakeClock(Mutex::new(Instant::now())));
        let p: Arc<dyn LlmProvider> = primary.clone();
        let s: Arc<dyn LlmProvider> = secondary.clone();
        let f = FallbackProvider::new(p, s)
            .with_cooldown(Duration::from_secs(900))
            .with_clock(clock.clone());
        (f, clock)
    }

    async fn call(f: &FallbackProvider) -> (LlmResult<serde_json::Value>, Option<Responder>) {
        capture_responder(f.complete_structured_raw(req(), serde_json::json!({}))).await
    }

    #[tokio::test]
    async fn primary_429_falls_back_and_is_attributed_to_secondary() {
        let primary = Scripted::new(
            "openai-compat",
            "poolside/laguna-s-2.1",
            vec![Some(status(429))],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);

        let (out, who) = call(&f).await;
        assert_eq!(out.unwrap()["answered_by"], "gemini");
        assert_eq!(
            who,
            Some(Responder {
                provider: "gemini",
                model: "gemini-2.5-flash".into()
            })
        );
        assert_eq!((primary.calls(), secondary.calls()), (1, 1));
    }

    #[tokio::test]
    async fn primary_500_falls_back_before_circuit_opens_and_is_attributed() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(500))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);

        let (out, who) = call(&f).await;
        assert!(out.is_ok());
        assert_eq!(who.unwrap().provider, "gemini");
        // One 5xx must not open the circuit: the next call tries the primary.
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "openai-compat");
        assert_eq!(primary.calls(), 2);
    }

    #[tokio::test]
    async fn truncation_falls_back_without_touching_the_circuit() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![Some(LlmError::Truncated {
                finish_reason: "length".into(),
            })],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "gemini");
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "openai-compat");
    }

    #[tokio::test]
    async fn primary_400_does_not_fall_back() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(400))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);

        let (out, who) = call(&f).await;
        match out {
            Err(LlmError::Provider { status: 400, .. }) => {}
            other => panic!("expected the primary's 400 unchanged, got {other:?}"),
        }
        assert_eq!(who.unwrap().provider, "openai-compat");
        assert_eq!(secondary.calls(), 0);
    }

    #[tokio::test]
    async fn schema_and_serde_errors_do_not_fall_back() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![
                Some(LlmError::Schema("bad".into())),
                Some(LlmError::Serde("nope".into())),
                Some(LlmError::UnexpectedShape("no tool".into())),
            ],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        for _ in 0..3 {
            assert!(call(&f).await.0.is_err());
        }
        assert_eq!(secondary.calls(), 0);
    }

    #[tokio::test]
    async fn circuit_opens_after_429_and_skips_primary() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);

        call(&f).await.0.unwrap();
        let (out, who) = call(&f).await;
        assert!(out.is_ok());
        assert_eq!(who.unwrap().provider, "gemini");
        assert_eq!(
            primary.calls(),
            1,
            "an open circuit must not touch the primary"
        );
        assert_eq!(secondary.calls(), 2);
    }

    #[tokio::test]
    async fn cooldown_expiry_probes_primary_again() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, clock) = build(&primary, &secondary);

        call(&f).await.0.unwrap();
        clock.advance(Duration::from_secs(899));
        call(&f).await.0.unwrap();
        assert_eq!(primary.calls(), 1, "still inside the cooldown");

        clock.advance(Duration::from_secs(2));
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "openai-compat");
        assert_eq!(
            primary.calls(),
            2,
            "cooldown elapsed: the primary is probed"
        );
        // The probe succeeded, so the circuit is closed again.
        call(&f).await.0.unwrap();
        assert_eq!(primary.calls(), 3);
    }

    #[tokio::test]
    async fn failed_probe_reopens_the_circuit() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![Some(status(429)), Some(status(429))],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, clock) = build(&primary, &secondary);
        call(&f).await.0.unwrap();
        clock.advance(Duration::from_secs(901));
        call(&f).await.0.unwrap(); // probe → 429 → fallback, reopen
        call(&f).await.0.unwrap();
        assert_eq!(primary.calls(), 2);
    }

    #[tokio::test]
    async fn three_consecutive_5xx_open_the_circuit() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![Some(status(500)), Some(status(502)), Some(status(503))],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        for _ in 0..3 {
            call(&f).await.0.unwrap();
        }
        assert_eq!(primary.calls(), 3);
        call(&f).await.0.unwrap();
        assert_eq!(
            primary.calls(),
            3,
            "third consecutive 5xx opened the circuit"
        );
    }

    #[tokio::test]
    async fn a_success_resets_the_5xx_count() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![
                Some(status(500)),
                Some(status(500)),
                None,
                Some(status(500)),
            ],
        );
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        for _ in 0..4 {
            call(&f).await.0.unwrap();
        }
        call(&f).await.0.unwrap();
        assert_eq!(primary.calls(), 5, "the success in between reset the count");
    }

    #[tokio::test]
    async fn secondary_failure_is_returned() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::new("gemini", "gemini-2.5-flash", vec![Some(status(503))]);
        let (f, _) = build(&primary, &secondary);
        match call(&f).await.0 {
            Err(LlmError::Provider { status: 503, .. }) => {}
            other => panic!("expected the secondary's 503, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn name_and_model_stay_the_configured_primary() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        call(&f).await.0.unwrap();
        assert_eq!((f.name(), f.model()), ("openai-compat", "laguna"));
    }

    #[tokio::test]
    async fn plain_provider_reports_no_responder() {
        let plain = Scripted::ok("openai-compat", "laguna");
        let (out, who) =
            capture_responder(plain.complete_structured_raw(req(), serde_json::json!({}))).await;
        assert!(out.is_ok());
        assert_eq!(who, None);
    }

    #[tokio::test]
    async fn operation_id_entry_point_also_falls_back() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _) = build(&primary, &secondary);
        let (out, who) = capture_responder(f.complete_structured_raw_with_operation_id(
            req(),
            serde_json::json!({}),
            LlmOperationId::new(),
        ))
        .await;
        assert_eq!(out.unwrap()["answered_by"], "gemini");
        assert_eq!(who.unwrap().provider, "gemini");
    }
}
