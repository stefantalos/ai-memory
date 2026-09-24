//! Runtime LLM lane chain with a circuit breaker per lane.
//!
//! The lane is chosen once, at service start, by whatever launches the
//! server. Without this wrapper a quota wall (`429 usage limit exceeded`) or a
//! provider outage on that lane stalls every LLM-backed pass until a restart,
//! even when approved further lanes are configured.
//!
//! [`FallbackProvider`] holds an ordered list of [`Lane`]s — e.g. Poolside
//! key A, Poolside key B, Gemini — and each request walks it in order:
//!
//! * a **transient / provider-side** failure (`429`, `5xx`, transport
//!   timeout / connect failure) or a **zero-yield** outcome (output cut at the
//!   limit, an empty answer) moves the same request to the next lane;
//! * a **deterministic** failure (any other `4xx`, auth, schema, serde,
//!   unexpected shape) is returned unchanged — another lane would only burn a
//!   second call on a request that cannot succeed as written;
//! * every lane has its own breaker. A `429` opens it at once;
//!   [`CIRCUIT_5XX_THRESHOLD`] consecutive server/transport failures open it;
//!   [`ZERO_YIELD_THRESHOLD`] consecutive zero-yield outcomes open it too —
//!   a truncated "success" is paid for, so it must be able to stop the spend
//!   exactly like an error does (measured 2026-09-24: 41/41 Gemini runs
//!   truncated and nothing paused the lane). While open the lane is skipped;
//!   after the cooldown the next request probes it (half-open) and any
//!   failure of that probe re-opens it at once.
//! * when every lane is paused no request is sent at all:
//!   [`LlmError::LanesPaused`].
//!
//! Every lane call is reported to an optional [`LaneObserver`] (the call
//! ledger, [`crate::ledger`]) with the tokens the provider reported, and every
//! breaker that opens is reported as a [`BreakerEvent`] plus a `warn!` alarm
//! line. Lanes are named by label and key fingerprint, never by key.
//!
//! Attribution. `name()` / `model()` are constant and always report the
//! first lane: they describe the *configured* lane. Which lane actually
//! answered a given request is recorded per task through
//! [`capture_responder`]. It is task-local, so concurrent callers sharing one
//! `Arc` never see each other's answer.
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
use crate::ledger::{
    BreakerEvent, CallRecord, LaneObserver, estimate_cost, funding_for, now_rfc3339,
};
use crate::metered_gate::{GateRequest, GateVerdict, MeteredGate};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};
use crate::usage::{ReportedUsage, capture_usage, current_caller};

/// Consecutive `5xx` / transport failures that open a lane's circuit.
pub const CIRCUIT_5XX_THRESHOLD: u32 = 3;

/// Consecutive zero-yield outcomes (truncated or empty) that open a lane's
/// circuit.
pub const ZERO_YIELD_THRESHOLD: u32 = 3;

/// How long a lane is skipped once its circuit opens.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// The lane that actually produced (or failed) a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Responder {
    /// Provider wire name (`"gemini"`, `"openai-compat"`, …).
    pub provider: &'static str,
    /// Model id that was hit.
    pub model: String,
    /// Lane label (`"key-a"`, `"key-b"`, `"gemini"`, …).
    pub lane: String,
}

tokio::task_local! {
    static RESPONDER: RefCell<Option<Responder>>;
}

/// Run `fut` and report which lane a [`FallbackProvider`] used inside it.
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

fn record_responder(lane: &Lane) {
    // Outside a capture scope there is nobody to tell; that is not an error.
    let _ = RESPONDER.try_with(|cell| {
        *cell.borrow_mut() = Some(Responder {
            provider: lane.provider.name(),
            model: lane.provider.model().to_string(),
            lane: lane.label.clone(),
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

/// One provider in the chain.
pub struct Lane {
    provider: Arc<dyn LlmProvider>,
    label: String,
    key_fp: Option<String>,
}

impl Lane {
    /// A lane named `label` (a short, key-free name such as `key-a`).
    #[must_use]
    pub fn new(provider: Arc<dyn LlmProvider>, label: impl Into<String>) -> Self {
        Self {
            provider,
            label: label.into(),
            key_fp: None,
        }
    }

    /// Attach the key's fingerprint ([`crate::ledger::key_fingerprint`]) for
    /// the ledger. Never pass the key itself.
    #[must_use]
    pub fn with_key_fp(mut self, fp: Option<String>) -> Self {
        self.key_fp = fp;
        self
    }

    /// The lane label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// How a lane failure is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    /// `429`: quota / rate wall. Next lane, open this lane's circuit now.
    Quota,
    /// `5xx` or transport timeout / connect: next lane, count toward the circuit.
    Server,
    /// Paid for, nothing usable (truncated or empty): next lane, count toward
    /// the zero-yield circuit.
    ZeroYield,
    /// Deterministic: return as-is, no further lane, circuit untouched.
    Fatal,
}

fn classify(err: &LlmError) -> FailureClass {
    match err {
        LlmError::Provider { status: 429, .. } => FailureClass::Quota,
        LlmError::Provider { status, .. } if (500..=599).contains(status) => FailureClass::Server,
        LlmError::Http(e) if e.is_timeout() || e.is_connect() => FailureClass::Server,
        LlmError::Truncated { .. } | LlmError::EmptyResponse(_) => FailureClass::ZeroYield,
        _ => FailureClass::Fatal,
    }
}

fn outcome_label(result: &Result<bool, &LlmError>) -> (&'static str, Option<u16>) {
    match result {
        Ok(false) => ("ok", None),
        Ok(true) => ("empty", None),
        Err(LlmError::Provider { status, .. }) if *status == 429 => ("quota", Some(*status)),
        Err(LlmError::Provider { status, .. }) if (500..=599).contains(status) => {
            ("server", Some(*status))
        }
        Err(LlmError::Provider { status, .. }) => ("fatal", Some(*status)),
        Err(LlmError::Truncated { .. }) => ("truncated", None),
        Err(LlmError::EmptyResponse(_)) => ("empty", None),
        Err(e) if classify(e) == FailureClass::Server => ("server", None),
        Err(_) => ("fatal", None),
    }
}

#[derive(Debug, Default)]
struct Circuit {
    open_until: Option<Instant>,
    half_open: bool,
    consecutive_server_failures: u32,
    consecutive_zero_yield: u32,
}

struct LaneState {
    lane: Lane,
    circuit: Mutex<Circuit>,
}

/// What a lane call produced, for the routing loop.
trait YieldCheck {
    /// Whether a successful value carries nothing usable.
    fn is_empty_yield(&self) -> bool;
}

impl YieldCheck for ChatResponse {
    fn is_empty_yield(&self) -> bool {
        self.text.trim().is_empty()
    }
}

impl YieldCheck for serde_json::Value {
    fn is_empty_yield(&self) -> bool {
        match self {
            serde_json::Value::Null => true,
            serde_json::Value::Object(m) => m.is_empty(),
            _ => false,
        }
    }
}

/// Ordered lane chain with a breaker per lane. See module docs.
pub struct FallbackProvider {
    lanes: Vec<LaneState>,
    cooldown: Duration,
    zero_yield_threshold: u32,
    clock: Arc<dyn Clock>,
    observer: Option<Arc<dyn LaneObserver>>,
    gate: Option<Arc<dyn MeteredGate>>,
    gated_callers: Vec<String>,
}

impl FallbackProvider {
    /// Two lanes: `primary`, then `secondary` as its runtime fallback.
    #[must_use]
    pub fn new(primary: Arc<dyn LlmProvider>, secondary: Arc<dyn LlmProvider>) -> Self {
        Self::chain(Lane::new(primary, "primary"), vec![Lane::new(secondary, "secondary")])
    }

    /// A chain of `first` followed by `rest`, tried in order.
    #[must_use]
    pub fn chain(first: Lane, rest: Vec<Lane>) -> Self {
        let lanes = std::iter::once(first)
            .chain(rest)
            .map(|lane| LaneState {
                lane,
                circuit: Mutex::new(Circuit::default()),
            })
            .collect();
        Self {
            lanes,
            cooldown: DEFAULT_COOLDOWN,
            zero_yield_threshold: ZERO_YIELD_THRESHOLD,
            clock: Arc::new(SystemClock),
            observer: None,
            gate: None,
            gated_callers: Vec::new(),
        }
    }

    /// Ask `gate` (Jev) before any request from one of `callers` reaches a
    /// metered lane. Availability stays deterministic; the gate only decides
    /// whether the request is worth paying for. See [`crate::metered_gate`].
    #[must_use]
    pub fn with_metered_gate(mut self, gate: Arc<dyn MeteredGate>, callers: Vec<String>) -> Self {
        self.gate = Some(gate);
        self.gated_callers = callers;
        self
    }

    /// Override the circuit cooldown.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Override how many consecutive zero-yield outcomes open a lane.
    #[must_use]
    pub fn with_zero_yield_threshold(mut self, threshold: u32) -> Self {
        self.zero_yield_threshold = threshold.max(1);
        self
    }

    /// Inject a clock (tests).
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Report every lane call and breaker event to `observer`.
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<dyn LaneObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Lane labels in order (for startup logging).
    #[must_use]
    pub fn lane_labels(&self) -> Vec<String> {
        self.lanes.iter().map(|l| l.lane.label.clone()).collect()
    }

    fn lock(state: &LaneState) -> std::sync::MutexGuard<'_, Circuit> {
        // A poisoned lock only means another request panicked mid-update;
        // the circuit state is still a valid (if stale) value.
        state
            .circuit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether this lane should be skipped for this request.
    fn skipped(&self, state: &LaneState) -> bool {
        let now = self.clock.now();
        let mut c = Self::lock(state);
        match c.open_until {
            Some(until) if now < until => true,
            Some(_) => {
                // Cooldown elapsed: half-open. Let this request probe the
                // lane; its outcome re-opens or closes the circuit.
                c.open_until = None;
                c.half_open = true;
                false
            }
            None => false,
        }
    }

    fn on_success(state: &LaneState) {
        let mut c = Self::lock(state);
        *c = Circuit::default();
    }

    fn on_failure(&self, state: &LaneState, class: FailureClass) {
        let now = self.clock.now();
        let tripped: Option<(&'static str, u32)> = {
            let mut c = Self::lock(state);
            let was_half_open = std::mem::take(&mut c.half_open);
            let trip = match class {
                FailureClass::Quota => Some(("quota", 1)),
                FailureClass::Server => {
                    c.consecutive_server_failures += 1;
                    (was_half_open || c.consecutive_server_failures >= CIRCUIT_5XX_THRESHOLD)
                        .then_some(("server", c.consecutive_server_failures))
                }
                FailureClass::ZeroYield => {
                    c.consecutive_zero_yield += 1;
                    (was_half_open || c.consecutive_zero_yield >= self.zero_yield_threshold)
                        .then_some(("zero_yield", c.consecutive_zero_yield))
                }
                FailureClass::Fatal => None,
            };
            if trip.is_some() {
                c.open_until = Some(now + self.cooldown);
                c.consecutive_server_failures = 0;
                c.consecutive_zero_yield = 0;
            }
            trip
        };
        if let Some((reason, consecutive)) = tripped {
            let lane = &state.lane;
            tracing::warn!(
                lane = %lane.label,
                provider = lane.provider.name(),
                model = lane.provider.model(),
                reason,
                consecutive,
                cooldown_secs = self.cooldown.as_secs(),
                "ALARM: LLM lane paused by its circuit breaker",
            );
            if let Some(observer) = &self.observer {
                observer.on_breaker_open(&BreakerEvent {
                    ts: now_rfc3339(),
                    kind: "breaker_open",
                    lane: lane.label.clone(),
                    provider: lane.provider.name().to_string(),
                    model: lane.provider.model().to_string(),
                    reason,
                    consecutive,
                    cooldown_secs: self.cooldown.as_secs(),
                });
            }
        }
    }

    fn observe_call(
        &self,
        lane: &Lane,
        outcome: (&'static str, Option<u16>),
        usage: Option<ReportedUsage>,
    ) {
        let Some(observer) = &self.observer else {
            return;
        };
        let provider = lane.provider.name();
        let model = lane.provider.model();
        observer.on_call(&CallRecord {
            ts: now_rfc3339(),
            kind: "call",
            caller: current_caller().to_string(),
            lane: lane.label.clone(),
            provider: provider.to_string(),
            model: model.to_string(),
            key_fp: lane.key_fp.clone(),
            outcome: outcome.0,
            status: outcome.1,
            input_tokens: usage.map(|u| u.input_tokens),
            output_tokens: usage.map(|u| u.output_tokens),
            cost_usd_est: usage
                .and_then(|u| estimate_cost(provider, model, u.input_tokens, u.output_tokens)),
            funding: funding_for(provider, model),
        });
    }

    /// Whether this request must pass the value gate before `lane`.
    fn gate_applies(&self, lane: &Lane) -> Option<&Arc<dyn MeteredGate>> {
        let gate = self.gate.as_ref()?;
        let caller = current_caller();
        (crate::ledger::is_metered(lane.provider.name())
            && self.gated_callers.iter().any(|c| c == caller))
        .then_some(gate)
    }

    fn observe_gate(&self, lane: &Lane, verdict: &GateVerdict) {
        if verdict.admit {
            tracing::info!(lane = %lane.label, reason = %verdict.reason, "metered lane admitted by value gate");
        } else {
            tracing::warn!(
                lane = %lane.label,
                reason = %verdict.reason,
                "metered lane skipped by value gate (skipped-by-jev); the request waits for a free lane",
            );
        }
        let Some(observer) = &self.observer else {
            return;
        };
        observer.on_call(&CallRecord {
            ts: now_rfc3339(),
            kind: "call",
            caller: current_caller().to_string(),
            lane: lane.label.clone(),
            provider: "jev".into(),
            model: format!("value-gate for {}", lane.provider.model()),
            key_fp: None,
            outcome: if verdict.admit {
                "gate_admitted"
            } else {
                "gate_declined"
            },
            status: None,
            input_tokens: None,
            output_tokens: None,
            cost_usd_est: verdict.cost_usd,
            funding: "Jev value gate (OpenRouter, recorded by floo authoriseMeteredCall)",
        });
    }

    /// Core routing, shared by all four trait entry points.
    async fn route<T, C, CF>(&self, evidence: String, call: C) -> LlmResult<T>
    where
        T: YieldCheck,
        C: Fn(Arc<dyn LlmProvider>) -> CF,
        CF: Future<Output = LlmResult<T>>,
    {
        let mut last_err: Option<LlmError> = None;
        let mut empty_ok: Option<T> = None;
        let mut skipped: Vec<&str> = Vec::new();
        let mut declined: Option<String> = None;
        for (index, state) in self.lanes.iter().enumerate() {
            if self.skipped(state) {
                skipped.push(&state.lane.label);
                continue;
            }
            let lane = &state.lane;
            if let Some(gate) = self.gate_applies(lane) {
                let verdict = gate
                    .judge(GateRequest {
                        caller: current_caller(),
                        provider: lane.provider.name(),
                        model: lane.provider.model(),
                        evidence: &evidence,
                    })
                    .await;
                self.observe_gate(lane, &verdict);
                if !verdict.admit {
                    declined = Some(verdict.reason);
                    continue;
                }
            }
            record_responder(lane);
            let (result, usage) = capture_usage(call(lane.provider.clone())).await;
            let is_last = index + 1 == self.lanes.len();
            match result {
                Ok(value) => {
                    let empty = value.is_empty_yield();
                    self.observe_call(lane, outcome_label(&Ok(empty)), usage);
                    if !empty {
                        Self::on_success(state);
                        return Ok(value);
                    }
                    self.on_failure(state, FailureClass::ZeroYield);
                    if is_last {
                        return Ok(value);
                    }
                    empty_ok = Some(value);
                }
                Err(err) => {
                    self.observe_call(lane, outcome_label(&Err(&err)), usage);
                    let class = classify(&err);
                    self.on_failure(state, class);
                    if class == FailureClass::Fatal {
                        return Err(err);
                    }
                    if !is_last {
                        tracing::warn!(
                            lane = %lane.label,
                            provider = lane.provider.name(),
                            model = lane.provider.model(),
                            error = %err,
                            "LLM lane failed transiently or yielded nothing; retrying this request on the next lane",
                        );
                    }
                    last_err = Some(err);
                    empty_ok = None;
                }
            }
        }
        if let Some(value) = empty_ok {
            return Ok(value);
        }
        if let Some(reason) = declined {
            return Err(LlmError::MeteredDeclined(reason));
        }
        Err(last_err.unwrap_or_else(|| {
            LlmError::LanesPaused(format!(
                "every lane's breaker is open ({}); retry after the cooldown",
                skipped.join(", ")
            ))
        }))
    }
}

/// The request's user-side text, bounded, for the value gate.
fn evidence_of(request: &ChatRequest) -> String {
    let joined: String = request
        .messages
        .iter()
        .filter(|m| m.role == crate::types::Role::User)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    crate::metered_gate::bound_evidence(&joined).to_string()
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        self.lanes[0].lane.provider.name()
    }

    fn model(&self) -> &str {
        self.lanes[0].lane.provider.model()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        self.route(evidence_of(&request), |p| {
            let r = request.clone();
            async move { p.complete(r).await }
        })
        .await
    }

    async fn complete_with_operation_id(
        &self,
        request: ChatRequest,
        operation_id: LlmOperationId,
    ) -> LlmResult<ChatResponse> {
        self.route(evidence_of(&request), |p| {
            let r = request.clone();
            async move { p.complete_with_operation_id(r, operation_id).await }
        })
        .await
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        self.route(evidence_of(&request), |p| {
            let (r, s) = (request.clone(), schema.clone());
            async move { p.complete_structured_raw(r, s).await }
        })
        .await
    }

    async fn complete_structured_raw_with_operation_id(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
        operation_id: LlmOperationId,
    ) -> LlmResult<serde_json::Value> {
        self.route(evidence_of(&request), |p| {
            let (r, s) = (request.clone(), schema.clone());
            async move {
                p.complete_structured_raw_with_operation_id(r, s, operation_id)
                    .await
            }
        })
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
            crate::usage::report(100, 10);
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
                model: "gemini-2.5-flash".into(),
                lane: "secondary".into(),
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
                partial: None,
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

    // ── Lane chain: two Poolside keys ahead of Gemini, per-lane breakers,
    // ── zero-yield breaker, ledger.

    #[derive(Default)]
    struct Recorder {
        calls: Mutex<Vec<CallRecord>>,
        breakers: Mutex<Vec<BreakerEvent>>,
    }
    impl LaneObserver for Recorder {
        fn on_call(&self, record: &CallRecord) {
            self.calls.lock().unwrap().push(record.clone());
        }
        fn on_breaker_open(&self, event: &BreakerEvent) {
            self.breakers.lock().unwrap().push(event.clone());
        }
    }

    fn truncated() -> LlmError {
        LlmError::Truncated {
            finish_reason: "MAX_TOKENS".into(),
            partial: None,
        }
    }

    fn three_lanes(
        a: &Arc<Scripted>,
        b: &Arc<Scripted>,
        g: &Arc<Scripted>,
    ) -> (FallbackProvider, Arc<FakeClock>, Arc<Recorder>) {
        let clock = Arc::new(FakeClock(Mutex::new(Instant::now())));
        let rec = Arc::new(Recorder::default());
        let (a, b, g): (Arc<dyn LlmProvider>, Arc<dyn LlmProvider>, Arc<dyn LlmProvider>) =
            (a.clone(), b.clone(), g.clone());
        let f = FallbackProvider::chain(
            Lane::new(a, "key-a").with_key_fp(Some("aaaa0000".into())),
            vec![
                Lane::new(b, "key-b").with_key_fp(Some("bbbb1111".into())),
                Lane::new(g, "gemini"),
            ],
        )
        .with_cooldown(Duration::from_secs(900))
        .with_clock(clock.clone())
        .with_observer(rec.clone());
        (f, clock, rec)
    }

    fn laguna(errors: Vec<Option<LlmError>>) -> Arc<Scripted> {
        Scripted::new("openai-compat", "poolside/laguna-s-2.1", errors)
    }

    #[tokio::test]
    async fn key_a_limited_retries_the_same_request_on_key_b() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![]),
            Scripted::ok("gemini", "gemini-2.5-flash"),
        );
        let (f, _, rec) = three_lanes(&a, &b, &g);
        let (out, who) = call(&f).await;
        assert!(out.is_ok());
        assert_eq!(who.unwrap().lane, "key-b");
        assert_eq!((a.calls(), b.calls(), g.calls()), (1, 1, 0));
        // Per-key cooldown: key A is not hammered on the next request.
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().lane, "key-b");
        assert_eq!(a.calls(), 1, "a limited key is skipped during its cooldown");
        let breakers = rec.breakers.lock().unwrap();
        assert_eq!(breakers.len(), 1);
        assert_eq!((breakers[0].lane.as_str(), breakers[0].reason), ("key-a", "quota"));
    }

    #[tokio::test]
    async fn both_keys_limited_goes_to_gemini() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![Some(status(429))]),
            Scripted::ok("gemini", "gemini-2.5-flash"),
        );
        let (f, _, _) = three_lanes(&a, &b, &g);
        let (out, who) = call(&f).await;
        assert_eq!(out.unwrap()["answered_by"], "gemini");
        assert_eq!(who.unwrap().lane, "gemini");
        call(&f).await.0.unwrap();
        assert_eq!((a.calls(), b.calls(), g.calls()), (1, 1, 2));
    }

    #[tokio::test]
    async fn key_a_recovers_after_its_cooldown() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![]),
            Scripted::ok("gemini", "gemini-2.5-flash"),
        );
        let (f, clock, _) = three_lanes(&a, &b, &g);
        call(&f).await.0.unwrap();
        clock.advance(Duration::from_secs(899));
        assert_eq!(call(&f).await.1.unwrap().lane, "key-b");
        clock.advance(Duration::from_secs(2));
        assert_eq!(call(&f).await.1.unwrap().lane, "key-a", "probed after cooldown");
        assert_eq!(call(&f).await.1.unwrap().lane, "key-a", "probe success closed it");
    }

    #[tokio::test]
    async fn consecutive_truncations_pause_the_last_lane_like_an_error() {
        let g = Scripted::new(
            "gemini",
            "gemini-2.5-flash",
            vec![Some(truncated()), Some(truncated()), Some(truncated())],
        );
        let (a, b) = (laguna(vec![]), laguna(vec![]));
        let clock = Arc::new(FakeClock(Mutex::new(Instant::now())));
        let rec = Arc::new(Recorder::default());
        let gd: Arc<dyn LlmProvider> = g.clone();
        let f = FallbackProvider::chain(Lane::new(gd, "gemini"), vec![])
            .with_cooldown(Duration::from_secs(900))
            .with_clock(clock.clone())
            .with_observer(rec.clone());
        let _ = (a, b);
        for _ in 0..3 {
            assert!(matches!(call(&f).await.0, Err(LlmError::Truncated { .. })));
        }
        match call(&f).await.0 {
            Err(LlmError::LanesPaused(msg)) => assert!(msg.contains("gemini"), "{msg}"),
            other => panic!("expected LanesPaused, got {other:?}"),
        }
        assert_eq!(g.calls(), 3, "a paused lane is not paid for again");
        {
            let breakers = rec.breakers.lock().unwrap();
            assert_eq!(breakers.len(), 1);
            assert_eq!((breakers[0].reason, breakers[0].consecutive), ("zero_yield", 3));
        }
        clock.advance(Duration::from_secs(901));
        assert!(call(&f).await.0.is_ok(), "retried after the cooldown");
        assert_eq!(g.calls(), 4);
    }

    #[tokio::test]
    async fn a_success_resets_the_zero_yield_count() {
        let g = Scripted::new(
            "gemini",
            "gemini-2.5-flash",
            vec![Some(truncated()), Some(truncated()), None, Some(truncated()), Some(truncated())],
        );
        let gd: Arc<dyn LlmProvider> = g.clone();
        let f = FallbackProvider::chain(Lane::new(gd, "gemini"), vec![]);
        for _ in 0..5 {
            let _ = call(&f).await;
        }
        let _ = call(&f).await;
        assert_eq!(g.calls(), 6, "never three in a row, so never paused");
    }

    #[tokio::test]
    async fn truncation_on_key_a_moves_to_key_b_and_pauses_a_after_three() {
        let a = laguna(vec![Some(truncated()), Some(truncated()), Some(truncated())]);
        let (b, g) = (laguna(vec![]), Scripted::ok("gemini", "gemini-2.5-flash"));
        let (f, _, _) = three_lanes(&a, &b, &g);
        for _ in 0..4 {
            assert_eq!(call(&f).await.1.unwrap().lane, "key-b");
        }
        assert_eq!(a.calls(), 3);
    }

    #[tokio::test]
    async fn an_empty_success_counts_as_zero_yield() {
        struct Empty(AtomicUsize);
        #[async_trait]
        impl LlmProvider for Empty {
            fn name(&self) -> &'static str {
                "gemini"
            }
            fn model(&self) -> &str {
                "gemini-2.5-flash"
            }
            async fn complete(&self, _r: ChatRequest) -> LlmResult<ChatResponse> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(ChatResponse {
                    text: "  ".into(),
                    usage: None,
                    model: "gemini-2.5-flash".into(),
                })
            }
            async fn complete_structured_raw(
                &self,
                _r: ChatRequest,
                _s: serde_json::Value,
            ) -> LlmResult<serde_json::Value> {
                unreachable!()
            }
        }
        let e = Arc::new(Empty(AtomicUsize::new(0)));
        let ed: Arc<dyn LlmProvider> = e.clone();
        let f = FallbackProvider::chain(Lane::new(ed, "gemini"), vec![]);
        for _ in 0..3 {
            assert!(f.complete(req()).await.is_ok());
        }
        assert!(matches!(f.complete(req()).await, Err(LlmError::LanesPaused(_))));
        assert_eq!(e.0.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn every_call_lands_in_the_ledger_with_lane_key_fp_and_tokens() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![Some(status(429))]),
            Scripted::ok("gemini", "gemini-2.5-flash"),
        );
        let (f, _, rec) = three_lanes(&a, &b, &g);
        crate::usage::with_caller("auto_improve", call(&f)).await.0.unwrap();
        let calls = rec.calls.lock().unwrap();
        let got: Vec<(&str, &str, Option<&str>)> = calls
            .iter()
            .map(|c| (c.lane.as_str(), c.outcome, c.key_fp.as_deref()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("key-a", "quota", Some("aaaa0000")),
                ("key-b", "quota", Some("bbbb1111")),
                ("gemini", "ok", None),
            ]
        );
        assert!(calls.iter().all(|c| c.caller == "auto_improve"));
        let gem = &calls[2];
        assert_eq!((gem.input_tokens, gem.output_tokens), (Some(100), Some(10)));
        assert!(gem.cost_usd_est.unwrap() > 0.0);
        assert_eq!(gem.funding, crate::ledger::METERED_FUNDING_TYPE);
        assert_eq!(calls[0].cost_usd_est, None);
    }

    #[tokio::test]
    async fn a_failed_half_open_probe_reopens_on_one_server_error() {
        let a = laguna(vec![
            Some(status(500)),
            Some(status(500)),
            Some(status(500)),
            Some(status(500)),
        ]);
        let (b, g) = (laguna(vec![]), Scripted::ok("gemini", "gemini-2.5-flash"));
        let (f, clock, _) = three_lanes(&a, &b, &g);
        for _ in 0..3 {
            call(&f).await.0.unwrap();
        }
        assert_eq!(a.calls(), 3);
        clock.advance(Duration::from_secs(901));
        call(&f).await.0.unwrap(); // probe: one 500
        call(&f).await.0.unwrap();
        assert_eq!(a.calls(), 4, "one failed probe re-opened the circuit");
    }

    // ── Value gate (Jev) in front of the metered lane.

    struct FakeGate {
        admit: bool,
        reason: &'static str,
        calls: AtomicUsize,
    }
    #[async_trait]
    impl MeteredGate for FakeGate {
        async fn judge(&self, request: GateRequest<'_>) -> GateVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.provider, "gemini", "only the metered lane is gated");
            GateVerdict {
                admit: self.admit,
                reason: self.reason.into(),
                cost_usd: Some(0.0001),
            }
        }
    }

    fn gated(
        a: &Arc<Scripted>,
        b: &Arc<Scripted>,
        g: &Arc<Scripted>,
        admit: bool,
        reason: &'static str,
    ) -> (FallbackProvider, Arc<FakeGate>, Arc<Recorder>) {
        let gate = Arc::new(FakeGate {
            admit,
            reason,
            calls: AtomicUsize::new(0),
        });
        let (f, _, rec) = three_lanes(a, b, g);
        let f = f.with_metered_gate(gate.clone(), vec!["auto_improve".into()]);
        (f, gate, rec)
    }

    fn walled() -> (Arc<Scripted>, Arc<Scripted>) {
        (laguna(vec![Some(status(429))]), laguna(vec![Some(status(429))]))
    }

    #[tokio::test]
    async fn jev_yes_lets_the_request_reach_gemini() {
        let (a, b) = walled();
        let g = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, gate, _) = gated(&a, &b, &g, true, "jev admitted");
        let (out, who) = crate::usage::with_caller("auto_improve", call(&f)).await;
        assert_eq!(out.unwrap()["answered_by"], "gemini");
        assert_eq!(who.unwrap().lane, "gemini");
        assert_eq!((gate.calls.load(Ordering::SeqCst), g.calls()), (1, 1));
    }

    #[tokio::test]
    async fn jev_no_keeps_the_request_off_gemini_and_is_recorded() {
        let (a, b) = walled();
        let g = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, gate, rec) = gated(&a, &b, &g, false, "jev declined: future_need=0.40 < 0.9");
        let (out, _) = crate::usage::with_caller("auto_improve", call(&f)).await;
        match out {
            Err(LlmError::MeteredDeclined(reason)) => assert!(reason.contains("future_need")),
            other => panic!("expected MeteredDeclined, got {other:?}"),
        }
        assert_eq!(g.calls(), 0, "a declined request is never paid for");
        assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
        let calls = rec.calls.lock().unwrap();
        let gate_row = calls.iter().find(|c| c.provider == "jev").expect("gate recorded");
        assert_eq!(gate_row.outcome, "gate_declined");
        assert_eq!(gate_row.cost_usd_est, Some(0.0001), "Jev's own cost is recorded");
    }

    #[tokio::test]
    async fn jev_unavailable_fails_closed() {
        let (a, b) = walled();
        let g = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, _, _) = gated(&a, &b, &g, false, "jev unavailable on future_need: timed out (fail closed)");
        let (out, _) = crate::usage::with_caller("auto_improve", call(&f)).await;
        assert!(matches!(out, Err(LlmError::MeteredDeclined(_))), "{out:?}");
        assert_eq!(g.calls(), 0);
    }

    #[tokio::test]
    async fn the_free_laguna_path_never_asks_jev() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![]),
            Scripted::ok("gemini", "gemini-2.5-flash"),
        );
        let (f, gate, _) = gated(&a, &b, &g, false, "would decline");
        for _ in 0..2 {
            let (out, who) = crate::usage::with_caller("auto_improve", call(&f)).await;
            assert!(out.is_ok());
            assert_eq!(who.unwrap().lane, "key-b");
        }
        assert_eq!(gate.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn an_ungated_caller_reaches_gemini_without_asking_jev() {
        let (a, b) = walled();
        let g = Scripted::ok("gemini", "gemini-2.5-flash");
        let (f, gate, _) = gated(&a, &b, &g, false, "would decline");
        let (out, _) = crate::usage::with_caller("consolidate", call(&f)).await;
        assert!(out.is_ok());
        assert_eq!((gate.calls.load(Ordering::SeqCst), g.calls()), (0, 1));
    }
}
