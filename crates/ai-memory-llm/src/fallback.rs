//! Runtime LLM lane chain with a circuit breaker per lane.
//!
//! The lane is chosen once, at service start, by whatever launches the
//! server. Without this wrapper a quota wall (`429 usage limit exceeded`) or a
//! provider outage on that lane stalls every LLM-backed pass until a restart,
//! even when further lanes are configured.
//!
//! [`FallbackProvider`] holds an ordered list of [`Lane`]s. The zero-metered
//! chain (FN8-9336, operator ruling 2026-09-26: "I refuse to pay for Gemini")
//! is Laguna key A → Laguna key B → the ChatGPT seat (`codex-oauth`) → no
//! LLM. Each request walks it in order:
//!
//! * a **transport** failure (`429`, `5xx`, transport timeout / connect
//!   failure) moves the same request to the next lane, including a
//!   [`Lane::transport_only`] lane (key B);
//! * a **content** failure (a loop the Laguna contract aborted, a cut at the
//!   output ceiling, an empty answer, no usable function call) moves the
//!   request to the next lane that is *not* transport-only. A second key of
//!   the same model would loop on the same input: key B is for walls and
//!   outages of key A, never for its loops. A content failure is a property of
//!   the input, not of the lane, so it never counts toward a breaker;
//! * a **deterministic** failure (any other `4xx`, auth, schema) is returned
//!   unchanged — another lane would only burn a second call on a request that
//!   cannot succeed as written;
//! * every lane has its own breaker. A `429` opens it at once;
//!   [`CIRCUIT_5XX_THRESHOLD`] consecutive server/transport failures open it.
//!   While open the lane is skipped; after the cooldown the next request
//!   probes it (half-open) and any failure of that probe re-opens it at once;
//! * a lane with a [`DailyCap`] (the ChatGPT seat) admits at most `max`
//!   requests per UTC day. A refused request sends nothing. When no lane
//!   answered and a cap refused, the chain returns
//!   [`LlmError::AllocationExhausted`], ahead of any content failure, so the
//!   caller defers the work to the reset instead of spending its retry budget.
//! * when every lane is paused no request is sent at all:
//!   [`LlmError::LanesPaused`].
//!
//! Every lane call is reported to an optional [`LaneObserver`] (the call
//! ledger, [`crate::ledger`]) with the tokens the provider reported and its
//! cost (`0.0` on a zero-metered lane), and every breaker that opens is
//! reported as a [`BreakerEvent`] plus a `warn!` alarm line. Lanes are named by
//! label and key fingerprint, never by key.
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
    BreakerEvent, CallRecord, LaneKind, LaneObserver, call_cost, funding_for, lane_kind,
    now_rfc3339,
};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};
use crate::usage::{ReportedUsage, capture_usage, current_caller};

/// Consecutive `5xx` / transport failures that open a lane's circuit.
pub const CIRCUIT_5XX_THRESHOLD: u32 = 3;

/// How long a lane is skipped once its circuit opens.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// Default per-UTC-day request cap on the flat ChatGPT-seat lane
/// (`AI_MEMORY_FLAT_FALLBACK_DAILY_MAX_REQUESTS`).
pub const DEFAULT_FLAT_DAILY_MAX_REQUESTS: u32 = 40;

const SECONDS_PER_DAY: i64 = 86_400;

/// The lane that actually produced (or failed) a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Responder {
    /// Provider wire name (`"openai-compat"`, `"openai-oauth"`, …).
    pub provider: &'static str,
    /// Model id that was hit.
    pub model: String,
    /// Lane label (`"key-a"`, `"key-b"`, `"codex-oauth"`, …).
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

/// Wall clock in Unix seconds, injectable so tests can cross a UTC midnight.
pub trait WallClock: Send + Sync {
    /// Seconds since the Unix epoch.
    fn unix_now(&self) -> i64;
}

/// The real wall clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemWallClock;

impl WallClock for SystemWallClock {
    fn unix_now(&self) -> i64 {
        jiff::Timestamp::now().as_second()
    }
}

/// Unix second of the 00:00 UTC that starts the day holding `unix`.
#[must_use]
pub fn utc_day_start(unix: i64) -> i64 {
    unix.div_euclid(SECONDS_PER_DAY) * SECONDS_PER_DAY
}

/// A per-UTC-day request allocation on one lane. Counted at admission: a
/// request that is let through counts whether or not it succeeds, because the
/// seat's allocation is spent by the request, not by the answer.
pub struct DailyCap {
    max: u32,
    wall: Arc<dyn WallClock>,
    /// `(day start, requests admitted that day)`.
    state: Mutex<(i64, u32)>,
}

impl DailyCap {
    /// At most `max` requests per UTC day, none admitted yet today.
    #[must_use]
    pub fn new(max: u32) -> Self {
        Self::with_wall_clock(max, Arc::new(SystemWallClock), 0)
    }

    /// At most `max` requests per UTC day on `wall`, with `used_today`
    /// requests already admitted today (seeded from the call ledger so a
    /// restart does not reset the allocation).
    #[must_use]
    pub fn with_wall_clock(max: u32, wall: Arc<dyn WallClock>, used_today: u32) -> Self {
        let day = utc_day_start(wall.unix_now());
        Self {
            max,
            wall,
            state: Mutex::new((day, used_today)),
        }
    }

    /// The configured cap.
    #[must_use]
    pub fn max(&self) -> u32 {
        self.max
    }

    /// Admit one request, or report `(used, cap, resets_at_unix)`.
    fn try_admit(&self) -> Result<(), (u32, u32, i64)> {
        let today = utc_day_start(self.wall.unix_now());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.0 != today {
            *state = (today, 0);
        }
        if state.1 >= self.max {
            return Err((state.1, self.max, today + SECONDS_PER_DAY));
        }
        state.1 += 1;
        Ok(())
    }
}

/// One provider in the chain.
pub struct Lane {
    provider: Arc<dyn LlmProvider>,
    label: String,
    key_fp: Option<String>,
    transport_only: bool,
    daily_cap: Option<DailyCap>,
}

impl Lane {
    /// A lane named `label` (a short, key-free name such as `key-a`).
    #[must_use]
    pub fn new(provider: Arc<dyn LlmProvider>, label: impl Into<String>) -> Self {
        Self {
            provider,
            label: label.into(),
            key_fp: None,
            transport_only: false,
            daily_cap: None,
        }
    }

    /// Attach the key's fingerprint ([`crate::ledger::key_fingerprint`]) for
    /// the ledger. Never pass the key itself.
    #[must_use]
    pub fn with_key_fp(mut self, fp: Option<String>) -> Self {
        self.key_fp = fp;
        self
    }

    /// Only try this lane after a *transport* failure (or a paused breaker)
    /// of the lanes before it — a second key of the same model, which would
    /// loop on the same input exactly as the first did.
    #[must_use]
    pub fn transport_only(mut self) -> Self {
        self.transport_only = true;
        self
    }

    /// Admit at most the cap's requests per UTC day on this lane.
    #[must_use]
    pub fn with_daily_cap(mut self, cap: DailyCap) -> Self {
        self.daily_cap = Some(cap);
        self
    }

    /// The lane label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// What this lane costs ([`crate::ledger::lane_kind`]).
    #[must_use]
    pub fn kind(&self) -> LaneKind {
        lane_kind(
            self.provider.name(),
            self.provider.model(),
            self.provider.endpoint(),
        )
    }
}

/// How a lane failure is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    /// `429`: quota / rate wall. Next lane, open this lane's circuit now.
    Quota,
    /// `5xx` or transport timeout / connect: next lane, count toward the circuit.
    Server,
    /// The answer, not the lane, failed: a loop, a cut, an empty answer, no
    /// usable function call. Next non-transport-only lane; circuit untouched.
    Content,
    /// Deterministic: return as-is, no further lane, circuit untouched.
    Fatal,
}

fn classify(err: &LlmError) -> FailureClass {
    match err {
        LlmError::Provider { status: 429, .. } => FailureClass::Quota,
        LlmError::Provider { status, .. } if (500..=599).contains(status) => FailureClass::Server,
        LlmError::Http(e) if e.is_timeout() || e.is_connect() => FailureClass::Server,
        LlmError::Truncated { .. }
        | LlmError::EmptyResponse(_)
        | LlmError::UnexpectedShape(_)
        | LlmError::Serde(_) => FailureClass::Content,
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
        Err(LlmError::UnexpectedShape(_) | LlmError::Serde(_)) => ("shape", None),
        Err(e) if classify(e) == FailureClass::Server => ("server", None),
        Err(_) => ("fatal", None),
    }
}

#[derive(Debug, Default)]
struct Circuit {
    open_until: Option<Instant>,
    half_open: bool,
    consecutive_server_failures: u32,
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
    clock: Arc<dyn Clock>,
    observer: Option<Arc<dyn LaneObserver>>,
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
            clock: Arc::new(SystemClock),
            observer: None,
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

    /// `(label, kind)` of every lane, in order.
    #[must_use]
    pub fn lane_kinds(&self) -> Vec<(String, LaneKind)> {
        self.lanes
            .iter()
            .map(|l| (l.lane.label.clone(), l.lane.kind()))
            .collect()
    }

    /// Labels of the lanes that are not allowed in the zero-metered chain.
    #[must_use]
    pub fn non_zero_metered_lanes(&self) -> Vec<String> {
        self.lane_kinds()
            .into_iter()
            .filter(|(_, kind)| !kind.is_zero_metered())
            .map(|(label, _)| label)
            .collect()
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
                // A content failure says nothing about the lane's health; a
                // half-open probe that got an answer (even a bad one) proves
                // the lane reachable, so it closes.
                FailureClass::Content | FailureClass::Fatal => None,
            };
            if trip.is_some() {
                c.open_until = Some(now + self.cooldown);
                c.consecutive_server_failures = 0;
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
        let endpoint = lane.provider.endpoint();
        let cost_usd_est = match usage {
            Some(u) => call_cost(provider, model, endpoint, u.input_tokens, u.output_tokens),
            None => call_cost(provider, model, endpoint, 0, 0),
        };
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
            http_calls: usage.map(|u| u.http_calls),
            cost_usd_est,
            funding: funding_for(provider, model),
        });
    }

    /// Core routing, shared by all four trait entry points.
    async fn route<T, C, CF>(&self, call: C) -> LlmResult<T>
    where
        T: YieldCheck,
        C: Fn(Arc<dyn LlmProvider>) -> CF,
        CF: Future<Output = LlmResult<T>>,
    {
        let mut last_err: Option<LlmError> = None;
        let mut empty_ok: Option<T> = None;
        let mut skipped: Vec<&str> = Vec::new();
        let mut capped: Option<LlmError> = None;
        // Set once any lane failed on the answer itself: from then on a
        // transport-only lane (a second key of the same model) is not worth
        // its call.
        let mut content_failed = false;
        for (index, state) in self.lanes.iter().enumerate() {
            let lane = &state.lane;
            if lane.transport_only && content_failed {
                tracing::info!(
                    lane = %lane.label,
                    "transport-only lane skipped: the previous lane failed on the answer, not the transport",
                );
                continue;
            }
            if self.skipped(state) {
                skipped.push(&state.lane.label);
                continue;
            }
            if let Some(cap) = &lane.daily_cap
                && let Err((used, max, resets_at_unix)) = cap.try_admit()
            {
                tracing::warn!(
                    lane = %lane.label,
                    used,
                    cap = max,
                    resets_at_unix,
                    "flat lane daily allocation used; request not sent (capped)",
                );
                self.observe_call(lane, ("capped", None), None);
                capped = Some(LlmError::AllocationExhausted {
                    lane: lane.label.clone(),
                    used,
                    cap: max,
                    resets_at_unix,
                });
                continue;
            }
            record_responder(lane);
            let (result, usage) = capture_usage(call(lane.provider.clone())).await;
            let is_last = index + 1 == self.lanes.len();
            match result {
                Ok(value) => {
                    let empty = value.is_empty_yield();
                    self.observe_call(lane, outcome_label(&Ok(empty)), usage);
                    Self::on_success(state);
                    if !empty {
                        return Ok(value);
                    }
                    content_failed = true;
                    if is_last {
                        return Ok(value);
                    }
                    empty_ok = Some(value);
                }
                Err(err) => {
                    self.observe_call(lane, outcome_label(&Err(&err)), usage);
                    let class = classify(&err);
                    self.on_failure(state, class);
                    match class {
                        FailureClass::Fatal => return Err(err),
                        FailureClass::Content => {
                            // The lane answered; its breaker state is healthy.
                            Self::on_success(state);
                            content_failed = true;
                        }
                        FailureClass::Quota | FailureClass::Server => {}
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
        if let Some(err) = capped {
            return Err(err);
        }
        Err(last_err.unwrap_or_else(|| {
            LlmError::LanesPaused(format!(
                "every lane's breaker is open ({}); retry after the cooldown",
                skipped.join(", ")
            ))
        }))
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        self.lanes[0].lane.provider.name()
    }

    fn model(&self) -> &str {
        self.lanes[0].lane.provider.model()
    }

    fn endpoint(&self) -> Option<&str> {
        self.lanes[0].lane.provider.endpoint()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        self.route(|p| {
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
        self.route(|p| {
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
        self.route(|p| {
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
        self.route(|p| {
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
        fn endpoint(&self) -> Option<&str> {
            (self.name == "openai-compat").then_some("https://inference.poolside.ai/v1")
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);

        let (out, who) = call(&f).await;
        assert_eq!(out.unwrap()["answered_by"], "openai-oauth");
        assert_eq!(
            who,
            Some(Responder {
                provider: "openai-oauth",
                model: "gpt-6-astra".into(),
                lane: "secondary".into(),
            })
        );
        assert_eq!((primary.calls(), secondary.calls()), (1, 1));
    }

    #[tokio::test]
    async fn primary_500_falls_back_before_circuit_opens_and_is_attributed() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(500))]);
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);

        let (out, who) = call(&f).await;
        assert!(out.is_ok());
        assert_eq!(who.unwrap().provider, "openai-oauth");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "openai-oauth");
        let (_, who) = call(&f).await;
        assert_eq!(who.unwrap().provider, "openai-compat");
    }

    #[tokio::test]
    async fn primary_400_does_not_fall_back() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(400))]);
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
    async fn schema_errors_do_not_fall_back_but_a_bad_answer_does() {
        let primary = Scripted::new(
            "openai-compat",
            "laguna",
            vec![
                Some(LlmError::Schema("bad".into())),
                Some(LlmError::Serde("nope".into())),
                Some(LlmError::UnexpectedShape("no tool".into())),
            ],
        );
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);
        assert!(matches!(call(&f).await.0, Err(LlmError::Schema(_))));
        assert_eq!(secondary.calls(), 0, "a request that cannot succeed is not re-sent");
        // A malformed answer is a property of that answer: the next lane may do better.
        assert!(call(&f).await.0.is_ok());
        assert!(call(&f).await.0.is_ok());
        assert_eq!(secondary.calls(), 2);
    }

    #[tokio::test]
    async fn circuit_opens_after_429_and_skips_primary() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);

        call(&f).await.0.unwrap();
        let (out, who) = call(&f).await;
        assert!(out.is_ok());
        assert_eq!(who.unwrap().provider, "openai-oauth");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
        let secondary = Scripted::new("openai-oauth", "gpt-6-astra", vec![Some(status(503))]);
        let (f, _) = build(&primary, &secondary);
        match call(&f).await.0 {
            Err(LlmError::Provider { status: 503, .. }) => {}
            other => panic!("expected the secondary's 503, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn name_and_model_stay_the_configured_primary() {
        let primary = Scripted::new("openai-compat", "laguna", vec![Some(status(429))]);
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
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
        let secondary = Scripted::ok("openai-oauth", "gpt-6-astra");
        let (f, _) = build(&primary, &secondary);
        let (out, who) = capture_responder(f.complete_structured_raw_with_operation_id(
            req(),
            serde_json::json!({}),
            LlmOperationId::new(),
        ))
        .await;
        assert_eq!(out.unwrap()["answered_by"], "openai-oauth");
        assert_eq!(who.unwrap().provider, "openai-oauth");
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
                Lane::new(b, "key-b")
                    .with_key_fp(Some("bbbb1111".into()))
                    .transport_only(),
                Lane::new(g, "codex-oauth"),
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
            Scripted::ok("openai-oauth", "gpt-6-astra"),
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
    async fn both_keys_limited_goes_to_the_codex_seat() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![Some(status(429))]),
            Scripted::ok("openai-oauth", "gpt-6-astra"),
        );
        let (f, _, _) = three_lanes(&a, &b, &g);
        let (out, who) = call(&f).await;
        assert_eq!(out.unwrap()["answered_by"], "openai-oauth");
        assert_eq!(who.unwrap().lane, "codex-oauth");
        call(&f).await.0.unwrap();
        assert_eq!((a.calls(), b.calls(), g.calls()), (1, 1, 2));
    }

    #[tokio::test]
    async fn key_a_recovers_after_its_cooldown() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![]),
            Scripted::ok("openai-oauth", "gpt-6-astra"),
        );
        let (f, clock, _) = three_lanes(&a, &b, &g);
        call(&f).await.0.unwrap();
        clock.advance(Duration::from_secs(899));
        assert_eq!(call(&f).await.1.unwrap().lane, "key-b");
        clock.advance(Duration::from_secs(2));
        assert_eq!(call(&f).await.1.unwrap().lane, "key-a", "probed after cooldown");
        assert_eq!(call(&f).await.1.unwrap().lane, "key-a", "probe success closed it");
    }

    struct FakeWall(Mutex<i64>);
    impl WallClock for FakeWall {
        fn unix_now(&self) -> i64 {
            *self.0.lock().unwrap()
        }
    }

    /// 2026-09-26T12:00:00Z.
    const NOON: i64 = 1_790_424_000;

    fn capped_chain(
        a: &Arc<Scripted>,
        b: &Arc<Scripted>,
        seat: &Arc<Scripted>,
        max: u32,
        used_today: u32,
    ) -> (FallbackProvider, Arc<FakeWall>, Arc<Recorder>) {
        let wall = Arc::new(FakeWall(Mutex::new(NOON)));
        let rec = Arc::new(Recorder::default());
        let (a, b, seat): (Arc<dyn LlmProvider>, Arc<dyn LlmProvider>, Arc<dyn LlmProvider>) =
            (a.clone(), b.clone(), seat.clone());
        let f = FallbackProvider::chain(
            Lane::new(a, "key-a"),
            vec![
                Lane::new(b, "key-b").transport_only(),
                Lane::new(seat, "codex-oauth").with_daily_cap(DailyCap::with_wall_clock(
                    max,
                    wall.clone(),
                    used_today,
                )),
            ],
        )
        .with_observer(rec.clone());
        (f, wall, rec)
    }

    #[tokio::test]
    async fn a_loop_on_key_a_skips_key_b_and_reaches_the_seat_without_a_breaker() {
        let a = laguna(vec![Some(truncated()), Some(truncated()), Some(truncated()), Some(truncated())]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, rec) = three_lanes(&a, &b, &seat);
        for _ in 0..4 {
            assert_eq!(call(&f).await.1.unwrap().lane, "codex-oauth");
        }
        assert_eq!(a.calls(), 4, "a looping input never pauses key A for other inputs");
        assert_eq!(b.calls(), 0, "key B is for transport failures only");
        assert!(rec.breakers.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_shape_fault_on_key_a_also_skips_key_b() {
        let a = laguna(vec![Some(LlmError::UnexpectedShape("no call".into()))]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, _) = three_lanes(&a, &b, &seat);
        assert_eq!(call(&f).await.1.unwrap().lane, "codex-oauth");
        assert_eq!((a.calls(), b.calls(), seat.calls()), (1, 0, 1));
    }

    #[tokio::test]
    async fn a_server_error_on_key_a_goes_to_key_b() {
        let a = laguna(vec![Some(status(503))]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, _) = three_lanes(&a, &b, &seat);
        assert_eq!(call(&f).await.1.unwrap().lane, "key-b");
        assert_eq!((a.calls(), b.calls(), seat.calls()), (1, 1, 0));
    }

    #[tokio::test]
    async fn a_paused_key_a_hands_its_traffic_to_key_b() {
        let a = laguna(vec![Some(status(429))]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, _) = three_lanes(&a, &b, &seat);
        call(&f).await.0.unwrap();
        assert_eq!(call(&f).await.1.unwrap().lane, "key-b", "key A skipped by its breaker");
        assert_eq!(a.calls(), 1);
    }

    #[tokio::test]
    async fn an_empty_success_moves_on_and_never_pauses_the_lane() {
        struct Empty(AtomicUsize);
        #[async_trait]
        impl LlmProvider for Empty {
            fn name(&self) -> &'static str {
                "openai-oauth"
            }
            fn model(&self) -> &str {
                "gpt-6-astra"
            }
            async fn complete(&self, _r: ChatRequest) -> LlmResult<ChatResponse> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(ChatResponse {
                    text: "  ".into(),
                    usage: None,
                    model: "gpt-6-astra".into(),
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
        let f = FallbackProvider::chain(Lane::new(ed, "codex-oauth"), vec![]);
        for _ in 0..4 {
            assert!(f.complete(req()).await.is_ok());
        }
        assert_eq!(e.0.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn the_seat_admits_its_cap_then_refuses_without_sending() {
        let a = laguna(vec![Some(truncated()), Some(truncated())]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, rec) = capped_chain(&a, &b, &seat, 40, 39);
        assert_eq!(call(&f).await.1.unwrap().lane, "codex-oauth", "request 40 of 40");
        match call(&f).await.0 {
            Err(LlmError::AllocationExhausted { lane, used, cap, resets_at_unix }) => {
                assert_eq!((lane.as_str(), used, cap), ("codex-oauth", 40, 40));
                assert_eq!(resets_at_unix, 1_790_467_200, "next 00:00 UTC");
            }
            other => panic!("expected AllocationExhausted, got {other:?}"),
        }
        assert_eq!(seat.calls(), 1, "request 41 sends nothing");
        let calls = rec.calls.lock().unwrap();
        let last = calls.last().unwrap();
        assert_eq!((last.lane.as_str(), last.outcome, last.http_calls), ("codex-oauth", "capped", None));
    }

    #[tokio::test]
    async fn the_cap_outranks_the_loop_that_sent_the_request_there() {
        // Key A looped (Truncated), the seat is capped: the caller must see the
        // allocation, not the loop, or it would park the job as a persistent cut.
        let a = laguna(vec![Some(truncated())]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, _, _) = capped_chain(&a, &b, &seat, 0, 0);
        assert!(matches!(call(&f).await.0, Err(LlmError::AllocationExhausted { .. })));
        assert_eq!(seat.calls(), 0);
    }

    #[tokio::test]
    async fn the_cap_resets_at_utc_midnight() {
        let a = laguna(vec![Some(truncated()), Some(truncated()), Some(truncated())]);
        let (b, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let (f, wall, _) = capped_chain(&a, &b, &seat, 1, 0);
        call(&f).await.0.unwrap();
        assert!(call(&f).await.0.is_err());
        *wall.0.lock().unwrap() = 1_790_467_200; // 2026-09-27T00:00:00Z
        call(&f).await.0.unwrap();
        assert_eq!(seat.calls(), 2);
    }

    #[test]
    fn utc_day_start_is_midnight() {
        assert_eq!(utc_day_start(NOON), 1_790_380_800);
        assert_eq!(utc_day_start(1_790_380_800), 1_790_380_800);
    }

    #[test]
    fn lane_kinds_mark_only_laguna_and_the_seat_zero_metered() {
        let (a, seat) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
        let metered = Scripted::ok("gemini", "gemini-2.5-flash");
        let (a, seat, metered): (Arc<dyn LlmProvider>, Arc<dyn LlmProvider>, Arc<dyn LlmProvider>) =
            (a, seat, metered);
        let f = FallbackProvider::chain(
            Lane::new(a, "key-a"),
            vec![Lane::new(seat, "codex-oauth"), Lane::new(metered, "gemini")],
        );
        assert_eq!(f.non_zero_metered_lanes(), vec!["gemini".to_string()]);
    }

    #[tokio::test]
    async fn every_call_lands_in_the_ledger_with_lane_key_fp_and_tokens() {
        let (a, b, g) = (
            laguna(vec![Some(status(429))]),
            laguna(vec![Some(status(429))]),
            Scripted::ok("openai-oauth", "gpt-6-astra"),
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
                ("codex-oauth", "ok", None),
            ]
        );
        assert!(calls.iter().all(|c| c.caller == "auto_improve"));
        let seat = &calls[2];
        assert_eq!((seat.input_tokens, seat.output_tokens), (Some(100), Some(10)));
        // Zero metered: every row is booked at 0.00, with its tokens and lane.
        assert!(calls.iter().all(|c| c.cost_usd_est == Some(0.0)), "{calls:?}");
        assert!(seat.funding.contains("subscription flat"), "{}", seat.funding);
    }

    #[tokio::test]
    async fn a_failed_half_open_probe_reopens_on_one_server_error() {
        let a = laguna(vec![
            Some(status(500)),
            Some(status(500)),
            Some(status(500)),
            Some(status(500)),
        ]);
        let (b, g) = (laguna(vec![]), Scripted::ok("openai-oauth", "gpt-6-astra"));
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
}
