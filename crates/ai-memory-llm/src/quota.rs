//! When a key's daily quota comes back, per key fingerprint, across restarts.
//!
//! Poolside answers an exhausted daily quota with `429` and the body
//! `{"error":"usage limit exceeded"}`, and sends no `Retry-After` and no
//! `x-ratelimit-reset*` header (measured 2026-09-24). The fixed breaker
//! cooldown therefore probed a walled key blind every 15 minutes all day.
//!
//! Where the reset time comes from. There is no server-side source: Poolside's
//! own client (`pool`, the ACP server) prints `Daily limit exceeded. Usage
//! limit resets in %d hour%s` from a function (0x100bb4650 in the 2026-08-22
//! build) that matches the body on `usage limit exceeded` and computes
//! `24 - (unix_seconds mod 86400) / 3600` in UTC: it *assumes* a reset at
//! 00:00Z. That assumption is the [`QuotaSource::Default`] here, and it is
//! only an assumption — the only evidence is what a key does around it. Every
//! time a walled key answers again shortly after a quota `429`, the pair brackets
//! the real reset; those brackets are the [`QuotaSource::Learned`] estimate.
//!
//! The policy, per key:
//!
//! * a quota `429` on a key that has not answered since the last estimated
//!   boundary is a boundary probe that failed: wait a short, growing window
//!   ([`BOUNDARY_LADDER`]) and probe once more; after the ladder, wait for the
//!   next boundary. Never the old blind 15-minute poll all day.
//! * a quota `429` on a key that answered after the boundary is a fresh
//!   exhaustion: wait until the next boundary plus [`RESET_MARGIN_SECS`].
//! * when the wait ends exactly one request probes the key
//!   ([`PROBE_GRACE_SECS`] holds the rest back); its answer decides.
//! * a straggler — a request admitted before the wall that answers after it —
//!   neither advances the ladder nor lifts the wall.
//!
//! The state is a small JSON file written atomically (temp file + rename in
//! the same directory). An absent or unreadable file means "no state", never
//! "blocked". Keys are fingerprints ([`crate::ledger::key_fingerprint`]),
//! never key values, and never lane labels: labels follow boot order and swap
//! when the boot gate binds the other key first.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Seconds in a day.
pub const DAY_SECS: i64 = 86_400;

/// The reset second-of-day (UTC) assumed until a key has learned its own:
/// 00:00Z, the rule Poolside's own client computes its "resets in N hours"
/// message from. See the module docs.
pub const DEFAULT_RESET_SECOND_OF_DAY: i64 = 0;

/// Waited past an estimated boundary before the probe, so the probe does not
/// land a moment before the reset.
pub const RESET_MARGIN_SECS: i64 = 5 * 60;

/// Successive waits after a boundary probe fails. After the last one the key
/// waits for the next boundary.
pub const BOUNDARY_LADDER: [i64; 3] = [30 * 60, 60 * 60, 120 * 60];

/// While a probe is in flight, other requests keep skipping the key this long.
pub const PROBE_GRACE_SECS: i64 = 120;

/// A `429 → 200` pair wider than this says too little about when the reset
/// happened to move the estimate.
pub const MAX_INFORMATIVE_BRACKET_SECS: i64 = 60 * 60;

/// Learned reset samples kept per key (most recent).
pub const MAX_RESET_SAMPLES: usize = 7;

/// The body substring of Poolside's daily-quota `429` — the same one its own
/// client matches. Any other `429` is a generic rate wall.
pub const DAILY_QUOTA_MARKER: &str = "usage limit exceeded";

/// Whether a `429` body is the daily-quota wall.
#[must_use]
pub fn is_daily_quota(body: &str) -> bool {
    body.to_ascii_lowercase().contains(DAILY_QUOTA_MARKER)
}

/// Where a reset estimate comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaSource {
    /// From this key's own `429 → 200` brackets.
    Learned,
    /// [`DEFAULT_RESET_SECOND_OF_DAY`]: an assumption, no observation yet.
    Default,
}

impl QuotaSource {
    /// Wire name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Learned => "learned",
            Self::Default => "default",
        }
    }
}

/// Why a key is walled until [`QuotaBlock::until`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Fresh exhaustion: waiting for the next boundary.
    NextBoundary,
    /// A boundary probe failed: a short window, step `n` (1-based) of the ladder.
    Ladder(usize),
}

/// A wall just placed on a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaBlock {
    /// Unix seconds until which the key is skipped.
    pub until: i64,
    /// The estimated reset instant this wait is aimed at (unix seconds).
    pub resets_at: i64,
    /// Where the estimate comes from.
    pub source: QuotaSource,
    /// Why.
    pub kind: BlockKind,
}

/// One key's persisted quota memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyQuota {
    /// Unix seconds until which the key is skipped, if walled.
    pub blocked_until: Option<i64>,
    /// The reset instant the current wall is aimed at.
    pub resets_at: Option<i64>,
    /// `learned` / `default` for the current wall.
    pub source: Option<String>,
    /// The latest quota `429` of the current wall.
    pub last_wall: Option<i64>,
    /// The latest successful answer.
    pub last_success: Option<i64>,
    /// Boundary probes failed in a row (ladder position).
    pub failed_boundary_probes: u32,
    /// A probe was let through and has not answered yet.
    pub probing: bool,
    /// Learned reset seconds-of-day (UTC), most recent last.
    pub reset_samples: Vec<i64>,
}

impl KeyQuota {
    /// The current reset estimate (second of day, UTC) and its source.
    #[must_use]
    pub fn estimate(&self) -> (i64, QuotaSource) {
        circular_mean(&self.reset_samples)
            .map_or((DEFAULT_RESET_SECOND_OF_DAY, QuotaSource::Default), |s| {
                (s, QuotaSource::Learned)
            })
    }

    /// Whether the key is walled at `now`. When the wall has just expired,
    /// the caller becomes the one probe: the wall is held [`PROBE_GRACE_SECS`]
    /// longer for everyone else. Returns `(blocked, state_changed)`.
    pub fn admit(&mut self, now: i64) -> (bool, bool) {
        match self.blocked_until {
            Some(until) if now < until => (true, false),
            Some(_) => {
                self.blocked_until = Some(now + PROBE_GRACE_SECS);
                self.probing = true;
                (false, true)
            }
            None => (false, false),
        }
    }

    /// A daily-quota `429` at `now`. `None` for a straggler (the key is
    /// already walled and this was not the probe).
    pub fn on_quota(&mut self, now: i64) -> Option<QuotaBlock> {
        if !self.probing && self.blocked_until.is_some_and(|u| now < u) {
            return None;
        }
        let (sod, source) = self.estimate();
        let prev_boundary = boundary_at_or_before(now, sod);
        let ladder_span: i64 = BOUNDARY_LADDER.iter().sum::<i64>() + RESET_MARGIN_SECS;
        let answered_since_boundary = self.last_success.is_some_and(|s| s >= prev_boundary);
        let step = usize::try_from(self.failed_boundary_probes).unwrap_or(usize::MAX);
        let block = if !answered_since_boundary
            && now - prev_boundary < ladder_span
            && step < BOUNDARY_LADDER.len()
        {
            self.failed_boundary_probes += 1;
            let until = now + BOUNDARY_LADDER[step];
            QuotaBlock {
                until,
                resets_at: until,
                source,
                kind: BlockKind::Ladder(step + 1),
            }
        } else {
            self.failed_boundary_probes = 0;
            let next = prev_boundary + DAY_SECS;
            QuotaBlock {
                until: next + RESET_MARGIN_SECS,
                resets_at: next,
                source,
                kind: BlockKind::NextBoundary,
            }
        };
        self.blocked_until = Some(block.until);
        self.resets_at = Some(block.resets_at);
        self.source = Some(source.as_str().to_string());
        self.last_wall = Some(now);
        self.probing = false;
        Some(block)
    }

    /// A usable answer at `now`. Returns whether the persisted state changed
    /// (so the caller writes the file only then, not on every call).
    pub fn on_success(&mut self, now: i64) -> bool {
        if !self.probing && self.blocked_until.is_some_and(|u| now < u) {
            // A straggler admitted before the wall: says nothing about the
            // reset, and must not lift a wall the probe has not tested.
            return false;
        }
        let (sod, _) = self.estimate();
        let first_since_boundary = self
            .last_success
            .is_none_or(|s| s < boundary_at_or_before(now, sod));
        let walled = self.blocked_until.is_some() || self.last_wall.is_some() || self.probing;
        if let Some(wall) = self.last_wall
            && now > wall
            && now - wall <= MAX_INFORMATIVE_BRACKET_SECS
        {
            // The reset happened in (wall, now]: take the middle.
            self.reset_samples
                .push((wall + (now - wall) / 2).rem_euclid(DAY_SECS));
            if self.reset_samples.len() > MAX_RESET_SAMPLES {
                self.reset_samples.remove(0);
            }
        }
        self.blocked_until = None;
        self.resets_at = None;
        self.source = None;
        self.last_wall = None;
        self.failed_boundary_probes = 0;
        self.probing = false;
        self.last_success = Some(now);
        walled || first_since_boundary
    }
}

/// The latest instant `<= t` whose UTC second-of-day is `sod`.
#[must_use]
pub fn boundary_at_or_before(t: i64, sod: i64) -> i64 {
    t - (t - sod).rem_euclid(DAY_SECS)
}

/// Mean of seconds-of-day on the 24-hour circle, so 23:58 and 00:02 average
/// to 00:00 and not to 12:00. `None` for no samples.
#[must_use]
pub fn circular_mean(samples: &[i64]) -> Option<i64> {
    if samples.is_empty() {
        return None;
    }
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    for &x in samples {
        #[allow(clippy::cast_precision_loss)]
        let a = (x.rem_euclid(DAY_SECS) as f64) / (DAY_SECS as f64) * std::f64::consts::TAU;
        s += a.sin();
        c += a.cos();
    }
    let a = s.atan2(c).rem_euclid(std::f64::consts::TAU);
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let secs = (a / std::f64::consts::TAU * DAY_SECS as f64).round() as i64;
    Some(secs.rem_euclid(DAY_SECS))
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct QuotaFile {
    version: u32,
    keys: BTreeMap<String, KeyQuota>,
}

/// Every key's quota memory, optionally persisted to one JSON file.
#[derive(Debug, Default)]
pub struct QuotaBook {
    path: Option<PathBuf>,
    keys: Mutex<BTreeMap<String, KeyQuota>>,
}

impl QuotaBook {
    /// In memory only.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Backed by `path`: read now (absent or unreadable = empty), written on
    /// every state change.
    #[must_use]
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let keys = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| match serde_json::from_str::<QuotaFile>(&s) {
                Ok(f) => Some(f.keys),
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "LLM quota state unreadable; starting empty");
                    None
                }
            })
            .unwrap_or_default();
        Self {
            path: Some(path),
            keys: Mutex::new(keys),
        }
    }

    /// The backing file, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Run `f` on `key`'s state; persist when it reports a change.
    pub fn with_key<R>(&self, key: &str, f: impl FnOnce(&mut KeyQuota) -> (R, bool)) -> R {
        let mut keys = self
            .keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (out, changed) = f(keys.entry(key.to_string()).or_default());
        if changed {
            self.persist(&keys);
        }
        out
    }

    /// A copy of `key`'s state (tests, diagnostics).
    #[must_use]
    pub fn get(&self, key: &str) -> Option<KeyQuota> {
        self.keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    fn persist(&self, keys: &BTreeMap<String, KeyQuota>) {
        let Some(path) = &self.path else {
            return;
        };
        let file = QuotaFile {
            version: 1,
            keys: keys.clone(),
        };
        if let Err(err) = write_atomic(path, &file) {
            tracing::warn!(path = %path.display(), error = %err, "LLM quota state write failed");
        }
    }
}

fn write_atomic(path: &Path, file: &QuotaFile) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .map_or_else(|| "quota".into(), |n| n.to_string_lossy().into_owned());
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        let body = serde_json::to_string_pretty(file).map_err(std::io::Error::other)?;
        f.write_all(body.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-24T00:00:00Z.
    const MIDNIGHT: i64 = 1_790_208_000;
    const H: i64 = 3600;
    const M: i64 = 60;

    #[test]
    fn midnight_constant_is_a_utc_midnight() {
        assert_eq!(MIDNIGHT.rem_euclid(DAY_SECS), 0);
    }

    #[test]
    fn only_the_usage_limit_body_is_a_daily_wall() {
        assert!(is_daily_quota(r#"{"error":"usage limit exceeded"}"#));
        assert!(is_daily_quota("Usage limit exceeded"));
        assert!(!is_daily_quota("status 429"));
        assert!(!is_daily_quota(r#"{"error":"rate limited"}"#));
    }

    #[test]
    fn circular_mean_wraps_midnight() {
        assert_eq!(circular_mean(&[DAY_SECS - 2 * M, 2 * M]), Some(0));
        assert_eq!(
            circular_mean(&[DAY_SECS - 4 * M, 2 * M]),
            Some(DAY_SECS - M)
        );
        assert_eq!(circular_mean(&[H, 3 * H]), Some(2 * H));
        assert_eq!(circular_mean(&[]), None);
    }

    #[test]
    fn fresh_wall_waits_for_the_next_default_boundary_not_a_cooldown() {
        let mut q = KeyQuota {
            last_success: Some(MIDNIGHT + 20 * H),
            ..KeyQuota::default()
        };
        let b = q.on_quota(MIDNIGHT + 21 * H + 23 * M).unwrap();
        assert_eq!(b.kind, BlockKind::NextBoundary);
        assert_eq!(b.resets_at, MIDNIGHT + DAY_SECS);
        assert_eq!(b.until, MIDNIGHT + DAY_SECS + RESET_MARGIN_SECS);
        assert_eq!(b.source, QuotaSource::Default);
        assert_eq!(q.admit(MIDNIGHT + 23 * H + 59 * M), (true, false));
    }

    #[test]
    fn exhaustion_after_answering_past_the_boundary_is_fresh_not_a_failed_probe() {
        // The key answered at 00:10 and ran out at 00:40: the reset already
        // came, so the next one is a day away — no ladder.
        let mut q = KeyQuota {
            last_success: Some(MIDNIGHT + 10 * M),
            ..KeyQuota::default()
        };
        let b = q.on_quota(MIDNIGHT + 40 * M).unwrap();
        assert_eq!(b.kind, BlockKind::NextBoundary);
        assert_eq!(b.resets_at, MIDNIGHT + DAY_SECS);
    }

    #[test]
    fn a_failed_boundary_probe_climbs_a_short_ladder_then_jumps_a_day() {
        // Walled since the evening, never answered since the boundary.
        let mut q = KeyQuota {
            last_success: Some(MIDNIGHT - 4 * H),
            ..KeyQuota::default()
        };
        q.on_quota(MIDNIGHT - 2 * H).unwrap();
        // The probe at 00:05 fails.
        assert_eq!(q.admit(MIDNIGHT + RESET_MARGIN_SECS), (false, true));
        let b1 = q.on_quota(MIDNIGHT + RESET_MARGIN_SECS).unwrap();
        assert_eq!(b1.kind, BlockKind::Ladder(1));
        assert_eq!(b1.until, MIDNIGHT + RESET_MARGIN_SECS + 30 * M);
        let t2 = b1.until;
        q.admit(t2);
        let b2 = q.on_quota(t2).unwrap();
        assert_eq!(b2.kind, BlockKind::Ladder(2));
        assert_eq!(b2.until, t2 + H);
        let t3 = b2.until;
        q.admit(t3);
        let b3 = q.on_quota(t3).unwrap();
        assert_eq!(b3.kind, BlockKind::Ladder(3));
        let t4 = b3.until;
        q.admit(t4);
        let b4 = q.on_quota(t4).unwrap();
        assert_eq!(b4.kind, BlockKind::NextBoundary, "ladder exhausted");
        assert_eq!(b4.resets_at, MIDNIGHT + DAY_SECS);
        // Probes in the whole day: 00:05, 00:35, 01:35, 03:35 — four, not 96.
    }

    #[test]
    fn stragglers_neither_advance_the_ladder_nor_lift_the_wall() {
        let mut q = KeyQuota::default();
        let b = q.on_quota(MIDNIGHT + 12 * H).unwrap();
        assert!(q.on_quota(MIDNIGHT + 12 * H + 5).is_none());
        assert!(!q.on_success(MIDNIGHT + 12 * H + 9));
        assert_eq!(q.blocked_until, Some(b.until));
        assert!(q.reset_samples.is_empty());
    }

    #[test]
    fn first_answer_after_a_close_wall_learns_the_reset() {
        let mut q = KeyQuota::default();
        // 429 at 23:57, the probe answers at 00:03 (as if the wall had been a
        // short ladder window).
        q.on_quota(MIDNIGHT - 3 * M).unwrap();
        q.blocked_until = Some(MIDNIGHT + 3 * M);
        assert_eq!(q.admit(MIDNIGHT + 3 * M), (false, true));
        assert!(q.on_success(MIDNIGHT + 3 * M));
        assert_eq!(q.reset_samples, vec![0]);
        assert_eq!(q.estimate(), (0, QuotaSource::Learned));
        assert_eq!(q.blocked_until, None);
    }

    #[test]
    fn a_wide_bracket_teaches_nothing() {
        let mut q = KeyQuota::default();
        q.on_quota(MIDNIGHT - 3 * H).unwrap();
        let until = q.blocked_until.unwrap();
        q.admit(until);
        assert!(q.on_success(until));
        assert!(q.reset_samples.is_empty());
        assert_eq!(q.estimate().1, QuotaSource::Default);
    }

    #[test]
    fn learned_boundary_moves_the_next_wall() {
        let mut q = KeyQuota {
            reset_samples: vec![25 * M],
            last_success: Some(MIDNIGHT + 8 * H),
            ..KeyQuota::default()
        };
        let b = q.on_quota(MIDNIGHT + 22 * H).unwrap();
        assert_eq!(b.source, QuotaSource::Learned);
        assert_eq!(b.resets_at, MIDNIGHT + DAY_SECS + 25 * M);
    }

    #[test]
    fn samples_are_bounded() {
        let mut q = KeyQuota::default();
        for i in 0..10 {
            let t = MIDNIGHT + i * DAY_SECS;
            q.on_quota(t - 10 * M).unwrap();
            q.blocked_until = Some(t);
            q.admit(t);
            q.on_success(t);
        }
        assert_eq!(q.reset_samples.len(), MAX_RESET_SAMPLES);
    }

    #[test]
    fn success_writes_once_per_boundary_not_per_call() {
        let mut q = KeyQuota::default();
        assert!(q.on_success(MIDNIGHT + H), "first ever");
        assert!(!q.on_success(MIDNIGHT + 2 * H), "same day, nothing new");
        assert!(
            q.on_success(MIDNIGHT + DAY_SECS + H),
            "first after the boundary"
        );
    }

    #[test]
    fn book_persists_and_reloads_by_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quota.json");
        let book = QuotaBook::load(&path);
        book.with_key("f6656650", |q| {
            let b = q.on_quota(MIDNIGHT + 22 * H);
            (b, b.is_some())
        });
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("f6656650"));
        let again = QuotaBook::load(&path);
        let q = again.get("f6656650").unwrap();
        assert_eq!(
            q.blocked_until,
            Some(MIDNIGHT + DAY_SECS + RESET_MARGIN_SECS)
        );
        assert!(again.get("153b1d4d").is_none());
    }

    #[test]
    fn corrupt_or_absent_state_is_empty_not_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quota.json");
        assert!(QuotaBook::load(&path).get("x").is_none());
        std::fs::write(&path, "{ not json").unwrap();
        let book = QuotaBook::load(&path);
        let blocked = book.with_key("x", |q| (q.admit(MIDNIGHT).0, false));
        assert!(!blocked);
    }
}
