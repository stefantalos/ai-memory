-- A scheduler claim used to be permanent the moment it was taken, before any
-- model call. A run that FAILED (provider 429 quota wall, 5xx, malformed
-- response) wrote no auto_improve_runs row but kept its claim, and the
-- candidate query excludes every claimed session, so the session was never
-- reviewed. Measured 2026-09-24: 32 substantive sessions (8..2749
-- observations) stranded this way in 24h, 24 of them by Poolside 429s.
--
-- A failed run now RELEASES its claim (released_at) so a later tick can take
-- it again. failed_attempts counts only failures attributable to the session
-- (5xx, timeout, malformed output, ...); a lane that is unavailable (429,
-- connect refused, auth, not configured) says nothing about the session and
-- does not spend the budget. At the bound the claim stays released but
-- ineligible, with the last error kept so the terminal state is named.
--
-- Constant defaults: ADD COLUMN is a metadata-only change in SQLite.
ALTER TABLE auto_improve_scheduler_claims ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE auto_improve_scheduler_claims ADD COLUMN released_at INTEGER;
ALTER TABLE auto_improve_scheduler_claims ADD COLUMN last_error TEXT;
