# Experiment Log — VCBench Task IV

All experiments are logged here in append-only format. Never delete entries.

---

## Step 1 — 2026-03-08
**Change:** Created `evaluate.py` (FIXED — never modify)
**Smoke test result:** f05=1.0, precision=1.0, recall=1.0, positive_rate=0.09, n_predicted_positive=9, threshold=0.804
**Verdict:** PASS — evaluate.py locked.

## Step 2 — 2026-03-08
**Change:** Created `data/split.py` — 80/20 stratified train/val split (random_state=42)
**Dataset:** 4,500 rows, 9.0% positive (405 success / 4,095 failure)
**Train:** 3,600 rows, 9.0% positive
**Val:** 900 rows, 9.0% positive
**Output:** `data/public_train.csv`, `data/public_val.csv`
**Verdict:** PASS — split locked. Do not re-run.

## Step 3 — 2026-03-08
**Change:** Created `features/extract_structured.py` — Tier 1–4 feature engineering (23 features)
**Null rates:** 0.0% across all 23 features
**Key findings:**
- exit_count: 106 founders with exits (2.9%). Success rate: 22.8% (1 exit), 60.0% (2 exits) vs 8.5% baseline
- edu_prestige_tier: top-10=421, top-50=479, top-100=219, ranked=2066, null=415
- max_seniority_reached: 67% are founder/C-level (5)
- prestige_sacrifice_score: good spread 0–45
**Bug fixed:** ipos/acquisitions fields use Python single-quote dict syntax, not JSON. Added `ast.literal_eval` fallback.
**Verdict:** PASS — all features extracted, no nulls.

## Step 3b — 2026-03-08
**Change:** Created `tests/test_extract_structured.py` — 90 unit + integration tests for all 23 features
**Bug found & fixed:** `_get_seniority` had substring false positives — "cto" matched inside "director", "ceo" inside "director". Fixed with word-boundary regex matching. Also moved junior/intern check before IC-level to prevent "Junior Developer" matching "developer" first.
**Test results:** 90/90 passed (2.77s)
**Output:** `features/FEATURE_SUMMARY.md` — full distribution and success-rate-by-value for all 23 features
**Verdict:** PASS — all features validated. See `features/FEATURE_SUMMARY.md` for detailed distributions.

## Step 5 — 2026-03-08
**Change:** Zero-shot Claude Sonnet 4.5 baseline on 900 val rows (`anonymised_prose` only)
**Model:** claude-sonnet-4-5-20250929
**F0.5:** 0.1265 (prev best: none — this is the first baseline)
**Precision:** 0.105
**Recall:** 0.7037
**Threshold:** 0.327
**Positive rate:** 60.3% (543/900 predicted positive)
**Notes:** Model outputs are poorly calibrated — 60% of predictions cluster at exactly 0.50, almost nothing above 0.60. The model predicts way too many positives, yielding very low precision. This is significantly worse than the GPT-4o zero-shot baseline on the leaderboard (25.7% F0.5). Possible causes: (1) Sonnet 4.5 is overly generous in success probability estimates, (2) the system prompt calibration is insufficient, (3) the anonymised prose doesn't contain enough discriminative signal for zero-shot approaches. This baseline strongly motivates the structured feature engineering approach.
**Verdict:** LOGGED — baseline established at F0.5=0.1265. All subsequent models must beat this.
