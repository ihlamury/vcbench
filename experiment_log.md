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

## Step 4 — 2026-03-08
**Change:** Created `features/high_precision_rules.py` — deterministic rule layer
**Rule performance on training set:**
- Rule 1 (prior_exit): fired=106, precision=24.5% — KEPT
- Rule 2 (top10_stem_founder): fired=134, precision=21.6% — KEPT
- Rule 3 (clevel_large_company_founder): fired=783, precision=10.0% — DISABLED (below 30% threshold)
**Verdict:** Rules 1 & 2 active. Rule 3 disabled due to low precision.

## Step 6 — 2026-03-08
**Change:** Created `classifier.py` — XGBoost on 23 structured features (Tiers 1–4) + rule layer
**F0.5:** 0.2203 (prev best: 0.1265 zero-shot)
**Precision:** 0.1957
**Recall:** 0.4444
**Threshold:** 0.512
**Positive rate:** 20.4% (184/900)
**Rule layer overrides:** prior_exit=36, top10_stem_founder=35
**Top 5 features by importance:**
1. exit_count (0.078)
2. edu_prestige_tier (0.076)
3. best_degree_prestige (0.070)
4. industry_alignment (0.055)
5. prestige_sacrifice_score (0.046)
**Notes:** Beats zero-shot baseline by +9.4pp. Below Phase 2 target of 28%. Feature importances show exit signals and education prestige dominate. Sacrifice signal (prestige_sacrifice_score) is in top-5. The threshold sweep shows F0.5 is fairly flat across 0.50–0.95, suggesting the model lacks confidence separation. Next steps: hyperparameter tuning, feature interactions, or threshold calibration.
**Verdict:** KEEP — first structured model. Needs improvement to hit 28% target.

## Step 7 — 2026-03-08
**Change:** Manual inspection of val set predictions (TP=36, FN=45, FP=148, TN=671)

**True Positives (36) — what the model gets right:**
Most TPs are caught by the rule layer (exit_count=1 or top10+STEM+founder). The model correctly identifies founders with prior exits, top-10 QS education, and high prestige_x_relevance scores. Average TP profile: exit_count=0.58, edu_prestige_tier=2.78, prestige_x_relevance=11.67. These are founders with clear, measurable pedigree signals.

**False Negatives (45) — successful founders the model misses:**
FNs tend to have low edu_prestige_tier (avg ~1.3), no exits, and lower prestige scores. Many are serial operators at 200+ QS schools with deep domain expertise — their signal comes from career trajectory patterns (long tenures, domain focus) that the current features don't weight enough. Several FNs have high prestige_sacrifice_score (45) but low education prestige — the sacrifice signal alone isn't enough without education prestige to boost it.

**False Positives (148) — the precision problem:**
The FP rate is the core issue (148 FPs vs 36 TPs). Many FPs look indistinguishable from TPs on our features — top-10 QS, STEM, founder role, high sacrifice score. The rule layer is responsible for many FPs: the top10_stem_founder rule fires on 35 val rows but many are failures. The model struggles to separate "impressive resume" from "actually successful founder." FPs cluster in biotech/research and VC/PE industries where elite credentials are common but success rates are still low.

**Feature comparison across groups:**
| Feature | TP avg | FN avg | FP avg |
|---|---|---|---|
| exit_count | 0.58 | 0.04 | 0.11 |
| edu_prestige_tier | 2.78 | 1.31 | 3.10 |
| prestige_sacrifice_score | 21.31 | 18.09 | 25.22 |
| industry_alignment | 0.47 | 0.33 | 0.38 |
| prestige_x_relevance | 11.67 | 3.91 | 13.24 |
| stem_flag | 0.56 | 0.40 | 0.68 |
| max_seniority_reached | 4.64 | 3.67 | 4.78 |
| founding_role_count | 2.22 | 1.11 | 1.84 |

**Key insight:** FPs actually have HIGHER edu_prestige_tier, prestige_sacrifice_score, and prestige_x_relevance than TPs. The model is over-indexing on education prestige. What separates TPs from FPs is primarily exit_count (0.58 vs 0.11) and founding_role_count (2.22 vs 1.84). The rule layer's top10_stem_founder rule is too aggressive.

**Hypotheses for Phase 4:**
1. Tighten the top10_stem_founder rule (add more conditions or disable it)
2. Add industry-specific base rate features (biotech/research has lower success despite elite credentials)
3. Add serial founder signal — founding_role_count interacted with tenure
4. Reduce scale_pos_weight to lower recall and boost precision
5. Add feature: ratio of founding tenure to total experience (commitment signal)
