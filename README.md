# VCBench Task IV — Founder Success Prediction

Structured feature engineering and signal-limit analysis for predicting startup founder success from career history. Companion code for the SecureFinAI Contest @ IEEE IDS 2026.

---

## Citation

```
Ihlamur, Y. (2026). When Career Data Runs Out: Structured Feature Engineering and Signal Limits for Founder Success Prediction. SecureFinAI Contest @ IEEE IDS 2026. arXiv:2604.00339. https://arxiv.org/abs/2604.00339
```

Camera-ready PDF: [`paper/paper_camera_ready.pdf`](paper/paper_camera_ready.pdf)

---

## Key Results

| Model | Val F₀.₅ | Private Test F₀.₅ | Precision | Recall |
|---|---|---|---|---|
| Zero-shot LLM baseline | 0.1265 | — | — | — |
| Rule layer (prior exit only) | 0.2889 | — | — | — |
| Structured features v1 (23 feat) | 0.2203 | — | — | — |
| **Structured v2 + HPO (28 feat, XGB)** | **0.3030** | **0.2811** | **0.3281** | **0.1802** |
| + LLM features (100% coverage) | 0.3065 | 0.2534 (CV) | — | — |

**Core finding:** LLM prose extraction is informationally redundant with structured JSON parsing — structured features alone match or exceed LLM-augmented approaches on this dataset. The private test ceiling (F₀.₅ ≈ 0.28) reflects a structural data limitation: 84% of positive founders (332/405) are statistically indistinguishable from failures on all available structured features.

---

## Repository Structure

```
vcbench/
├── classifier.py              # Main XGBoost training + 5-fold CV evaluation
├── predict.py                 # Generate predictions on a new CSV
├── evaluate.py                # Evaluation harness — F₀.₅, precision, recall (DO NOT MODIFY)
├── features/
│   ├── extract_structured.py  # 28-feature structured extraction pipeline (Tiers 1–4)
│   ├── extract_llm.py         # LLM feature extraction via Claude Haiku (experimental)
│   ├── high_precision_rules.py# High-precision rule layer (prior exit override)
│   └── FEATURE_SUMMARY.md     # Feature distributions and lift statistics on train set
├── baselines/
│   └── zero_shot_baseline.py  # Zero-shot Claude Sonnet baseline (F₀.₅ = 0.1265)
├── tests/
│   └── test_extract_structured.py  # 90 unit tests for feature extraction
├── data/
│   ├── split.py               # Stratified 80/20 split script (run once, locked)
│   ├── public_train.csv        # Training split (3,600 rows) — not included, see Data note
│   └── public_val.csv          # Validation split (900 rows) — not included, see Data note
├── paper/
│   ├── paper_final.tex             # Original submission LaTeX source
│   ├── paper_final.pdf             # Original submission PDF
│   ├── paper_camera_ready.tex      # Camera-ready LaTeX source (accepted)
│   └── paper_camera_ready.pdf      # Camera-ready compiled PDF
├── experiments/
│   ├── run_calibration.py     # Platt scaling experiment (not used in final model)
│   └── platt_scaler.pkl       # Calibration artifact from above
├── docs/
│   ├── plan.md                # Initial 4-phase implementation plan
│   ├── plan_v2.md             # Revised plan post-Step 7 analysis
│   ├── program.md             # Phase 4 overnight loop specification
│   └── references.md          # Related papers (Vela Research, RRF, policy induction)
├── experiment_log.md          # Append-only log of 120+ experiments with F₀.₅ deltas
├── requirements.txt
└── LICENSE
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/ihlamury/vcbench
cd vcbench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Obtain the dataset

The VCBench dataset is a commercial dataset available at [vcbench.com](https://vcbench.com). Place the files as follows:

```
data/vcbench_final_public.csv   # 4,500 rows (public split)
```

Then regenerate train/val splits:

```bash
python data/split.py
# Outputs: data/public_train.csv (3,600 rows), data/public_val.csv (900 rows)
```

### 3. Train the model and reproduce validation results

```bash
python classifier.py
```

Expected output (approximately):
```
=== 5-fold CV (with rules) ===
CV F0.5: 0.2539 ± 0.0348
Rule layer overrides: {'prior_exit': ...}
=== Val set result ===
{'f05': 0.3030, 'precision': 0.3333, 'recall': 0.2222, 'threshold': 0.738, ...}
```

### 4. Run inference on new data

```bash
python predict.py
```

`predict.py` expects `data/vcbench_final_private.csv` by default. To run on your own CSV, edit `PRIVATE_DATA_PATH` at the top of `predict.py`. The script validates that the positive prediction rate falls in the expected 4–15% range before saving.

---

## Feature Documentation

All 28 features are extracted by `features/extract_structured.py` from the `educations_json`, `jobs_json`, `ipos`, and `acquisitions` columns of the VCBench dataset.

### Tier 1 — Prior Exit Signals (3 features)

The strongest predictors. Rare (2.9% of founders) but highly discriminative.

| Feature | Description | Lift |
|---|---|---|
| `has_prior_ipo` | Binary: founder previously led a company to IPO | 4.7× |
| `has_prior_acquisition` | Binary: founder previously led an acquisition | 2.7× |
| `exit_count` | Sum of above (0–2) | 2.8× at ≥1 |

### Tier 2 — Sacrifice Signal (5 features)

Captures opportunity cost: founders who left prestigious, high-stability positions.

| Feature | Description |
|---|---|
| `max_company_size_before_founding` | Ordinal (0–9): largest employer size before founding |
| `prestige_sacrifice_score` | `max_company_size × max_seniority` before founding |
| `years_in_large_company` | Total years at 501+ employee companies pre-founding |
| `comfort_index` | Weighted tenure in high-stability industries (banking, consulting, government) |
| `founding_timing` | Total inferred years of experience before first founding role |

> Note: Tier 2 features show inverted signal on this dataset — false positives have *higher* sacrifice scores than true positives. The model uses these features but with low weight.

### Tier 3 — Education × Prestige Interaction (6 features)

| Feature | Description | Lift |
|---|---|---|
| `edu_prestige_tier` | QS ranking tier: top-10=4, top-50=3, top-100=2, ranked=1, none=0 | 3.0× at tier 4 |
| `field_relevance_score` | Field relevance to startup industry (1–5) | 1.7× at score 5 |
| `prestige_x_relevance` | `edu_prestige_tier × field_relevance_score` (key interaction term) | 5.5× at score 20 |
| `degree_level` | PhD=4, MBA/JD/MD=3, MS=2, BS/BA=1, other=0 | 1.5× at PhD |
| `stem_flag` | Binary: any STEM degree | 1.4× |
| `best_degree_prestige` | Alias of `edu_prestige_tier` (highest-prestige institution) | — |

> `prestige_x_relevance` (score=20: top-10 QS + STEM field + tech startup) → 24.2% success rate vs 4.4% at score=1. This is the strongest education signal.

### Tier 4 — Career Trajectory (9 features)

| Feature | Description |
|---|---|
| `max_seniority_reached` | Highest seniority level: Founder/C-level=5, VP=4, Director=3, Senior=2, IC=1, Intern=0 |
| `seniority_is_monotone` | Binary: seniority non-decreasing throughout career |
| `company_size_is_growing` | Binary: company size non-decreasing throughout career |
| `restlessness_score` | Count of roles with duration < 2 years |
| `founding_role_count` | Total founding-stage roles (company size ≤ 10 or Founder/C-level title) |
| `longest_founding_tenure` | Duration (years) of longest founding-stage role |
| `industry_pivot_count` | Count of distinct industries across career |
| `industry_alignment` | Binary: any prior job in same industry as current startup |
| `total_inferred_experience` | Total career years (sum of duration midpoints) |

### v2 Interaction Features (5 features)

Derived features added in Step 7 based on cross-tier signal analysis.

| Feature | Formula | Rationale |
|---|---|---|
| `is_serial_founder` | `founding_role_count ≥ 2` | Serial founders (6+ roles) have 17.5% success rate |
| `exit_x_serial` | `exit_count × founding_role_count` | Amplifies exit signal for serial founders |
| `sacrifice_x_serial` | `prestige_sacrifice_score × is_serial_founder` | Sacrifice signal only meaningful for serials |
| `industry_prestige_penalty` | `edu_prestige_tier × is_biotech_or_vc` | Biotech/VC sectors inflate prestige without signal |
| `persistence_score` | `longest_founding_tenure / (total_inferred_experience + 0.01)` | Fraction of career spent in founding roles |

---

## Model Configuration

Final model (XGBoost decision stumps, Optuna-optimized, 200 trials):

```python
MODEL_PARAMS = dict(
    n_estimators=227,
    max_depth=1,               # Decision stumps — forced by heavy regularization
    learning_rate=0.0674,
    subsample=0.949,
    colsample_bytree=0.413,
    scale_pos_weight=10,       # Compensates for 91%/9% class imbalance
    min_child_weight=14,
    gamma=4.19,
    reg_alpha=0.73,
    reg_lambda=15.0,
    eval_metric="logloss",
    random_state=42,
)
FINAL_THRESHOLD = 0.738        # Optimized for F₀.₅ on validation set
```

**Rule layer** (applied after model scoring): if `exit_count > 0`, override probability to 1.0. This fires on ~3% of rows with precision 24.5% (2.7× base rate).

All deeper tree configurations (max_depth 2–6), ensemble methods, and LightGBM variants converged to worse CV performance. The dataset's scale (4,500 rows, 405 positives) and the two-population structure (see RESULTS.md) favour additive stumps.

---

## Reproducing the Validation Result (F₀.₅ = 0.3030)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset files in data/ (see Quickstart above)

# 3. Train and evaluate
python classifier.py
# Prints: Val F0.5: 0.3030, Precision: 0.3333, Recall: 0.2222, threshold: 0.738
```

The random seeds are fixed (`random_state=42` in both the data split and the model). Minor floating-point variation across platforms is possible but should not exceed ±0.001 F₀.₅.

To run feature extraction in isolation:

```bash
python features/extract_structured.py
# Prints null rates and feature distributions for all 28 features on public_train.csv
```

---

## Data Note

The VCBench dataset is a commercial dataset provided by Vela Partners through the VCBench benchmark ([vcbench.com](https://vcbench.com)). The public split (4,500 rows) is available to registered contest participants; the private test set (4,500 rows) is held by the organizers.

This repository does **not** include any dataset files. You must obtain the data independently and place it in the `data/` directory as described in the Quickstart.

The train/val split (`data/split.py`, stratified 80/20, `random_state=42`) is locked and must not be re-generated if you wish to reproduce the reported numbers.

---

## Roadmap & Next Steps

**Paper**
- [x] Camera-ready submitted — see `paper/paper_camera_ready.pdf`
- [x] Section V updated with private test results (F₀.₅ = 0.2811, fold breakdown, confusion matrix)
- [x] Independent work footnote added (work conducted independently of Amazon)
- [x] arXiv preprint posted: https://arxiv.org/abs/2604.00339

**Model & Features**
- [x] Full feature extraction pipeline open-sourced (`features/extract_structured.py`)
- [ ] Ablation study on Tier 4 features — several showed marginal CV gains that warrant deeper analysis with a larger dataset
- [ ] `repeat_founding_gap` feature dropped due to 73.8% null rate — worth investigating with imputation or a richer dataset

**Benchmark Suggestions for VCBench v2**

The structured ceiling analysis revealed that 84% of positive founders (332/405 on private test) are statistically near-identical to failures on all available structured features. The following new fields could break this ceiling:
- [ ] Founding year (temporal market context)
- [ ] Co-founder count and network quality signals
- [ ] Investor tier at first funding round
- [ ] Finer industry granularity (current encoding loses signal at sub-sector level)
- [ ] Exit type detail (acqui-hire vs. strategic acquisition vs. IPO)

**Research Directions**
- [ ] Formal two-population analysis as a standalone benchmark paper (NeurIPS 2026 Datasets & Benchmarks track target)
- [ ] Cross-benchmark generalization: do structured features transfer to other founder success datasets beyond VCBench?

---

## License

MIT License — see [LICENSE](LICENSE). Note: the VCBench dataset itself is not covered by this license.

## Contact

Yagiz Ihlamur — https://www.linkedin.com/in/yagizihlamur/
