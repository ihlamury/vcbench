# classifier.py
"""
Main classifier. This file is iterated on by Claude Code in Phase 4.
Current version: XGBoost on structured features (Tiers 1-4).
evaluate.py is FIXED and defines the metric. Do not modify evaluate.py.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from evaluate import evaluate, sweep_thresholds, best_threshold
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules

FEATURE_COLS = [
    # Tier 1
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    # Tier 2 — sacrifice signal
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    # Tier 3 — education x QS
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    # Tier 4 — trajectory
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
]


def train_and_evaluate():
    train = extract_features(pd.read_csv("data/public_train.csv"))
    val = extract_features(pd.read_csv("data/public_val.csv"))

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["success"]
    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["success"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,   # ~91/9 class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Get base probabilities from model
    probs = model.predict_proba(X_val)[:, 1].tolist()

    # Apply high-precision rule layer — override probabilities for rule hits
    rule_stats = {}
    for idx_pos, (idx, row) in enumerate(val.iterrows()):
        rule_pred, rule_name = apply_rules(row)
        if rule_pred == 1:
            probs[idx_pos] = 1.0
            if rule_name not in rule_stats:
                rule_stats[rule_name] = 0
            rule_stats[rule_name] += 1

    print(f"Rule layer overrides: {rule_stats}")

    result = best_threshold(y_val.tolist(), probs)
    print(f"\n=== Structured features baseline ===")
    print(result)

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(importances.head(10))

    # Also show full sweep
    print("\n=== Threshold sweep (top 10) ===")
    sweep = sweep_thresholds(y_val.tolist(), probs)
    for r in sweep[:10]:
        print(r)

    joblib.dump(model, "model.pkl")
    return result


if __name__ == "__main__":
    train_and_evaluate()
