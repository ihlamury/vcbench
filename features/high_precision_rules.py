# features/high_precision_rules.py
def apply_rules(row) -> tuple[int | None, str]:
    """
    Returns (prediction, rule_name) if a high-confidence rule fires.
    Returns (None, None) if no rule fires — falls through to classifier.
    """
    # Rule 1: Prior exit history — near-deterministic
    if row.get('exit_count', 0) > 0:
        return 1, "prior_exit"

    # Rule 2: Top-10 QS + STEM + founding role
    if (row.get('edu_prestige_tier', 0) >= 4 and
        row.get('stem_flag', 0) == 1 and
        row.get('founding_role_count', 0) > 0):
        return 1, "top10_stem_founder"

    # Rule 3: DISABLED — precision 10% on train set (below 30% threshold)
    # if (row.get('max_seniority_reached', 0) >= 4 and
    #     row.get('max_company_size_before_founding', 0) >= 6 and
    #     row.get('founding_role_count', 0) > 0):
    #     return 1, "clevel_large_company_founder"

    return None, None
