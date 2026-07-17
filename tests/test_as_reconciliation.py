"""Smoke test for the bucket-classification logic in the AS-reconciliation notebook."""


def assign_bucket(c4_elig, c4_AS, c6_AS):
    if c4_AS and c6_AS:
        return "both"
    if c4_AS and not c6_AS:
        return "causal4_only"
    if c6_AS and c4_elig:
        return "causal6_only_eligible"
    if c6_AS and not c4_elig:
        return "causal6_only_newly_eligible"
    return "neither_AS"


def test_buckets():
    assert assign_bucket(True, True, True) == "both"
    assert assign_bucket(True, True, False) == "causal4_only"
    assert assign_bucket(True, False, True) == "causal6_only_eligible"
    assert assign_bucket(False, False, True) == "causal6_only_newly_eligible"
    assert assign_bucket(True, False, False) == "neither_AS"
    # Edge: c4_AS True implies c4_eligible True in practice; classifier still
    # routes c4_AS+c6_AS to "both" regardless of c4_eligible.
    assert assign_bucket(False, True, True) == "both"


if __name__ == "__main__":
    test_buckets()
    print("OK")
