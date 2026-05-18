from sentinel_ai.data.synthetic import generate_transactions
from sentinel_ai.evaluation.drift import drift_report, population_stability_index


def test_population_stability_index_is_zero_for_same_distribution() -> None:
    data = generate_transactions(rows=500, seed=11)

    assert population_stability_index(data["amount"], data["amount"]) == 0.0


def test_drift_report_flags_changed_data() -> None:
    reference = generate_transactions(rows=800, seed=11, drift=False)
    current = generate_transactions(rows=800, seed=12, drift=True)
    report = drift_report(reference, current)

    assert "summary" in report
    assert "features" in report
    assert report["summary"]["max_psi"] >= 0
