from sentinel_ai.data.synthetic import FEATURE_COLUMNS, TARGET_COLUMN, generate_transactions


def test_generate_transactions_has_expected_shape() -> None:
    data = generate_transactions(rows=250, seed=7)

    assert list(data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert len(data) == 250
    assert data[TARGET_COLUMN].isin([0, 1]).all()


def test_drifted_data_changes_distribution() -> None:
    reference = generate_transactions(rows=500, seed=7, drift=False)
    current = generate_transactions(rows=500, seed=7, drift=True)

    assert current["is_new_device"].mean() > reference["is_new_device"].mean()
