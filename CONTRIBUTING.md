# Contributing

Thank you for reviewing Sentinel AI Risk Platform.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make bootstrap
make test
```

## Pull Request Checklist

- Add or update tests for changed behavior.
- Run `ruff check .`, `mypy src`, and `pytest`.
- Update documentation when APIs, model behavior, or operational assumptions change.
- Keep synthetic data only in the public repository.

## Design Principles

- Prefer explicit, auditable ML workflows over hidden automation.
- Keep model decisions explainable enough for analyst review.
- Treat monitoring and data quality as first-class production concerns.
