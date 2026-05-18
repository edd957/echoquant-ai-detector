.PHONY: bootstrap test lint typecheck run-api train evaluate drift clean

bootstrap:
	sentinel generate-data --rows 5000
	sentinel train
	sentinel evaluate

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

run-api:
	uvicorn sentinel_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

train:
	sentinel train

evaluate:
	sentinel evaluate

drift:
	sentinel drift-report

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov']]"
