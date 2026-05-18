# Architecture

Sentinel AI Risk Platform is organized as a small but production-shaped AI system.

## Components

1. Data generation
   - Creates synthetic transaction records with calibrated behavioral risk factors.
   - Produces a reference dataset and a drifted current dataset.

2. Model training
   - Uses a scikit-learn `Pipeline` with explicit preprocessing and a histogram gradient boosting classifier.
   - Exports a single joblib artifact for serving.

3. Inference API
   - FastAPI validates inputs with Pydantic v2 models.
   - `/v1/predict` returns calibrated risk probability, risk band, recommended action, and top signals.
   - `/v1/analyze` adds retrieved policy context and analyst workflow actions.

4. Retrieval layer
   - Uses local TF-IDF retrieval over a policy corpus.
   - Keeps the default project self-contained and auditable.

5. Drift monitoring
   - Computes Population Stability Index for each feature.
   - Flags distribution changes that deserve investigation before model quality degrades.

## Extension Points

- Replace TF-IDF retrieval with a vector database.
- Add MLflow tracking with the optional `mlops` dependency group.
- Add a human feedback table and retraining scheduler.
- Add OpenTelemetry traces for API and model latency.
- Add provider-backed LLM summaries behind the deterministic analyst workflow.
