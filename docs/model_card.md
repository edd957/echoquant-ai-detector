# Model Card

## Model Details

- Name: Sentinel Transaction Risk Classifier
- Version: 0.1.0
- Model type: Gradient boosted decision tree classifier
- Framework: scikit-learn
- Intended use: Transaction risk prioritization and analyst decision support

## Intended Use

The model estimates whether a transaction should be treated as high risk. It is designed for demos, architecture reviews, and engineering evaluation of ML/AI platform patterns.

It should not be used as a real fraud, credit, sanctions, or compliance system without domain validation, governance review, bias testing, monitoring, and human oversight.

## Inputs

- Transaction amount
- Customer age
- Account age
- Merchant and country risk scores
- Hour of day
- Device trust score
- Failed login count
- Cross-border and new-device indicators

## Outputs

- Risk probability
- Risk band
- Operational recommendation
- Human-readable top signals

## Limitations

- Training data is synthetic.
- Labels are generated from a transparent simulation, not observed fraud outcomes.
- The default explanation method is signal-based rather than SHAP-based.
- The deterministic analyst workflow is not a substitute for regulated decisioning controls.

## Monitoring

The repository includes a PSI-based drift report. In production, this should be expanded with outcome monitoring, calibration checks, fairness slices, latency metrics, and human review outcomes.
