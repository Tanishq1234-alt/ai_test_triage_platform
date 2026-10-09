# AI-Powered Test Automation and Failure Triage Platform

An enterprise-grade, end-to-end platform for automated testing of wireless IoT / embedded devices, failure log ingestion, test-result normalization, ML-powered root cause classification, and interactive triage analytics.

---

## 🏛 Platform Architecture

```
                                    +-------------------------------------------------+
                                    |              Wireless Device DUT                |
                                    |        (State Machine, Fault Injector)          |
                                    +-----------------------+-------------------------+
                                                            |
                                                            v
                                    +-------------------------------------------------+
                                    |           Pytest Test Automation Suite          |
                                    |  (Connectivity, Reconnect, Latency, Loss, RSSI) |
                                    +-----------------------+-------------------------+
                                                            |
                                                            v
                                    +-------------------------------------------------+
                                    |       JUnit XML & Plain-Text Log Parser         |
                                    |  (JUnitParser, PlainTextLogParser, Normalizer)  |
                                    +-----------------------+-------------------------+
                                                            |
                                                            v
                                    +-------------------------------------------------+
                                    |           Normalized Failure Records            |
                                    |       (JSON, CSV, Pandas DataFrame Export)      |
                                    +-----------------------+-------------------------+
                                                            |
                                   +------------------------+------------------------+
                                   |                                                 |
                                   v                                                 v
            +------------------------------------+                +------------------------------------+
            |      Feature Engineering & ML      |                |       SQLite Database Store        |
            |     Failure Triage Classifier      |                |   (Runs, Tests, Logs, Triage)      |
            |   (TF-IDF, Logistic, LinearSVC)    |                +------------------+-----------------+
            +-----------------+------------------+                                   |
                              |                                                      |
                              +--------------------------+---------------------------+
                                                         |
                                                         v
                                    +-------------------------------------------------+
                                    |            FastAPI Backend REST API             |
                                    |       (/tests/run, /triage/classify, /history)  |
                                    +--------------------+----------------------------+
                                                         |
                                                         v
                                    +-------------------------------------------------+
                                    |           Streamlit Triage Dashboard            |
                                    |     (Analytics, Telemetry, Root-Cause Reports)  |
                                    +-------------------------------------------------+
```

---

## 📂 Project Structure

```
ai_test_triage_platform/
├── README.md                          # Platform documentation and architecture
├── requirements.txt                    # Project dependencies
├── pytest.ini                         # Pytest configuration and logging options
├── src/
│   ├── __init__.py
│   ├── log_parser/                    # Phase 2: Log parsing & normalization engine
│   │   ├── __init__.py                # Package facade & correlation engine
│   │   ├── junit_parser.py            # JUnit XML parser (Pytest reports)
│   │   ├── text_parser.py             # Plain-text parser with context windows
│   │   ├── normalizer.py              # NormalizedTestRecord schema & deterministic rules
│   │   └── exporter.py                # JSON & CSV export with historical preservation
│   ├── ml/                            # Phase 3: Machine Learning Failure Triage
│   │   ├── __init__.py                # ML package exports
│   │   ├── dataset.py                 # Dataset loader, label normalizer, synthetic generator
│   │   ├── feature_engineering.py     # TF-IDF, text combiners, ColumnTransformer, pipelines
│   │   ├── model_registry.py          # Model persistence, versioning, safe loading
│   │   ├── evaluate.py                # Cross-validation, metrics, baseline comparisons
│   │   ├── predict.py                 # Inference interface, confidence threshold, explanations
│   │   └── train.py                   # Model training and leakage-safe evaluation workflow
│   ├── models/                        # Domain models
│   │   ├── __init__.py
│   │   └── schemas.py                 # Core Pydantic schemas (TestStatus, FailureCategory)
│   ├── simulator/                     # Wireless device simulation engine
│   │   ├── __init__.py
│   │   ├── device.py                  # WirelessDevice simulation model
│   │   ├── exceptions.py              # Domain exception hierarchy (5 failure modes)
│   │   └── runner.py                  # Automated test runner with fault injection
│   ├── logging_utils/                 # Structured log collector
│   │   ├── __init__.py
│   │   └── logger.py                  # Subsystem tagger, buffer, and file logger
│   └── parser/                        # Phase 1 log parser
│       ├── __init__.py
│       └── log_parser.py              # Telemetry extraction & DataFrame mapper
├── tests/                             # Phase 1–4 regression and unit tests
│   ├── __init__.py
│   ├── test_wireless_suite.py         # Wireless device integration tests (19 tests)
│   ├── test_simulator.py              # Device simulation unit tests (6 tests)
│   ├── test_parser.py                 # Telemetry parser unit tests (6 tests)
│   ├── test_phase4.py                 # Historical pagination, human review, and metric availability
│   ├── fixtures/                      # Representative test fixtures
│   │   ├── sample_junit.xml           # Multi-suite XML (passed, failed, error, skipped)
│   │   ├── malformed.xml              # Malformed XML for resilience testing
│   │   ├── empty.xml                  # Empty XML file
│   │   ├── sample_app.log             # Multi-test plain-text log with stack traces
│   │   └── noisy_logs.log             # Informational logs with word "error"
│   └── unit/                          # Unit tests
│       ├── __init__.py
│       ├── test_junit_parser.py       # JUnit XML parser tests (6 tests)
│       ├── test_text_parser.py        # Plain-text log parser tests (5 tests)
│       ├── test_normalizer.py         # Normalizer & rule categorization tests (3 tests)
│       ├── test_exporter.py           # JSON/CSV exporter & history append tests (4 tests)
│       ├── test_dataset.py            # Dataset loader, labels, synthetic tests (6 tests)
│       ├── test_feature_engineering.py# Feature extractor & pipeline tests (3 tests)
│       ├── test_training.py           # Training, leakage prevention tests (3 tests)
│       ├── test_prediction.py         # Inference, confidence, review tests (4 tests)
│       └── test_evaluation.py         # Metrics, cross-validation, registry tests (3 tests)
├── scripts/
│   ├── generate_sample_run.py         # Generates sample execution data and logs
│   ├── parse_reports.py               # CLI tool to parse existing reports without rerunning
│   ├── run_and_parse.py               # Executes tests and generates normalized records
│   ├── triage_failures.py             # CLI tool applying ML triage to normalized failures
│   ├── run_prioritized_tests.py       # Historical ranking and opt-in bounded test execution
│   └── generate_ci_report.py          # JUnit parsing, ML triage, persistence, JSON/Markdown reporting
├── .github/workflows/ci.yml           # Full-suite CI, reports, and failure-preserving artifacts
├── src/orchestration/prioritizer.py   # Deterministic history-based score and fallback
└── src/notifications/webhook.py      # Optional dry-run-first webhook delivery
├── models/                            # Serialized ML model artifacts
│   ├── failure_classifier_v1.0.0.joblib
│   ├── model_metadata_v1.0.0.json
│   ├── latest.joblib
│   └── latest_metadata.json
├── reports/                           # Model evaluation and benchmark reports
│   └── evaluation_report.json
└── data/
    ├── training/
    │   └── failure_dataset.json       # Documented balanced training dataset
    ├── logs/
    │   ├── current_test_run.log       # Current wireless execution logs
    │   └── sample_wireless_run.log    # Sample wireless execution logs
    ├── normalized/
    │   ├── normalized_results.json    # Normalized failure records (JSON)
    │   └── normalized_results.csv     # Normalized failure records (CSV)
    └── triaged/
        ├── triaged_results.json       # ML-triaged failure records (JSON)
        └── triaged_results.csv        # ML-triaged failure records (CSV)
```

---

## 🤖 Canonical Failure Categories (Phase 3)

The platform maps Phase 2 labels to canonical categories:

1. **`timeout`**: Handshake timeouts, probe response timeouts, ping echo deadlines, DHCP lease acquisition timer expirations.
2. **`assertion_failure`**: SLA violations (RTT latency threshold breaches, packet loss rate limits, RSSI minimum sensitivity limits).
3. **`connection_error`**: Physical RF drops, AP beacon frame losses, carrier wave degradation, link disassociations.
4. **`configuration_error`**: Invalid SSID formats, short WPA2/WPA3 keys, mismatched cipher suites, unsupported RF channels.
5. **`dependency_error`**: Firmware HAL driver lockups, DMA TX queue buffer overruns, I2C telemetry sensor bus faults, kernel crashes.
6. **`unknown`**: Unrecognized exception patterns, unmapped diagnostic codes, or low-confidence predictions triggering human review.

---

## 🛡️ Leakage Prevention & Training Methodology

1. **Strict Temporal/Template Isolation**:
   - Training and test sets are split (`StratifiedShuffleSplit`, 75/25 split, `random_state=42`) *before* any feature extraction.
   - Text vectorizers (`TfidfVectorizer`), scalers (`StandardScaler`), and imputers (`SimpleImputer`) are fitted **strictly on the training split**. Unseen vocabulary in test or inference data never contaminates model parameters.
2. **Grouped Synthetic Templates**:
   - Every synthetic example carries a `template_group` and `is_synthetic=True` marker, allowing clear tracking of synthetic vs real records.
3. **Reproducibility**:
   - Random seed is pinned (`seed=42`) across dataset generation, stratified splitting, cross-validation, and model fitting.

---

## 📈 Measured Model Evaluation vs Baseline

Evaluated on 75 unseen test samples (stratified holdout split from 300 balanced records):

| Canonical Category | Test Support | ML Precision | ML Recall | ML F1-Score | Baseline Precision | Baseline Recall | Baseline F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **assertion_failure** | 13 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| **configuration_error** | 11 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| **connection_error** | 13 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 0.9231 | 0.9600 |
| **dependency_error** | 13 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| **timeout** | 13 | **1.0000** | **1.0000** | **1.0000** | 0.9286 | 1.0000 | 0.9630 |
| **unknown** | 12 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| **Overall Macro-F1** | 75 | **1.0000** | **1.0000** | **1.0000** | 0.9881 | 0.9872 | **0.9872** |
| **Weighted F1** | 75 | **1.0000** | **1.0000** | **1.0000** | 0.9876 | 0.9867 | **0.9866** |

- **5-Fold Cross Validation Mean Macro-F1**: `1.0000 (+/- 0.0000)`
- **Confusion Matrix**: Zero misclassifications by the ML model on the holdout evaluation set.

---

## 🚀 Commands & Workflows

### 1. Run Complete Automated Test Suite (Phases 1 + 2 + 3)
```powershell
python -m pytest tests -v
```
Runs the integration, unit, ML pipeline, and Phase 4 regression tests.

### 2. Train and Evaluate ML Models
```powershell
python -m src.ml.train --data data/normalized/normalized_results.json --samples-per-class 50 --seed 42 --version v1.0.0
```
Outputs model artifact to `models/failure_classifier_v1.0.0.joblib` and evaluation metrics to `reports/evaluation_report.json`.

### 3. Triage Test Failures with ML Classifier
```powershell
python scripts/triage_failures.py --input data/normalized/normalized_results.json --output-dir data/triaged --threshold 0.50
```
Enriches each failure record with `ml_triage` metadata (`predicted_category`, `confidence`, `needs_review`, `top_features`, `rule_based_category`).

### 4. Parse Reports Without Rerunning Tests
```powershell
python scripts/parse_reports.py --xml data/test_junit.xml --log data/logs/current_test_run.log --output-dir data/normalized
```

### 5. Start the Phase 4 API and dashboard
```powershell
uvicorn src.api.main:app --reload
```
In a second terminal, run:
```powershell
streamlit run frontend/dashboard.py
```
The dashboard reads historical metrics and failures from the API, supports human review corrections, and can submit a log for triage. Set `TRIAGE_API_URL` if the API is not running at `http://127.0.0.1:8000`. Set `SQLITE_DB_PATH` to change the SQLite database location.

### 6. Phase 5 CI and consolidated reports

GitHub Actions runs the complete Pytest suite on pushes and pull requests, writes `reports/junit.xml`, parses the results, applies the Phase 3 classifier when its artifact loads, persists records into a CI-local SQLite database, and creates `reports/ci_report.json` and `reports/ci_report.md`. The reports and JUnit XML are uploaded even when tests fail; the job still ends as failed when tests or report generation fail. Workflow execution on GitHub must be confirmed in the Actions tab.

Run the full suite and produce local reports with:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
python -m pytest tests --junitxml=reports/junit.xml
python scripts/generate_ci_report.py --junit reports/junit.xml --output-dir reports
```

Pytest returns a nonzero exit code on test failures, so in a shell that stops on errors, run report generation in a `finally`/always step as the workflow does. The JSON report includes counts, durations, failed test details and excerpts, rule and ML categories, confidence/review state, SQLite persistence counts, model version and the existing Phase 3 evaluation report when present. Missing model/evaluation data is recorded as a warning. Failure recall is left unmeasured because the repository does not contain an independent labeled defect set.

### 7. Historical test prioritization

Full-suite execution is the default and remains the CI behavior. To see rankings without changing which tests run:

```powershell
python scripts/run_prioritized_tests.py --mode rank-only --report-json reports/prioritization.json -- tests
```

To explicitly execute a bounded subset, use the prioritized mode:

```powershell
python scripts/run_prioritized_tests.py --mode prioritized --limit 20 --full-suite-every 7 --report-json reports/prioritization.json -- --junitxml=reports/prioritized.xml tests
python scripts/generate_ci_report.py --junit reports/prioritized.xml --output-dir reports/prioritized --prioritization-json reports/prioritization.json
```

This mode ranks by historical failure rate, failure recency, and execution duration. A test without history receives a zero score and follows known-risk tests; with no usable history, test node IDs are ordered deterministically. Every seventh prioritized invocation runs the full suite as a safety check (adjustable with `--full-suite-every`). The full mode is the default. Ranking alone does not skip tests; only the explicit `--mode prioritized` execution applies the test-count limit. The Phase 3 classifier classifies failure causes, not the probability that a test will fail, so it is not used as a ranking model. No learned ranker is trained from the currently available history.

The report generator can compare two actual report files for selected test counts and summed test durations:

```powershell
python scripts/generate_ci_report.py --junit reports/prioritized.xml --output-dir reports/prioritized --prioritization-json reports/prioritization.json --comparison-report reports/full/ci_report.json
```

This is an execution-time comparison only. It does not establish equivalent code coverage or defect detection, and failure recall remains null without reliable labels. The dashboard shows the most recent local `reports/ci_report.json` and offers JSON/Markdown downloads when those files exist. Copy a downloaded GitHub Actions artifact into `reports/` to inspect it locally.

### 8. Optional webhook notifications

Notifications are disabled unless `TRIAGE_NOTIFICATION_ENABLED=true`. Configure `TRIAGE_WEBHOOK_URL` through a local environment variable or a GitHub Actions secret, never a committed credential. Dry-run mode is enabled by default and can be set with `TRIAGE_NOTIFICATION_DRY_RUN=true`; the report records skipped, dry-run, delivered, or failed status without exposing the webhook destination. Set `TRIAGE_REPORT_URL` if the report is available at a shareable URL. No notification service is required for CI or local report generation.

For GitHub Actions delivery, set the repository variable `TRIAGE_NOTIFICATION_ENABLED` to `true`, add secret `TRIAGE_WEBHOOK_URL`, and set repository variable `TRIAGE_NOTIFICATION_DRY_RUN` to `false` only when ready to send. `TRIAGE_REPORT_URL` can be set as a repository variable when a stable report link exists.

---

## ⚙️ Confidence Thresholding & Human Review

To prevent unreliable classifications:
- Predictions with probability $< \text{threshold}$ (default: `0.50`) are flagged with `needs_review = True`, and the category is marked as `unknown`.
- Failures matching the `unknown` pattern explicitly set `needs_review = True` and trigger a review reason: `"Model classified failure as unknown/unrecognized pattern"`.
- Empty/corrupted input records set `needs_review = True` with `"Insufficient log content"`.

---

## ⚠️ Known Limitations

1. **Probability Calibration**: Softmax probabilities in Logistic Regression represent decision confidence within the trained feature space and are not strictly calibrated true probabilities.
2. **Out-of-Distribution Vocabulary**: Novel failure modes outside the trained TF-IDF vocabulary will yield low confidence scores and correctly trigger `needs_review = True`.
3. **Model Artifact Trust**: Serialized `joblib` artifacts should only be loaded from trusted local paths (`models/`).

---

## 🎯 Implementation Roadmap
- [x] **Phase 1: Foundation & Test Automation Suite** (31 tests passing)
- [x] **Phase 2: Test Log Parser & Structured Failure Records** (49 total tests passing)
- [x] **Phase 3: Machine Learning Failure Classifier vs Baseline** (68 total tests passing)
  - [x] Canonical categories and normalization
  - [x] Labeled synthetic dataset generation with group tracking
  - [x] TF-IDF + ColumnTransformer scikit-learn pipeline
  - [x] Candidate comparison: Logistic Regression vs Calibrated Linear SVC
  - [x] Leakage prevention (strict pre-split fitting)
  - [x] Precision, recall, F1, confusion matrix, and rule-based baseline comparison
  - [x] Prediction interface with confidence thresholding and feature explanations
  - [x] Model registry with versioning and safe loading
  - [x] Integration CLI (`scripts/triage_failures.py`)
- [x] **Phase 4: SQLite Persistence, FastAPI Backend, and Streamlit Triage Dashboard**
- [x] **Phase 5: CI/CD, historical prioritization, and consolidated reporting**
