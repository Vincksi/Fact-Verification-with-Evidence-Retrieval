# User Guide - Testing and Code Quality

## Tool Installation

Testing and quality tools are already included in `requirements.txt`:

```bash
pip install pytest pytest-cov coverage pylint
```

Or use the full installation script:

```bash
./scripts/setup/install_dependencies.sh
```

## Test Suite Overview

Here is a breakdown of the tests included in the project:

### Data & Loading
- **`tests/data/test_dataset_loader.py`**: Validates the loading of the SciFact dataset (Corpus, Claims) and the integrity of data objects (`Claim`, `Document`).

### Retrieval System
- **`tests/retrieval/test_retrieval.py`**: Functional tests for the BM25 retriever (index building, search).
- **`tests/retrieval/test_retrievers.py`**: Unit tests for `DenseRetriever` and `HybridRetriever` (initialization, retrieval logic).
- **`tests/retrieval/test_bm25_extended.py`**: Specific tests for BM25 edge cases (tokenization, empty corpus) and persistence.
- **`tests/retrieval/test_persistence.py`**: Verifies that retriever indices (Dense/Hybrid) can be saved to and loaded from disk correctly.

### Verification System
- **`tests/verification/test_nli_model.py`**: Unit tests for the NLI model (DeBERTa), mocking the Hugging Face pipeline to test entailment logic without heavy model loading.
- **`tests/verification/test_multi_hop.py`**: Verification of the Graph Neural Network (GNN) reasoner and graph construction.
- **`tests/verification/test_verification.py`**: General verification logic tests.
- **`tests/verification/test_aggregator_extended.py`**: Tests for evidence aggregation strategies (majority vote, confidence weighted, etc.).

### Pipeline & Integration
- **`tests/pipeline/test_pipeline.py`**: Integration tests for the full `FactVerificationPipeline`, ensuring all components work together (Loader -> Retriever -> Verifier).
- **`tests/pipeline/test_pipeline_gat.py`**: Specific integration test for the pipeline with GAT (Graph Attention Network) enabled.

### Metrics
- **`tests/metrics/test_retrieval_metrics.py`**: Tests for retrieval metrics (Precision@k, Recall@k, MAP, NDCG, etc.).
- **`tests/metrics/test_verification_metrics.py`**: Tests for verification metrics (Accuracy, F1, FEVER score).

---

## Testing with Coverage

### 1. Quick Execution

```bash
./scripts/quality/test_coverage.sh
```

This command:
- Runs all tests
- Generates a coverage report in the terminal
- Creates an HTML report in `htmlcov/index.html`

### 2. Full Quality Analysis

```bash
./scripts/quality/run_quality_checks.sh
```

This command executes:
- Tests with full coverage
- Pylint analysis of source code
- Generates detailed reports

### 3. Manual Commands

**Tests with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

**Tests for a specific module:**
```bash
pytest tests/retrieval/test_retrieval.py --cov=src/retrieval --cov-report=term
```

**Tests with markers:**
```bash
# Unit tests only
pytest -m unit

# Exclude slow tests
pytest -m "not slow"
```

## Pylint Analysis

### Full Analysis

```bash
pylint src/ --rcfile=.pylintrc
```

### Specific Module Analysis

```bash
pylint src/retrieval/ --rcfile=.pylintrc
```

### Generate Detailed Report

```bash
pylint src/ --rcfile=.pylintrc --output-format=json > pylint_report.json
```

### Score Only

```bash
pylint src/ --rcfile=.pylintrc | grep "Your code has been rated"
```

## Coverage Reports

### Available Formats

1. **Terminal** (--cov-report=term-missing):
   - Immediate display in console
   - Shows missing lines

2. **HTML** (--cov-report=html):
   - Interactive report in `htmlcov/index.html`
   - File navigation
   - Highlighting of covered/uncovered lines

3. **XML** (--cov-report=xml):
   - Standard format for CI/CD
   - `coverage.xml` file

### Viewing HTML Report

```bash
# Linux
xdg-open htmlcov/index.html

# Mac
open htmlcov/index.html

# Windows
start htmlcov/index.html

# Or directly in browser
firefox htmlcov/index.html
```

## Configuration

### pytest.ini

Configures:
- Test discovery patterns
- Coverage options
- Test markers
- Exclusions (warnings, etc.)

### .pylintrc

Configures:
- Style rules (PEP 8)
- Complexity limits
- Ignored warnings
- Output format

### Customization

**Modify minimum coverage threshold:**

In `pytest.ini`:
```ini
[tool:pytest]
addopts = --cov-fail-under=80  # Fails if < 80%
```

**Disable specific Pylint rules:**

In `.pylintrc` section `[MESSAGES CONTROL]`:
```ini
disable=
    C0111,  # missing-docstring
    # Add your rules here
```

## CI/CD Integration

### GitHub Actions example

```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ --cov=src --cov-report=xml
    
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### GitLab CI example

```yaml
test:
  script:
    - pip install -r requirements.txt
    - pytest tests/ --cov=src --cov-report=term --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Best Practices

### Testing

1. **Naming**: `test_<function>_<scenario>.py`
2. **Organization**: One test file per module
3. **Target Coverage**: Aim for 80%+ for critical code
4. **Markers**: Use `@pytest.mark.slow` for slow tests

### Code Quality

1. **Pylint Score**: Aim for 8.0/10 minimum
2. **Complexity**: Keep functions simple (max 15 branches)
3. **Documentation**: Docstrings for public classes and functions
4. **Style**: Follow PEP 8 (max 120 chars per line)

## Useful Commands

```bash
# Install quality tools
pip install pytest-cov pylint coverage

# Quick tests without coverage
pytest tests/ -v

# Coverage for specific module
pytest tests/test_retrieval.py --cov=src/retrieval

# Generate coverage badge
coverage-badge -o coverage.svg

# Clean cache files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf htmlcov/ .coverage coverage.xml
```

## Troubleshooting

### Import errors during tests

Ensure PYTHONPATH includes the project directory:
```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```
*(Note: `scripts/quality/test_coverage.sh` handles this automatically)*

### Pylint finds too many errors

1. Check `.pylintrc` for disabled rules
2. Use `# pylint: disable=<rule>` on specific lines
3. Adjust thresholds in config

### Coverage too low

1. Identify untested modules: `coverage report`
2. Add tests for critical cases
3. Use `# pragma: no cover` for debug-only code
