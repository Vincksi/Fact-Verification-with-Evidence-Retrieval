# Code Quality and Testing - Quick Reference

## Quick Commands

### Test Coverage
```bash
# Quick test with coverage
./scripts/quality/test_coverage.sh

# View HTML report
open htmlcov/index.html
```

### Code Quality (Pylint)
```bash
# Full quality check (tests + pylint)
./scripts/quality/run_quality_checks.sh

# Pylint only
pylint src/ --rcfile=.pylintrc
```

### Standard Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_retrieval.py -v

# With coverage details
pytest tests/ --cov=src --cov-report=term-missing
```

## Generated Reports

- **HTML Coverage**: `htmlcov/index.html` (interactive, detailed)
- **XML Coverage**: `coverage.xml` (CI/CD compatible)
- **Terminal**: Real-time coverage display

## Configuration Files

- `.pylintrc` - Pylint rules and settings
- `pytest.ini` - Pytest and coverage configuration  
- `.gitignore` - Files to exclude from git

## See TESTING_GUIDE.md for more details
