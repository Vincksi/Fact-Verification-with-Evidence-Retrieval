#!/bin/bash
# Run all code quality checks

# Activate virtual environment
source nlp_env/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"

echo "=========================================="
echo "Code Quality Checks"
echo "=========================================="

# Run tests with coverage
echo ""
echo "Running tests with coverage..."
PYTHONPATH=. pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

if [ $? -eq 0 ]; then
    echo "✓ Tests passed"
else
    echo "✗ Some tests failed"
    exit 1
fi

# Run pylint on source code
echo ""
echo "=========================================="
echo "Running Pylint on source code..."
echo "=========================================="

pylint src/ --rcfile=.pylintrc --output-format=colorized

PYLINT_SCORE=$?

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

# Display coverage summary
echo ""
echo "Coverage Report:"
if [ -f "htmlcov/index.html" ]; then
    echo "  HTML report: htmlcov/index.html"
fi
if [ -f "coverage.xml" ]; then
    echo "  XML report: coverage.xml"
fi

echo ""
echo "Pylint Analysis:"
if [ $PYLINT_SCORE -eq 0 ]; then
    echo "  ✓ Code quality: Excellent (10/10)"
else
    echo "  ⚠ See pylint output above for details"
fi

echo ""
echo "=========================================="
echo "Quality checks complete!"
echo "=========================================="
