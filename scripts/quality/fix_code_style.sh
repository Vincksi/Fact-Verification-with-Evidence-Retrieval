#!/bin/bash
# Comprehensive Pylint fixes to achieve 9.5/10

source nlp_env/bin/activate

echo "========================================"
echo "AUTO-FIXING PYLINT ISSUES"
echo "========================================"

# Fix 1: Remove unused imports automatically
echo "1. Installing autoflake to remove unused imports..."
pip install -q autoflake

echo "2. Removing unused imports and variables..."
autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/

# Fix 2: Fix import order
echo "3. Installing isort for import ordering..."
pip install -q isort

echo "4. Fixing import order (PEP 8)..."
isort src/ --profile black --line-length 120

# Fix 3: Fix trailing whitespaces and basic formatting
echo "5. Removing trailing whitespaces..."
find src/ -name "*.py" -type f -exec sed -i 's/[[:space:]]*$//' {} +

# Fix 6: Auto-format code style
echo "6. Installing autopep8..."
pip install -q autopep8

echo "7. Auto-formatting with autopep8..."
autopep8 --in-place --aggressive --aggressive --max-line-length 120 --recursive src/

echo ""
echo "========================================"
echo "Fixes Applied! Checking score..."
echo "========================================"

# Run pylint to see new score
pylint src/ --rcfile=.pylintrc --score-only

echo ""
echo "Run './run_quality_checks.sh' for full report"
