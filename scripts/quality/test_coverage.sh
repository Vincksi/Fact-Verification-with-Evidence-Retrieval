#!/bin/bash
# Quick test run with coverage summary

source nlp_env/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"

echo "Running tests with coverage..."
PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

echo ""
echo "Coverage report generated in htmlcov/index.html"
