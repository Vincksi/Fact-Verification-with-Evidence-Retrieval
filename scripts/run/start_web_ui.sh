#!/bin/bash
# Script to launch the Custom Web UI (FastAPI)

echo "Starting Premium Fact Verification UI..."
export PYTHONPATH=$PYTHONPATH:.
# Launch Uvicorn with reload enabled for development
# Host 0.0.0.0 is accessible from browser typically via localhost
./nlp_env/bin/python3 -m uvicorn src.ui.server:app --host 0.0.0.0 --port 8000 --reload
