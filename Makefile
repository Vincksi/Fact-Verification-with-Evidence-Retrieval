.PHONY: install test lint run clean docker-build docker-run help

# Variables
IMAGE_NAME = fact-verification-system
PORT = 8000

help:
	@echo "Available commands:"
	@echo "  install      : Install dependencies locally"
	@echo "  test         : Run tests with coverage"
	@echo "  lint         : Run Pylint checks"
	@echo "  run          : Start the Web UI locally"
	@echo "  clean        : Remove temporary files"
	@echo "  docker-build : Build Docker image"
	@echo "  docker-run   : Run Docker container"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	./scripts/quality/test_coverage.sh

lint:
	./scripts/quality/run_quality_checks.sh

run:
	./scripts/run/start_web_ui.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "Cleanup complete."

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)
