# Fact Verification System - Containerization Guide

This document describes how to use Docker and Docker Compose to develop and deploy the Fact Verification System.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed.
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+ recommended).

## Docker Setup

### Building the Image
To build the Docker image manually:
```bash
docker build -t fact-verification-system .
```
Or using the provided Makefile:
```bash
make docker-build
```

### Running the Container
To run a single container:
```bash
docker run -p 8000:8000 fact-verification-system
```

## Docker Compose
Using Docker Compose is the recommended way to manage the application, especially for persistence and volume mounting.

### Start the Service
```bash
docker-compose up -d
```
The Web UI will be available at `http://localhost:8000`.

### Stop the Service
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

## Makefile Reference
The project includes a `Makefile` to simplify common operations.

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies in your current environment. |
| `make test` | Run the complete test suite with coverage report. |
| `make lint` | Perform Pylint quality checks (Score target > 9.5). |
| `make run` | Launch the Web UI locally. |
| `make clean` | Remove temporary cache and build files. |
| `make docker-build` | Build the Fact Verification Docker image. |
| `make docker-run` | Launch the application inside a Docker container. |

## Persistence
The `docker-compose.yml` file is configured to mount the `./data` and `./models` directories. This ensures that:
- Downloaded datasets are preserved across container restarts.
- Trained GNN models remain available.
- FAISS/BM25 indices are stored on the host.

## Development with Docker
To enable hot-reloading while developing (source code mapping), uncomment the `./src:/app/src` volume in `docker-compose.yml`.
