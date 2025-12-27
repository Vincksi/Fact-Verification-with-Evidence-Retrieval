# Fact Verification with Evidence Retrieval

A CPU-friendly, 2-stage fact verification system that retrieves evidence from scientific literature and verifies claims using Natural Language Inference (NLI) and **Optimized Graph Attention Networks (GATv2)** for multi-hop reasoning.

## Latest Results (SciFact Dev)
- **Retrieval Recall@10**: **80.2%** (Hybrid HNSW-RRF)
- **Bias Correction**: Successfully resolved "Supports Machine" bias; model now actively predicts and distinguishes between `SUPPORTS`, `REFUTES`, and `NEI` labels.
- **CPU Latency**: **<250ms** end-to-end processing (Intel i7).

## Features

**Multi-Strategy Evidence Retrieval**
- BM25 sparse retrieval (lexical matching)
- Dense retrieval using Sentence-BERT (semantic similarity)
- Hybrid retrieval with Reciprocal Rank Fusion

**Advanced Claim Verification**
- NLI-based verification with DeBERTa (**INT8 ONNX Optimized**)
- **GATv2-based** multi-hop reasoning (dynamic attention + edge embeddings)
- **Bias Mitigation**: Balanced oversampling and NLI threshold recalibration (0.1)
- Entity extraction and graph construction
- Multiple evidence aggregation strategies
- **LLM-powered explanations** via Groq API (Llama 3.3, Mixtral)

**CPU-Optimized**
- Lightweight models (all-MiniLM-L6-v2, DeBERTa-v3-small)
- **HNSW Indexing** (FAISS) for logarithmic search complexity
- Optimized GATv2 architecture (2 layers, 256-dim hidden, 4 heads)

**Comprehensive Evaluation**
- Retrieval metrics: P@k, R@k (80.2% @ k=10), MAP, MRR, NDCG@k
- Verification metrics: Accuracy, F1 (macro/micro/per-class)
- FEVER score (joint retrieval + verification)
- **87% test coverage** with 89 passing tests
- Detailed **Research Report** (included in `reports/`)

## Installation

### Prerequisites
- Python 3.8+
- CPU (GPU optional but not required)

### Setup

1. **Clone and navigate to the project:**
```bash
cd /home/vincksi/nlp_project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model (for entity extraction):**
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Build Retrieval Index

Build indices for all retrieval methods:

```bash
python scripts/data/manage_index.py build
```

This will create BM25, dense, and hybrid indices from the SciFact corpus (~5,184 documents).

### 2. Run Evaluation

Evaluate the complete pipeline on the dev set:

```bash
python scripts/evaluation/evaluate.py --split dev --load-index --eval-mode both
```

Options:
- `--split`: Dataset split (`train`, `dev`, `test`)
- `--eval-mode`: What to evaluate (`retrieval`, `verification`, `both`)
- `--load-index`: Load pre-built index (faster)
- `--limit`: Limit number of claims (for quick testing)
- `--save-predictions`: Save predictions to JSON file

### 3. Use the Pipeline Programmatically

```python
from src.pipeline import FactVerificationPipeline

# Initialize pipeline
pipeline = FactVerificationPipeline('config/config.yaml')
pipeline.load_corpus()
pipeline.build_retriever(method='hybrid')  # or 'bm25', 'dense'

# Verify a claim
claim = "Aspirin reduces the risk of heart attack"
result = pipeline.process_claim(0, claim)

print(f"Predicted: {result.predicted_label}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Top evidence: {result.retrieved_docs[0].title}")
```

### 4. Interactive Web Interface

The system comes with a modern dashboard to verify claims and visualize graph attention in real-time.

```bash
bash scripts/run/start_web_ui.sh
```

Then visit `http://localhost:8000` in your browser.

> [!TIP]
> For more details on the interface features (GAT graphs, etc.), see the [Web UI Guide](docs/web_ui.md).

## Project Structure

```
nlp_project/
├── config/
│   └── config.yaml                     # Configuration (models, hyperparameters)
├── data/
│   ├── corpus.jsonl                    # 5,184 scientific abstracts
│   ├── claims_train.jsonl              # Training claims
│   ├── claims_dev.jsonl                # Development claims
│   ├── claims_test.jsonl               # Test claims
│   └── preprocessed/                   # Built indices
├── src/
│   ├── data/
│   │   └── dataset_loader.py           # SciFact dataset loader
│   ├── retrieval/
│   │   ├── base_retriever.py           # Abstract retriever interface
│   │   ├── bm25_retriever.py           # BM25 sparse retrieval
│   │   ├── dense_retriever.py          # Dense semantic retrieval
│   │   └── hybrid_retriever.py         # Hybrid RRF fusion
│   ├── verification/
│   │   ├── nli_model.py                # NLI-based verifier
│   │   ├── evidence_aggregator.py      # Evidence fusion strategies
│   │   └── multi_hop_reasoner.py       # GNN-based multi-hop reasoning
│   ├── evaluation/
│   │   ├── retrieval_metrics.py        # P@k, R@k, MAP, MRR, NDCG
│   │   └── verification_metrics.py     # Accuracy, F1, FEVER score
│   └── pipeline.py                     # End-to-end pipeline
├── scripts/
│   ├── setup/                          # Environment setup
│   ├── data/                           # Indexing and data management
│   ├── training/                       # Model training (GAT)
│   ├── evaluation/                     # Evaluation metrics and scripts
│   ├── quality/                        # Code quality, linting, and tests
│   └── run/                            # Application and demo scripts
├── tests/                              # Unit and integration tests
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Architecture

### Stage 1: Evidence Retrieval (Hybrid HNSW-RRF)

The system supports three retrieval methods:

1. **BM25**: Traditional term-based retrieval using lexical matching.
2. **Dense**: Semantic retrieval using MiniLM-L6 embeddings and **FAISS HNSW** index.
3. **Hybrid**: Combines BM25 + Dense using **Reciprocal Rank Fusion (RRF)**.

### Stage 2: Claim Verification (GATv2)

Two verification modes:

1. **NLI Mode** (default):
   - Uses cross-encoder NLI model (DeBERTa-v3-small)
   - **INT8 ONNX Quantization** for 2x faster CPU inference.
   - Fast and accurate for single-hop reasoning.

2. **GNN Mode** (multi-hop):
   - Constructs heterogeneous graph (claim, sentences, entities).
   - **2-layer GATv2** with dynamic attention and Edge Type Embeddings.
   - **Balanced Training**: Uses oversampling to resolve the "Supports" bias.
   - Better for complex multi-hop scientific reasoning.

## Configuration

Edit `config/config.yaml` to customize:

**Retrieval settings:**
```yaml
retrieval:
  method: "hybrid"  # bm25, dense, or hybrid
  top_k: 10
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"
```

**Verification settings:**
```yaml
verification:
  nli_model: "cross-encoder/nli-deberta-v3-small"
  aggregation: "confidence_weighted"

multi_hop:
  use_gnn: false  # Set to true for GNN-based verification
  gnn:
    num_layers: 2
    hidden_dim: 256
    num_heads: 4
```

## Evaluation Metrics

### Retrieval Metrics
- **Precision@k**: What fraction of top-k results are relevant?
- **Recall@k**: What fraction of relevant docs are in top-k?
- **MAP**: Mean Average Precision across all queries
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain

### Verification Metrics
- **Accuracy**: Overall label prediction accuracy
- **F1 Score**: Macro and micro F1 across all labels
- **FEVER Score**: Accuracy with correct evidence retrieved

## CPU Optimization Strategies

1. **Model Selection**:
   - Sentence-BERT: `all-MiniLM-L6-v2` (80MB, 384-dim)
   - NLI: `DeBERTa-v3-small` (~140MB)

2. **GNN Optimizations**:
   - Limited graph size (max 20 evidence sentences)
   - Lightweight architecture (2 layers, 256 hidden dim)
   - Efficient sparse operations with PyG

3. **Quantization & Runtime**:
   - **ONNX Runtime (CPU)**: Models quantized to INT8 for 2-3x faster inference.
   - **HNSW Indexing**: FAISS HNSW index for O(log N) retrieval speed.
   - **Persistent Caching**: Embeddings and NLI results cached to disk.

4. **Docker Optimization**:
   - **Multi-stage builds**: Final image contains only runtime dependencies.
   - **CPU-only Wheels**: Optimized PyTorch installation (no CUDA overhead).

## SciFact Dataset

The project uses the **SciFact** dataset, which is designed for verifying scientific claims against abstracts from the scientific literature (PubMed).

### Dataset Statistics
- **Corpus**: 5,184 scientific paper abstracts (each containing 5-15 sentences).
- **Claims**: Over 1,400 scientific claims, split into:
  - `train`: 809 claims
  - `dev`: 300 claims
  - `test`: 300 claims

### Data Structure
- `corpus.jsonl`: Contains `doc_id`, `title`, and `abstract` (list of sentences).
- `claims_*.jsonl`: Contains `id`, `claim`, and `evidence` (mapping of doc IDs to sentence indices and labels).

### Labels
Claims are categorized into three classes:
- **SUPPORTS**: The evidence supports the claim.
- **REFUTES**: The evidence contradicts the claim.
- **NOT_ENOUGH_INFO**: No conclusive evidence found in the corpus.

Reference: [SciFact: Verifying Scientific Claims](https://github.com/allenai/scifact)

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Retrieval Methods

1. Subclass `BaseRetriever` in `src/retrieval/`
2. Implement `build_index()` and `retrieve()` methods
3. Update configuration and pipeline

### Adding New Verification Methods

1. Create new module in `src/verification/`
2. Update `pipeline.py` to support the new method
3. Add evaluation metrics if needed

## Performance Expectations

On CPU (Intel i5/i7):

| Component | Time (per query) | Metric (Recall@10 / Acc) |
|-----------|------------------|--------------------------|
| BM25 Retrieval | ~12ms | 0.654 |
| Dense Retrieval | ~48ms | 0.712 |
| **Hybrid Retrieval** | **~55ms** | **0.802** |
| NLI (INT8 ONNX) | ~100ms | ~0.70 |
| **GATv2 Reasoning** | **~180ms** | **~0.72** |

**Total Pipeline Latency**: ~225ms (end-to-end).

## Troubleshooting

**Issue**: `ModuleNotFoundError` when running scripts

**Solution**: Run from project root or add to PYTHONPATH:
```bash
export PYTHONPATH=/home/vincksi/nlp_project:$PYTHONPATH
```

**Issue**: spaCy model not found

**Solution**: Download the model:
```bash
python -m spacy download en_core_web_sm
```

**Issue**: Out of memory with GNN

**Solution**: Reduce graph size in `config.yaml`:
```yaml
multi_hop:
  graph:
    max_evidence_sentences: 10  # Reduce from 20
```

## Future Work

1.  **Contrastive Hard-Negative Mining**: Forcing the model to distinguish between extremely similar sentences with inverted polarities.
2.  **Symbolic Numerical Layer**: Integrating a calculator or logic engine for quantitative claims.
3.  **Cross-Abstract GNNs**: Linking entities across multiple papers to improve global scientific synthesis.

## Citation

If you use this code, please cite:

```bibtex
@misc{nlp_fact_verification,
  title={CPU-Friendly Fact Verification with GATv2 and ONNX Optimization},
  author={Kerrian Le Bars},
  institution={CentraleSupélec},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- SciFact dataset: Allen Institute for AI
- Sentence-BERT: UKP Lab
- PyTorch Geometric: PyG Team