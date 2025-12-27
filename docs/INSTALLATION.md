# Installation Guide

## Quick Installation

The easiest way to install all dependencies in the correct order:

```bash
# Make the script executable (if not already)
chmod +x install_dependencies.sh

# Run the installation script
./install_dependencies.sh
```

This script handles the dependency ordering automatically, ensuring PyTorch is installed before PyTorch Geometric extensions.

## Manual Installation (Step-by-Step)

If you prefer to install manually or the script doesn't work:

### 1. Install Core PyTorch and NLP Libraries

```bash
# PyTorch (CPU version)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Transformers and Sentence Transformers
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.2
```

### 2. Install PyTorch Geometric

```bash
pip install torch-geometric>=2.3.0
```

### 3. Install PyG Extensions

**Important**: These require PyTorch to be installed first!

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 4. Install spaCy and Language Model

```bash
pip install spacy>=3.6.0
python3 -m spacy download en_core_web_sm
```

### 5. Install Remaining Dependencies

```bash
pip install rank-bm25>=0.2.2
pip install faiss-cpu>=1.7.4
pip install datasets>=2.14.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install networkx>=3.1
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
pip install jsonlines>=3.1.0
pip install pytest>=7.4.0
pip install jupyter>=1.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

## Troubleshooting

### Issue: `torch-scatter` build fails

**Cause**: PyTorch not installed before building `torch-scatter`.

**Solution**: Install PyTorch first:
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install PyG extensions:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue: spaCy model not found

**Solution**: Download the English model:
```bash
python3 -m spacy download en_core_web_sm
```

### Alternative: Skip GNN Dependencies

If you don't need GNN-based multi-hop reasoning, you can skip the PyTorch Geometric dependencies:

```bash
# Install everything except PyTorch Geometric
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.2
pip install rank-bm25>=0.2.2
pip install faiss-cpu>=1.7.4
# ... rest of dependencies
```

Then set `use_gnn: false` in `config/config.yaml`.

## Verification

Test the installation:

```bash
# Test dataset loading
python3 src/data/dataset_loader.py

# Run unit tests
python3 -m pytest tests/ -v

# Quick retrieval test
python3 -c "from src.retrieval.bm25_retriever import BM25Retriever; print('✓ Import successful')"
```
