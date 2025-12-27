#!/bin/bash
# Installation script for fact verification project
# Handles PyTorch Geometric dependency ordering

echo "=========================================="
echo "Installing Fact Verification Dependencies"
echo "=========================================="

# Step 1: Install core PyTorch and transformers first
echo ""
echo "Step 1: Installing PyTorch and core NLP libraries..."
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.2

# Step 2: Install PyTorch Geometric dependencies
echo ""
echo "Step 2: Installing PyTorch Geometric..."
pip install torch-geometric>=2.3.0

# Step 3: Install PyG extensions (now that torch is available)
echo ""
echo "Step 3: Installing PyG extensions (torch-scatter, torch-sparse)..."
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Step 4: Install spaCy and download model
echo ""
echo "Step 4: Installing spaCy..."
pip install spacy>=3.6.0
python3 -m spacy download en_core_web_sm

# Step 5: Install remaining dependencies
echo ""
echo "Step 5: Installing remaining dependencies..."
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
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install fastapi>=0.68.0
pip install uvicorn>=0.15.0
pip install python-multipart>=0.0.5
pip install jinja2>=3.0.0
pip install groq>=0.5.0
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To verify installation, run:"
echo "  python3 src/data/dataset_loader.py"
echo "  python3 -m pytest tests/ -v"
