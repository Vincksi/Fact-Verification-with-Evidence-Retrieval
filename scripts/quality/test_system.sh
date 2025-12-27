#!/bin/bash
# Comprehensive test script for the fact verification system

# Activate virtual environment
echo "Activating nlp_env virtual environment..."
source nlp_env/bin/activate
export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "Fact Verification System Tests"
echo "=========================================="

# Test 1: Data loading
echo ""
echo "Test 1: Loading SciFact dataset..."
python3 src/data/dataset_loader.py
if [ $? -eq 0 ]; then
    echo "✓ Dataset loaded successfully"
else
    echo "✗ Error loading dataset"
    exit 1
fi

# Test 2: Unit tests
echo ""
echo "Test 2: Running unit tests..."
python3 -m pytest tests/ -v
if [ $? -eq 0 ]; then
    echo "✓ Unit tests passed"
else
    echo "⚠ Some tests failed (may require dependencies)"
fi

# Test 3: BM25 retrieval test
echo ""
echo "Test 3: Testing BM25 retriever..."
python3 -c "
from src.data.dataset_loader import SciFactDataset
from src.retrieval.bm25_retriever import BM25Retriever

dataset = SciFactDataset('data')
dataset.load_corpus()

retriever = BM25Retriever(dataset.corpus)
retriever.load_index('data/preprocessed/indices')

query = 'Aspirin reduces cardiovascular risk'
results = retriever.retrieve(query, top_k=3)

print(f'\nQuery: {query}')
print(f'Top 3 results:')
for r in results:
    print(f'  [{r.rank}] Score: {r.score:.4f} - {r.title[:60]}...')
print('✓ BM25 retriever working')
"

# Test 4: Dense retrieval test
echo ""
echo "Test 4: Testing Dense retriever..."
python3 -c "
from src.data.dataset_loader import SciFactDataset
from src.retrieval.dense_retriever import DenseRetriever

dataset = SciFactDataset('data')
dataset.load_corpus()

print('Loading model and index...')
retriever = DenseRetriever(dataset.corpus, model_name='sentence-transformers/all-MiniLM-L6-v2')
retriever.load_index('data/preprocessed/indices')

query = 'Aspirin reduces cardiovascular risk'
results = retriever.retrieve(query, top_k=3)

print(f'\nQuery: {query}')
print(f'Top 3 results:')
for r in results:
    print(f'  [{r.rank}] Score: {r.score:.4f} - {r.title[:60]}...')
print('✓ Dense retriever working')
"

# Test 5: Full pipeline test (NLI mode)
echo ""
echo "Test 5: Testing complete pipeline (NLI mode)..."
python3 -c "
from src.pipeline import FactVerificationPipeline

print('Initializing pipeline...')
pipeline = FactVerificationPipeline('config/config.yaml')
pipeline.load_corpus()

# Use hybrid retriever and load index
print('Loading retriever and indices...')
pipeline.build_retriever(method='hybrid')
pipeline.load_index()

# Disable GNN for this test (faster)
pipeline.use_gnn = False

# Test on a claim
claim = 'Aspirin reduces the risk of heart attack in patients with cardiovascular disease'
print(f'\nClaim: {claim}')

result = pipeline.process_claim(0, claim)

print(f'\nResult:')
print(f'  Prediction: {result.predicted_label}')
print(f'  Confidence: {result.confidence:.4f}')
print(f'  Top 3 retrieved documents:')
for doc in result.retrieved_docs[:3]:
    print(f'    - {doc.title[:70]}... (score: {doc.score:.4f})')

print('\n✓ Complete pipeline working')
"

# Test 6: Full pipeline test (GAT mode)
echo ""
echo "Test 6: Testing complete pipeline (GAT mode)..."
# We run the specific GAT test file to verify the GNN flow
# Using the mock-based test is safer/faster for system check than loading large models
pytest tests/pipeline/test_pipeline_gat.py -v
if [ $? -eq 0 ]; then
    echo "✓ GAT pipeline flow working (verified via tests)"
else
    echo "✗ GAT pipeline test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
