"""
Basic tests for the retrieval module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import SciFactDataset, Document
from src.retrieval.base_retriever import RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def sample_corpus():
    """Create a small sample corpus for testing."""
    corpus = {
        1: Document(
            doc_id=1,
            title="Aspirin and Heart Disease",
            abstract=["Aspirin reduces heart attack risk.", "It inhibits platelet aggregation."],
            structured=False
        ),
        2: Document(
            doc_id=2,
            title="Diabetes Treatment",
            abstract=["Metformin is used for diabetes.", "It improves insulin sensitivity."],
            structured=False
        ),
        3: Document(
            doc_id=3,
            title="Aspirin Side Effects",
            abstract=["Aspirin can cause stomach bleeding.", "Some patients experience side effects."],
            structured=False
        )
    }
    return corpus


def test_bm25_retriever_initialization(sample_corpus):
    """Test BM25 retriever initialization."""
    retriever = BM25Retriever(sample_corpus)
    assert retriever.corpus is not None
    assert len(retriever.corpus) == 3


def test_bm25_build_index(sample_corpus):
    """Test BM25 index building."""
    retriever = BM25Retriever(sample_corpus)
    retriever.build_index()
    
    assert retriever.bm25 is not None
    assert len(retriever.doc_ids) == 3
    assert len(retriever.tokenized_corpus) == 3


def test_bm25_retrieve(sample_corpus):
    """Test BM25 retrieval."""
    retriever = BM25Retriever(sample_corpus)
    retriever.build_index()
    
    # Query about aspirin
    results = retriever.retrieve("aspirin heart attack", top_k=2)
    
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].doc_id in [1, 3]  # Should retrieve aspirin-related docs
    assert results[0].score > 0


def test_bm25_no_results_for_unknown_terms(sample_corpus):
    """Test retrieval with unknown terms."""
    retriever = BM25Retriever(sample_corpus)
    retriever.build_index()
    
    results = retriever.retrieve("quantum physics", top_k=2)
    
    # Should still return results (all docs with score 0 or very low)
    assert len(results) == 2


def test_retrieval_result_structure(sample_corpus):
    """Test that retrieval results have correct structure."""
    retriever = BM25Retriever(sample_corpus)
    retriever.build_index()
    
    results = retriever.retrieve("aspirin", top_k=1)
    result = results[0]
    
    assert hasattr(result, 'doc_id')
    assert hasattr(result, 'score')
    assert hasattr(result, 'title')
    assert hasattr(result, 'text')
    assert hasattr(result, 'rank')
    assert result.rank == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
