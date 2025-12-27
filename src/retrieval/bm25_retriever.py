"""
BM25-based sparse retrieval for evidence documents.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from .base_retriever import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """BM25 sparse retrieval using lexical matching."""

    def __init__(self, corpus: Dict[int, Any], **kwargs):
        """
        Initialize BM25 retriever.

        Args:
            corpus: Dictionary mapping doc_id to Document objects
            **kwargs: Additional parameters (unused for BM25)
        """
        super().__init__(corpus, **kwargs)
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and lowercasing."""
        # Use simple replacement for punctuation
        for char in ".,!?;:()[]{}":
            text = text.replace(char, " ")
        return text.lower().split()

    def build_index(self):
        """Build BM25 index from corpus."""
        print("Building BM25 index...")

        # Sort documents by ID for consistent ordering
        sorted_docs = sorted(self.corpus.items(), key=lambda x: x[0])

        self.doc_ids = []
        self.tokenized_corpus = []

        for doc_id, doc in sorted_docs:
            self.doc_ids.append(doc_id)
            # Combine title and abstract for full-text search
            full_text = doc.full_text
            self.tokenized_corpus.append(self._tokenize(full_text))

        if not self.tokenized_corpus:
            print("Warning: Empty corpus, skipping BM25 index build.")
            return

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index built with {len(self.doc_ids)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents using BM25 scoring.

        Args:
            query: The query text (claim)
            top_k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects, sorted by BM25 score
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self.doc_ids[idx]
            doc = self.corpus[doc_id]

            result = RetrievalResult(
                doc_id=doc_id,
                score=float(scores[idx]),
                title=doc.title,
                text=doc.full_text,
                rank=rank
            )
            results.append(result)

        return results

    def save_index(self, path: str):
        """
        Save BM25 index to disk.

        Args:
            path: Directory path to save index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        index_data = {
            'bm25': self.bm25,
            'doc_ids': self.doc_ids,
            'tokenized_corpus': self.tokenized_corpus
        }

        with open(path / 'bm25_index.pkl', 'wb') as f:
            pickle.dump(index_data, f)

        print(f"BM25 index saved to {path / 'bm25_index.pkl'}")

    def load_index(self, path: str):
        """
        Load BM25 index from disk.

        Args:
            path: Directory path containing the index
        """
        path = Path(path)
        index_file = path / 'bm25_index.pkl'

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.bm25 = index_data['bm25']
        self.doc_ids = index_data['doc_ids']
        self.tokenized_corpus = index_data['tokenized_corpus']

        print(f"BM25 index loaded from {index_file}")


def main():  # pragma: no cover
    """Example usage of BM25Retriever."""
    from src.data.dataset_loader import SciFactDataset

    # Load dataset
    dataset_example = SciFactDataset("data")
    dataset_example.load_corpus()

    # Create and build BM25 retriever
    retriever_example = BM25Retriever(dataset_example.corpus)
    retriever_example.build_index()

    # Test retrieval
    query_text = "Aspirin reduces the risk of heart attack"
    results_list = retriever_example.retrieve(query_text, top_k=5)

    print(f"\nTop 5 results for: '{query_text}'")
    for result in results_list:
        print(f"\nRank {result.rank}: {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Doc ID: {result.doc_id}")


if __name__ == "__main__":  # pragma: no cover
    main()
