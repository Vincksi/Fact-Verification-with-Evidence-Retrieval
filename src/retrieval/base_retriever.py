"""
Base retriever interface for evidence retrieval.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    doc_id: int
    score: float
    title: str
    text: str
    rank: int = 0

    def __repr__(self):
        return f"RetrievalResult(doc_id={self.doc_id}, score={self.score:.4f}, rank={self.rank})"


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods."""

    def __init__(self, corpus: Dict[int, Any], **kwargs):
        """
        Initialize the retriever.

        Args:
            corpus: Dictionary mapping doc_id to Document objects
            **kwargs: Additional configuration parameters
        """
        self.corpus = corpus
        self.config = kwargs

    @abstractmethod
    def build_index(self):
        """Build the retrieval index from the corpus."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a given query.

        Args:
            query: The query text (claim)
            top_k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """

    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query

        Returns:
            List of retrieval results for each query
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results

    def save_index(self, path: str):
        """Save the index to disk."""
        raise NotImplementedError("Subclass must implement save_index")

    def load_index(self, path: str):
        """Load the index from disk."""
        raise NotImplementedError("Subclass must implement load_index")
