"""
Hybrid retrieval combining BM25 and dense retrieval using Reciprocal Rank Fusion.
"""

from collections import defaultdict
from typing import Any, Dict, List

from .base_retriever import BaseRetriever, RetrievalResult
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining lexical (BM25) and semantic (dense) methods."""

    def __init__(self, corpus: Dict[int, Any],
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 fusion_method: str = "rrf",  # "rrf" or "weighted"
                 **kwargs):
        """
        Initialize hybrid retriever.

        Args:
            corpus: Dictionary mapping doc_id to Document objects
            dense_model: Name of the sentence transformer model
            bm25_weight: Weight for BM25 scores (for weighted fusion)
            dense_weight: Weight for dense scores (for weighted fusion)
            fusion_method: Method for combining scores ("rrf" or "weighted")
            **kwargs: Additional parameters
        """
        super().__init__(corpus, **kwargs)

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method

        # Initialize both retrievers
        print("Initializing BM25 retriever...")
        self.bm25_retriever = BM25Retriever(corpus, **kwargs)

        print("Initializing dense retriever...")
        self.dense_retriever = DenseRetriever(corpus, model_name=dense_model, **kwargs)

    def build_index(self):
        """Build indices for both BM25 and dense retrieval."""
        print("Building hybrid index...")
        self.bm25_retriever.build_index()
        self.dense_retriever.build_index()
        print("Hybrid index built successfully")

    def _reciprocal_rank_fusion(self,
                                bm25_results: List[RetrievalResult],
                                dense_results: List[RetrievalResult],
                                k: int = 60) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each retriever

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            k: Constant for RRF (default 60)

        Returns:
            Fused and re-ranked results
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_info = {}  # Store doc info for final results

        # Add BM25 contributions
        for result in bm25_results:
            rrf_scores[result.doc_id] += 1.0 / (k + result.rank)
            doc_info[result.doc_id] = (result.title, result.text)

        # Add dense contributions
        for result in dense_results:
            rrf_scores[result.doc_id] += 1.0 / (k + result.rank)
            doc_info[result.doc_id] = (result.title, result.text)

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            title, text = doc_info[doc_id]
            result = RetrievalResult(
                doc_id=doc_id,
                score=score,
                title=title,
                text=text,
                rank=rank
            )
            results.append(result)

        return results

    def _weighted_fusion(self,
                         bm25_results: List[RetrievalResult],
                         dense_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine results using weighted score fusion.

        Final score = bm25_weight * normalized_bm25_score + dense_weight * dense_score

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever

        Returns:
            Fused and re-ranked results
        """
        combined_scores = defaultdict(float)
        doc_info = {}

        # Normalize BM25 scores (min-max normalization)
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            min_bm25 = min(bm25_scores)
            max_bm25 = max(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

            for result in bm25_results:
                normalized_score = (result.score - min_bm25) / bm25_range
                combined_scores[result.doc_id] += self.bm25_weight * normalized_score
                doc_info[result.doc_id] = (result.title, result.text)

        # Dense scores are already normalized (cosine similarity in [0, 1])
        for result in dense_results:
            combined_scores[result.doc_id] += self.dense_weight * result.score
            doc_info[result.doc_id] = (result.title, result.text)

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            title, text = doc_info[doc_id]
            result = RetrievalResult(
                doc_id=doc_id,
                score=score,
                title=title,
                text=text,
                rank=rank
            )
            results.append(result)

        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents using hybrid method.

        Args:
            query: The query text (claim)
            top_k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects, sorted by fused score
        """
        # Retrieve from both methods (more candidates for fusion)
        retrieval_k = top_k * 2  # Retrieve more candidates for better fusion

        bm25_results = self.bm25_retriever.retrieve(query, top_k=retrieval_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_k)

        # Fuse results
        if self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(bm25_results, dense_results)
        elif self.fusion_method == "weighted":
            fused_results = self._weighted_fusion(bm25_results, dense_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Return top-k
        return fused_results[:top_k]

    def save_index(self, path: str):
        """Save both BM25 and dense indices."""
        self.bm25_retriever.save_index(path)
        self.dense_retriever.save_index(path)
        print(f"Hybrid index saved to {path}")

    def load_index(self, path: str):
        """Load both BM25 and dense indices."""
        self.bm25_retriever.load_index(path)
        self.dense_retriever.load_index(path)
        print(f"Hybrid index loaded from {path}")


def main():  # pragma: no cover
    """Example usage of HybridRetriever."""
    from src.data.dataset_loader import SciFactDataset

    # Load dataset
    dataset_example = SciFactDataset("data")
    dataset_example.load_corpus()

    # Create and build hybrid retriever
    retriever_example = HybridRetriever(dataset_example.corpus, fusion_method="rrf")
    retriever_example.build_index()

    # Test retrieval
    query_text = "Aspirin reduces the risk of heart attack"
    results_list = retriever_example.retrieve(query_text, top_k=5)

    print(f"\nTop 5 hybrid results for: '{query_text}'")
    for result in results_list:
        print(f"\nRank {result.rank}: {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Doc ID: {result.doc_id}")


if __name__ == "__main__":  # pragma: no cover
    main()
