"""
Retrieval evaluation metrics: Precision@k, Recall@k, MAP, MRR, NDCG.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np


class RetrievalMetrics:
    """Compute retrieval evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """
        Precision@k: Proportion of retrieved documents that are relevant.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            Precision@k score
        """
        if k == 0 or not retrieved:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)

        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """
        Recall@k: Proportion of relevant documents that are retrieved.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            Recall@k score
        """
        if not relevant or not retrieved:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)

        return relevant_retrieved / len(relevant)

    @staticmethod
    def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
        """
        Average Precision: Average of precision values at ranks where relevant docs appear.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs

        Returns:
            Average precision score
        """
        if not relevant or not retrieved:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0

        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision_sum += relevant_count / k

        return precision_sum / len(relevant) if relevant_count > 0 else 0.0

    @staticmethod
    def mean_average_precision(results: List[Tuple[List[int], Set[int]]]) -> float:
        """
        Mean Average Precision (MAP): Mean of AP across all queries.

        Args:
            results: List of (retrieved, relevant) tuples for each query

        Returns:
            MAP score
        """
        if not results:
            return 0.0

        ap_scores = [RetrievalMetrics.average_precision(ret, rel)
                     for ret, rel in results]

        return np.mean(ap_scores)

    @staticmethod
    def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
        """
        Reciprocal Rank: 1 / rank of first relevant document.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs

        Returns:
            Reciprocal rank score
        """
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(results: List[Tuple[List[int], Set[int]]]) -> float:
        """
        Mean Reciprocal Rank (MRR): Mean of RR across all queries.

        Args:
            results: List of (retrieved, relevant) tuples for each query

        Returns:
            MRR score
        """
        if not results:
            return 0.0

        rr_scores = [RetrievalMetrics.reciprocal_rank(ret, rel)
                     for ret, rel in results]

        return np.mean(rr_scores)

    @staticmethod
    def dcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """
        Discounted Cumulative Gain@k.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            DCG@k score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            gain = 1.0 if doc_id in relevant else 0.0
            dcg += gain / np.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@k.

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            NDCG@k score
        """
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevant, k)

        # Ideal DCG: all relevant docs at the top
        ideal_retrieved = list(relevant) + [0] * (k - len(relevant))
        idcg = RetrievalMetrics.dcg_at_k(ideal_retrieved, relevant, k)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
        """
        Calculate Hit Rate at k (Success@k).
        Checks if at least one relevant document is retrieved in the top k results.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            1.0 if any relevant document is in top k, else 0.0
        """
        top_k = set(retrieved_ids[:k])
        # Intersection is non-empty means we have a hit
        if not top_k.isdisjoint(relevant_ids):
            return 1.0
        return 0.0

    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[List[int]],
                           relevant_docs: List[Set[int]],
                           k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, float]:
        """
        Compute all retrieval metrics for a set of queries.

        Args:
            retrieved_docs: List of retrieved doc IDs for each query
            relevant_docs: List of relevant doc ID sets for each query
            k_values: List of k values for P@k, R@k, NDCG@k

        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}

        # Precision@k and Recall@k for each k
        for k in k_values:
            precisions = [RetrievalMetrics.precision_at_k(ret, rel, k)
                          for ret, rel in zip(retrieved_docs, relevant_docs)]
            recalls = [RetrievalMetrics.recall_at_k(ret, rel, k)
                       for ret, rel in zip(retrieved_docs, relevant_docs)]
            ndcgs = [RetrievalMetrics.ndcg_at_k(ret, rel, k)
                     for ret, rel in zip(retrieved_docs, relevant_docs)]

            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'ndcg@{k}'] = np.mean(ndcgs)

        # MAP and MRR
        results = list(zip(retrieved_docs, relevant_docs))
        metrics['map'] = RetrievalMetrics.mean_average_precision(results)
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(results)

        return metrics


def main():  # pragma: no cover
    """Example usage of RetrievalMetrics."""
    retrieved_example = [1, 3, 5, 7, 9]
    relevant_example = {1, 5, 8}

    print("Example retrieval evaluation:")
    print(f"Retrieved: {retrieved_example}")
    print(f"Relevant: {relevant_example}")
    print(f"\nPrecision@5: {RetrievalMetrics.precision_at_k(retrieved_example, relevant_example, 5):.4f}")
    print(f"Recall@5: {RetrievalMetrics.recall_at_k(retrieved_example, relevant_example, 5):.4f}")
    print(f"Average Precision: {RetrievalMetrics.average_precision(retrieved_example, relevant_example):.4f}")
    print(f"Reciprocal Rank: {RetrievalMetrics.reciprocal_rank(retrieved_example, relevant_example):.4f}")
    print(f"NDCG@5: {RetrievalMetrics.ndcg_at_k(retrieved_example, relevant_example, 5):.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
