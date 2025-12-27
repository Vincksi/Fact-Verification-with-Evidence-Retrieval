
import pytest
import numpy as np
from src.evaluation.retrieval_metrics import RetrievalMetrics

class TestRetrievalMetrics:
    
    def test_precision_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        # k=1 (hit)
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 1) == 1.0
        # k=2 (hit, miss) -> 1/2
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 2) == 0.5
        # k=5 (hit, miss, hit, miss, hit) -> 3/5
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 5) == 0.6
        # k=0
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 0) == 0.0
        # empty retrieved
        assert RetrievalMetrics.precision_at_k([], relevant, 5) == 0.0

    def test_recall_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 99} # 4 relevant docs
        
        # k=1 (1 relevant retrieved / 4 total relevant)
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 1) == 0.25
        # k=5 (3 relevant retrieved / 4 total relevant)
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 5) == 0.75
        # empty relevant
        assert RetrievalMetrics.recall_at_k(retrieved, set(), 5) == 0.0

    def test_average_precision(self):
        retrieved = [1, 2, 3, 4, 5, 6]
        relevant = {1, 3, 6}
        
        # Ranks of relevant docs: 1, 3, 6
        # Precisions at ranks:
        # k=1: 1/1 = 1.0
        # k=3: 2/3 = 0.666...
        # k=6: 3/6 = 0.5
        # AP = (1.0 + 0.6666 + 0.5) / 3 = 2.1666 / 3 = 0.7222
        
        ap = RetrievalMetrics.average_precision(retrieved, relevant)
        assert abs(ap - 0.7222) < 0.0001
        
        # No relevant documents retrieved
        assert RetrievalMetrics.average_precision([10, 11], relevant) == 0.0
        
        # Empty inputs
        assert RetrievalMetrics.average_precision([], relevant) == 0.0
        assert RetrievalMetrics.average_precision([1], set()) == 0.0

    def test_mean_average_precision(self):
        # Result 1: AP = 1.0 (perfect)
        r1_ret = [1, 2]
        r1_rel = {1, 2}
        
        # Result 2: AP = 0.0 (none)
        r2_ret = [3, 4]
        r2_rel = {1, 2}
        
        results = [(r1_ret, r1_rel), (r2_ret, r2_rel)]
        
        # MAP = (1.0 + 0.0) / 2 = 0.5
        assert RetrievalMetrics.mean_average_precision(results) == 0.5
        
        assert RetrievalMetrics.mean_average_precision([]) == 0.0

    def test_reciprocal_rank(self):
        retrieved = [2, 5, 1, 9]
        relevant = {1, 3}
        
        # First relevant doc is '1' at rank 3 (0-indexed -> index 2, 1-based rank 3)
        # RR = 1/3
        assert abs(RetrievalMetrics.reciprocal_rank(retrieved, relevant) - 0.3333) < 0.0001
        
        # No relevant
        assert RetrievalMetrics.reciprocal_rank([4, 5], relevant) == 0.0

    def test_mean_reciprocal_rank(self):
        # Q1: RR = 1.0
        r1 = ([1, 2], {1})
        # Q2: RR = 0.5
        r2 = ([2, 1], {1})
        
        assert RetrievalMetrics.mean_reciprocal_rank([r1, r2]) == 0.75
        assert RetrievalMetrics.mean_reciprocal_rank([]) == 0.0

    def test_ndcg_at_k(self):
        """Test NDCG calculation."""
        # Query 1: relevant={1, 3}, retrieved=[1, 2, 3]
        # IDCG @ 3: rel items 1(3), 3(3) -> ideal order [1, 3, x]
        # DCG = 1 + 0 + 1/log2(4) = 1.5
        # IDCG = 1 + 1/log2(3) = 1.63... -> NDCG ~ 0.91
        
        # Simple case: Perfect ranking
        retrieved = [1, 3, 2]
        relevant = {1, 3}
        # DCG: 1 (for 1) + 1/log2(3) (for 3) = 1 + 0.6309 = 1.6309
        # IDCG: same
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, 3) == 1.0
        
        # Zero case
        assert RetrievalMetrics.ndcg_at_k([5, 6], relevant, 2) == 0.0

    def test_hit_rate_at_k(self):
        """Test Hit Rate (Success@k) calculation."""
        relevant = {1, 5, 8}
        
        # Case 1: Hit at rank 1
        assert RetrievalMetrics.hit_rate_at_k([1, 2, 3], relevant, 1) == 1.0
        
        # Case 2: Hit at rank 3
        assert RetrievalMetrics.hit_rate_at_k([2, 3, 5], relevant, 3) == 1.0
        
        # Case 3: Miss at rank 2 (but hit at 3, so 0 for k=2)
        assert RetrievalMetrics.hit_rate_at_k([2, 3, 5], relevant, 2) == 0.0
        
        # Case 4: Complete miss
        assert RetrievalMetrics.hit_rate_at_k([2, 4, 6], relevant, 3) == 0.0

    def test_evaluate_retrieval(self):
        retrieved_docs = [[1, 2, 3], [4, 5, 6]]
        relevant_docs = [{1, 3}, {4}]
        k_values = [1, 3]
        
        metrics = RetrievalMetrics.evaluate_retrieval(retrieved_docs, relevant_docs, k_values)
        
        assert 'precision@1' in metrics
        assert 'recall@3' in metrics
        assert 'map' in metrics
        assert 'mrr' in metrics
        
        # Check values are floats
        assert isinstance(metrics['map'], float)
