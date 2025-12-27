
import pytest
from src.verification.evidence_aggregator import EvidenceAggregator

class TestEvidenceAggregatorExtended:
    
    def test_initialization_error(self):
        # Test default init
        agg = EvidenceAggregator()
        assert agg.strategy == "confidence_weighted"
        
        # Test invalid strategy
        agg = EvidenceAggregator(strategy="invalid_strategy")
        
        # Should raise ValueError when calling aggregate with data
        with pytest.raises(ValueError):
            agg.aggregate([("SUPPORTS", 1.0, {})])

    def test_contradiction_filtering(self):
        agg = EvidenceAggregator()
        
        # Tuples: (label, confidence, probs)
        evidence = [
            ("SUPPORTS", 0.9, {"SUPPORTS": 0.9}),
            ("REFUTES", 0.8, {"REFUTES": 0.8}), # Contradiction but lower conf
            ("SUPPORTS", 0.7, {"SUPPORTS": 0.7})
        ]
        
        # filter_contradictory_evidence expects list of tuples
        filtered = agg.filter_contradictory_evidence(evidence)
        
        labels = [e[0] for e in filtered]
        # Should likely keep high confidence ones or all if Logic isn't filtering strictly
        # Based on code: if both support and refutes exist:
        # threshold = 75th percentile. 
        # confs: 0.9, 0.8, 0.7. 
        # 75th percentile of [0.7, 0.8, 0.9] is 0.85
        # So only >= 0.85 kept -> 0.9 (SUPPORTS). REFUTES (0.8) should be removed.
        
        assert "REFUTES" not in labels 
        assert "SUPPORTS" in labels

    def test_aggregate_max_confidence(self):
        agg = EvidenceAggregator(strategy="max_confidence")
        
        evidence = [
            ("SUPPORTS", 0.5, {}),
            ("REFUTES", 0.9, {})
        ]
        
        label, conf, probs = agg.aggregate(evidence)
        assert label == "REFUTES"
        assert conf == 0.9
