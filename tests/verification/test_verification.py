"""
Basic tests for the verification module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.evidence_aggregator import EvidenceAggregator


def test_aggregator_majority_vote():
    """Test majority voting aggregation."""
    aggregator = EvidenceAggregator(strategy="majority")
    
    predictions = [
        ("SUPPORTS", 0.9, {"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05}),
        ("SUPPORTS", 0.8, {"SUPPORTS": 0.8, "REFUTES": 0.1, "NOT_ENOUGH_INFO": 0.1}),
        ("REFUTES", 0.7, {"SUPPORTS": 0.2, "REFUTES": 0.7, "NOT_ENOUGH_INFO": 0.1}),
    ]
    
    label, confidence, probs = aggregator.aggregate(predictions)
    
    assert label == "SUPPORTS"  # Majority
    assert 0 <= confidence <= 1
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-2)


def test_aggregator_confidence_weighted():
    """Test confidence-weighted aggregation."""
    aggregator = EvidenceAggregator(strategy="confidence_weighted")
    
    predictions = [
        ("SUPPORTS", 0.9, {"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05}),
        ("REFUTES", 0.6, {"SUPPORTS": 0.2, "REFUTES": 0.6, "NOT_ENOUGH_INFO": 0.2}),
    ]
    
    label, confidence, probs = aggregator.aggregate(predictions)
    
    # Should weight towards the more confident prediction (SUPPORTS at 0.9)
    assert label == "SUPPORTS"
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1


def test_aggregator_max_confidence():
    """Test max confidence aggregation."""
    aggregator = EvidenceAggregator(strategy="max_confidence")
    
    predictions = [
        ("SUPPORTS", 0.7, {"SUPPORTS": 0.7, "REFUTES": 0.2, "NOT_ENOUGH_INFO": 0.1}),
        ("REFUTES", 0.9, {"SUPPORTS": 0.05, "REFUTES": 0.9, "NOT_ENOUGH_INFO": 0.05}),
    ]
    
    label, confidence, probs = aggregator.aggregate(predictions)
    
    # Should select the prediction with highest confidence
    assert label == "REFUTES"
    assert confidence == pytest.approx(0.9, rel=1e-6)


def test_aggregator_empty_predictions():
    """Test aggregation with empty predictions list."""
    aggregator = EvidenceAggregator(strategy="majority")
    
    label, confidence, probs = aggregator.aggregate([])
    
    assert label == "NOT_ENOUGH_INFO"
    assert confidence == 1.0


def test_filter_contradictory_evidence():
    """Test filtering of contradictory evidence."""
    aggregator = EvidenceAggregator(strategy="majority")
    
    predictions = [
        ("SUPPORTS", 0.9, {"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05}),
        ("SUPPORTS", 0.8, {"SUPPORTS": 0.8, "REFUTES": 0.1, "NOT_ENOUGH_INFO": 0.1}),
        ("REFUTES", 0.7, {"SUPPORTS": 0.2, "REFUTES": 0.7, "NOT_ENOUGH_INFO": 0.1}),
        ("REFUTES", 0.6, {"SUPPORTS": 0.3, "REFUTES": 0.6, "NOT_ENOUGH_INFO": 0.1}),
    ]
    
    filtered = aggregator.filter_contradictory_evidence(predictions)
    
    # Should detect contradiction and filter (return high-confidence predictions)
    assert len(filtered) <= len(predictions)
    assert all(isinstance(p, tuple) and len(p) == 3 for p in filtered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
