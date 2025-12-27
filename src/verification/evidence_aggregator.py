"""
Evidence aggregation strategies for combining multiple evidence pieces.
"""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class EvidenceAggregator:
    """
    Aggregates evidence from multiple sources for claim verification.
    """

    LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    def __init__(self, strategy: str = "confidence_weighted"):
        """
        Initialize evidence aggregator.

        Args:
            strategy: Aggregation strategy
                - "majority": Simple majority voting
                - "confidence_weighted": Weight by prediction confidence
                - "max_confidence": Take prediction with highest confidence
        """
        self.strategy = strategy

    def aggregate(self,
                  predictions: List[Tuple[str, float, Dict[str, float]]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Aggregate multiple predictions into a single verdict.

        Args:
            predictions: List of (label, confidence, probabilities) tuples

        Returns:
            Aggregated (label, confidence, probabilities)
        """
        if not predictions:
            return "NOT_ENOUGH_INFO", 1.0, {"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0}

        if self.strategy == "majority":
            return self._majority_vote(predictions)
        elif self.strategy == "confidence_weighted":
            return self._confidence_weighted(predictions)
        elif self.strategy == "max_confidence":
            return self._max_confidence(predictions)
        elif self.strategy == "fact_verification":
            return self._fact_verification_priority(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _majority_vote(self, predictions: List[Tuple[str, float, Dict[str, float]]]
                       ) -> Tuple[str, float, Dict[str, float]]:
        """Simple majority voting across predictions."""
        labels = [pred[0] for pred in predictions]
        label_counts = Counter(labels)

        # Get most common label
        majority_label = label_counts.most_common(1)[0][0]
        confidence = label_counts[majority_label] / len(predictions)

        # Calculate average probabilities
        avg_probs = {label: 0.0 for label in self.LABELS}
        for _, _, probs in predictions:
            for label, prob in probs.items():
                avg_probs[label] += prob / len(predictions)

        return majority_label, confidence, avg_probs

    def _confidence_weighted(self, predictions: List[Tuple[str, float,
                             Dict[str, float]]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Weight predictions by their confidence scores.

        This gives more influence to predictions the model is more confident about.
        """
        # Accumulate weighted probabilities
        weighted_probs = {label: 0.0 for label in self.LABELS}
        total_confidence = 0.0

        for label, confidence, probs in predictions:
            total_confidence += confidence
            for lbl, prob in probs.items():
                weighted_probs[lbl] += prob * confidence

        # Normalize
        if total_confidence > 0:
            weighted_probs = {
                lbl: prob / total_confidence
                for lbl, prob in weighted_probs.items()
            }

        # Get label with highest weighted probability
        final_label = max(weighted_probs.items(), key=lambda x: x[1])[0]
        final_confidence = weighted_probs[final_label]

        return final_label, final_confidence, weighted_probs

    def _max_confidence(self, predictions: List[Tuple[str, float, Dict[str, float]]]
                        ) -> Tuple[str, float, Dict[str, float]]:
        """Take the prediction with the highest confidence."""
        best_pred = max(predictions, key=lambda x: x[1])
        return best_pred

    def _fact_verification_priority(self, 
                                   predictions: List[Tuple[str, float, Dict[str, float]]],
                                   priority_threshold: float = 0.35) -> Tuple[str, float, Dict[str, float]]:
        """
        Priority-based aggregation for SciFact.
        
        Logic:
        1. If any evidence SUPPORTS with high confidence, return SUPPORTS.
        2. Else if any evidence REFUTES with high confidence, return REFUTES.
        3. Else, fall back to max_confidence or majority.
        """
        supports = [p for p in predictions if p[0] == "SUPPORTS" and p[1] >= priority_threshold]
        refutes = [p for p in predictions if p[0] == "REFUTES" and p[1] >= priority_threshold]
        
        if supports:
            # Take the most confident support
            return max(supports, key=lambda x: x[1])
            
        if refutes:
            # Take the most confident refute
            return max(refutes, key=lambda x: x[1])
            
        # Fallback: take the best max_confidence result
        return self._max_confidence(predictions)

    def aggregate_with_threshold(self,
                                 predictions: List[Tuple[str, float, Dict[str, float]]],
                                 threshold: float = 0.5) -> Tuple[str, float, Dict[str, float]]:
        """
        Aggregate predictions with a confidence threshold.

        If the aggregated confidence is below threshold, return NOT_ENOUGH_INFO.

        Args:
            predictions: List of (label, confidence, probabilities) tuples
            threshold: Minimum confidence threshold

        Returns:
            Aggregated (label, confidence, probabilities)
        """
        label, confidence, probs = self.aggregate(predictions)

        if confidence < threshold:
            return "NOT_ENOUGH_INFO", confidence, probs

        return label, confidence, probs

    def filter_contradictory_evidence(
            self, predictions: List[Tuple[str, float, Dict[str, float]]]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Detect and handle contradictory evidence (both SUPPORTS and REFUTES).

        Args:
            predictions: List of (label, confidence, probabilities) tuples

        Returns:
            Filtered predictions or indication of contradiction
        """
        labels = [pred[0] for pred in predictions]

        has_supports = "SUPPORTS" in labels
        has_refutes = "REFUTES" in labels

        if has_supports and has_refutes:
            # Evidence is contradictory
            # Strategy: Keep only the predictions with highest confidence
            threshold = np.percentile([p[1] for p in predictions], 75)
            filtered = [p for p in predictions if p[1] >= threshold]
            return filtered if filtered else predictions

        return predictions


def main():  # pragma: no cover
    """Example usage of EvidenceAggregator."""
    aggregator = EvidenceAggregator(strategy="confidence_weighted")

    predictions_example = [
        ("SUPPORTS", 0.9, {"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05}),
        ("SUPPORTS", 0.7, {"SUPPORTS": 0.7, "REFUTES": 0.2, "NOT_ENOUGH_INFO": 0.1}),
        ("REFUTES", 0.6, {"SUPPORTS": 0.2, "REFUTES": 0.6, "NOT_ENOUGH_INFO": 0.2}),
    ]

    label, confidence, probs = aggregator.aggregate(predictions_example)

    print(f"Aggregated label: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probs}")


if __name__ == "__main__":  # pragma: no cover
    main()
