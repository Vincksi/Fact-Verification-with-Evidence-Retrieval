"""
Verification evaluation metrics: Accuracy, F1, Precision, Recall, Confusion Matrix.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


class VerificationMetrics:
    """Compute verification evaluation metrics."""

    LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    @staticmethod
    def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Compute all verification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (macro and micro)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=VerificationMetrics.LABELS, zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', labels=VerificationMetrics.LABELS, zero_division=0
        )

        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=VerificationMetrics.LABELS, zero_division=0
        )

        for i, label in enumerate(VerificationMetrics.LABELS):
            metrics[f'precision_{label.lower()}'] = precision_per_class[i]
            metrics[f'recall_{label.lower()}'] = recall_per_class[i]
            metrics[f'f1_{label.lower()}'] = f1_per_class[i]
            metrics[f'support_{label.lower()}'] = support[i]

        return metrics

    @staticmethod
    def compute_confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(y_true, y_pred, labels=VerificationMetrics.LABELS)

    @staticmethod
    def print_classification_report(y_true: List[str], y_pred: List[str]) -> str:
        """
        Generate classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, labels=VerificationMetrics.LABELS, zero_division=0)

    @staticmethod
    def compute_fever_score(y_true: List[str],
                            y_pred: List[str],
                            evidence_retrieved: List[bool]) -> float:
        """
        Compute FEVER score: accuracy where correct evidence was retrieved.

        FEVER score = proportion of claims where:
        1. The predicted label is correct AND
        2. At least one piece of correct evidence was retrieved

        Args:
            y_true: True labels
            y_pred: Predicted labels
            evidence_retrieved: Whether correct evidence was retrieved for each claim

        Returns:
            FEVER score
        """
        correct_with_evidence = sum(
            1 for true, pred, evid in zip(y_true, y_pred, evidence_retrieved)
            if true == pred and evid
        )

        return correct_with_evidence / len(y_true) if y_true else 0.0

    @staticmethod
    def compute_ece(y_true: List[str], y_probs: np.ndarray, num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_true: List of true labels (strings).
            y_probs: Array of predicted probabilities [N, num_classes].
            num_bins: Number of confidence bins.

        Returns:
            ECE score (lower is better).
        """
        # Convert labels to indices
        label_map = {label: i for i, label in enumerate(VerificationMetrics.LABELS)}
        # Filter out labels not in our set if necessary, but assume consistency
        y_true_indices = np.array([label_map[l] for l in y_true])

        # Get predicted confidence and predicted class
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)

        accuracies = predictions == y_true_indices

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, num_bins + 1)

        for i in range(num_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prob_in_bin = np.mean(in_bin)

            if prob_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])

                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prob_in_bin

        return ece


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):  # pragma: no cover
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metric names to scores
        title: Title for the metrics display
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    # Overall metrics first
    overall_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                    'precision_micro', 'recall_micro', 'f1_micro', 'fever_score']

    print("\nOverall Metrics:")
    print("-" * 60)
    for key in overall_keys:
        if key in metrics:
            print(f"  {key:25s}: {metrics[key]:.4f}")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 60)
    for label in VerificationMetrics.LABELS:
        label_lower = label.lower()
        print(f"\n  {label}:")
        for metric_type in ['precision', 'recall', 'f1', 'support']:
            key = f"{metric_type}_{label_lower}"
            if key in metrics:
                value = metrics[key]
                if metric_type == 'support':
                    print(f"    {metric_type:15s}: {int(value)}")
                else:
                    print(f"    {metric_type:15s}: {value:.4f}")

    print(f"{'=' * 60}\n")


def main():  # pragma: no cover
    """Example usage of VerificationMetrics."""
    y_true_example = ["SUPPORTS", "REFUTES", "SUPPORTS", "NOT_ENOUGH_INFO", "SUPPORTS"]
    y_pred_example = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "NOT_ENOUGH_INFO", "REFUTES"]
    evidence_retrieved_example = [True, False, True, True, False]

    print("Example verification evaluation:")
    metrics = VerificationMetrics.compute_metrics(y_true_example, y_pred_example)
    metrics['fever_score'] = VerificationMetrics.compute_fever_score(
        y_true_example, y_pred_example, evidence_retrieved_example
    )

    print_metrics(metrics)

    print("\nConfusion Matrix:")
    cm = VerificationMetrics.compute_confusion_matrix(y_true_example, y_pred_example)
    print(cm)

    print("\nClassification Report:")
    print(VerificationMetrics.print_classification_report(y_true_example, y_pred_example))


if __name__ == "__main__":  # pragma: no cover
    main()
