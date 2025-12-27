
import pytest
from src.evaluation.verification_metrics import VerificationMetrics

class TestVerificationMetrics:
    
    def test_compute_metrics(self):
        y_true = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        y_pred = ["SUPPORTS", "SUPPORTS", "NOT_ENOUGH_INFO"]
        
        metrics = VerificationMetrics.compute_metrics(y_true, y_pred)
        
        # 2/3 correct = 0.6666
        assert abs(metrics['accuracy'] - 0.6666) < 0.0001
        assert 'f1_macro' in metrics
        assert 'precision_supports' in metrics

    def test_compute_metrics_empty(self):
        # sklearn raises ValueError for empty input
        with pytest.raises(ValueError):
            VerificationMetrics.compute_metrics([], [])

    def test_compute_ece(self):
        """Test Expected Calibration Error (ECE) calculation."""
        import numpy as np
        
        # Case 1: Perfect Calibration
        # Probability 1.0 -> Correct, Probability 0.6 -> Correct
        # (This is a simplified check, ECE checks if avg conf matches accuracy in bin)
        
        # Let's construct a case where accuracy matches confidence
        y_true = ["SUPPORTS", "SUPPORTS"]
        # Pred 1: SUPPORTS (idx 0 assuming label map order) with 0.9 conf
        # Pred 2: SUPPORTS with 0.9 conf
        # Bin 0.8-1.0 has 2 samples. Avg conf = 0.9. Accuracy = 1.0 (2/2 correct).
        # Diff = |1.0 - 0.9| = 0.1
        # ECE = 0.1 * (2/2) = 0.1
        
        # NOTE: VerificationMetrics.LABELS depends on initialization.
        # Assuming defaults ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        # SUPPORTS is index 0.
        
        y_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05]
        ])
        
        # If accuracy is 1.0 and conf is 0.9, ECE is 0.1
        ece = VerificationMetrics.compute_ece(y_true, y_probs, num_bins=10)
        assert abs(ece - 0.1) < 0.001
        
        # Case 2: Perfect match
        # Conf 1.0, Accuracy 1.0
        y_probs_perf = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        ece_perf = VerificationMetrics.compute_ece(y_true, y_probs_perf, num_bins=10)
        assert ece_perf == 0.0
        
    def test_fever_score(self):
        y_true = ["SUPPORTS", "SUPPORTS", "REFUTES"]
        y_pred = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
        evidence_retrieved = [True, False, True] # 1st: Correct+Retrieved, 2nd: Correct+NotRetrieved, 3rd: Wrong+Retrieved
        
        # Only 1st one counts for FEVER score
        # Score = 1/3 = 0.3333
        
        score = VerificationMetrics.compute_fever_score(y_true, y_pred, evidence_retrieved)
        assert abs(score - 0.3333) < 0.001
        
    def test_confusion_matrix(self):
        y_true = ["SUPPORTS", "REFUTES"]
        y_pred = ["SUPPORTS", "SUPPORTS"]
        
        cm = VerificationMetrics.compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_classification_report(self):
        y_true = ["SUPPORTS", "REFUTES"]
        y_pred = ["SUPPORTS", "SUPPORTS"]
        
        report = VerificationMetrics.print_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "SUPPORTS" in report
