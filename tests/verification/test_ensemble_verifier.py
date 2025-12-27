import pytest
from unittest.mock import Mock, MagicMock
from src.verification.ensemble_verifier import EnsembleVerifier
from src.verification.nli_model import VerificationResult

class TestEnsembleVerifier:
    def test_initialization(self):
        mock_nli = Mock()
        mock_gnn = Mock()
        verifier = EnsembleVerifier(mock_nli, mock_gnn, nli_weight=0.7, gnn_weight=0.3)
        
        assert verifier.nli_model == mock_nli
        assert verifier.gnn_model == mock_gnn
        assert verifier.nli_weight == 0.7
        assert verifier.gnn_weight == 0.3

    def test_verify_with_evidence_no_evidence(self):
        mock_nli = Mock()
        mock_gnn = Mock()
        verifier = EnsembleVerifier(mock_nli, mock_gnn)
        
        result = verifier.verify_with_evidence(1, "Claim", [])
        
        assert result.label == "NOT_ENOUGH_INFO"
        assert result.confidence == 1.0
        assert result.label_probabilities["NOT_ENOUGH_INFO"] == 1.0

    def test_verify_with_evidence_fusion(self):
        mock_nli = Mock()
        mock_gnn = Mock()
        
        # Mock results
        nli_res = VerificationResult(
            claim_id=1,
            label="SUPPORTS",
            confidence=0.8,
            label_probabilities={"SUPPORTS": 0.8, "REFUTES": 0.1, "NOT_ENOUGH_INFO": 0.1},
            evidence_scores=[(101, "Text 1", 0.8)]
        )
        
        gnn_res = VerificationResult(
            claim_id=1,
            label="REFUTES",
            confidence=0.6,
            label_probabilities={"SUPPORTS": 0.2, "REFUTES": 0.6, "NOT_ENOUGH_INFO": 0.2},
            evidence_scores=[(101, "Text 1", 0.4)]
        )
        # GNN result might have graph_data
        gnn_res.graph_data = {"nodes": [], "edges": []}
        
        mock_nli.verify_with_evidence.return_value = nli_res
        mock_gnn.verify_with_evidence.return_value = gnn_res
        
        # Equal weights
        verifier = EnsembleVerifier(mock_nli, mock_gnn, nli_weight=1.0, gnn_weight=1.0)
        
        evidence_list = [(101, "Text 1")]
        result = verifier.verify_with_evidence(1, "Claim", evidence_list)
        
        # Probabilities:
        # SUPPORTS: (0.8 + 0.2) / 2 = 0.5
        # REFUTES: (0.1 + 0.6) / 2 = 0.35
        # NEI: (0.1 + 0.2) / 2 = 0.15
        
        assert result.label == "SUPPORTS"
        assert result.confidence == pytest.approx(0.5)
        assert result.label_probabilities["SUPPORTS"] == pytest.approx(0.5)
        assert result.label_probabilities["REFUTES"] == pytest.approx(0.35)
        
        # Evidence scores:
        # NLI: (101, "Text 1", 0.8) -> 0.8 * 1.0 = 0.8
        # GNN: (101, "Text 1", 0.4) -> 0.4 * 1.0 = 0.4
        # Final: (0.8 + 0.4) / 2.0 = 0.6
        assert len(result.evidence_scores) == 1
        assert result.evidence_scores[0][0] == 101
        assert result.evidence_scores[0][2] == pytest.approx(0.6)
        
        # Graph data should be preserved
        assert result.graph_data == gnn_res.graph_data

    def test_verify_with_evidence_weighted_fusion(self):
        mock_nli = Mock()
        mock_gnn = Mock()
        
        nli_res = VerificationResult(
            claim_id=1,
            label="SUPPORTS",
            confidence=0.9,
            label_probabilities={"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05},
            evidence_scores=[]
        )
        
        gnn_res = VerificationResult(
            claim_id=1,
            label="REFUTES",
            confidence=0.8,
            label_probabilities={"SUPPORTS": 0.1, "REFUTES": 0.8, "NOT_ENOUGH_INFO": 0.1},
            evidence_scores=[]
        )
        
        mock_nli.verify_with_evidence.return_value = nli_res
        mock_gnn.verify_with_evidence.return_value = gnn_res
        
        # Weight towards GNN
        verifier = EnsembleVerifier(mock_nli, mock_gnn, nli_weight=1.0, gnn_weight=3.0)
        
        evidence_list = [(101, "Text 1")]
        result = verifier.verify_with_evidence(1, "Claim", evidence_list)
        
        # Probabilities:
        # SUPPORTS: (0.9 * 1 + 0.1 * 3) / 4 = 1.2 / 4 = 0.3
        # REFUTES: (0.05 * 1 + 0.8 * 3) / 4 = 2.45 / 4 = 0.6125
        
        assert result.label == "REFUTES"
        assert result.confidence == pytest.approx(0.6125)
