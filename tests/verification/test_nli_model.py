
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from src.verification.nli_model import NLIModel, VerificationResult

class TestNLIModel:
    
    @pytest.fixture
    def mock_transformers(self):
        with patch('src.verification.nli_model.AutoTokenizer') as mock_tok, \
             patch('src.verification.nli_model.AutoModelForSequenceClassification') as mock_model_cls:
            yield mock_tok, mock_model_cls

    @pytest.fixture
    def nli_model(self, mock_transformers):
        mock_tok, mock_model_cls = mock_transformers
        
        # Setup mock model instance
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Setup tokenizer
        mock_tok_instance = MagicMock()
        mock_tok.from_pretrained.return_value = mock_tok_instance
        
        # make tokenizer return an object with .to() method (like BatchEncoding)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding # .to() returns self
        mock_encoding.__getitem__.side_effect = lambda k: torch.tensor([[1]]) # simplified access
        mock_tok_instance.return_value = mock_encoding
        
        # Setup prediction return
        # Logits for [REFUTES (0), SUPPORTS (1), NotEnough (2)]
        # Default to SUPPORTS (index 1 high)
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9, 0.1]])
        mock_model.return_value = mock_outputs
        
        model = NLIModel(model_name="test-model", device="cpu")
        return model

    def test_initialization(self, mock_transformers):
        mock_tok, mock_model_cls = mock_transformers
        model = NLIModel(model_name="test-model")
        mock_tok.from_pretrained.assert_called_with("test-model")
        mock_model_cls.from_pretrained.assert_called_with("test-model")

    def test_predict_single(self, nli_model):
        claim = "Aspirin helps heart."
        evidence = "Aspirin reduces heart attack risk."
        
        # Default mock returns SUPPORTS (idx 1)
        label, conf, probs = nli_model.predict_single(claim, evidence)
        
        assert label == "SUPPORTS"
        # 0.9 / (0.1+0.9+0.1) is not simple softmax, torch.softmax will differ
        # but we mock the result of softmax logic via logits? No, code calls softmax on logits.
        # logits=[0.1, 0.9, 0.1]. Softmax([0.1, 0.9, 0.1]) -> idx 1 will be highest.
        
    def test_predict_contradiction(self, nli_model):
        # Modify mock for contradiction (idx 0 high)
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.9, 0.1, 0.0]])
        nli_model.model.return_value = mock_outputs
        
        label, conf, probs = nli_model.predict_single("A", "B")
        assert label == "REFUTES"

    def test_predict_neutral(self, nli_model):
        # Neural/NEI is idx 2
        # Note: With neutral bias correction enabled, even neutral predictions
        # will be reassigned to SUPPORTS or REFUTES if they have any signal
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.1, 0.9]])
        nli_model.model.return_value = mock_outputs
        
        label, conf, probs = nli_model.predict_single("A", "B")
        # After bias correction, this should now be REFUTES or SUPPORTS (both equal, REFUTES is higher idx)
        assert label in ["SUPPORTS", "REFUTES"]

    def test_predict_batch(self, nli_model):
        evidences = ["E1", "E2"]
        # Batch size 2
        mock_outputs = Mock()
        # 2 samples: 1st SUPPORTS, 2nd REFUTES
        mock_outputs.logits = torch.tensor([
            [0.1, 0.9, 0.1], # SUPPORTS
            [0.9, 0.05, 0.05] # REFUTES
        ])
        nli_model.model.return_value = mock_outputs
        
        results = nli_model.predict_batch("Claim", evidences, batch_size=2)
        assert len(results) == 2
        assert results[0][0] == "SUPPORTS"
        assert results[1][0] == "REFUTES"

    def test_verify_with_evidence(self, nli_model):
        claim = "Claim"
        evidence_list = [(1, "Evidence 1")]
        
        # Mock predict_batch return directly to simplify
        nli_model.predict_batch = Mock(return_value=[("SUPPORTS", 0.9, {})])
        
        result = nli_model.verify_with_evidence(0, claim, evidence_list)
        
        assert result.label == "SUPPORTS"
        assert result.confidence == 0.9
        
    def test_verify_no_evidence(self, nli_model):
        result = nli_model.verify_with_evidence(0, "Claim", [])
        assert result.label == "NOT_ENOUGH_INFO"
