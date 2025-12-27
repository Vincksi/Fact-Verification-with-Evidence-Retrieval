
import pytest
import json
import jsonlines
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from src.data.dataset_loader import SciFactDataset, Claim, Document

class TestSciFactDataset:
    
    @pytest.fixture
    def mock_fs(self):
        with patch('src.data.dataset_loader.jsonlines.open') as mock_jsonlines, \
             patch('src.data.dataset_loader.Path.exists') as mock_exists:
            yield mock_jsonlines, mock_exists

    def test_claim_model(self):
        claim = Claim(
            id=1,
            claim="Text",
            evidence={"1": [{"sentences": [0], "label": "SUPPORTS"}]},
            cited_doc_ids=[1]
        )
        assert claim.label == "SUPPORTS"
        
        # Test default label
        claim_no_ev = Claim(1, "Text", {}, [])
        assert claim_no_ev.label == "NOT_ENOUGH_INFO"
        
        # Test empty evidence list
        claim_empty_list = Claim(1, "Text", {"1": []}, [1])
        assert claim_empty_list.label == "NOT_ENOUGH_INFO"

    def test_document_model(self):
        doc = Document(
            doc_id=1,
            title="Title",
            abstract=["Sent 1", "Sent 2"],
            structured=False
        )
        assert doc.full_text == "Title Sent 1 Sent 2"

    def test_load_corpus(self, mock_fs):
        mock_reader, _ = mock_fs
        
        # Mock data
        mock_reader.return_value.__enter__.return_value = [
            {"doc_id": 1, "title": "T1", "abstract": ["S1"], "structured": False},
            {"doc_id": 2, "title": "T2", "abstract": ["S2"], "structured": True}
        ]
        
        dataset = SciFactDataset("data_dir")
        corpus = dataset.load_corpus("dummy_path")
        
        assert len(corpus) == 2
        assert corpus[1].title == "T1"
        assert corpus[2].structured is True
        
        # Test default path
        dataset.load_corpus() # Uses default path construction
        # Just ensure no error, path construction is simple

    def test_load_claims(self, mock_fs):
        mock_reader, _ = mock_fs
        
        mock_reader.return_value.__enter__.return_value = [
            {"id": 1, "claim": "C1", "evidence": {}, "cited_doc_ids": []}
        ]
        
        dataset = SciFactDataset("data_dir")
        
        # Train
        claims = dataset.load_claims("train")
        assert len(dataset.claims_train) == 1
        assert claims[0].claim == "C1"
        
        # Dev
        dataset.load_claims("dev")
        assert len(dataset.claims_dev) == 1
        
        # Test
        dataset.load_claims("test")
        assert len(dataset.claims_test) == 1

    def test_load_all(self, mock_fs):
        mock_reader, _ = mock_fs
        # Reuse same mock iterator for all calls
        mock_reader.return_value.__enter__.return_value = []
        
        dataset = SciFactDataset("data_dir")
        
        # Mock load_corpus and load_claims to avoid complex side effects if preferred
        # or simplified: just let it run with empty data
        dataset.load_all()
        # Should not raise error

    def test_get_claim(self):
        dataset = SciFactDataset("data")
        c1 = Claim(1, "C1", {}, [])
        dataset.claims_train = [c1]
        
        assert dataset.get_claim(1, "train") == c1
        assert dataset.get_claim(99, "train") is None
        assert dataset.get_claim(1, "dev") is None

    def test_get_evidence_sentences(self):
        dataset = SciFactDataset("data")
        
        # Setup corpus
        dataset.corpus = {
            1: Document(1, "T1", ["S0", "S1", "S2"], False)
        }
        
        # Setup claim with evidence pointing to doc 1, sentence 1
        claim = Claim(
            id=10, 
            claim="C", 
            evidence={"1": [{"sentences": [1]}]},
            cited_doc_ids=[1]
        )
        
        evidence = dataset.get_evidence_sentences(claim)
        assert 1 in evidence
        assert evidence[1] == ["S1"]
        
        # Test with missing doc
        claim_missing = Claim(11, "C", {"99": [{"sentences": [0]}]}, [])
        evidence_missing = dataset.get_evidence_sentences(claim_missing)
        assert 99 not in evidence_missing
