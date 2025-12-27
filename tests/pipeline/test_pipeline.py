
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from src.pipeline import FactVerificationPipeline, PipelineResult, VerificationResult
from src.retrieval.base_retriever import RetrievalResult

class TestPipeline:
    
    @pytest.fixture
    def mock_config(self):
        return {
            'data': {
                'scifact_dir': 'data',
                'corpus_path': 'data/corpus.jsonl',
                'index_dir': 'data/indices'
            },
            'retrieval': {
                'method': 'bm25',
                'top_k': 3,
                'dense_model': 'top-model',
                'batch_size': 32,
                'bm25_weight': 0.5,
                'dense_weight': 0.5
            },
            'verification': {
                'nli_model': 'test-model',
                'aggregation': 'majority'
            },
            'multi_hop': {
                'enabled': True,
                'use_gnn': False,
                'gnn': {
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'num_heads': 2,
                    'dropout': 0.1
                },
                'graph': {
                    'sentence_similarity_threshold': 0.7,
                    'max_evidence_sentences': 5,
                    'use_entity_extraction': False
                }
            }
        }

    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.NLIModel')
    def test_initialization(self, mock_nli, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_yaml.return_value = mock_config
        
        pipeline = FactVerificationPipeline("config.yaml")
        
        assert pipeline.use_gnn is False
        mock_nli.assert_called_once()
        mock_dataset.assert_called_once()

    @patch('src.pipeline.os.path.exists')
    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.MultiHopReasoner')
    def test_initialization_gnn(self, mock_gnn, mock_dataset, mock_open, mock_yaml, mock_exists, mock_config):
        mock_exists.return_value = False
        mock_config['multi_hop']['use_gnn'] = True
        mock_yaml.return_value = mock_config
        
        pipeline = FactVerificationPipeline("config.yaml")
        
        assert pipeline.use_gnn is True
        mock_gnn.assert_called_once()

    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.NLIModel')
    @patch('src.pipeline.BM25Retriever')
    def test_build_retriever(self, mock_bm25, mock_nli, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_yaml.return_value = mock_config
        pipeline = FactVerificationPipeline("config.yaml")
        
        pipeline.build_retriever('bm25')
        mock_bm25.assert_called_once()
        
        # Test unknown method
        with pytest.raises(ValueError):
            pipeline.build_retriever('unknown')

    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.NLIModel')
    def test_retrieve_evidence(self, mock_nli, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_yaml.return_value = mock_config
        pipeline = FactVerificationPipeline("config.yaml")
        
        # No retriever yet
        with pytest.raises(ValueError):
            pipeline.retrieve_evidence("query")
            
        # Mock retriever
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [RetrievalResult(1, 1.0, "text", 1)]
        
        results = pipeline.retrieve_evidence("query")
        assert len(results) == 1
        pipeline.retriever.retrieve.assert_called_with("query", top_k=3)

    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.NLIModel')
    def test_process_claim_nli(self, mock_nli, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_yaml.return_value = mock_config
        pipeline = FactVerificationPipeline("config.yaml")
        
        # Setup mock retriever
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [
            RetrievalResult(1, 0.9, "Evidence 1", 1)
        ]
        
        # Setup mock verifier
        pipeline.verifier.predict_batch.return_value = [
            ("SUPPORTS", 0.9, {"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05})
        ]
        
        result = pipeline.process_claim(123, "Test Claim")
        
        assert isinstance(result, PipelineResult)
        assert result.claim_id == 123
        assert result.predicted_label == "SUPPORTS"
        assert len(result.retrieved_docs) == 1

    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.MultiHopReasoner')
    def test_process_claim_gnn(self, mock_gnn, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_config['multi_hop']['use_gnn'] = True
        mock_yaml.return_value = mock_config
        pipeline = FactVerificationPipeline("config.yaml")
        
        # Setup mock retriever
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [
            RetrievalResult(1, 0.9, "Evidence 1", 1)
        ]
        
        # Mock dataset get_document
        mock_doc = Mock()
        mock_doc.abstract = ["Sentence 1", "Sentence 2"]
        pipeline.dataset.get_document.return_value = mock_doc
        
        # Setup mock verifier predict
        pipeline.verifier.predict.return_value = ("REFUTES", 0.8, {"REFUTES": 0.8}, {"nodes": [], "edges": []})
        
        result = pipeline.process_claim(123, "Test Claim")
        
        assert result.predicted_label == "REFUTES"
        # Check that sentences were passed to verifier
        pipeline.verifier.predict.assert_called()
        
    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.NLIModel')
    def test_process_dataset(self, mock_nli, mock_dataset, mock_open, mock_yaml, mock_config):
        mock_yaml.return_value = mock_config
        pipeline = FactVerificationPipeline("config.yaml")
        
        # Mock load_claims
        mock_claim = Mock()
        mock_claim.id = 1
        mock_claim.claim = "Claim"
        mock_claim.label = "SUPPORTS"
        pipeline.dataset.load_claims.return_value = [mock_claim]
        
        # Mock process_claim
        pipeline.process_claim = Mock()
        pipeline.process_claim.return_value = PipelineResult(1, "Claim", "SUPPORTS", 0.9, [], [])
        
        results = pipeline.process_dataset(split='dev', limit=1)
        
        assert len(results) == 1
        pipeline.dataset.load_claims.assert_called_with('dev')
