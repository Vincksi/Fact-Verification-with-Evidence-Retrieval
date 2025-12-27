
import pytest
from unittest.mock import Mock, patch
from src.pipeline import FactVerificationPipeline, PipelineResult
from src.retrieval.base_retriever import RetrievalResult

class TestPipelineGAT:
    
    @pytest.fixture
    def gat_config(self):
        return {
            'data': {
                'scifact_dir': 'data',
                'corpus_path': 'data/corpus.jsonl',
                'index_dir': 'data/indices'
            },
            'retrieval': {
                'method': 'bm25',
                'top_k': 3,
                'dense_model': 'dummy-model',
                'batch_size': 32,
                'bm25_weight': 0.5,
                'dense_weight': 0.5
            },
            'verification': {
                'nli_model': 'dummy-model',
                'aggregation': 'majority'
            },
            'multi_hop': {
                'enabled': True,
                'use_gnn': True,  # FORCE GNN
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

    @patch('src.pipeline.os.path.exists')
    @patch('src.pipeline.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.pipeline.SciFactDataset')
    @patch('src.pipeline.MultiHopReasoner')
    def test_full_gat_flow(self, mock_gnn_class, mock_dataset_class, mock_open, mock_yaml, mock_exists, gat_config):
        """Test the full pipeline flow with GAT enabled."""
        mock_exists.return_value = False
        mock_yaml.return_value = gat_config
        
        # 1. Initialize Pipeline with GAT
        pipeline = FactVerificationPipeline("config.yaml")
        assert pipeline.use_gnn is True
        
        # Verify GNN reasoner was initialized
        mock_gnn_class.assert_called_once()
        mock_verifier = mock_gnn_class.return_value
        
        # 2. Setup Mock Data
        # Mock Retriever
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [
            RetrievalResult(doc_id=1, score=0.9, text="Doc 1 text", title="Doc 1"),
            RetrievalResult(doc_id=2, score=0.8, text="Doc 2 text", title="Doc 2")
        ]
        
        # Mock Dataset (for fetching abstracts)
        mock_doc_1 = Mock()
        mock_doc_1.abstract = ["Sentence 1.1", "Sentence 1.2"]
        mock_doc_2 = Mock()
        mock_doc_2.abstract = ["Sentence 2.1"]
        
        # Dataset side_effect to return correct doc
        def get_doc_side_effect(doc_id):
            if doc_id == 1: return mock_doc_1
            if doc_id == 2: return mock_doc_2
            return None
            
        pipeline.dataset.get_document.side_effect = get_doc_side_effect
        
        # Mock GNN Prediction
        # Return: label, confidence, probabilities, graph_data
        mock_verifier.predict.return_value = ("REFUTES", 0.95, {"SUPPORTS": 0.05, "REFUTES": 0.95}, {"nodes": [], "edges": []})
        
        # 3. Execution
        result = pipeline.process_claim(claim_id=101, claim_text="GATs are useless.")
        
        # 4. Assertions
        
        # Check that sentences were collected and passed to verifier
        # Expected: Abstract 1 (2 sents) + Abstract 2 (1 sent) = 3 sentences
        call_args = mock_verifier.predict.call_args
        assert call_args is not None
        claim_arg, sentences_arg = call_args[0]
        
        assert claim_arg == "GATs are useless."
        assert len(sentences_arg) == 3
        assert "Sentence 1.1" in sentences_arg
        assert "Sentence 2.1" in sentences_arg
        
        # Check Result Object
        assert isinstance(result, PipelineResult)
        assert result.predicted_label == "REFUTES"
        assert result.confidence == 0.95
        assert result.claim_id == 101
        
        print("\nGAT Pipeline Test Passed!")
        print(f"Input: {claim_arg}")
        print(f"Evidence Sentences: {len(sentences_arg)}")
        print(f"Prediction: {result.predicted_label}")
