import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from src.verification.multi_hop_reasoner import MultiHopReasoner, GraphBuilder, GNNEncoder

class TestGraphBuilder:
    @patch('src.verification.multi_hop_reasoner.SentenceTransformer')
    @patch('src.verification.multi_hop_reasoner.spacy.load')
    def test_build_graph(self, mock_spacy_load, mock_st):
        # Mock embeddings
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.zeros((3, 10)) # 3 nodes, 10 dim
        mock_st.return_value = mock_encoder
        
        # Mock Spacy
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.text = "Entity"
        mock_ent.label_ = "ORG"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        builder = GraphBuilder(use_entities=True)
        
        claim = "Claim text"
        evidence = ["Sentence 1"]
        
        data = builder.build_graph(claim, evidence)
        
        # Nodes: Claim (1) + Evidence (1) + Entity (1) = 3
        assert data.num_nodes == 3
        # Check node types
        assert data.node_type[0] == 0 # Claim
        assert data.node_type[1] == 1 # Evidence
        assert data.node_type[2] == 2 # Entity


class TestMultiHopReasoner:
    @patch('src.verification.multi_hop_reasoner.GraphBuilder')
    def test_predict(self, mock_builder_cls):
        # Mock GraphBuilder instance
        mock_builder = Mock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.get_sentence_embedding_dimension.return_value = 10
        
        # Mock Graph Data
        mock_data = Mock()
        mock_data.x = torch.randn(3, 10)
        mock_data.edge_index = torch.zeros((2, 2), dtype=torch.long)
        mock_data.num_nodes = 3
        mock_data.node_type = torch.tensor([0, 1, 2])
        mock_data.node_texts = ["Claim", "Evidence", "Entity"]
        mock_builder.build_graph.return_value = mock_data
        
        reasoner = MultiHopReasoner(
            embedding_model="dummy",
            hidden_dim=16,
            num_layers=1
        )
        
        # Mock Graph flow execution to avoid actual GNN errors if complexities arise
        # or simple forward pass if torch is handled correctly
        # Here we let it run because we mocked inputs to be valid torch tensors
        
        label, conf, probs, graph_data = reasoner.predict("Claim", ["Evidence"])
        
        assert label in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        assert 0.0 <= conf <= 1.0
        assert "nodes" in graph_data
        assert "edges" in graph_data

    def test_predict_no_evidence(self):
        with patch('src.verification.multi_hop_reasoner.GraphBuilder') as mb:
            mb.return_value.get_sentence_embedding_dimension.return_value = 10
            reasoner = MultiHopReasoner(hidden_dim=4)
            
            label, conf, probs, graph_data = reasoner.predict("Claim", [])
            
            assert label == "NOT_ENOUGH_INFO"
            assert conf == 1.0
            assert graph_data is None

    def test_train_step(self):
        with patch('src.verification.multi_hop_reasoner.GraphBuilder') as mb:
            # Setup bridge
            mb.return_value.get_sentence_embedding_dimension.return_value = 10
            
            # Mock graph data
            mock_data = Mock()
            mock_data.x = torch.randn(2, 10)
            mock_data.edge_index = torch.tensor([[0, 1], [1, 0]])
            mock_data.num_nodes = 2
            mock_data.node_type = torch.tensor([0, 1])
            mb.return_value.build_graph.return_value = mock_data
            
            reasoner = MultiHopReasoner(hidden_dim=16)
            optimizer = torch.optim.Adam(reasoner.parameters())
            
            loss = reasoner.train_step("Claim", ["Evidence"], 0, optimizer)
            
            assert isinstance(loss, float)
            assert loss >= 0

    def test_verify_with_evidence(self):
        # Test the compatibility method
        reasoner = MagicMock(spec=MultiHopReasoner)
        # We need to test the actual method, so we should patch MultiHopReasoner but keep method
        # Easier: Create instance with mocks
        
        with patch('src.verification.multi_hop_reasoner.GraphBuilder') as mb:
            mb.return_value.get_sentence_embedding_dimension.return_value = 10
            real_reasoner = MultiHopReasoner(hidden_dim=4)
            
            # Mock predict method to isolate verify_with_evidence logic
            real_reasoner.predict = Mock(return_value=("SUPPORTS", 0.9, {}, {"nodes": [], "edges": []}))
            
            evidence_list = [(1, "Ev 1")]
            result = real_reasoner.verify_with_evidence(0, "Claim", evidence_list)
            
            assert result.label == "SUPPORTS"
            assert result.confidence == 0.9
