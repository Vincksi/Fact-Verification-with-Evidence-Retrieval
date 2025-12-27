"""
Tests for GATDataset and DataLoader.
"""

import pytest
import torch
from unittest.mock import MagicMock, Mock
from src.data.gat_dataset import GATDataset, get_gat_dataloader
from src.data.dataset_loader import Claim, SciFactDataset

class TestGATDataset:
    def setup_method(self):
        # Mock dependencies
        self.mock_dataset_loader = MagicMock(spec=SciFactDataset)
        self.mock_verifier = MagicMock()
        self.mock_graph_builder = MagicMock()
        self.mock_verifier.graph_builder = self.mock_graph_builder
        
        # Mock claims
        self.claims = [
            Claim(id=1, claim="Claim 1", evidence={"100": [{"sentences": [1], "label": "SUPPORT"}]}, cited_doc_ids=[100]),
            Claim(id=2, claim="Claim 2", evidence={}, cited_doc_ids=[200]) # NEI
        ]
        
        # Mock evidence sentences
        self.mock_dataset_loader.get_evidence_sentences.side_effect = [
            {100: ["Evidence 1"]},
            {} # Empty for claim 2
        ]
        
        # Mock graph construction
        self.mock_graph = MagicMock()
        self.mock_graph.num_nodes = 5
        self.mock_graph_builder.build_graph.return_value = self.mock_graph

    def test_dataset_len(self):
        dataset = GATDataset(self.claims, self.mock_dataset_loader, self.mock_verifier)
        assert len(dataset) == 2

    def test_dataset_getitem_with_evidence(self):
        dataset = GATDataset(self.claims, self.mock_dataset_loader, self.mock_verifier)
        
        # Test first claim
        graph = dataset[0]
        assert graph == self.mock_graph
        # Check if label was added (SUPPORT mapped to 0)
        assert hasattr(graph, 'y')
        assert graph.y.item() == 0 
        
        self.mock_graph_builder.build_graph.assert_called_once_with("Claim 1", ["Evidence 1"])

    def test_dataset_getitem_no_evidence(self):
        # We need to ensure the side_effect for this call returns {}
        self.mock_dataset_loader.get_evidence_sentences.side_effect = [
            {100: ["Evidence 1"]},
            {} # Empty for the second item
        ]
        dataset = GATDataset(self.claims, self.mock_dataset_loader, self.mock_verifier)
        
        # Skip the first one 
        _ = dataset[0]
        
        # Test second claim (no evidence)
        graph = dataset[1]
        assert graph is None

    def test_dataloader_batching(self):
        # Setup multiple claims including some with no evidence
        many_claims = [
            Claim(id=i, claim=f"Claim {i}", evidence={"1": []}, cited_doc_ids=[]) 
            for i in range(5)
        ]
        
        # Reset the mock from earlier tests
        self.mock_dataset_loader.get_evidence_sentences = MagicMock()
        # Alternate: i=0 (Ev), i=1 ({}), i=2 (Ev), i=3 ({}), i=4 (Ev)
        self.mock_dataset_loader.get_evidence_sentences.side_effect = [
            {1: ["Ev"]} if i % 2 == 0 else {} for i in range(5)
        ]
        
        # Mock build_graph to return a real-looking Data object
        from torch_geometric.data import Data
        def side_effect_build(claim_text, sentences):
            d = Data()
            d.x = torch.randn(1, 10)
            d.edge_index = torch.zeros((2, 0), dtype=torch.long)
            # Add num_nodes attribute which PyG Batch needs
            d.num_nodes = 1
            return d
            
        self.mock_graph_builder.build_graph.side_effect = side_effect_build

        dataloader = get_gat_dataloader(
            many_claims, 
            self.mock_dataset_loader, 
            self.mock_verifier, 
            # batch_size=2: 
            # Items 0, 1 -> Dataset returns [G0, None]. Collate returns Batch([G0]) (1 graph)
            # Items 2, 3 -> Dataset returns [G2, None]. Collate returns Batch([G2]) (1 graph)
            # Items 4 -> Dataset returns [G4]. Collate returns Batch([G4]) (1 graph)
            # TOTAL: 3 batches of 1 graph each
            batch_size=2,
            shuffle=False
        )
        
        batches = list(dataloader)
        # Should be 3 batches because None values are filtered out AFTER batching
        assert len(batches) == 3
        
        for batch in batches:
            assert batch.num_graphs == 1
