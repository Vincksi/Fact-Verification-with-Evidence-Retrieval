"""
PyTorch Dataset and DataLoader for GAT training.
"""

from typing import Any, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from src.data.dataset_loader import Claim, SciFactDataset


class GATDataset(Dataset):
    """
    Dataset wrapper for GAT training.
    """

    def __init__(self, claims: List[Claim], dataset_loader: SciFactDataset, verifier: Any):
        """
        Initialize GAT dataset.

        Args:
            claims: List of Claim objects
            dataset_loader: SciFactDataset instance
            verifier: MultiHopReasoner instance (to use its graph builder)
        """
        self.claims = claims
        self.dataset_loader = dataset_loader
        self.verifier = verifier
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = self.claims[idx]

        # Get evidence sentences
        evidence_dict = self.dataset_loader.get_evidence_sentences(claim)
        all_sentences = []
        for doc_sentences in evidence_dict.values():
            all_sentences.extend(doc_sentences)

        if not all_sentences:
            return None

        # Build graph
        graph = self.verifier.graph_builder.build_graph(claim.claim, all_sentences)

        # Add label
        label_idx = self.label_map.get(claim.label, 2)
        graph.y = torch.tensor([label_idx], dtype=torch.long)

        return graph


def collate_fn(batch):
    """
    Collate function for GATDataset to handle None values (skipped samples).
    """
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)


def get_gat_dataloader(claims: List[Claim], dataset_loader: SciFactDataset,
                       verifier: Any, batch_size: int = 4, shuffle: bool = True):
    """
    Create a DataLoader for GAT training.
    """
    dataset = GATDataset(claims, dataset_loader, verifier)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
