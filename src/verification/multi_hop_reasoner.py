"""
GNN-based multi-hop reasoning for claim verification.

This module constructs a graph from claims, evidence sentences, and entities,
then uses Graph Attention Networks (GAT) to aggregate information across the graph.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GraphBuilder:
    """
    Constructs graphs from claims, evidence, and entities for GNN processing.
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 max_sentences: int = 20,
                 use_entities: bool = True):
        """
        Initialize graph builder.

        Args:
            embedding_model: Model for sentence embeddings
            similarity_threshold: Threshold for sentence similarity edges
            max_sentences: Maximum evidence sentences to include
            use_entities: Whether to extract and use entities
        """
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences
        self.use_entities = use_entities

        self.embedding_model = embedding_model
        # Load embedding model
        self.encoder = SentenceTransformer(embedding_model)

        # Load spaCy for entity extraction (if enabled)
        if use_entities:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_entities = False

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        if not self.use_entities:
            return []

        doc = self.nlp(text)
        entities = [ent.text.lower() for ent in doc.ents
                    if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]]
        return list(set(entities))  # Remove duplicates

    def build_graph(self,
                    claim: str,
                    evidence_sentences: List[str],
                    doc_ids: Optional[List[int]] = None) -> Data:
        """
        Build a heterogeneous graph from claim and evidence.

        Graph structure:
        - Node 0: Claim
        - Nodes 1 to N: Evidence sentences (limited to max_sentences)
        - Nodes N+1 onwards: Entities (if enabled)

        Edge types:
        - claim_to_evidence: From claim to each evidence sentence
        - sentence_similarity: Between similar evidence sentences
        - entity_coref: From sentences/claim to their entities

        Args:
            claim: The claim text
            evidence_sentences: List of evidence sentence texts
            doc_ids: Optional list of document IDs for each sentence

        Returns:
            PyTorch Geometric Data object
        """
        # Limit number of evidence sentences for CPU efficiency
        if len(evidence_sentences) > self.max_sentences:
            evidence_sentences = evidence_sentences[:self.max_sentences]
            if doc_ids:
                doc_ids = doc_ids[:self.max_sentences]

        # Build node list
        texts = [claim] + evidence_sentences
        num_sentences = len(texts)

        # Extract entities
        entities = []
        entity_to_idx = {}
        sentence_entities = defaultdict(list)  # Maps sentence idx to entity indices

        if self.use_entities:
            for i, text in enumerate(texts):
                text_entities = self.extract_entities(text)
                for entity in text_entities:
                    if entity not in entity_to_idx:
                        entity_to_idx[entity] = len(entities) + num_sentences
                        entities.append(entity)
                    sentence_entities[i].append(entity_to_idx[entity])

        total_nodes = num_sentences + len(entities)

        # Encode all texts (claim + sentences + entities)
        all_texts = texts + entities
        node_features = self.encoder.encode(
            all_texts,
            convert_to_numpy=False,
            normalize_embeddings=True
        )
        node_features = torch.tensor(np.array(node_features), dtype=torch.float)

        # Build edges
        edge_index = []
        edge_type = []  # 0: claim_to_evidence, 1: similarity, 2: entity

        # 1. Claim to evidence edges (bidirectional)
        for i in range(1, num_sentences):
            edge_index.append([0, i])
            edge_type.append(0)
            edge_index.append([i, 0])
            edge_type.append(0)

        # 2. Sentence similarity edges
        if num_sentences > 2:  # Only if we have evidence
            # Compute similarity matrix for evidence sentences
            evidence_features = node_features[1:num_sentences]
            similarity_matrix = torch.mm(evidence_features, evidence_features.t())

            # Add edges for similar sentences
            for i in range(len(evidence_features)):
                for j in range(i + 1, len(evidence_features)):
                    if similarity_matrix[i, j] > self.similarity_threshold:
                        # Bidirectional edges
                        edge_index.append([i + 1, j + 1])
                        edge_type.append(1)
                        edge_index.append([j + 1, i + 1])
                        edge_type.append(1)

        # 3. Entity co-reference edges
        if self.use_entities and entities:
            for sent_idx, entity_indices in sentence_entities.items():
                for ent_idx in entity_indices:
                    # Bidirectional edges between sentence and entity
                    edge_index.append([sent_idx, ent_idx])
                    edge_type.append(2)
                    edge_index.append([ent_idx, sent_idx])
                    edge_type.append(2)

        # Convert to tensor
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        else:
            # No edges, create empty tensors
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        # Create node type labels (0: claim, 1: evidence, 2: entity)
        node_type = torch.zeros(total_nodes, dtype=torch.long)
        node_type[0] = 0  # Claim
        node_type[1:num_sentences] = 1  # Evidence sentences
        if entities:
            node_type[num_sentences:] = 2  # Entities

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            num_nodes=total_nodes,
            node_texts=all_texts
        )

        return data


class GNNEncoder(nn.Module):
    """
    Graph Attention Network encoder for multi-hop reasoning.
    """

    def __init__(self,
                 input_dim: int = 384,  # all-MiniLM-L6-v2 dimension
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize GNN encoder.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
            else:
                # Subsequent layers
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch=None, return_attention=False):
        """
        Forward pass through GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for each node (for batched graphs)
            return_attention: Whether to return attention weights

        Returns:
            Updated node features [num_nodes, hidden_dim]
            (Optional) Attention weights
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        all_attentions = []

        # GAT layers
        for i, conv in enumerate(self.conv_layers):
            if return_attention:
                x_new, (edge_idx, att_weights) = conv(x, edge_index, return_attention_weights=True)
                all_attentions.append((edge_idx, att_weights))
            else:
                x_new = conv(x, edge_index)

            x_new = F.elu(x_new)  # ELU activation for GAT
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new

        # Output projection
        x = self.output_proj(x)

        if return_attention:
            return x, all_attentions
        return x


class MultiHopReasoner(nn.Module):
    """
    Complete multi-hop reasoning model using GNN.
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 3,  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
                 **graph_config):
        """
        Initialize multi-hop reasoner.

        Args:
            embedding_model: Sentence embedding model
            hidden_dim: GNN hidden dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_classes: Number of output classes
            **graph_config: Configuration for GraphBuilder
        """
        super().__init__()

        # Graph builder
        self.graph_builder = GraphBuilder(
            embedding_model=embedding_model,
            similarity_threshold=graph_config.get('sentence_similarity_threshold', 0.6),
            max_sentences=graph_config.get('max_evidence_sentences', 20),
            use_entities=graph_config.get('use_entity_extraction', True)
        )

        # Get embedding dimension from model
        input_dim = self.graph_builder.encoder.get_sentence_embedding_dimension()

        # GNN encoder
        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, claim: str, evidence_sentences: List[str]) -> torch.Tensor:
        """
        Forward pass for claim verification.

        Args:
            claim: Claim text
            evidence_sentences: List of evidence sentences

        Returns:
            Logits for each class [num_classes]
        """
        # Build graph
        graph = self.graph_builder.build_graph(claim, evidence_sentences)

        # Run GNN
        node_embeddings = self.gnn(graph.x, graph.edge_index)

        # Get claim node embedding (node 0)
        claim_embedding = node_embeddings[0]

        # Classify
        logits = self.classifier(claim_embedding)

        return logits

    def predict(self, claim: str, evidence_sentences: List[str]) -> Tuple[str, float, Dict[str, float], Dict[str, Any]]:
        """
        Predict label for a claim given evidence.

        Args:
            claim: Claim text
            evidence_sentences: List of evidence sentences

        Returns:
            (label, confidence, probabilities, graph_data)
        """
        if not evidence_sentences:
            return "NOT_ENOUGH_INFO", 1.0, {"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0}, None

        self.eval()

        with torch.no_grad():
            # Build graph
            graph = self.graph_builder.build_graph(claim, evidence_sentences)

            # Run GNN with attention
            node_embeddings, attentions = self.gnn(graph.x, graph.edge_index, return_attention=True)

            # Get claim node embedding (node 0)
            claim_embedding = node_embeddings[0]

            # Classify
            logits = self.classifier(claim_embedding)
            probs = F.softmax(logits, dim=-1)

            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

            label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}
            label = label_map[pred_idx]

            prob_dict = {
                label_map[i]: probs[i].item()
                for i in range(len(probs))
            }

            # Extract graph data for visualization
            # Just take the last layer attention for simplicity
            edge_index, att_weights = attentions[-1]
            # Average attention over heads
            if att_weights.dim() > 1:
                att_weights = att_weights.mean(dim=-1)

            graph_data = {
                "nodes": [
                    {
                        "id": i,
                        "label": "Claim" if graph.node_type[i] == 0 else ("Evidence" if graph.node_type[i] == 1 else "Entity"),
                        "text": graph.node_texts[i] if hasattr(graph, 'node_texts') else ""
                    }
                    for i in range(graph.num_nodes)
                ],
                "edges": [
                    {"source": u.item(), "target": v.item(), "weight": w.item()}
                    for (u, v), w in zip(edge_index.t(), att_weights)
                ]
            }

        return label, confidence, prob_dict, graph_data

    def verify_with_evidence(self, claim_id: int, claim: str,
                             evidence_list: List[Tuple[int, str]]) -> 'VerificationResult':
        """
        Verify a claim with evidence (compatible with NLIModel interface).

        Args:
            claim_id: ID of the claim
            claim: Claim text
            evidence_list: List of (doc_id, evidence_text) tuples

        Returns:
            VerificationResult object
        """
        from src.verification.nli_model import VerificationResult

        if not evidence_list:
            return VerificationResult(
                claim_id=claim_id,
                label="NOT_ENOUGH_INFO",
                confidence=1.0,
                label_probabilities={"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0},
                evidence_scores=[]
            )

        # Extract evidence sentences
        evidence_sentences = [text for _, text in evidence_list]

        # Get prediction
        label, confidence, probs, graph_data = self.predict(claim, evidence_sentences)

        # Build evidence scores
        evidence_scores = [(doc_id, text, confidence) for doc_id, text in evidence_list]

        res = VerificationResult(
            claim_id=claim_id,
            label=label,
            confidence=confidence,
            label_probabilities=probs,
            evidence_scores=evidence_scores
        )
        res.graph_data = graph_data  # Attach graph data
        return res

    def train_step(self, claim: str, evidence: List[str], label_idx: int, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform a single training step.

        Args:
            claim: Claim text
            evidence: List of evidence sentences
            label_idx: Index of the true label (0: SUPPORTS, 1: REFUTES, 2: NEI)
            optimizer: Torch optimizer

        Returns:
            Loss value
        """
        self.train()
        optimizer.zero_grad()

        # Forward pass
        logits = self.forward(claim, evidence)

        # Loss (CrossEntropy expects target as long tensor)
        target = torch.tensor([label_idx], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()


    def save_model(self, path: str):
        """
        Save the model weights and configuration to disk.

        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save state dict and config
        save_dict = {
            'state_dict': self.state_dict(),
            'config': {
                'embedding_model': self.graph_builder.embedding_model,
                'hidden_dim': self.gnn.input_proj.out_features,
                'num_layers': self.gnn.num_layers,
                'num_heads': self.gnn.conv_layers[0].heads,
                'dropout': self.gnn.dropout,
                'graph_config': {
                    'sentence_similarity_threshold': self.graph_builder.similarity_threshold,
                    'max_evidence_sentences': self.graph_builder.max_sentences,
                    'use_entity_extraction': self.graph_builder.use_entities
                }
            }
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> 'MultiHopReasoner':
        """
        Load a model from disk.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded MultiHopReasoner instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Initialize model with saved config
        instance = cls(
            embedding_model=config['embedding_model'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            **config['graph_config']
        )
        
        # Load weights
        instance.load_state_dict(checkpoint['state_dict'])
        instance.to(device)
        instance.eval()
        
        print(f"Model loaded from {path}")
        return instance

def main():  # pragma: no cover
    """Example usage of MultiHopReasoner."""
    print("Testing Multi-Hop Reasoner...")

    # Initialize reasoner
    reasoner_example = MultiHopReasoner(
        hidden_dim=256,
        num_layers=2,
        max_sentences=10,
        use_entities=True
    )

    # Example claim and evidence
    claim_text = "Aspirin reduces the risk of heart attack in high-risk patients"
    evidence_sentences = [
        "Studies have shown that low-dose aspirin can reduce cardiovascular events.",
        "Aspirin inhibits platelet aggregation, reducing blood clot formation.",
        "Some patients may experience side effects from aspirin use."
    ]

    # Build and inspect graph
    graph_obj = reasoner_example.graph_builder.build_graph(claim_text, evidence_sentences)
    print("\nGraph statistics:")
    print(f"  Nodes: {graph_obj.num_nodes}")
    print(f"  Edges: {graph_obj.edge_index.size(1)}")
    print(f"  Node types: {graph_obj.node_type.unique().tolist()}")
    print(f"  Edge types: {graph_obj.edge_type.unique().tolist()}")

    # Make prediction (random at initialization)
    label_pred, conf_pred, probs_pred, _ = reasoner_example.predict(claim_text, evidence_sentences)
    print("\nPrediction (untrained):")
    print(f"  Label: {label_pred}")
    print(f"  Confidence: {conf_pred:.4f}")
    print(f"  Probabilities: {probs_pred}")


if __name__ == "__main__":  # pragma: no cover
    main()
