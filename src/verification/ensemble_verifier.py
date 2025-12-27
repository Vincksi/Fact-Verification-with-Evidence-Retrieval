"""
Ensemble verification strategy combining NLI and GNN models.
"""

from typing import List, Tuple

from src.verification.multi_hop_reasoner import MultiHopReasoner
from src.verification.nli_model import NLIModel, VerificationResult


class EnsembleVerifier:
    """
    Ensemble verifier that combines predictions from NLI and GNN models.
    """

    def __init__(self, nli_model: NLIModel, gnn_model: MultiHopReasoner,
                 nli_weight: float = 0.5, gnn_weight: float = 0.5):
        """
        Initialize ensemble verifier.

        Args:
            nli_model: Initialized NLI model
            gnn_model: Initialized GNN model
            nli_weight: Weight for NLI model predictions
            gnn_weight: Weight for GNN model predictions
        """
        self.nli_model = nli_model
        self.gnn_model = gnn_model
        self.nli_weight = nli_weight
        self.gnn_weight = gnn_weight
        self.label_map = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    def verify_with_evidence(self, claim_id: int, claim: str,
                             evidence_list: List[Tuple[int, str]]) -> VerificationResult:
        """
        Verify a claim using ensemble of models.

        Args:
            claim_id: ID of the claim
            claim: Claim text
            evidence_list: List of (doc_id, evidence_text) tuples

        Returns:
            VerificationResult object
        """
        if not evidence_list:
            return VerificationResult(
                claim_id=claim_id,
                label="NOT_ENOUGH_INFO",
                confidence=1.0,
                label_probabilities={"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0},
                evidence_scores=[]
            )

        # Get NLI result (uses aggregator internally in the pipeline context,
        # but here we might need to handle aggregation or use the verifier directly)
        # However, the pipeline usually calls verifier.verify_with_evidence.

        # NLI prediction (batch)
        nli_res = self.nli_model.verify_with_evidence(claim_id, claim, evidence_list)

        # GNN prediction
        gnn_res = self.gnn_model.verify_with_evidence(claim_id, claim, evidence_list)

        # Merge probabilities
        merged_probs = {}
        for label in self.label_map:
            p_nli = nli_res.label_probabilities.get(label, 0.0)
            p_gnn = gnn_res.label_probabilities.get(label, 0.0)
            merged_probs[label] = (p_nli * self.nli_weight + p_gnn * self.gnn_weight) / \
                (self.nli_weight + self.gnn_weight)

        # Get final label and confidence
        final_label = max(merged_probs, key=merged_probs.get)
        final_confidence = merged_probs[final_label]

        # Average evidence scores for robustness
        # doc_id -> score
        evidence_dict = {}
        for doc_id, text, score in nli_res.evidence_scores:
            evidence_dict[doc_id] = (text, score * self.nli_weight)

        for doc_id, text, score in gnn_res.evidence_scores:
            if doc_id in evidence_dict:
                text, s = evidence_dict[doc_id]
                evidence_dict[doc_id] = (text, s + score * self.gnn_weight)
            else:
                evidence_dict[doc_id] = (text, score * self.gnn_weight)

        # Final evidence scores (normalized weights)
        final_evidence_scores = [
            (doc_id, text, score / (self.nli_weight + self.gnn_weight))
            for doc_id, (text, score) in evidence_dict.items()
        ]

        res = VerificationResult(
            claim_id=claim_id,
            label=final_label,
            confidence=final_confidence,
            label_probabilities=merged_probs,
            evidence_scores=final_evidence_scores
        )
        res.graph_data = getattr(gnn_res, 'graph_data', None)
        return res
