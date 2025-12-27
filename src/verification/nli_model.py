"""
Natural Language Inference (NLI) model for claim verification.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim_id: int
    label: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    confidence: float
    label_probabilities: Dict[str, float]
    evidence_scores: List[Tuple[int, str, float]]  # (doc_id, sentence, score)
    graph_data: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None  # LLM-generated explanation

    def __repr__(self):
        return f"VerificationResult(label={self.label}, confidence={self.confidence:.4f})"


class NLIModel:
    """
    Natural Language Inference model for verifying claims against evidence.
    """

    LABEL_MAP = {
        0: "REFUTES",
        1: "SUPPORTS",
        2: "NOT_ENOUGH_INFO"
    }

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cpu"):
        """
        Initialize NLI model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device

        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        print(f"NLI model loaded on {device}")

    def _adjust_neutral_bias(self, label: str, confidence: float, prob_dict: Dict[str, float]) -> str:
        """
        Adjust predictions to correct neutral bias in DeBERTa models.
        
        If the model predicts NOT_ENOUGH_INFO but SUPPORTS or REFUTES have
        reasonable confidence, reassign to the stronger non-neutral label.
        
        Args:
            label: Predicted label
            confidence: Confidence in predicted label
            prob_dict: Probability distribution over labels
            
        Returns:
            Adjusted label
        """
        if label == "NOT_ENOUGH_INFO":
            supports_conf = prob_dict.get("SUPPORTS", 0)
            refutes_conf = prob_dict.get("REFUTES", 0)
            neutral_conf = prob_dict.get("NOT_ENOUGH_INFO", 0)
            
            # More aggressive correction: even if one of SUPPORTS/REFUTES
            # has minimal signal (>0.01%), consider it over neutral
            max_non_neutral = max(supports_conf, refutes_conf)
            
            # If there's ANY non-trivial signal in SUPPORTS or REFUTES
            if max_non_neutral > 0.0001:  # Extremely low threshold
                # Reassign to stronger non-neutral label
                return "SUPPORTS" if supports_conf > refutes_conf else "REFUTES"
        
        return label

    def predict_single(self, claim: str, evidence: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the relationship between a claim and a single piece of evidence.

        Args:
            claim: The claim text
            evidence: The evidence text

        Returns:
            Tuple of (label, confidence, probabilities_dict)
        """
        # Tokenize input
        inputs = self.tokenizer(
            claim,
            evidence,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Get predicted label and confidence
        predicted_idx = torch.argmax(probs).item()
        label = self.LABEL_MAP[predicted_idx]
        confidence = probs[predicted_idx].item()

        # Build probabilities dictionary
        prob_dict = {
            self.LABEL_MAP[i]: probs[i].item()
            for i in range(len(probs))
        }

        # Apply neutral bias correction
        label = self._adjust_neutral_bias(label, confidence, prob_dict)

        return label, confidence, prob_dict

    def predict_batch(self, claim: str, evidences: List[str],
                      batch_size: int = 16) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Predict claim-evidence relationships for multiple evidence pieces.

        Args:
            claim: The claim text
            evidences: List of evidence texts
            batch_size: Batch size for processing

        Returns:
            List of (label, confidence, probabilities_dict) tuples
        """
        results = []

        for i in range(0, len(evidences), batch_size):
            batch_evidences = evidences[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                [claim] * len(batch_evidences),
                batch_evidences,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Process each result
            for j in range(len(batch_evidences)):
                predicted_idx = torch.argmax(probs[j]).item()
                label = self.LABEL_MAP[predicted_idx]
                confidence = probs[j][predicted_idx].item()

                prob_dict = {
                    self.LABEL_MAP[k]: probs[j][k].item()
                    for k in range(len(probs[j]))
                }

                results.append((label, confidence, prob_dict))

        return results

    def verify_with_evidence(self,
                             claim_id: int,
                             claim: str,
                             evidence_list: List[Tuple[int, str]]) -> VerificationResult:
        """
        Verify a claim against multiple evidence pieces.

        Args:
            claim_id: ID of the claim
            claim: The claim text
            evidence_list: List of (doc_id, evidence_text) tuples

        Returns:
            VerificationResult object with aggregated prediction
        """
        if not evidence_list:
            # No evidence available
            return VerificationResult(
                claim_id=claim_id,
                label="NOT_ENOUGH_INFO",
                confidence=1.0,
                label_probabilities={"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0},
                evidence_scores=[]
            )

        # Get predictions for each evidence
        evidence_texts = [text for _, text in evidence_list]
        predictions = self.predict_batch(claim, evidence_texts)

        # Store evidence scores
        evidence_scores = []
        for (doc_id, text), (label, conf, _) in zip(evidence_list, predictions):
            evidence_scores.append((doc_id, text, conf))

        # Simple aggregation: take the prediction with highest confidence
        best_idx = max(range(len(predictions)), key=lambda i: predictions[i][1])
        best_label, best_conf, best_probs = predictions[best_idx]

        return VerificationResult(
            claim_id=claim_id,
            label=best_label,
            confidence=best_conf,
            label_probabilities=best_probs,
            evidence_scores=evidence_scores
        )


def main():  # pragma: no cover
    """Example usage of NLIModel."""
    nli_model_example = NLIModel()

    claim_example = "Aspirin reduces the risk of heart attack"
    evidence_example = "Studies have shown that low-dose aspirin can reduce cardiovascular events in high-risk patients."

    label, confidence, probs = nli_model_example.predict_single(claim_example, evidence_example)

    print(f"Claim: {claim_example}")
    print(f"Evidence: {evidence_example}")
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probs}")


if __name__ == "__main__":  # pragma: no cover
    main()
