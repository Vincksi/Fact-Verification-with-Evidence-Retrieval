"""
End-to-end fact verification pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.data.dataset_loader import SciFactDataset
from src.retrieval.base_retriever import BaseRetriever, RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.verification.evidence_aggregator import EvidenceAggregator
from src.verification.multi_hop_reasoner import MultiHopReasoner
from src.verification.nli_model import NLIModel, VerificationResult
from src.verification.ensemble_verifier import EnsembleVerifier

try:
    from src.explanation.groq_explainer import GroqExplainer
    EXPLANATION_AVAILABLE = True
except ImportError:
    EXPLANATION_AVAILABLE = False
    GroqExplainer = None


@dataclass
class PipelineResult:
    """Result from the complete pipeline."""
    claim_id: int
    claim: str
    predicted_label: str
    confidence: float
    retrieved_docs: List[RetrievalResult]
    evidence_used: List[Tuple[int, str, float]]  # (doc_id, sentence, score)
    true_label: Optional[str] = None
    graph_data: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None  # LLM-generated explanation

    def __repr__(self):
        return (f"PipelineResult(claim_id={self.claim_id}, "
                f"label={self.predicted_label}, confidence={self.confidence:.4f})")


class FactVerificationPipeline:
    """Complete fact verification pipeline combining retrieval and verification."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize dataset
        data_dir = self.config['data']['scifact_dir']
        self.dataset = SciFactDataset(data_dir)

        # Initialize retriever (will be built later)
        self.retriever: Optional[BaseRetriever] = None

        # Initialize verifier based on config
        verification_config = self.config['verification']
        self.use_gnn = self.config['multi_hop'].get('use_gnn', False)

        if self.use_gnn:
            model_path = self.config['multi_hop']['gnn'].get('model_path', 'models/gat_model.pt')
            use_onnx = self.config.get('optimization', {}).get('use_onnx', False)
            dense_model = self.config['retrieval']['dense_model']

            if os.path.exists(model_path):
                print(f"Loading trained GNN model from {model_path}...")
                self.verifier = MultiHopReasoner.load_model(
                    model_path, 
                    use_onnx=use_onnx,
                    embedding_model=dense_model
                )
            else:
                print(f"Initializing fresh GNN-based verifier (no saved model found at {model_path})...")
                gnn_config = self.config['multi_hop']['gnn']
                graph_config = self.config['multi_hop']['graph']

                self.verifier = MultiHopReasoner(
                    embedding_model=dense_model,
                    hidden_dim=gnn_config['hidden_dim'],
                    num_layers=gnn_config['num_layers'],
                    num_heads=gnn_config['num_heads'],
                    dropout=gnn_config['dropout'],
                    similarity_threshold=graph_config['sentence_similarity_threshold'],
                    max_sentences=graph_config['max_evidence_sentences'],
                    use_entities=graph_config['use_entity_extraction'],
                    use_onnx=use_onnx
                )
        else:
            print("Initializing NLI-based verifier...")
            self.verifier = NLIModel(
                model_name=verification_config['nli_model'],
                device='cpu',
                use_onnx=self.config.get('optimization', {}).get('use_onnx', False)
            )

        # Evidence aggregator (only for NLI mode)
        if not self.use_gnn:
            self.aggregator = EvidenceAggregator(
                strategy=verification_config['aggregation']
            )

        # Initialize explanation generator if enabled
        self.explainer = None
        explanation_config = self.config.get('explanation', {})
        if explanation_config.get('enabled', False) and EXPLANATION_AVAILABLE:
            try:
                self.explainer = GroqExplainer(
                    model=explanation_config.get('model', 'llama-3.3-70b-versatile'),
                    max_tokens=explanation_config.get('max_tokens', 512),
                    temperature=explanation_config.get('temperature', 0.3)
                )
                print("Explanation generation enabled with Groq")
            except Exception as e:
                print(f"Warning: Could not initialize explainer: {e}")
                self.explainer = None

    def load_corpus(self):
        """Load the document corpus."""
        print("Loading corpus...")
        self.dataset.load_corpus(self.config['data']['corpus_path'])

    def build_retriever(self, method: Optional[str] = None, build: bool = True):
        """
        Build the retrieval index.

        Args:
            method: Retrieval method ('bm25', 'dense', 'hybrid').
                   If None, uses config default.
            build: Whether to build the index immediately. Default True.
        """
        if method is None:
            method = self.config['retrieval']['method']

        print(f"Building {method} retriever...")

        if method == 'bm25':
            self.retriever = BM25Retriever(self.dataset.corpus)
        elif method == 'dense':
            self.retriever = DenseRetriever(
                self.dataset.corpus,
                model_name=self.config['retrieval']['dense_model'],
                batch_size=self.config['retrieval']['batch_size'],
                use_onnx=self.config.get('optimization', {}).get('use_onnx', False)
            )
        elif method == 'hybrid':
            self.retriever = HybridRetriever(
                self.dataset.corpus,
                dense_model=self.config['retrieval']['dense_model'],
                bm25_weight=self.config['retrieval']['bm25_weight'],
                dense_weight=self.config['retrieval']['dense_weight'],
                batch_size=self.config['retrieval']['batch_size'],
                use_onnx=self.config.get('optimization', {}).get('use_onnx', False)
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        if build:
            self.retriever.build_index()

    def save_index(self, path: Optional[str] = None):
        """Save retrieval index to disk."""
        if path is None:
            path = self.config['data']['index_dir']

        Path(path).mkdir(parents=True, exist_ok=True)
        self.retriever.save_index(path)

    def load_index(self, path: Optional[str] = None):
        """Load retrieval index from disk."""
        if path is None:
            path = self.config['data']['index_dir']

        self.retriever.load_index(path)

    def retrieve_evidence(self, claim: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve evidence for a claim.

        Args:
            claim: The claim text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_retriever() first.")

        if top_k is None:
            top_k = self.config['retrieval']['top_k']

        return self.retriever.retrieve(claim, top_k=top_k)

    def verify_claim(self, claim: str, evidence: List[RetrievalResult]) -> VerificationResult:
        """
        Verify a claim given retrieved evidence.

        Args:
            claim: The claim text
            evidence: List of retrieval results

        Returns:
            Verification result
        """
        if self.use_gnn:
            # GNN-based verification
            evidence_sentences = []
            for result_item in evidence:
                document = self.dataset.get_document(result_item.doc_id)
                if document:
                    # Use all sentences from the abstract
                    evidence_sentences.extend(document.abstract)

            # Limit sentences for CPU efficiency
            max_sentences = self.config['multi_hop']['graph']['max_evidence_sentences']
            evidence_sentences = evidence_sentences[:max_sentences]

            label, confidence, probs, graph_data = self.verifier.predict(claim, evidence_sentences)

            # Build verification result
            res = VerificationResult(
                claim_id=0,  # Will be set by caller
                label=label,
                confidence=confidence,
                label_probabilities=probs,
                evidence_scores=[(r.doc_id, r.text, r.score) for r in evidence]
            )
            res.graph_data = graph_data
            return res

        # NLI-based or Ensemble verification
        evidence_list = [(r.doc_id, r.text) for r in evidence]

        if not evidence_list:
            return VerificationResult(
                claim_id=0,
                label="NOT_ENOUGH_INFO",
                confidence=1.0,
                label_probabilities={"NOT_ENOUGH_INFO": 1.0, "SUPPORTS": 0.0, "REFUTES": 0.0},
                evidence_scores=[]
            )

        # Handle EnsembleVerifier specially
        if isinstance(self.verifier, EnsembleVerifier):
            return self.verifier.verify_with_evidence(0, claim, evidence_list)

        # Get predictions for each evidence piece
        evidence_texts = [text for _, text in evidence_list]
        predictions = self.verifier.predict_batch(claim, evidence_texts)

        # Aggregate predictions using the pipeline's aggregator
        label, confidence, probs = self.aggregator.aggregate(predictions)

        # Build evidence scores list (doc_id, text, confidence)
        evidence_scores = []
        for (doc_id, text), (_, pred_conf, _) in zip(evidence_list, predictions):
            evidence_scores.append((doc_id, text, pred_conf))

        return VerificationResult(
            claim_id=0,
            label=label,
            confidence=confidence,
            label_probabilities=probs,
            evidence_scores=evidence_scores
        )

    def process_claim(self, claim_id: int, claim_text: str,
                      true_label: Optional[str] = None) -> PipelineResult:
        """
        Process a single claim through the complete pipeline.

        Args:
            claim_id: ID of the claim
            claim_text: The claim text
            true_label: True label (optional, for evaluation)

        Returns:
            Pipeline result
        """
        # Retrieve evidence
        retrieved = self.retrieve_evidence(claim_text)

        # Verify claim
        verification = self.verify_claim(claim_text, retrieved)

        # Generate explanation if enabled
        explanation = None
        if self.explainer and verification.evidence_scores:
            try:
                explanation = self.explainer.explain_verification(
                    claim=claim_text,
                    evidence_list=[(doc_id, text) for doc_id, text, _ in verification.evidence_scores],
                    predicted_label=verification.label,
                    confidence=verification.confidence,
                    label_probabilities=verification.label_probabilities
                )
            except Exception as e:
                print(f"Warning: Failed to generate explanation: {e}")
                explanation = None

        return PipelineResult(
            claim_id=claim_id,
            claim=claim_text,
            predicted_label=verification.label,
            confidence=verification.confidence,
            retrieved_docs=retrieved,
            evidence_used=verification.evidence_scores,
            true_label=true_label,
            graph_data=verification.graph_data,
            explanation=explanation
        )

    def process_dataset(self, split: str = 'dev',
                        limit: Optional[int] = None) -> List[PipelineResult]:
        """
        Process all claims in a dataset split.

        Args:
            split: Dataset split ('train', 'dev', 'test')
            limit: Maximum number of claims to process

        Returns:
            List of pipeline results
        """
        # Load claims
        claims = self.dataset.load_claims(split)

        if limit:
            claims = claims[:limit]

        print(f"\nProcessing {len(claims)} claims from {split} split...")

        results = []
        for i, claim in enumerate(claims, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(claims)} claims...")

            result = self.process_claim(
                claim_id=claim.id,
                claim_text=claim.claim,
                true_label=claim.label
            )
            results.append(result)

        print(f"Finished processing {len(results)} claims")
        return results


def main():  # pragma: no cover
    """Example usage of the FactVerificationPipeline."""
    pipeline = FactVerificationPipeline()

    # Load corpus and build retriever
    pipeline.load_corpus()
    pipeline.build_retriever(method='hybrid')

    # Process a single claim
    example_claim_text = "Aspirin reduces the risk of heart attack in high-risk patients"
    pipeline_result = pipeline.process_claim(0, example_claim_text)

    print(f"\nClaim: {pipeline_result.claim}")
    print(f"Predicted: {pipeline_result.predicted_label} "
          f"(confidence: {pipeline_result.confidence:.4f})")
    print("\nTop 3 retrieved documents:")
    for doc_result in pipeline_result.retrieved_docs[:3]:
        print(f"  - {doc_result.title} (score: {doc_result.score:.4f})")


if __name__ == "__main__":  # pragma: no cover
    main()
