"""
Evaluate the fact verification pipeline on SciFact dataset.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FactVerificationPipeline, PipelineResult
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.verification_metrics import VerificationMetrics, print_metrics


def extract_relevant_docs(claim) -> Set[int]:
    """Extract relevant document IDs from a claim."""
    relevant = set()
    
    # Get cited docs
    if hasattr(claim, 'cited_doc_ids') and claim.cited_doc_ids:
        relevant.update(claim.cited_doc_ids)
    
    # Get docs from evidence
    if hasattr(claim, 'evidence') and claim.evidence:
        for doc_id_str in claim.evidence.keys():
            relevant.add(int(doc_id_str))
    
    return relevant


def evaluate_retrieval(pipeline: FactVerificationPipeline, 
                      split: str = 'dev',
                      k_values: List[int] = [1, 3, 5, 10, 20]):
    """
    Evaluate retrieval performance.
    
    Args:
        pipeline: Fact verification pipeline
        split: Dataset split to evaluate
        k_values: List of k values for metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating Retrieval on {split.upper()} set")
    print(f"{'=' * 70}")
    
    # Load claims
    claims = pipeline.dataset.load_claims(split)
    
    retrieved_docs = []
    relevant_docs = []
    
    print(f"\nRetrieving evidence for {len(claims)} claims...")
    
    for i, claim in enumerate(claims, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(claims)} claims...")
        
        # Retrieve documents
        results = pipeline.retrieve_evidence(claim.claim, top_k=max(k_values))
        retrieved_ids = [r.doc_id for r in results]
        
        # Get relevant documents
        relevant_ids = extract_relevant_docs(claim)
        
        retrieved_docs.append(retrieved_ids)
        relevant_docs.append(relevant_ids)
    
    # Compute metrics
    metrics = RetrievalMetrics.evaluate_retrieval(retrieved_docs, relevant_docs, k_values)
    
    # Print results
    print(f"\n{'-' * 70}")
    print("Retrieval Results:")
    print(f"{'-' * 70}")
    
    for k in k_values:
        print(f"\nk = {k}:")
        print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
        print(f"  NDCG@{k}:      {metrics[f'ndcg@{k}']:.4f}")
    
    print(f"\nAggregate Metrics:")
    print(f"  MAP: {metrics['map']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    
    return metrics


def evaluate_verification(pipeline: FactVerificationPipeline,
                         split: str = 'dev',
                         limit: int = None):
    """
    Evaluate end-to-end verification performance.
    
    Args:
        pipeline: Fact verification pipeline
        split: Dataset split to evaluate
        limit: Maximum number of claims to evaluate
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating End-to-End Verification on {split.upper()} set")
    print(f"{'=' * 70}")
    
    # Process dataset
    results = pipeline.process_dataset(split, limit=limit)
    
    # Extract predictions and ground truth
    y_true = [r.true_label for r in results if r.true_label]
    y_pred = [r.predicted_label for r in results if r.true_label]
    
    # Check which claims had correct evidence retrieved
    evidence_retrieved = []
    for result in results:
        if result.true_label is None:
            continue
        
        # Get claim object
        claim = pipeline.dataset.get_claim(result.claim_id, split)
        if claim is None:
            evidence_retrieved.append(False)
            continue
        
        # Get relevant docs for this claim
        relevant_docs = extract_relevant_docs(claim)
        
        # Check if any retrieved doc is relevant
        retrieved_doc_ids = {r.doc_id for r in result.retrieved_docs}
        has_relevant = bool(relevant_docs & retrieved_doc_ids)
        evidence_retrieved.append(has_relevant)
    
    # Compute metrics
    metrics = VerificationMetrics.compute_metrics(y_true, y_pred)
    
    # Compute FEVER score
    if evidence_retrieved:
        metrics['fever_score'] = VerificationMetrics.compute_fever_score(
            y_true, y_pred, evidence_retrieved
        )
    
    # Print results
    print_metrics(metrics, title="Verification Results")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print("-" * 70)
    cm = VerificationMetrics.compute_confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(VerificationMetrics.print_classification_report(y_true, y_pred))
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description='Evaluate fact verification pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--split', type=str, default='dev', 
                       choices=['train', 'dev', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--eval-mode', type=str, default='both',
                       choices=['retrieval', 'verification', 'both'],
                       help='What to evaluate')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of claims to evaluate')
    parser.add_argument('--load-index', action='store_true',
                       help='Load pre-built index instead of building')
    parser.add_argument('--save-predictions', type=str, default=None,
                       help='Path to save predictions JSON')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = FactVerificationPipeline(args.config)
    pipeline.load_corpus()
    
    # Build or load index
    if args.load_index:
        print("Loading pre-built index...")
        pipeline.build_retriever()  # Initialize retriever
        pipeline.load_index()
    else:
        print("Building retrieval index...")
        pipeline.build_retriever()
    
    # Run evaluation
    if args.eval_mode in ['retrieval', 'both']:
        retrieval_metrics = evaluate_retrieval(pipeline, args.split)
    
    if args.eval_mode in ['verification', 'both']:
        verification_metrics, results = evaluate_verification(
            pipeline, args.split, args.limit
        )
        
        # Save predictions if requested
        if args.save_predictions:
            predictions = []
            for r in results:
                predictions.append({
                    'claim_id': r.claim_id,
                    'claim': r.claim,
                    'predicted_label': r.predicted_label,
                    'confidence': r.confidence,
                    'true_label': r.true_label,
                    'retrieved_doc_ids': [doc.doc_id for doc in r.retrieved_docs]
                })
            
            with open(args.save_predictions, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            print(f"\nPredictions saved to {args.save_predictions}")
    
    print(f"\n{'=' * 70}")
    print("Evaluation complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
