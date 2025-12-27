#!/usr/bin/env python3
"""
Advanced Metrics Report Script for Fact Verification.
This script provides a unified view of retrieval and verification metrics.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Set, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import FactVerificationPipeline
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.verification_metrics import VerificationMetrics, print_metrics


import json


class Tee:
    """Utility to write to multiple files (like terminal and a log file)."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def extract_relevant_docs(claim) -> Set[int]:
    """Extract relevant document IDs from a claim."""
    relevant = set()
    if hasattr(claim, 'cited_doc_ids') and claim.cited_doc_ids:
        relevant.update(claim.cited_doc_ids)
    if hasattr(claim, 'evidence') and claim.evidence:
        for doc_id_str in claim.evidence.keys():
            relevant.add(int(doc_id_str))
    return relevant


def display_boxed_text(text: str, width: int = 70):
    """Display text inside an ASCII box."""
    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│ {text.center(width - 4)} │")
    print(f"└{'─' * (width - 2)}┘")


def main():
    parser = argparse.ArgumentParser(description='Fact Verification Unified Metrics Report')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'], help='Dataset split')
    parser.add_argument('--limit', type=int, default=20, help='Number of claims to evaluate')
    parser.add_argument('--top_k', type=int, default=5, help='K value for retrieval metrics')
    parser.add_argument('--output', type=str, help='Path to save the text report')
    parser.add_argument('--json-output', type=str, help='Path to save metrics as JSON')
    parser.add_argument('--load-index', action='store_true', help='Load pre-built index if available')
    parser.add_argument('--use-gnn', action='store_true', help='Force GNN verifier')
    parser.add_argument('--use-nli', action='store_true', help='Force NLI verifier')
    args = parser.parse_args()

    # Setup output capture if requested
    original_stdout = sys.stdout
    log_file = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(output_path, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_file)

    try:
        # ... (rest of the try block remains same until JSON output)
        display_boxed_text("FACT VERIFICATION SYSTEM - PERFORMANCE REPORT")

        # Initialize pipeline
        print(f"Loading pipeline from {args.config}...")
        pipeline = FactVerificationPipeline(args.config)
        
        # Override GNN/NLI if requested
        if args.use_gnn:
            print("Forcing GNN verifier...")
            pipeline.config['multi_hop']['use_gnn'] = True
            # We need to re-initialize since pipeline.__init__ already ran
            gnn_config = pipeline.config['multi_hop']['gnn']
            graph_config = pipeline.config['multi_hop']['graph']
            pipeline.verifier = MultiHopReasoner(
                embedding_model=pipeline.config['retrieval']['dense_model'],
                hidden_dim=gnn_config['hidden_dim'],
                num_layers=gnn_config['num_layers'],
                num_heads=gnn_config['num_heads'],
                dropout=gnn_config['dropout'],
                **graph_config
            )
            pipeline.use_gnn = True
        elif args.use_nli:
            print("Forcing NLI verifier...")
            pipeline.config['multi_hop']['use_gnn'] = False
            from src.verification.nli_model import NLIModel
            pipeline.verifier = NLIModel(
                model_name=pipeline.config['verification']['nli_model'],
                device='cpu'
            )
            from src.verification.evidence_aggregator import EvidenceAggregator
            pipeline.aggregator = EvidenceAggregator(
                strategy=pipeline.config['verification']['aggregation']
            )
            pipeline.use_gnn = False

        pipeline.load_corpus()
        
        # Build or load retriever
        if args.load_index:
            pipeline.build_retriever(method='hybrid', build=False)
            try:
                pipeline.load_index()
                print("Retrieval index loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load index ({e}). Building fresh index...")
                pipeline.build_retriever(method='hybrid', build=True)
        else:
            pipeline.build_retriever(method='hybrid', build=True)
        
        # Load claims
        claims = pipeline.dataset.load_claims(args.split)
        if args.limit:
            claims = claims[:args.limit]
        
        print(f"Evaluating on {len(claims)} claims from '{args.split}' split...\n")

        # 1. Retrieval Evaluation
        retrieved_ids_list = []
        relevant_ids_list = []
        
        # 2. Verification Evaluation
        y_true = []
        y_pred = []
        evidence_found = []

        for i, claim in enumerate(claims, 1):
            if i % 5 == 0 or i == 1:
                print(f"[{i}/{len(claims)}] Processing claim: {claim.claim[:60]}...")
            
            # Retrieval
            retrieval_results = pipeline.retrieve_evidence(claim.claim, top_k=args.top_k)
            ret_ids = [r.doc_id for r in retrieval_results]
            rel_ids = extract_relevant_docs(claim)
            
            retrieved_ids_list.append(ret_ids)
            relevant_ids_list.append(rel_ids)
            
            # End-to-End Verification
            evidence_context = [(r.doc_id, r.text) for r in retrieval_results]
            verification_result = pipeline.verifier.verify_with_evidence(claim.id, claim.claim, evidence_context)
            
            y_true.append(claim.label)
            y_pred.append(verification_result.label)
            
            # Check if correct evidence was in top_k
            has_rel = bool(rel_ids & set(ret_ids))
            evidence_found.append(has_rel)

        # --- COMPUTE RETRIEVAL METRICS ---
        ret_metrics = RetrievalMetrics.evaluate_retrieval(retrieved_ids_list, relevant_ids_list, k_values=[1, 3, 5, 10])
        
        # --- COMPUTE VERIFICATION METRICS ---
        ver_metrics = VerificationMetrics.compute_metrics(y_true, y_pred)
        ver_metrics['fever_score'] = VerificationMetrics.compute_fever_score(y_true, y_pred, evidence_found)

        # --- DISPLAY REPORT ---
        display_boxed_text("RETRIEVAL PERFORMANCE")
        print(f"{'Metric':<20} | {'Score':<10}")
        print("-" * 35)
        for m in ['precision@1', 'precision@3', 'precision@5', 'recall@1', 'recall@3', 'recall@5', 'map', 'mrr']:
            if m in ret_metrics:
                print(f"{m.replace('_', ' ').title():<20} | {ret_metrics[m]:.4f}")

        display_boxed_text("VERIFICATION PERFORMANCE")
        print_metrics(ver_metrics, title="End-to-End Metrics")
        
        print("\nConfusion Matrix:")
        print(VerificationMetrics.compute_confusion_matrix(y_true, y_pred))

        display_boxed_text("SUMMARY")
        total_claims = len(claims)
        print(f"Total claims evaluated: {total_claims}")
        print(f"Overall Accuracy:       {ver_metrics['accuracy']:.4f}")
        print(f"FEVER Score:            {ver_metrics['fever_score']:.4f}")
        print(f"Evidence Hit Rate:      {sum(evidence_found)/total_claims:.4f}")
        
        # --- JSON OUTPUT ---
        if args.json_output:
            # Helper to convert nested values to native Python types
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                elif hasattr(obj, 'item'):  # Numpy types, Torch tensors
                    return obj.item()
                return obj

            combined_metrics = {
                "retrieval": convert_types(ret_metrics),
                "verification": convert_types(ver_metrics),
                "summary": {
                    "total_claims": total_claims,
                    "accuracy": float(ver_metrics['accuracy']),
                    "fever_score": float(ver_metrics['fever_score']),
                    "evidence_hit_rate": float(sum(evidence_found)/total_claims)
                },
                "config": {
                    "split": args.split,
                    "limit": args.limit,
                    "top_k": args.top_k
                }
            }
            json_path = Path(args.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_metrics, f, indent=4)
            print(f"\nSaved raw metrics to {args.json_output}")

        if args.output:
            print(f"\nSaved text report to {args.output}")

    finally:
        # Restore stdout and close file
        sys.stdout = original_stdout
        if log_file:
            log_file.close()

    print("\nReport generation complete.")


if __name__ == "__main__":
    main()
