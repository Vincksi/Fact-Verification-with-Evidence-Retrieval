#!/usr/bin/env python3
"""
Script to execute the Pipeline using GAT (Graph Attention Network) for verification.
"""

import sys
import os
from pathlib import Path

# Ensure src is in python path
sys.path.append(os.getcwd())

from src.pipeline import FactVerificationPipeline
from src.verification.multi_hop_reasoner import MultiHopReasoner

def run_gat_pipeline():
    print("Initializing Pipeline in GAT Mode...")
    
    # 1. Initialize standard pipeline
    # This loads data, corpus, and config
    pipeline = FactVerificationPipeline("config/config.yaml")
    
    # 2. FORCE GAT Mode
    # Since config.yaml might have use_gnn: False, we explicitly override it here
    # and manually re-initialize the verifier with GNN parameters.
    print("Switching to GAT Verification...")
    pipeline.use_gnn = True
    
    gnn_config = pipeline.config['multi_hop']['gnn']
    graph_config = pipeline.config['multi_hop']['graph']
    
    pipeline.verifier = MultiHopReasoner(
        embedding_model=pipeline.config['retrieval']['dense_model'],
        hidden_dim=gnn_config['hidden_dim'],
        num_layers=gnn_config['num_layers'],
        num_heads=gnn_config['num_heads'],
        dropout=gnn_config['dropout'],
        similarity_threshold=graph_config['sentence_similarity_threshold'],
        max_sentences=graph_config['max_evidence_sentences'],
        use_entities=graph_config['use_entity_extraction'],
        use_onnx=pipeline.config.get('optimization', {}).get('use_onnx', False)
    )
    
    # 3. Load Data
    pipeline.load_corpus()
    
    # 4. Build/Load Retriever
    # Using 'dense' for speed in this demo, or 'hybrid' as per config
    print("Building Retriever (this may take a moment)...")
    # For a quick run, we try to load existing index if available
    try:
        pipeline.build_retriever(method='dense') 
        pipeline.load_index() # Try loading first
    except Exception as e:
        print(f"Index load failed ({e}), building from scratch...")
        pipeline.build_retriever(method='dense')
        pipeline.retriever.build_index()

    # 5. Run on an Example
    claim_text = "Aspirin reduces the risk of heart attack."
    print(f"\nProcessing Claim: '{claim_text}'")
    
    result = pipeline.process_claim(claim_id=999, claim_text=claim_text)
    
    print("\nExecution Complete!")
    print(f"Claim:      {result.claim}")
    print(f"Prediction: {result.predicted_label}")
    print(f"Confidence: {result.confidence:.4f}")
    
    if result.evidence_used:
        print(f"Evidence Used ({len(result.evidence_used)} items):")
        for i, (doc_id, text, score) in enumerate(result.evidence_used[:3]):
             print(f"  {i+1}. [Doc {doc_id}] {text[:80]}...")

if __name__ == "__main__":
    run_gat_pipeline()
