#!/usr/bin/env python3
"""
CLI script to train the GAT (Graph Attention Network) model.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.pipeline import FactVerificationPipeline
from src.training.trainer import GATTrainer
from src.verification.multi_hop_reasoner import MultiHopReasoner

def main():
    parser = argparse.ArgumentParser(description="Train the GAT model for claim verification.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    
    args = parser.parse_args()

    print(f"Starting GAT Training (Epochs: {args.epochs}, LR: {args.lr})")
    
    # 1. Initialize pipeline
    pipeline = FactVerificationPipeline(args.config)
    pipeline.load_corpus()
    
    # 2. Ensure MultiHopReasoner is initialized
    gnn_config = pipeline.config['multi_hop']['gnn']
    pipeline.verifier = MultiHopReasoner(
        embedding_model=pipeline.config['retrieval']['dense_model'],
        hidden_dim=gnn_config['hidden_dim'],
        num_layers=gnn_config['num_layers'],
        num_heads=gnn_config['num_heads'],
        dropout=gnn_config['dropout'],
        **pipeline.config['multi_hop']['graph']
    )
    pipeline.use_gnn = True

    # 3. Start Trainer
    trainer = GATTrainer(pipeline, learning_rate=args.lr)
    
    # Observe training progress (the trainer usually runs in background, 
    # but for CLI we want visibility)
    # We can override the background thread or just poll here.
    
    print("Loading training data...")
    trainer.start_training(epochs=args.epochs)
    
    try:
        while trainer.is_training:
            import time
            time.sleep(1)
        print("\nTraining Complete!")
    except KeyboardInterrupt:
        print("\nStopping training...")
        trainer.stop_training()
        while trainer.is_training:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
