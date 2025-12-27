#!/usr/bin/env python
"""
Standalone utility for managing document indices (BM25 and Dense).
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.pipeline import FactVerificationPipeline

def main():
    parser = argparse.ArgumentParser(description="Manage document indices for Fact Verification.")
    parser.add_argument("action", choices=["build", "inspect", "rebuild"], help="Action to perform")
    parser.add_argument("--method", choices=["bm25", "dense", "all"], default="all", help="Retrieval method")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    print(f"--- Index Manager ---")
    print(f"Action: {args.action}")
    print(f"Method: {args.method}")
    print(f"Config: {args.config}")
    
    pipeline = FactVerificationPipeline(args.config)
    pipeline.load_corpus()
    
    methods = ["bm25", "dense"] if args.method == "all" else [args.method]
    
    for method in methods:
        print(f"\nProcessing {method} index...")
        
        if args.action == "build":
            # Build if not exists
            pipeline.build_retriever(method=method)
            try:
                pipeline.load_index()
                print(f"Index for {method} already exists.")
            except:
                print(f"Building {method} index...")
                pipeline.retriever.build_index()
                pipeline.save_index()
                print(f"Index for {method} built and saved.")
                
        elif args.action == "rebuild":
            print(f"Rebuilding {method} index...")
            pipeline.build_retriever(method=method)
            pipeline.retriever.build_index()
            pipeline.save_index()
            print(f"Index for {method} rebuilt and saved.")
            
        elif args.action == "inspect":
            pipeline.build_retriever(method=method)
            try:
                pipeline.load_index()
                print(f"Index for {method} is valid.")
                # Show some stats if possible
                if method == "bm25":
                    print(f"  Docs in corpus: {len(pipeline.dataset.corpus)}")
                elif method == "dense":
                    print(f"  Index size: {pipeline.retriever.index.ntotal}")
            except Exception as e:
                print(f"Error loading {method} index: {e}")

if __name__ == "__main__":
    main()
