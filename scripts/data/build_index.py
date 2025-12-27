"""
Build retrieval indices from the SciFact corpus.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import SciFactDataset
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description='Build retrieval indices for SciFact corpus')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing SciFact data')
    parser.add_argument('--output-dir', type=str, default='data/preprocessed/indices',
                       help='Directory to save indices')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['bm25', 'dense', 'hybrid', 'all'],
                       help='Which retrieval method to build index for')
    parser.add_argument('--dense-model', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Dense retrieval model name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    print("=" * 60)
    print("Building Retrieval Indices for SciFact")
    print("=" * 60)
    
    dataset = SciFactDataset(args.data_dir)
    dataset.load_corpus()
    
    print(f"\nCorpus loaded: {len(dataset.corpus)} documents")
    print(f"Output directory: {output_path}")
    
    # Build indices based on method
    methods_to_build = ['bm25', 'dense', 'hybrid'] if args.method == 'all' else [args.method]
    
    for method in methods_to_build:
        print(f"\n{'-' * 60}")
        print(f"Building {method.upper()} index...")
        print(f"{'-' * 60}")
        
        if method == 'bm25':
            retriever = BM25Retriever(dataset.corpus)
            retriever.build_index()
            retriever.save_index(str(output_path))
            
        elif method == 'dense':
            retriever = DenseRetriever(
                dataset.corpus,
                model_name=args.dense_model,
                batch_size=args.batch_size
            )
            retriever.build_index()
            retriever.save_index(str(output_path))
            
        elif method == 'hybrid':
            retriever = HybridRetriever(
                dataset.corpus,
                dense_model=args.dense_model,
                batch_size=args.batch_size
            )
            retriever.build_index()
            retriever.save_index(str(output_path))
    
    print(f"\n{'=' * 60}")
    print("Index building complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
