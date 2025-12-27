"""
Dense retrieval using sentence embeddings and FAISS for efficient search.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """Dense retrieval using semantic embeddings."""

    def __init__(self, corpus: Dict[int, Any], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        """
        Initialize dense retriever.

        Args:
            corpus: Dictionary mapping doc_id to Document objects
            model_name: Name of the sentence transformer model
            **kwargs: Additional parameters (batch_size, etc.)
        """
        super().__init__(corpus, **kwargs)
        self.model_name = model_name
        self.batch_size = kwargs.get('batch_size', 32)

        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # Set to evaluation mode

        self.index = None
        self.doc_ids = []
        self.embeddings = None

    def build_index(self):
        """Build FAISS index from document embeddings."""
        print("Building dense retrieval index...")

        # Sort documents by ID for consistent ordering
        sorted_docs = sorted(self.corpus.items(), key=lambda x: x[0])

        self.doc_ids = []
        texts = []

        for doc_id, doc in sorted_docs:
            self.doc_ids.append(doc_id)
            texts.append(doc.full_text)

        # Try to load cached embeddings first to save time
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        embeddings_loaded = False
        if cache_file.exists():
            try:
                print(f"Loading cached embeddings from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Verify cache matches current corpus size/content (simplistic check)
                if len(cache_data['doc_ids']) == len(self.doc_ids):
                    self.embeddings = cache_data['embeddings']
                    # Ensure doc_ids order matches cache, or re-align
                    if self.doc_ids == cache_data['doc_ids']:
                         embeddings_loaded = True
                         print("Cached embeddings loaded successfully.")
                    else:
                        print("Cache doc_id mismatch. Recomputing...")
                else:
                    print(f"Cache size mismatch ({len(cache_data['doc_ids'])} vs {len(self.doc_ids)}). Recomputing...")
            except Exception as e:
                print(f"Failed to load cache: {e}")

        if not embeddings_loaded:
            # Encode documents in batches
            print(f"Encoding {len(texts)} documents...")
            self.embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            # Save to cache
            print(f"Saving embeddings to cache: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'doc_ids': self.doc_ids,
                        'embeddings': self.embeddings
                    }, f)
            except Exception as e:
                print(f"Failed to write cache: {e}")

        # Build FAISS index using HNSW (Hierarchical Navigable Small World)
        # HNSW provides O(log N) complexity vs O(N) for FlatIP
        embedding_dim = self.embeddings.shape[1]
        
        # M is the number of neighbors for each node in the HNSW graph (higher = more memory/accuracy)
        # ef_construction controls index build depth/quality
        M = 32
        self.index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 40  # Trade-off between build time and accuracy
        
        # Train not needed for simple HNSWFlat, but good practice to check
        if not self.index.is_trained:
            self.index.train(self.embeddings)
            
        self.index.add(self.embeddings)

        print(f"Dense index built with {len(self.doc_ids)} documents (dim={embedding_dim})")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents using dense similarity.

        Args:
            query: The query text (claim)
            top_k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects, sorted by similarity score
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            doc_id = self.doc_ids[idx]
            doc = self.corpus[doc_id]

            result = RetrievalResult(
                doc_id=doc_id,
                score=float(score),  # Cosine similarity score
                title=doc.title,
                text=doc.full_text,
                rank=rank
            )
            results.append(result)

        return results

    def save_index(self, path: str):
        """
        Save dense index and embeddings to disk.

        Args:
            path: Directory path to save index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / 'faiss_index.bin'))

        # Save metadata
        metadata = {
            'doc_ids': self.doc_ids,
            'model_name': self.model_name,
            'embeddings': self.embeddings
        }

        with open(path / 'dense_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Dense index saved to {path}")

    def load_index(self, path: str):
        """
        Load dense index from disk.

        Args:
            path: Directory path containing the index
        """
        path = Path(path)

        # Load FAISS index
        index_file = path / 'faiss_index.bin'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        # faiss.read_index handles all index types (Flat, HNSW, etc.) automatically
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = path / 'dense_metadata.pkl'
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        self.doc_ids = metadata['doc_ids']
        self.embeddings = metadata['embeddings']

        # Verify model name matches
        if metadata['model_name'] != self.model_name:
            print(f"Warning: Loaded index was built with {metadata['model_name']}, "
                  f"but current model is {self.model_name}")

        print(f"Dense index loaded from {path}")


def main():  # pragma: no cover
    """Example usage of DenseRetriever."""
    from src.data.dataset_loader import SciFactDataset

    # Load dataset
    dataset_example = SciFactDataset("data")
    dataset_example.load_corpus()

    # Create and build dense retriever
    retriever_example = DenseRetriever(dataset_example.corpus)
    retriever_example.build_index()

    # Test retrieval
    query_text = "Aspirin reduces the risk of heart attack"
    results_list = retriever_example.retrieve(query_text, top_k=5)

    print(f"\nTop 5 results for: '{query_text}'")
    for result in results_list:
        print(f"\nRank {result.rank}: {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Doc ID: {result.doc_id}")


if __name__ == "__main__":  # pragma: no cover
    main()
