
import pytest
from unittest.mock import Mock, patch
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.data.dataset_loader import Document

class TestPersistence:
    
    @patch('src.retrieval.dense_retriever.SentenceTransformer')
    def test_dense_save_load(self, mock_st_cls, tmp_path):
        import numpy as np
        # Mock ST
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1]]) # Numpy array
        mock_model.get_sentence_embedding_dimension.return_value = 1
        mock_st_cls.return_value = mock_model
        
        corpus = {1: Document(1, "Title", ["Text"], False)}
        retriever = DenseRetriever(corpus)
        retriever.build_index()
        
        save_dir = tmp_path / "dense_index"
        retriever.save_index(str(save_dir))
        
        # Load
        # Mock faiss to avoid file issues
        with patch('src.retrieval.dense_retriever.faiss') as mock_faiss:
            mock_faiss.read_index.return_value = Mock()
            
            new_retriever = DenseRetriever(corpus)
            new_retriever.load_index(str(save_dir))
            
            assert new_retriever.index is not None
            
    def test_hybrid_save_load(self, tmp_path):
        corpus = {}
        retriever = HybridRetriever(corpus)
        retriever.bm25_retriever = Mock()
        retriever.dense_retriever = Mock()
        
        save_dir = tmp_path / "hybrid_index"
        retriever.save_index(str(save_dir))
        
        retriever.bm25_retriever.save_index.assert_called_with(str(save_dir))
        retriever.dense_retriever.save_index.assert_called_with(str(save_dir))
        
        # Load
        # We need to mock the subclass instances in new_retriever or mock the classes
        
        with patch('src.retrieval.hybrid_retriever.BM25Retriever') as mock_bm25_cls, \
             patch('src.retrieval.hybrid_retriever.DenseRetriever') as mock_dense_cls:
            
            mock_bm25_instance = Mock()
            mock_dense_instance = Mock()
            mock_bm25_cls.return_value = mock_bm25_instance
            mock_dense_cls.return_value = mock_dense_instance
            
            new_retriever = HybridRetriever(corpus)
            new_retriever.load_index(str(save_dir))
            
            mock_bm25_instance.load_index.assert_called()
            mock_dense_instance.load_index.assert_called()
