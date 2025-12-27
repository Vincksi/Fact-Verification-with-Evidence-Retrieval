import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.base_retriever import RetrievalResult

class TestDenseRetriever:
    @patch('src.retrieval.dense_retriever.SentenceTransformer')
    def test_initialization(self, mock_st):
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        corpus = {1: Mock(full_text="doc1"), 2: Mock(full_text="doc2")}
        retriever = DenseRetriever(corpus, "model-name")
        
        mock_st.assert_called_with("model-name")
        assert retriever.corpus == corpus

    @patch('src.retrieval.dense_retriever.SentenceTransformer')
    @patch('src.retrieval.dense_retriever.faiss')
    def test_build_index(self, mock_faiss, mock_st):
        mock_model = Mock()
        # Mock encode returns numpy array
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_st.return_value = mock_model
        
        corpus = {1: Mock(full_text="doc1"), 2: Mock(full_text="doc2")}
        retriever = DenseRetriever(corpus)
        retriever.build_index()
        
        mock_faiss.IndexFlatIP.assert_called_with(2)
        assert len(retriever.doc_ids) == 2

    @patch('src.retrieval.dense_retriever.SentenceTransformer')
    @patch('src.retrieval.dense_retriever.faiss')
    def test_retrieve(self, mock_faiss, mock_st):
        # Setup index mock
        mock_index = Mock()
        # Search returns distances, indices
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]), # scores
            np.array([[0, 1]])      # indices
        )
        
        retriever = DenseRetriever({1: Mock(title="T1", full_text="D1"), 2: Mock(title="T2", full_text="D2")})
        retriever.index = mock_index
        retriever.doc_ids = [1, 2] # map index 0->doc 1, 1->doc 2
        retriever.model = Mock()
        retriever.model.encode.return_value = np.array([[0.1]])
        
        results = retriever.retrieve("query", top_k=2)
        
        assert len(results) == 2
        assert results[0].doc_id == 1
        assert results[0].score == 0.9
        assert results[1].doc_id == 2


    @patch('src.retrieval.dense_retriever.SentenceTransformer')
    @patch('src.retrieval.dense_retriever.faiss')
    def test_save_load_index(self, mock_faiss, mock_st, tmp_path):
        retriever = DenseRetriever({1: Mock(full_text="D1")})
        retriever.index = Mock()
        retriever.doc_ids = [1]
        retriever.embeddings = np.array([[0.1]])
        
        path = tmp_path / "index_dir"
        path.mkdir()
        retriever.save_index(str(path))
        
        # Ensure files were "created" (mocked write_index won't create them, so we create them)
        (path / 'faiss_index.bin').touch()
        (path / 'dense_metadata.pkl').touch()
        
        # Mock faiss.read_index for load
        mock_faiss.read_index.return_value = Mock()
        
        new_retriever = DenseRetriever({1: Mock(full_text="D1")})
        with patch('src.retrieval.dense_retriever.pickle.load') as mock_pickle_load:
            mock_pickle_load.return_value = {
                'doc_ids': [1],
                'model_name': new_retriever.model_name,
                'embeddings': np.array([[0.1]])
            }
            new_retriever.load_index(str(path))
        
        assert new_retriever.doc_ids == [1]

class TestHybridRetriever:
    def test_initialization(self):
        corpus = {}
        retriever = HybridRetriever(corpus)
        assert retriever.bm25_weight == 0.5
        assert retriever.dense_weight == 0.5

    @patch('src.retrieval.hybrid_retriever.BM25Retriever')
    @patch('src.retrieval.hybrid_retriever.DenseRetriever')
    def test_build_index(self, mock_dense_cls, mock_bm25_cls):
        mock_bm25 = Mock()
        mock_dense = Mock()
        mock_bm25_cls.return_value = mock_bm25
        mock_dense_cls.return_value = mock_dense
        
        retriever = HybridRetriever({})
        retriever.build_index()
        
        mock_bm25.build_index.assert_called_once()
        mock_dense.build_index.assert_called_once()

    @patch('src.retrieval.hybrid_retriever.BM25Retriever')
    @patch('src.retrieval.hybrid_retriever.DenseRetriever')
    def test_retrieve(self, mock_dense_cls, mock_bm25_cls):
        # Setup mocks
        mock_bm25 = Mock()
        mock_dense = Mock()
        mock_bm25_cls.return_value = mock_bm25
        mock_dense_cls.return_value = mock_dense
        
        # BM25 returns doc 1 (rank 1) and doc 2 (rank 2)
        mock_bm25.retrieve.return_value = [
            RetrievalResult(1, 10.0, "D1", 1),
            RetrievalResult(2, 5.0, "D2", 2)
        ]
        
        # Dense returns doc 2 (rank 1) and doc 3 (rank 2)
        mock_dense.retrieve.return_value = [
            RetrievalResult(2, 0.9, "D2", 1),
            RetrievalResult(3, 0.8, "D3", 2)
        ]
        
        retriever = HybridRetriever({
            1: Mock(title="T1", full_text="D1"), 
            2: Mock(title="T2", full_text="D2"),
            3: Mock(title="T3", full_text="D3")
        })
        retriever.bm25_retriever = mock_bm25
        retriever.dense_retriever = mock_dense
        retriever.bm25_weight = 1.0 
        retriever.dense_weight = 1.0
        
        # Helper to bypass build_index
        
        results = retriever.retrieve("query", top_k=3)
        
        # RRF Formula: score = weight * (1 / (constants + rank))
        # Doc 1: BM25 rank 1 -> 1.0 * (1/61) = 0.01639
        # Doc 2: BM25 rank 2, Dense rank 1 -> 1.0 * (1/62) + 1.0 * (1/61) = 0.0161 + 0.01639 = 0.0325
        # Doc 3: Dense rank 2 -> 1.0 * (1/62) = 0.0161
        
        # Order should be Doc 2, Doc 1, Doc 3
        assert results[0].doc_id == 2
        assert results[1].doc_id == 1
        assert results[2].doc_id == 3

    @patch('src.retrieval.hybrid_retriever.BM25Retriever')
    @patch('src.retrieval.hybrid_retriever.DenseRetriever')
    def test_weighted_fusion(self, mock_dense_cls, mock_bm25_cls):
        mock_bm25 = Mock()
        mock_dense = Mock()
        mock_bm25_cls.return_value = mock_bm25
        mock_dense_cls.return_value = mock_dense
        
        # Test weighted fusion logic
        retriever = HybridRetriever({1: Mock(title="T1", full_text="D1"), 2: Mock(title="T2", full_text="D2")})
        retriever.fusion_method = "weighted"
        retriever.bm25_weight = 0.5
        retriever.dense_weight = 0.5
        
        bm25_res = [RetrievalResult(1, 10.0, "T1", "D1", 1)]
        dense_res = [RetrievalResult(2, 0.9, "T2", "D2", 1)]
        
        with patch.object(retriever.bm25_retriever, 'retrieve', return_value=bm25_res):
            with patch.object(retriever.dense_retriever, 'retrieve', return_value=dense_res):
                results = retriever.retrieve("query", top_k=2)
                
        assert len(results) == 2
        