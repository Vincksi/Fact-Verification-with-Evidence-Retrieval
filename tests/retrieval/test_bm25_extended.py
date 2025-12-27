
import pytest
from src.retrieval.bm25_retriever import BM25Retriever
from src.data.dataset_loader import Document

class TestBM25Extended:
    
    def test_save_load_index(self, tmp_path):
        # Use tmp_path fixture for real file system interaction (safer/easier than mocking)
        corpus = {1: Document(1, "T", ["S"], False)}
        retriever = BM25Retriever(corpus)
        retriever.build_index() # Actually build it
        
        save_dir = tmp_path / "index"
        retriever.save_index(str(save_dir))
        
        # Load back
        new_retriever = BM25Retriever(corpus)
        new_retriever.load_index(str(save_dir))
        
        assert new_retriever.bm25 is not None
        assert new_retriever.doc_ids == [1]

    def test_preprocess(self):
        text = "Hello World! This is a test."
        retriever = BM25Retriever({})
        # The method to test is _tokenize
        tokens = retriever._tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "!" not in tokens
        
    def test_build_index_empty(self):
        retriever = BM25Retriever({})
        # Should now print warning and return, bm25 remains None
        retriever.build_index()
        assert retriever.bm25 is None
