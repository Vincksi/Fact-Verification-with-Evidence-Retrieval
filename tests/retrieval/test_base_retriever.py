import pytest
from unittest.mock import Mock
from src.retrieval.base_retriever import BaseRetriever, RetrievalResult

class TestRetrievalResult:
    def test_creation(self):
        res = RetrievalResult(
            doc_id=1,
            score=0.95,
            title="Title",
            text="Text",
            rank=1
        )
        assert res.doc_id == 1
        assert res.score == 0.95
        assert res.title == "Title"
        assert res.text == "Text"
        assert res.rank == 1

class DummyRetriever(BaseRetriever):
    def build_index(self):
        pass
    def retrieve(self, query, top_k=10):
        return []

class TestBaseRetriever:
    def test_initialization(self):
        corpus = {1: Mock(), 2: Mock()}
        retriever = DummyRetriever(corpus)
        assert retriever.corpus == corpus

    def test_save_load_stubs(self):
        # Base class might have stubs or basic impl
        retriever = DummyRetriever({})
        with pytest.raises(NotImplementedError):
            retriever.save_index("dummy")
        with pytest.raises(NotImplementedError):
            retriever.load_index("dummy")
