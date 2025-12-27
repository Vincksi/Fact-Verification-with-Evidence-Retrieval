"""
Data loading and preprocessing utilities for SciFact dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines


@dataclass
class Claim:
    """Represents a claim from the SciFact dataset."""
    id: int
    claim: str
    evidence: Dict[str, List[Dict]]  # doc_id -> list of evidence
    cited_doc_ids: List[int]

    @property
    def label(self) -> str:
        """Get the claim label (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)."""
        if not self.evidence:
            return "NOT_ENOUGH_INFO"

        # Mapping from SciFact raw labels to unified labels
        label_map = {
            "SUPPORT": "SUPPORTS",
            "CONTRADICT": "REFUTES",
            "SUPPORTS": "SUPPORTS",
            "REFUTES": "REFUTES"
        }

        # Get the first evidence label
        for _, evidence_list in self.evidence.items():
            if evidence_list:
                raw_label = evidence_list[0].get("label", "NOT_ENOUGH_INFO")
                return label_map.get(raw_label, "NOT_ENOUGH_INFO")

        return "NOT_ENOUGH_INFO"


@dataclass
class Document:
    """Represents a document from the corpus."""
    doc_id: int
    title: str
    abstract: List[str]  # List of sentences
    structured: bool

    @property
    def full_text(self) -> str:
        """Get full text of the document."""
        return f"{self.title} {' '.join(self.abstract)}"


class SciFactDataset:
    """Loader and preprocessor for SciFact dataset."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Path to the directory containing SciFact files
        """
        self.data_dir = Path(data_dir)
        self.corpus: Dict[int, Document] = {}
        self.claims_train: List[Claim] = []
        self.claims_dev: List[Claim] = []
        self.claims_test: List[Claim] = []

    def load_corpus(self, corpus_path: Optional[str] = None) -> Dict[int, Document]:
        """
        Load the corpus of scientific abstracts.

        Args:
            corpus_path: Path to corpus.jsonl file

        Returns:
            Dictionary mapping doc_id to Document
        """
        if corpus_path is None:
            corpus_path = self.data_dir / "corpus.jsonl"
        else:
            corpus_path = Path(corpus_path)

        print(f"Loading corpus from {corpus_path}...")
        self.corpus = {}

        with jsonlines.open(corpus_path) as reader:
            for obj in reader:
                doc = Document(
                    doc_id=obj["doc_id"],
                    title=obj["title"],
                    abstract=obj["abstract"],
                    structured=obj.get("structured", False)
                )
                self.corpus[doc.doc_id] = doc

        print(f"Loaded {len(self.corpus)} documents")
        return self.corpus

    def load_claims(self, split: str = "train") -> List[Claim]:
        """
        Load claims from specified split.

        Args:
            split: One of 'train', 'dev', 'test'

        Returns:
            List of Claim objects
        """
        claims_path = self.data_dir / f"claims_{split}.jsonl"
        print(f"Loading {split} claims from {claims_path}...")

        claims = []
        with jsonlines.open(claims_path) as reader:
            for obj in reader:
                claim = Claim(
                    id=obj["id"],
                    claim=obj["claim"],
                    evidence=obj.get("evidence", {}),
                    cited_doc_ids=obj.get("cited_doc_ids", [])
                )
                claims.append(claim)

        print(f"Loaded {len(claims)} {split} claims")

        # Store in the appropriate attribute
        if split == "train":
            self.claims_train = claims
        elif split == "dev":
            self.claims_dev = claims
        elif split == "test":
            self.claims_test = claims

        return claims

    def load_all(self):
        """Load corpus and all claim splits."""
        self.load_corpus()
        self.load_claims("train")
        self.load_claims("dev")
        self.load_claims("test")

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a document by ID."""
        return self.corpus.get(doc_id)

    def get_claim(self, claim_id: int, split: str = "train") -> Optional[Claim]:
        """Get a claim by ID from specified split."""
        claims = {
            "train": self.claims_train,
            "dev": self.claims_dev,
            "test": self.claims_test
        }.get(split, [])

        for claim in claims:
            if claim.id == claim_id:
                return claim
        return None

    def get_evidence_sentences(self, claim: Claim) -> Dict[int, List[str]]:
        """
        Get the evidence sentences for a claim.

        Args:
            claim: Claim object

        Returns:
            Dictionary mapping doc_id to list of evidence sentences
        """
        evidence_sentences = {}

        for doc_id_str, evidence_list in claim.evidence.items():
            doc_id = int(doc_id_str)
            doc = self.get_document(doc_id)

            if doc is None:
                continue

            sentences = []
            for evidence in evidence_list:
                sentence_ids = evidence.get("sentences", [])
                for sent_id in sentence_ids:
                    if sent_id < len(doc.abstract):
                        sentences.append(doc.abstract[sent_id])

            if sentences:
                evidence_sentences[doc_id] = sentences

        return evidence_sentences


if __name__ == "__main__":  # pragma: no cover # pragma: no cover
    # Example usage
    dataset = SciFactDataset("data")
    dataset.load_all()

    # Print some statistics
    print("\n=== Dataset Statistics ===")
    print(f"Corpus size: {len(dataset.corpus)}")
    print(f"Train claims: {len(dataset.claims_train)}")
    print(f"Dev claims: {len(dataset.claims_dev)}")
    print(f"Test claims: {len(dataset.claims_test)}")

    # Show example claim
    if dataset.claims_train:
        example_claim = dataset.claims_train[0]
        print("\n=== Example Claim ===")
        print(f"ID: {example_claim.id}")
        print(f"Claim: {example_claim.claim}")
        print(f"Label: {example_claim.label}")
        print(f"Cited docs: {example_claim.cited_doc_ids}")
