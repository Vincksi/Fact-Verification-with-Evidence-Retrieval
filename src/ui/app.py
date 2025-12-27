
from src.verification.multi_hop_reasoner import MultiHopReasoner
from src.pipeline import FactVerificationPipeline
import os
import sys

import streamlit as st

# Add project root to path
sys.path.append(os.getcwd())


# Set page config
st.set_page_config(
    page_title="Fact Verification Agent",
    layout="wide"
)

# --- CACHED RESOURCES ---


@st.cache_resource
def load_pipeline():
    """Load the base pipeline components (Config, Corpus, Retriever if available)."""
    with st.spinner("Loading Corpus and Pipeline..."):
        pipeline = FactVerificationPipeline("config/config.yaml")
        pipeline.load_corpus()

        # Try to load existing index
        try:
            pipeline.build_retriever(method='dense')
            pipeline.load_index()
        except BaseException:
            st.warning("Could not load dense index. Building retriever from scratch...")
            pipeline.build_retriever()
            pipeline.retriever.build_index()

        return pipeline


@st.cache_resource
def get_gnn_verifier(_pipeline):
    """Initialize GNN verifier (heavy model)."""
    gnn_config = _pipeline.config['multi_hop']['gnn']
    graph_config = _pipeline.config['multi_hop']['graph']

    return MultiHopReasoner(
        embedding_model=_pipeline.config['retrieval']['dense_model'],
        hidden_dim=gnn_config['hidden_dim'],
        num_layers=gnn_config['num_layers'],
        num_heads=gnn_config['num_heads'],
        dropout=gnn_config['dropout'],
        similarity_threshold=graph_config['sentence_similarity_threshold'],
        max_sentences=graph_config['max_evidence_sentences'],
        use_entities=graph_config['use_entity_extraction']
    )

# --- MAIN UI ---


st.title("Fact Verification Agent")
st.markdown("Enter a scientific claim below to verify it against the SciFact corpus.")

# Load Pipeline
try:
    pipeline = load_pipeline()
    st.success(f"Pipeline loaded with {len(pipeline.dataset.corpus)} documents in corpus.")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Sidebar Config
st.sidebar.header("Configuration")
verification_mode = st.sidebar.radio("Verification Model", ["NLI (Transformer)", "GAT (Graph Neural Network)"])

# Input
claim_text = st.text_area("Claim", placeholder="e.g. Aspirin reduces the risk of heart attack.")

if st.button("Verify Claim", type="primary"):
    if not claim_text:
        st.warning("Please enter a claim.")
    else:
        # Setup Verification Mode
        if verification_mode == "GAT (Graph Neural Network)":
            pipeline.use_gnn = True
            if not getattr(pipeline, "gnn_loaded", False):
                with st.spinner("Initializing GAT Verifier..."):
                    pipeline.verifier = get_gnn_verifier(pipeline)
                    pipeline.gnn_loaded = True
        else:
            pipeline.use_gnn = False
            # NLI is loaded by default in __init__ but let's ensure it's there
            # Actually pipeline.__init__ loads NLI. So if we switch back from GAT, we might need to restore it.
            # Ideally we should keep both or reload. For simplicity in this script,
            # we assume NLI was initial state. If missing/overwritten, we reload NLI.
            # Check if current verifier is MultiHopReasoner
            if isinstance(pipeline.verifier, MultiHopReasoner):
                from src.verification.nli_model import NLIModel
                pipeline.verifier = NLIModel(
                    model_name=pipeline.config['verification']['nli_model'],
                    device='cpu'
                )

        # Process
        with st.spinner("Retrieving evidence and verifying..."):
            result = pipeline.process_claim(0, claim_text)

        # Results Display
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Verdict")
            label_color = {
                "SUPPORTS": "green",
                "REFUTES": "red",
                "NOT_ENOUGH_INFO": "orange"
            }.get(result.predicted_label, "gray")

            st.markdown(f":{label_color}[**{result.predicted_label}**]")
            st.metric("Confidence", f"{result.confidence:.4f}")

        with col2:
            st.subheader("Evidence Used")
            if result.evidence_used:
                for doc_id, text, score in result.evidence_used:
                    with st.expander(f"Doc {doc_id} (Score: {score:.4f})"):
                        st.write(text)
            else:
                st.info("No sufficient evidence found.")

        # Debug info
        with st.expander("Retrieved Documents"):
            for doc in result.retrieved_docs:
                st.markdown(f"**{doc.title}** (Score: {doc.score:.4f})")
                st.caption(doc.text[:200] + "...")
