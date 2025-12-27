
from src.verification.nli_model import NLIModel
from src.verification.multi_hop_reasoner import MultiHopReasoner
from src.verification.evidence_aggregator import EvidenceAggregator
from src.verification.ensemble_verifier import EnsembleVerifier
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pipeline import FactVerificationPipeline
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.getcwd())


# --- Data Models ---


class RetrievalConfig(BaseModel):
    method: str  # bm25, dense, hybrid
    top_k: int
    bm25_weight: float
    dense_weight: float


class VerificationConfig(BaseModel):
    model_type: str  # nli, gnn, ensemble
    threshold: float
    gnn_layers: int = 2
    gnn_heads: int = 4
    gnn_hidden_dim: int = 64
    gnn_dropout: float = 0.1
    aggregation_strategy: str = "confidence_weighted"  # majority, confidence_weighted, max_confidence


class VerifyRequest(BaseModel):
    claim: str
    retrieval: RetrievalConfig
    verification: VerificationConfig


# --- Global State ---

class PipelineState:
    instance: Optional[FactVerificationPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load pipeline on startup
    print("Loading pipeline...")
    pipeline = FactVerificationPipeline("config/config.yaml")
    pipeline.load_corpus()

    # Pre-load deafult retriever (dense) if index exists
    try:
        pipeline.build_retriever(method='dense', build=False)
        pipeline.load_index()
        print("Default dense retriever loaded.")
    except Exception as e:
        print(f"Could not load default retriever: {e}")
        # Only build manually if load failed
        # pipeline.retriever.build_index()

    PipelineState.instance = pipeline
    yield
    # Clean up if needed
    print("Shutting down...")

# --- App Definition ---

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/ui/templates")

# --- Helpers ---


def update_retriever(pipeline: FactVerificationPipeline, config: RetrievalConfig):
    """Update retriever if verification method or weights changed."""
    # Logic to switch retriever type
    # Note: reconstructing retriever is expensive if index building is needed.
    # We assume indices are pre-built/loaded.

    current_method = pipeline.config['retrieval']['method']

    # Ideally should check actual instance type, but config helps tracking
    if config.method != current_method:
        print(f"Switching retriever to {config.method}")
        pipeline.build_retriever(method=config.method, build=False)
        try:
            pipeline.load_index()
        except BaseException:
            print("Warning: Index loading failed during switch.")

    # Update runtime parameters
    # Note: Some retrievers might need re-init to change weights (like Hybrid)
    if isinstance(pipeline.retriever, HybridRetriever):
        pipeline.retriever.bm25_weight = config.bm25_weight
        pipeline.retriever.dense_weight = config.dense_weight

    pipeline.config['retrieval']['top_k'] = config.top_k


def update_verifier(pipeline: FactVerificationPipeline, config: VerificationConfig):
    """Switch between NLI and GNN verifier."""

    use_gnn = (config.model_type == 'gnn')

    # Update aggregation strategy (for NLI)
    if not use_gnn:
        if not hasattr(pipeline, 'aggregator') or pipeline.aggregator is None:
            pipeline.aggregator = EvidenceAggregator(strategy=config.aggregation_strategy)
        else:
            pipeline.aggregator.strategy = config.aggregation_strategy

    # so we just re-init if using GNN or switching to it.
    use_onnx = pipeline.config.get('optimization', {}).get('use_onnx', False)
    
    if use_gnn:
        print(f"Initializing GNN with layers={config.gnn_layers}, heads={config.gnn_heads} (ONNX={use_onnx})")
        pipeline.use_gnn = True
        graph_config = pipeline.config['multi_hop']['graph']
        pipeline.verifier = MultiHopReasoner(
            embedding_model=pipeline.config['retrieval']['dense_model'],
            hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_layers,
            num_heads=config.gnn_heads,
            dropout=config.gnn_dropout,
            similarity_threshold=graph_config['sentence_similarity_threshold'],
            max_sentences=graph_config['max_evidence_sentences'],
            use_entities=graph_config['use_entity_extraction'],
            use_onnx=use_onnx
        )
    elif pipeline.use_gnn:
        # Switch back to NLI
        print(f"Switching back to NLI (ONNX={use_onnx})")
        pipeline.use_gnn = False
        pipeline.verifier = NLIModel(
            model_name=pipeline.config['verification']['nli_model'],
            device='cpu',
            use_onnx=use_onnx
        )
        # Ensure aggregator is present
        if not hasattr(pipeline, 'aggregator') or pipeline.aggregator is None:
            pipeline.aggregator = EvidenceAggregator(strategy=config.aggregation_strategy)
    elif config.model_type == 'ensemble':
        print(f"Initializing Ensemble Verifier (ONNX={use_onnx})")
        # We need both models initialized
        # If NLI not there, init it
        if not isinstance(pipeline.verifier, NLIModel) and not pipeline.use_gnn:
            nli = NLIModel(model_name=pipeline.config['verification']['nli_model'], device='cpu', use_onnx=use_onnx)
        elif isinstance(pipeline.verifier, NLIModel):
            nli = pipeline.verifier
        else:
            nli = NLIModel(model_name=pipeline.config['verification']['nli_model'], device='cpu', use_onnx=use_onnx)

        # GNN
        graph_config = pipeline.config['multi_hop']['graph']
        gnn = MultiHopReasoner(
            embedding_model=pipeline.config['retrieval']['dense_model'],
            hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_layers,
            num_heads=config.gnn_heads,
            dropout=config.gnn_dropout,
            similarity_threshold=graph_config['sentence_similarity_threshold'],
            max_sentences=graph_config['max_evidence_sentences'],
            use_entities=graph_config['use_entity_extraction'],
            use_onnx=use_onnx
        )
        pipeline.verifier = EnsembleVerifier(nli, gnn)
        pipeline.use_gnn = False  # Ensemble handles both

# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/defaults")
async def get_defaults():
    """Return default configuration."""
    pipeline = PipelineState.instance
    gnn_defaults = pipeline.config['multi_hop']['gnn']
    return {
        "retrieval": {
            "method": pipeline.config['retrieval']['method'],
            "top_k": pipeline.config['retrieval']['top_k'],
            "bm25_weight": pipeline.config['retrieval'].get('bm25_weight', 0.5),
            "dense_weight": pipeline.config['retrieval'].get('dense_weight', 0.5)
        },
        "verification": {
            "model_type": "gnn" if pipeline.use_gnn else "nli",
            "threshold": pipeline.config['verification']['threshold'],
            "gnn_layers": gnn_defaults.get('num_layers', 2),
            "gnn_heads": gnn_defaults.get('num_heads', 4),
            "gnn_hidden_dim": gnn_defaults.get('hidden_dim', 64),
            "gnn_dropout": gnn_defaults.get('dropout', 0.1),
            "aggregation_strategy": pipeline.config['verification'].get('aggregation', 'confidence_weighted')
        }
    }


@app.post("/verify")
async def verify_claim(data: VerifyRequest):
    pipeline = PipelineState.instance
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # Apply Configuration Updates
    update_retriever(pipeline, data.retrieval)
    update_verifier(pipeline, data.verification)

    # Run Pipeline
    # Using pipeline.process_claim but with patched config if needed
    # process_claim uses self.retrieve_evidence which uses self.retriever.retrieve(..., top_k)
    # The top_k is passed to retrieve(), we need to ensure retrieve matches the requested top_k

    # We can override config dict directly as process_claim reads from it for top_k default
    pipeline.config['retrieval']['top_k'] = data.retrieval.top_k

    result = pipeline.process_claim(0, data.claim)

    return {
        "claim": result.claim,
        "predicted_label": result.predicted_label,
        "confidence": result.confidence,
        "evidence": [
            {"doc_id": r[0], "text": r[1], "score": float(r[2])}
            for r in result.evidence_used
        ],
        "retrieved_docs": [
            {"title": doc.title, "score": doc.score}
            for doc in result.retrieved_docs
        ],
        "graph_data": getattr(result, 'graph_data', None),
        "explanation": getattr(result, 'explanation', None)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8000, reload=True)
