# Web Interface Guide

This project includes a modern, interactive web interface for exploring the fact verification pipeline. It allows you to verify claims and visualize evidence graphs.

## Features

### Claim Verification
- **Input**: Enter any scientific claim in the text area.
- **Model Selection**: Choose between different verification strategies:
  - **NLI (Standard)**: Fast, single-hop verification using DeBERTa.
  - **GAT (Multi-hop)**: Uses Graph Attention Networks to reason over multiple evidence sentences and entities.
  - **Ensemble**: Combines both models for maximum robustness.
- **Retrieval Method**: Toggle between BM25, Dense (Sentence-BERT), or Hybrid (RRF) retrieval.

### Interactive Evidence Graph
When using GAT or Ensemble modes, the UI renders a heterogeneous graph:
- **Purple Node**: The claim.
- **Dark Nodes**: Retrieved evidence sentences.
- **Gray Nodes**: Extracted entities connecting the evidence.
- **Edges**: The thickness of the lines represents the **Attention Weight** (importance) the GAT model assigned to that specific connection.
- **Hover**: Hover over nodes to see the full text/title of the document.

### AI-Powered Explanations
When enabled, the system generates natural language explanations of verification reasoning:
- **Provider**: Groq API with Llama 3.3 70B or Mixtral models
- **Content**: Clear, concise explanations (under 150 words) of:
  - How evidence supports or contradicts the claim
  - Key scientific facts or relationships identified
  - Limitations or uncertainties in the reasoning
- **Configuration**: Enable in `config.yaml` by setting `explanation.enabled: true` and providing `GROQ_API_KEY` environment variable

## How to Run

1. **Start the Server**:
   ```bash
   bash scripts/run/start_web_ui.sh
   ```
   *Note: This script automatically handles process management and environment activation.*

2. **Access the UI**:
   Open your browser and navigate to `http://localhost:8000`.

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism design), and JavaScript.
- **Visualization**: 
  - **Graphs**: Custom Canvas-based force-directed layout.
- **Icons**: Lucide Icons for a modern look.

## Performance Notes

- The Web UI is optimized for CPU usage.
- Initializing the NLI model may take a few seconds on the first request.
