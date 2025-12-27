#!/usr/bin/env python3
"""
Script to export NLI and Embedding models to ONNX format for optimized CPU inference.
Uses Hugging Face Optimum.

Usage:
    python scripts/optimization/export_onnx.py --model_id cross-encoder/nli-deberta-v3-small --task text-classification
    python scripts/optimization/export_onnx.py --model_id sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction
"""

import argparse
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def export_model(model_id: str, task: str, output_dir: str):
    print(f"Exporting {model_id} to ONNX (Task: {task})...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if task == "text-classification":
        model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    elif task == "feature-extraction":
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    else:
        raise ValueError(f"Unsupported task: {task}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save to directory
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--task", type=str, required=True, choices=["text-classification", "feature-extraction"], help="Model task")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save ONNX model")
    
    args = parser.parse_args()
    
    export_model(args.model_id, args.task, args.output_dir)

if __name__ == "__main__":
    main()
