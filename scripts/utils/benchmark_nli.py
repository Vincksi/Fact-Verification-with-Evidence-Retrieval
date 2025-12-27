import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def benchmark_model(model_name, claim, evidence):
    print(f"\nBenchmarking {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    
    # Standard NLI labels usually: 0: contradiction, 1: entailment, 2: neutral
    # But some models might differ. Let's check config.id2label
    id2label = model.config.id2label
    for i, p in enumerate(probs[0]):
        label = id2label.get(i, f"Label {i}")
        print(f"  {label}: {p.item():.4f}")

claim = "0-dimensional biomaterials show inductive properties."
evidence = "0-dimensional biomaterials, such as hydroxyapatite nanoparticles, have been shown to induce osteogenic differentiation of mesenchymal stem cells through their high surface area and reactivity."

models = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-distilroberta-base"
]

for m in models:
    try:
        benchmark_model(m, claim, evidence)
    except Exception as e:
        print(f"Error for {m}: {e}")
