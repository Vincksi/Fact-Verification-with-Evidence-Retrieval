#!/usr/bin/env python3
"""
Demo script to test Groq-based explanation generation.
"""

import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.explanation.groq_explainer import GroqExplainer


def main():
    print("=" * 60)
    print("Groq Explanation Generator - Demo")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\nError: GROQ_API_KEY environment variable not set")
        print("\nTo use this demo:")
        print("1. Get a free API key from: https://console.groq.com/")
        print("2. Set it: export GROQ_API_KEY='your-key-here'")
        print("3. Run this script again")
        return
    
    # Initialize explainer
    print("\nInitializing Groq explainer...")
    explainer = GroqExplainer(
        model="llama-3.3-70b-versatile",
        max_tokens=512,
        temperature=0.3
    )
    print("✓ Explainer ready")
    
    # Test claims
    test_cases = [
        {
            "claim": "Aspirin reduces the risk of heart attacks.",
            "evidence": [
                (1, "Regular aspirin use has been shown to reduce the risk of heart attacks in clinical trials."),
                (2, "Aspirin inhibits platelet aggregation, which helps prevent blood clots.")
            ],
            "label": "SUPPORTS",
            "confidence": 0.89,
            "probs": {"SUPPORTS": 0.89, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.06}
        },
        {
            "claim": "Vitamin C can cure the common cold.",
            "evidence": [
                (3, "Studies show vitamin C may slightly reduce the duration of cold symptoms."),
                (4, "There is no evidence that vitamin C prevents or cures the common cold.")
            ],
            "label": "REFUTES",
            "confidence": 0.75,
            "probs": {"SUPPORTS": 0.15, "REFUTES": 0.75, "NOT_ENOUGH_INFO": 0.10}
        }
    ]
    
    # Generate explanations
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}")
        print(f"{'=' * 60}")
        print(f"\nClaim: {case['claim']}")
        print(f"Verdict: {case['label']} ({case['confidence']:.0%} confidence)")
        print(f"\nGenerating explanation...")
        
        explanation = explainer.explain_verification(
            claim=case['claim'],
            evidence_list=case['evidence'],
            predicted_label=case['label'],
            confidence=case['confidence'],
            label_probabilities=case['probs']
        )
        
        print(f"\nExplanation:\n{explanation}")
    
    print(f"\n{'=' * 60}")
    print("Demo complete! ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
