"""
Groq-based explanation generator for claim verification.

Uses Groq's inference API for fast LLM-generated explanations.
"""

import os
from typing import List, Tuple, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


class GroqExplainer:
    """Generate natural language explanations using Groq API."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3
    ):
        """
        Initialize Groq explainer.

        Args:
            model: Groq model name
            api_key: Groq API key (or set GROQ_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more focused)
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Groq(api_key=self.api_key)

    def explain_verification(
        self,
        claim: str,
        evidence_list: List[Tuple[int, str]],
        predicted_label: str,
        confidence: float,
        label_probabilities: dict
    ) -> str:
        """
        Generate explanation for a verification result.

        Args:
            claim: The claim being verified
            evidence_list: List of (doc_id, evidence_text) tuples
            predicted_label: Predicted label (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)
            confidence: Confidence score
            label_probabilities: Probability distribution over labels

        Returns:
            Natural language explanation
        """
        # Build prompt
        prompt = self._build_prompt(
            claim, evidence_list, predicted_label, confidence, label_probabilities
        )

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert scientific fact-checker. "
                        "Explain claim verification reasoning clearly and concisely."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            explanation = response.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def _build_prompt(
        self,
        claim: str,
        evidence_list: List[Tuple[int, str]],
        predicted_label: str,
        confidence: float,
        label_probabilities: dict
    ) -> str:
        """Build prompt for explanation generation."""
        # Format evidence
        evidence_text = "\n".join(
            f"{i+1}. {text}" for i, (_, text) in enumerate(evidence_list)
        )

        # Format probabilities
        prob_text = ", ".join(
            f"{label}: {prob:.2%}" for label, prob in label_probabilities.items()
        )

        prompt = f"""Analyze the following scientific claim verification:

**Claim:** {claim}

**Retrieved Evidence:**
{evidence_text if evidence_text else "No evidence retrieved."}

**Verification Result:** {predicted_label} (Confidence: {confidence:.2%})
**Label Probabilities:** {prob_text}

Provide a clear, concise explanation of why this verdict was reached. Focus on:
1. How the evidence supports or contradicts the claim
2. Key scientific facts or relationships identified
3. Any limitations or uncertainties in the reasoning

Keep the explanation under 150 words."""

        return prompt


def main():  # pragma: no cover
    """Example usage of GroqExplainer."""
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Set GROQ_API_KEY environment variable")
        return

    explainer = GroqExplainer()

    claim = "Aspirin reduces heart attack risk."
    evidence = [
        (1, "Regular aspirin use has been shown to reduce the risk of heart attacks in clinical trials."),
        (2, "Aspirin inhibits platelet aggregation, which helps prevent blood clots.")
    ]

    explanation = explainer.explain_verification(
        claim=claim,
        evidence_list=evidence,
        predicted_label="SUPPORTS",
        confidence=0.89,
        label_probabilities={"SUPPORTS": 0.89, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.06}
    )

    print("Explanation:")
    print(explanation)


if __name__ == "__main__":  # pragma: no cover
    main()
