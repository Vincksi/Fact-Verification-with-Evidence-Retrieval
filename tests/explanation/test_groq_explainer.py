"""
Tests for Groq-based explanation generator.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.explanation.groq_explainer import GroqExplainer


class TestGroqExplainer:
    """Test suite for GroqExplainer."""

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_initialization_with_api_key(self, mock_groq_class):
        """Test initialization with explicit API key."""
        mock_client = Mock()
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(api_key="test-key-123")
        
        assert explainer.model == "llama-3.3-70b-versatile"
        assert explainer.max_tokens == 512
        assert explainer.temperature == 0.3
        mock_groq_class.assert_called_once_with(api_key="test-key-123")

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    @patch.dict(os.environ, {'GROQ_API_KEY': 'env-key-456'})
    def test_initialization_with_env_key(self, mock_groq_class):
        """Test initialization with API key from environment."""
        mock_client = Mock()
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer()
        
        mock_groq_class.assert_called_once_with(api_key="env-key-456")

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_no_api_key_raises(self, mock_groq_class):
        """Test that initialization fails without API key."""
        with pytest.raises(ValueError, match="Groq API key required"):
            GroqExplainer()

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', False)
    def test_initialization_groq_not_available(self):
        """Test that initialization fails if groq package not installed."""
        with pytest.raises(ImportError, match="groq package not installed"):
            GroqExplainer(api_key="test-key")

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_explain_verification_success(self, mock_groq_class):
        """Test successful explanation generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This claim is supported by evidence."
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(api_key="test-key")
        
        # Test explanation generation
        explanation = explainer.explain_verification(
            claim="Aspirin reduces heart attack risk.",
            evidence_list=[(1, "Clinical trials show aspirin reduces risk.")],
            predicted_label="SUPPORTS",
            confidence=0.89,
            label_probabilities={"SUPPORTS": 0.89, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.06}
        )
        
        assert explanation == "This claim is supported by evidence."
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify API call parameters
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == "llama-3.3-70b-versatile"
        assert call_kwargs['max_tokens'] == 512
        assert call_kwargs['temperature'] == 0.3
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['messages'][0]['role'] == 'system'
        assert call_kwargs['messages'][1]['role'] == 'user'

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_explain_verification_no_evidence(self, mock_groq_class):
        """Test explanation with no evidence."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "No evidence available."
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(api_key="test-key")
        
        explanation = explainer.explain_verification(
            claim="Test claim",
            evidence_list=[],
            predicted_label="NOT_ENOUGH_INFO",
            confidence=1.0,
            label_probabilities={"NOT_ENOUGH_INFO": 1.0}
        )
        
        assert explanation == "No evidence available."

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_explain_verification_api_error(self, mock_groq_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API connection error")
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(api_key="test-key")
        
        explanation = explainer.explain_verification(
            claim="Test claim",
            evidence_list=[(1, "Test evidence")],
            predicted_label="SUPPORTS",
            confidence=0.8,
            label_probabilities={"SUPPORTS": 0.8}
        )
        
        assert "Error generating explanation" in explanation
        assert "API connection error" in explanation

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_custom_model_parameters(self, mock_groq_class):
        """Test initialization with custom model parameters."""
        mock_client = Mock()
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(
            model="mixtral-8x7b-32768",
            api_key="test-key",
            max_tokens=1024,
            temperature=0.7
        )
        
        assert explainer.model == "mixtral-8x7b-32768"
        assert explainer.max_tokens == 1024
        assert explainer.temperature == 0.7

    @patch('src.explanation.groq_explainer.GROQ_AVAILABLE', True)
    @patch('src.explanation.groq_explainer.Groq')
    def test_build_prompt_format(self, mock_groq_class):
        """Test that prompt is properly formatted."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test explanation"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        explainer = GroqExplainer(api_key="test-key")
        
        explainer.explain_verification(
            claim="Test claim",
            evidence_list=[
                (1, "First evidence"),
                (2, "Second evidence")
            ],
            predicted_label="SUPPORTS",
            confidence=0.75,
            label_probabilities={"SUPPORTS": 0.75, "REFUTES": 0.15, "NOT_ENOUGH_INFO": 0.10}
        )
        
        # Check that prompt contains key elements
        call_args = mock_client.chat.completions.create.call_args[1]
        prompt = call_args['messages'][1]['content']
        
        assert "Test claim" in prompt
        assert "First evidence" in prompt
        assert "Second evidence" in prompt
        assert "SUPPORTS" in prompt
        assert "75.00%" in prompt  # Formatted as percentage with 2 decimals
