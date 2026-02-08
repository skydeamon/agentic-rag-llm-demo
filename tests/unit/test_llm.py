
import pytest
import sys
from unittest.mock import MagicMock, patch

# We need to mock the provider-specific libraries before importing/using them
# because they might not be installed in the test environment or we don't want side effects.

@pytest.fixture
def mock_providers():
    with patch.dict(sys.modules, {
        "langchain_openai": MagicMock(),
        "langchain_anthropic": MagicMock(),
        "langchain_huggingface": MagicMock(),
    }):
        # We also need to mock the specific classes they export
        sys.modules["langchain_openai"].ChatOpenAI = MagicMock()
        sys.modules["langchain_anthropic"].ChatAnthropic = MagicMock()
        sys.modules["langchain_huggingface"].HuggingFaceEndpoint = MagicMock()
        
        yield

def test_get_llm_openai(mock_providers):
    # Import inside test to ensure mocks are active
    from app.llm.provider import get_llm, OpenAIProvider
    
    # We must patch os.environ or pass api_key to avoid validation errors if the real classes checked
    # But since we mocked the classes, they are MagicMocks, so they accept anything.
    
    llm = get_llm("openai", "gpt-4-turbo")
    assert isinstance(llm, OpenAIProvider)
    # Verify the underlying LangChain model was initialized (mocked)
    from langchain_openai import ChatOpenAI
    ChatOpenAI.assert_called()

def test_get_llm_anthropic(mock_providers):
    from app.llm.provider import get_llm, AnthropicProvider
    
    llm = get_llm("anthropic", "claude-3-opus")
    assert isinstance(llm, AnthropicProvider)
    from langchain_anthropic import ChatAnthropic
    ChatAnthropic.assert_called()

def test_get_llm_huggingface(mock_providers):
    from app.llm.provider import get_llm, HuggingFaceProvider
    
    # HuggingFace provider init likely needs an API token even if mocked?
    # No, MagicMock eats arguments.
    llm = get_llm("huggingface", "meta-llama/Llama-3-70b")
    assert isinstance(llm, HuggingFaceProvider)
    from langchain_huggingface import HuggingFaceEndpoint
    HuggingFaceEndpoint.assert_called()

def test_get_llm_invalid():
    from app.llm.provider import get_llm
    with pytest.raises(ValueError):
        get_llm("unknown_provider", "model")
