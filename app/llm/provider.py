from abc import ABC, abstractmethod
from typing import Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for a given prompt string."""
        pass
    
    def log_usage(self, response: Any):
        """Log token usage if available in response metadata."""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            logger.info(f"LLM Usage: prompt={usage.get('prompt_tokens')}, "
                        f"completion={usage.get('completion_tokens')}, "
                        f"total={usage.get('total_tokens')}")
        # Deprecated/other formats
        elif hasattr(response, 'additional_kwargs') and 'token_usage' in response.additional_kwargs:
            usage = response.additional_kwargs['token_usage']
            logger.info(f"LLM Usage: {usage}")

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=3,
            request_timeout=30
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        self.log_usage(response)
        return response.content

class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            max_retries=3,
            timeout=30
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        self.log_usage(response)
        # Type check: Anthropic response content is usually distinct
        if isinstance(response.content, str):
            return response.content
        return str(response.content)

class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from langchain_huggingface import HuggingFaceEndpoint
        from langchain_core.prompts import PromptTemplate
        # Wrapper for HF Endpoint which is generally completion, not chat
        # But we can treat it as generation
        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            timeout=30
        )

    def generate(self, prompt: str) -> str:
        # HF Endpoint returns string directly usually via invoke
        return self.llm.invoke(prompt)

def get_llm(provider: str, model_name: str) -> LLMProvider:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIProvider(model_name)
    elif provider == "anthropic":
        return AnthropicProvider(model_name)
    elif provider == "huggingface":
        return HuggingFaceProvider(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
