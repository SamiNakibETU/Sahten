"""
LLM Client Abstraction Layer
Provides a unified interface for different LLM providers
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self):
        self.client = None
        self.model = None

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion"""
        pass


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and development"""
    
    class MockClient:
        class Chat:
            class Completions:
                def create(self, *args, **kwargs):
                    # Return a mock response object that mimics OpenAI's structure
                    class Choice:
                        class Message:
                            def __init__(self):
                                self.content = "Mock response content"
                                self.tool_calls = None
                        message = Message()
                    
                    class Response:
                        choices = [Choice()]
                    
                    return Response()
            completions = Completions()
        chat = Chat()

    def __init__(self):
        super().__init__()
        self.client = self.MockClient()
        self.model = "mock-model"

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return mock responses based on message content"""
        last_message = messages[-1]["content"].lower()

        # Mock classification responses
        if "classify" in last_message or "intent" in last_message:
            return json.dumps({
                "intent": "recipe_by_name",
                "language": "fr",
                "slots": {
                    "dishes": [],
                    "ingredients": [],
                    "methods": [],
                    "occasions": []
                }
            })

        # Mock general responses
        if response_format == "json_object":
            return json.dumps({"response": "Mock JSON response"})

        return "Mock text response from LLM"


class OpenAIClient(LLMClient):
    """OpenAI LLM client"""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.llm_model

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion using OpenAI"""
        response_format_param = (
            {"type": "json_object"} if response_format == "json_object" else {"type": "text"}
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format_param,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )

        return completion.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Anthropic (Claude) LLM client"""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        self.client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = settings.llm_model

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion using Anthropic"""
        # Convert messages to Anthropic format
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        # Add JSON instruction if needed
        if response_format == "json_object" and user_messages:
            user_messages[-1]["content"] += "\n\nRespond with valid JSON only."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            temperature=temperature or settings.llm_temperature,
            system=system_message or "",
            messages=user_messages,
        )

        return response.content[0].text


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory function to get the appropriate LLM client"""
    provider = provider or settings.llm_provider

    if provider == "mock":
        return MockLLMClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
