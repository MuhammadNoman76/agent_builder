"""
OpenRouter LLM model component - access multiple models through one API.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List, Optional
from .base import BaseLLMModel, LLMProvider
from components.field_types import (
    FieldDefinition,
    ApiKeyField,
    ModelSelectField,
    StringField,
)


class OpenRouterModel(BaseLLMModel):
    """
    OpenRouter model component - provides access to multiple LLM providers
    through a single API endpoint.
    
    This component configures a model via OpenRouter and outputs it for use by Agent.
    """
    
    _component_type = "llm_openrouter"
    _name = "OpenRouter Model"
    _description = "Access multiple LLM providers through OpenRouter"
    _category = "models"
    _icon = "globe"
    _color = "#6366f1"
    _provider = LLMProvider.OPENROUTER
    _supported_models = [
        # OpenAI Models
        {"value": "openai/gpt-4o", "label": "OpenAI GPT-4o", "group": "OpenAI"},
        {"value": "openai/gpt-4-turbo", "label": "OpenAI GPT-4 Turbo", "group": "OpenAI"},
        {"value": "openai/gpt-3.5-turbo", "label": "OpenAI GPT-3.5 Turbo", "group": "OpenAI"},
        # Anthropic Models
        {"value": "anthropic/claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "group": "Anthropic"},
        {"value": "anthropic/claude-3-opus", "label": "Claude 3 Opus", "group": "Anthropic"},
        {"value": "anthropic/claude-3-haiku", "label": "Claude 3 Haiku", "group": "Anthropic"},
        # Google Models
        {"value": "google/gemini-pro", "label": "Google Gemini Pro", "group": "Google"},
        {"value": "google/gemini-pro-1.5", "label": "Google Gemini 1.5 Pro", "group": "Google"},
        # Meta Models
        {"value": "meta-llama/llama-3-70b-instruct", "label": "Llama 3 70B", "group": "Meta"},
        {"value": "meta-llama/llama-3-8b-instruct", "label": "Llama 3 8B", "group": "Meta"},
        # Mistral Models
        {"value": "mistralai/mistral-large", "label": "Mistral Large", "group": "Mistral"},
        {"value": "mistralai/mixtral-8x7b-instruct", "label": "Mixtral 8x7B", "group": "Mistral"},
        # Free Models
        {"value": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo:free", "label": "Nous Hermes 2 (Free)", "group": "Free"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get OpenRouter-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="openrouter",
                description="Your OpenRouter API key",
                required=True,
                placeholder="sk-or-v1-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="openrouter",
                description="Model to use via OpenRouter",
                default="openai/gpt-4o",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="custom_model",
                label="Custom Model ID",
                description="Enter a custom model ID if not in the list",
                placeholder="provider/model-name",
                order=3,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="OpenRouter API base URL",
                default="https://openrouter.ai/api/v1",
                order=4,
                group="connection",
            ),
            StringField.create(
                name="site_url",
                label="Site URL",
                description="Your site URL for OpenRouter ranking",
                placeholder="https://yoursite.com",
                order=5,
                group="connection",
            ),
            StringField.create(
                name="site_name",
                label="Site Name",
                description="Your site name for OpenRouter ranking",
                placeholder="Your App Name",
                order=6,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the OpenRouter client using LangChain OpenAI."""
        from langchain_openai import ChatOpenAI
        
        # Use custom model if provided, otherwise use selected model
        model = self.get_parameter("custom_model") or self.get_parameter("model", "openai/gpt-4o")
        
        # Build extra headers for OpenRouter
        extra_headers = {}
        if site_url := self.get_parameter("site_url"):
            extra_headers["HTTP-Referer"] = site_url
        if site_name := self.get_parameter("site_name"):
            extra_headers["X-Title"] = site_name
        
        self._client = ChatOpenAI(
            api_key=self.get_parameter("api_key"),
            base_url=self.get_parameter("base_url", "https://openrouter.ai/api/v1"),
            model=model,
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            default_headers=extra_headers if extra_headers else None,
            model_kwargs={
                "top_p": self.get_parameter("top_p", 1.0),
                "frequency_penalty": self.get_parameter("frequency_penalty", 0.0),
                "presence_penalty": self.get_parameter("presence_penalty", 0.0),
            }
        )
