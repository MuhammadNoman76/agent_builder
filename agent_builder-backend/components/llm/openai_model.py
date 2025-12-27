"""
OpenAI LLM model component.
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


class OpenAIModel(BaseLLMModel):
    """
    OpenAI model component supporting GPT-4, GPT-3.5, and other OpenAI models.
    
    This component configures an OpenAI model and outputs it for use by Agent.
    """
    
    _component_type = "llm_openai"
    _name = "OpenAI Model"
    _description = "OpenAI language models (GPT-4, GPT-3.5, etc.)"
    _category = "models"
    _icon = "brain"
    _color = "#10a37f"
    _provider = LLMProvider.OPENAI
    _supported_models = [
        {"value": "gpt-4o", "label": "GPT-4o (Latest)", "group": "GPT-4"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "GPT-4"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "GPT-4"},
        {"value": "gpt-4", "label": "GPT-4", "group": "GPT-4"},
        {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "group": "GPT-3.5"},
        {"value": "gpt-3.5-turbo-16k", "label": "GPT-3.5 Turbo 16K", "group": "GPT-3.5"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get OpenAI-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="openai",
                description="Your OpenAI API key",
                required=True,
                placeholder="sk-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="openai",
                description="OpenAI model to use",
                default="gpt-4o",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="Custom API base URL (optional)",
                default="https://api.openai.com/v1",
                placeholder="https://api.openai.com/v1",
                order=3,
                group="connection",
            ),
            StringField.create(
                name="organization",
                label="Organization ID",
                description="OpenAI organization ID (optional)",
                placeholder="org-...",
                order=4,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        from langchain_openai import ChatOpenAI
        
        self._client = ChatOpenAI(
            api_key=self.get_parameter("api_key"),
            model=self.get_parameter("model", "gpt-4o"),
            base_url=self.get_parameter("base_url"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            model_kwargs={
                "top_p": self.get_parameter("top_p", 1.0),
                "frequency_penalty": self.get_parameter("frequency_penalty", 0.0),
                "presence_penalty": self.get_parameter("presence_penalty", 0.0),
            }
        )
