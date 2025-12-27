"""
Anthropic LLM model component.
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


class AnthropicModel(BaseLLMModel):
    """
    Anthropic model component supporting Claude models.
    
    This component configures an Anthropic model and outputs it for use by Agent.
    """
    
    _component_type = "llm_anthropic"
    _name = "Anthropic Model"
    _description = "Anthropic Claude language models"
    _category = "models"
    _icon = "brain"
    _color = "#d97757"
    _provider = LLMProvider.ANTHROPIC
    _supported_models = [
        {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet (Latest)", "group": "Claude 3.5"},
        {"value": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku", "group": "Claude 3.5"},
        {"value": "claude-3-opus-20240229", "label": "Claude 3 Opus", "group": "Claude 3"},
        {"value": "claude-3-sonnet-20240229", "label": "Claude 3 Sonnet", "group": "Claude 3"},
        {"value": "claude-3-haiku-20240307", "label": "Claude 3 Haiku", "group": "Claude 3"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get Anthropic-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="anthropic",
                description="Your Anthropic API key",
                required=True,
                placeholder="sk-ant-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="anthropic",
                description="Claude model to use",
                default="claude-3-5-sonnet-20241022",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="Custom API base URL (optional)",
                default="https://api.anthropic.com",
                placeholder="https://api.anthropic.com",
                order=3,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        from langchain_anthropic import ChatAnthropic
        
        self._client = ChatAnthropic(
            api_key=self.get_parameter("api_key"),
            model=self.get_parameter("model", "claude-3-5-sonnet-20241022"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
        )
