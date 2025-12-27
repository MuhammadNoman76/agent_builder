"""
Base LLM model class - foundation for all LLM providers.
Uses centralized field types for configuration.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from pydantic import BaseModel, Field
from enum import Enum

from components.base import BaseComponent, ComponentPort, PortType, FieldDefinition, FieldGroup
from components.field_types import (
    NumberField,
    IntegerField,
    BooleanField,
    ApiKeyField,
    ModelSelectField,
    StringField,
    SliderField,
)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class LLMConfig(BaseModel):
    """Configuration for LLM model."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: List[str] = Field(default_factory=list)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM model."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMModel(BaseComponent, ABC):
    """
    Abstract base class for all LLM model components.
    
    LLM models are configuration components that output a model instance
    for use by Agent components. They have a single "model" output port.
    """
    
    _component_type = "llm_model"
    _name = "LLM Model"
    _description = "Base LLM model component"
    _category = "models"
    _icon = "brain"
    _color = "#8b5cf6"
    _provider: LLMProvider = LLMProvider.CUSTOM
    _supported_models: List[Dict[str, str]] = []
    
    def __init__(
        self, 
        node_id: Optional[str] = None, 
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None
    ):
        """Initialize the LLM model component."""
        super().__init__(node_id, parameters)
        self.config = config
        self._client = None
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """
        LLM model components have NO input ports.
        They are configuration nodes that output a model instance.
        """
        return []
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """
        LLM models have a single output port: the model instance.
        This connects to an Agent's 'model' input port.
        """
        return [
            ComponentPort(
                id="model",
                name="Model",
                type=PortType.OUTPUT,
                data_type="llm_model",
                required=True,
                description="LLM model instance for Agent to use"
            )
        ]
    
    @classmethod
    def _get_base_fields(cls) -> List[FieldDefinition]:
        """Get base fields common to all LLM models."""
        return [
            SliderField.create(
                name="temperature",
                label="Temperature",
                description="Controls randomness in output (0 = deterministic, 2 = creative)",
                min_value=0.0,
                max_value=2.0,
                default=0.7,
                step=0.1,
                order=10,
                group="generation",
            ),
            IntegerField.create(
                name="max_tokens",
                label="Max Tokens",
                description="Maximum tokens in response",
                default=4096,
                min_value=1,
                max_value=128000,
                order=11,
                group="generation",
            ),
            SliderField.create(
                name="top_p",
                label="Top P",
                description="Nucleus sampling parameter (alternative to temperature)",
                min_value=0.0,
                max_value=1.0,
                default=1.0,
                step=0.05,
                order=12,
                group="advanced",
            ),
            SliderField.create(
                name="frequency_penalty",
                label="Frequency Penalty",
                description="Penalize frequent tokens (-2 to 2)",
                min_value=-2.0,
                max_value=2.0,
                default=0.0,
                step=0.1,
                order=13,
                group="advanced",
            ),
            SliderField.create(
                name="presence_penalty",
                label="Presence Penalty",
                description="Penalize tokens already present (-2 to 2)",
                min_value=-2.0,
                max_value=2.0,
                default=0.0,
                step=0.1,
                order=14,
                group="advanced",
            ),
        ]
    
    @classmethod
    @abstractmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get provider-specific fields (to be implemented by subclasses)."""
        pass
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Combine base and provider-specific fields."""
        return cls._get_provider_fields() + cls._get_base_fields()
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organization."""
        return [
            FieldGroup(
                id="connection",
                label="Connection",
                description="API connection settings",
                order=0,
            ),
            FieldGroup(
                id="model",
                label="Model",
                description="Model selection",
                order=1,
            ),
            FieldGroup(
                id="generation",
                label="Generation Settings",
                description="Control output generation",
                collapsible=True,
                order=2,
            ),
            FieldGroup(
                id="advanced",
                label="Advanced",
                description="Advanced model parameters",
                collapsible=True,
                collapsed_by_default=True,
                order=3,
            ),
        ]
    
    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the LLM client (provider-specific)."""
        pass
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM model component.
        
        This initializes and returns the model instance for use by Agent.
        The model doesn't process messages directly - that's the Agent's job.
        """
        if self._client is None:
            await self._initialize_client()
        
        # Return the model instance for the Agent to use
        return {
            "model": self  # Return self so Agent can access _client
        }
    
    def get_llm_config(self) -> LLMConfig:
        """Build LLMConfig from parameters."""
        return LLMConfig(
            provider=self._provider,
            model=self.get_parameter("model", ""),
            api_key=self.get_parameter("api_key"),
            base_url=self.get_parameter("base_url"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            top_p=self.get_parameter("top_p", 1.0),
            frequency_penalty=self.get_parameter("frequency_penalty", 0.0),
            presence_penalty=self.get_parameter("presence_penalty", 0.0)
        )
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema with LLM-specific fields."""
        schema = super().to_schema()
        schema["provider"] = self._provider.value
        schema["model_config"] = self.get_llm_config().model_dump()
        return schema
