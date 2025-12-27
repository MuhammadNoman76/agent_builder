"""
LLM Model Registry - manages available LLM providers and models.
"""
from typing import Dict, Type, List, Optional, Any
from .base import BaseLLMModel, LLMProvider
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .openrouter_model import OpenRouterModel


class LLMRegistry:
    """
    Registry for LLM model components.
    
    This class manages the registration and retrieval of LLM model
    components, allowing for dynamic addition of new providers.
    """
    
    _instance: Optional["LLMRegistry"] = None
    _models: Dict[str, Type[BaseLLMModel]] = {}
    
    def __new__(cls) -> "LLMRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_models()
        return cls._instance
    
    def _initialize_default_models(self) -> None:
        """Register default LLM models."""
        self.register(OpenAIModel)
        self.register(AnthropicModel)
        self.register(OpenRouterModel)
    
    def register(self, model_class: Type[BaseLLMModel]) -> None:
        """
        Register a new LLM model class.
        
        Args:
            model_class: The model class to register
        """
        self._models[model_class._component_type] = model_class
    
    def unregister(self, component_type: str) -> bool:
        """
        Unregister an LLM model.
        
        Args:
            component_type: The component type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if component_type in self._models:
            del self._models[component_type]
            return True
        return False
    
    def get(self, component_type: str) -> Optional[Type[BaseLLMModel]]:
        """
        Get an LLM model class by component type.
        
        Args:
            component_type: The component type identifier
            
        Returns:
            The model class or None if not found
        """
        return self._models.get(component_type)
    
    def create(
        self, 
        component_type: str, 
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseLLMModel]:
        """
        Create an instance of an LLM model.
        
        Args:
            component_type: The component type identifier
            node_id: Optional node ID
            parameters: Component parameters
            
        Returns:
            Model instance or None if not found
        """
        model_class = self.get(component_type)
        if model_class:
            return model_class(node_id=node_id, parameters=parameters)
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered LLM models with their configurations.
        
        Returns:
            List of model configurations
        """
        return [
            {
                "component_type": component_type,
                "config": model_class.get_config().model_dump()
            }
            for component_type, model_class in self._models.items()
        ]
    
    def get_by_provider(self, provider: LLMProvider) -> List[Type[BaseLLMModel]]:
        """
        Get all models for a specific provider.
        
        Args:
            provider: The LLM provider
            
        Returns:
            List of model classes for the provider
        """
        return [
            model_class 
            for model_class in self._models.values() 
            if model_class._provider == provider
        ]
    
    @property
    def available_types(self) -> List[str]:
        """Get list of available component types."""
        return list(self._models.keys())


# Global registry instance
llm_registry = LLMRegistry()
