"""
Component Registry - manages all available components.
"""
from typing import Dict, Type, List, Optional, Any
from .base import BaseComponent, ComponentConfig
from .input_component import InputComponent
from .output_component import OutputComponent
from .agent_component import AgentComponent
from .composio_component import ComposioToolComponent
from .llm import LLMRegistry, OpenAIModel, AnthropicModel, OpenRouterModel


class ComponentRegistry:
    """
    Central registry for all components in the agent builder.
    
    This class manages component registration, retrieval, and instantiation,
    providing a unified interface for the flow builder.
    """
    
    _instance: Optional["ComponentRegistry"] = None
    _components: Dict[str, Type[BaseComponent]] = {}
    
    def __new__(cls) -> "ComponentRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_components()
        return cls._instance
    
    def _initialize_default_components(self) -> None:
        """Register all default components."""
        # Core components
        self.register(InputComponent)
        self.register(OutputComponent)
        self.register(AgentComponent)
        self.register(ComposioToolComponent)
        
        # LLM models
        self.register(OpenAIModel)
        self.register(AnthropicModel)
        self.register(OpenRouterModel)
    
    def register(self, component_class: Type[BaseComponent]) -> None:
        """
        Register a new component class.
        
        Args:
            component_class: The component class to register
        """
        self._components[component_class._component_type] = component_class
    
    def unregister(self, component_type: str) -> bool:
        """
        Unregister a component.
        
        Args:
            component_type: The component type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if component_type in self._components:
            del self._components[component_type]
            return True
        return False
    
    def get(self, component_type: str) -> Optional[Type[BaseComponent]]:
        """
        Get a component class by type.
        
        Args:
            component_type: The component type identifier
            
        Returns:
            The component class or None if not found
        """
        return self._components.get(component_type)
    
    def create(
        self, 
        component_type: str, 
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseComponent]:
        """
        Create an instance of a component.
        
        Args:
            component_type: The component type identifier
            node_id: Optional node ID
            parameters: Component parameters
            
        Returns:
            Component instance or None if not found
        """
        component_class = self.get(component_type)
        if component_class:
            return component_class(node_id=node_id, parameters=parameters)
        return None
    
    def list_components(self) -> List[Dict[str, Any]]:
        """
        List all registered components with their configurations.
        
        Returns:
            List of component configurations
        """
        return [
            {
                "component_type": component_type,
                "config": component_class.get_config().model_dump()
            }
            for component_type, component_class in self._components.items()
        ]
    
    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        List components filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of component configurations in the category
        """
        return [
            {
                "component_type": component_type,
                "config": component_class.get_config().model_dump()
            }
            for component_type, component_class in self._components.items()
            if component_class._category == category
        ]
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        categories = set()
        for component_class in self._components.values():
            categories.add(component_class._category)
        return sorted(list(categories))
    
    @property
    def available_types(self) -> List[str]:
        """Get list of available component types."""
        return list(self._components.keys())


# Global registry instance
component_registry = ComponentRegistry()
