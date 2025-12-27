"""
Base component class - foundation for all components.
Uses centralized field types for consistent input handling.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum
import uuid

# Import from field_types package
from .field_types import (
    FieldDefinition,
    FieldGroup,
    FieldTypeEnum,
    FieldValidation,
    FieldOption,
    ComponentSchemaGenerator,
    validate_fields,
    get_validation_errors,
    # Import specific field types for convenience
    StringField,
    TextField,
    NumberField,
    IntegerField,
    BooleanField,
    SelectField,
    MultiSelectField,
    PasswordField,
    JsonField,
    SliderField,
    ApiKeyField,
    ModelSelectField,
    PromptField,
    VariableField,
)


class PortType(str, Enum):
    """Port types for component connections."""
    INPUT = "input"
    OUTPUT = "output"


class ComponentPort(BaseModel):
    """Represents a connection port on a component."""
    id: str
    name: str
    type: PortType
    data_type: str = "any"
    required: bool = False
    multiple: bool = False
    description: Optional[str] = None


class ComponentConfig(BaseModel):
    """Configuration schema for a component."""
    component_type: str
    name: str
    description: str
    category: str
    icon: str = "box"
    color: str = "#6366f1"
    input_ports: List[ComponentPort] = Field(default_factory=list)
    output_ports: List[ComponentPort] = Field(default_factory=list)
    # Updated to use FieldDefinition
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    field_groups: List[Dict[str, Any]] = Field(default_factory=list)


class BaseComponent(ABC):
    """
    Abstract base class for all components in the agent builder.
    
    Uses the centralized field types system for input definitions.
    """
    
    _component_type: str = "base"
    _name: str = "Base Component"
    _description: str = "Base component class"
    _category: str = "general"
    _icon: str = "box"
    _color: str = "#6366f1"
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Initialize the component."""
        self.node_id = node_id or str(uuid.uuid4())
        self.parameters = parameters or {}
        self._fields: List[FieldDefinition] = []
        self._field_groups: List[FieldGroup] = []
        self._initialize_fields()
        self._validate_and_set_defaults()
    
    def _initialize_fields(self) -> None:
        """Initialize field definitions. Called once during init."""
        self._fields = self._get_fields()
        self._field_groups = self._get_field_groups()
    
    def _validate_and_set_defaults(self) -> None:
        """Validate parameters and set defaults for missing values."""
        for field in self._fields:
            if field.name not in self.parameters:
                if field.default is not None:
                    self.parameters[field.name] = field.default
                elif field.validation.required:
                    raise ValueError(
                        f"Required field '{field.name}' is missing for component '{self._name}'"
                    )
    
    @classmethod
    def get_config(cls) -> ComponentConfig:
        """Get the component configuration."""
        instance = cls.__new__(cls)
        instance._fields = []
        instance._field_groups = []
        instance._initialize_fields = lambda: None
        instance.parameters = {}
        instance.node_id = "temp"
        
        # Get fields using class method
        fields = cls._get_fields()
        groups = cls._get_field_groups()
        
        return ComponentConfig(
            component_type=cls._component_type,
            name=cls._name,
            description=cls._description,
            category=cls._category,
            icon=cls._icon,
            color=cls._color,
            input_ports=cls._get_input_ports(),
            output_ports=cls._get_output_ports(),
            fields=[f.to_schema() for f in fields],
            field_groups=[g.model_dump() for g in groups],
        )
    
    @classmethod
    @abstractmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Define input ports for the component."""
        pass
    
    @classmethod
    @abstractmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Define output ports for the component."""
        pass
    
    @classmethod
    @abstractmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields for the component using field types."""
        pass
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organizing fields. Override in subclasses."""
        return []
    
    def validate_parameters(self) -> List[str]:
        """Validate component parameters using field validators."""
        return get_validation_errors(self.parameters, self._fields)
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component logic."""
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert component to JSON schema representation."""
        config = self.get_config()
        return {
            "node_id": self.node_id,
            "component_type": self._component_type,
            "name": config.name,
            "parameters": self.parameters,
            "input_ports": [port.model_dump() for port in config.input_ports],
            "output_ports": [port.model_dump() for port in config.output_ports],
            "fields": config.fields,
            "field_groups": config.field_groups,
        }
    
    def get_field_schema(self) -> Dict[str, Any]:
        """Get complete field schema for frontend rendering."""
        return ComponentSchemaGenerator.generate(
            fields=self._fields,
            groups=self._field_groups,
            component_info={
                "type": self._component_type,
                "name": self._name,
                "description": self._description,
                "category": self._category,
                "icon": self._icon,
                "color": self._color,
            }
        )
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update component parameters."""
        self.parameters.update(parameters)
        self._validate_and_set_defaults()
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a specific parameter value."""
        return self.parameters.get(name, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_id={self.node_id}, parameters={self.parameters})"


# Re-export field types for convenience
__all__ = [
    "BaseComponent",
    "ComponentConfig",
    "ComponentPort",
    "PortType",
    # Field types
    "FieldDefinition",
    "FieldGroup",
    "FieldTypeEnum",
    "FieldValidation",
    "FieldOption",
    "StringField",
    "TextField",
    "NumberField",
    "IntegerField",
    "BooleanField",
    "SelectField",
    "MultiSelectField",
    "PasswordField",
    "JsonField",
    "SliderField",
    "ApiKeyField",
    "ModelSelectField",
    "PromptField",
    "VariableField",
]
