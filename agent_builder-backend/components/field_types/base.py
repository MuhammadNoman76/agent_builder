"""
Base classes for field type definitions.

This module contains the foundational classes and types used to define
input fields across all components in the agent builder.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re


class FieldTypeEnum(str, Enum):
    """Enumeration of all available field types."""
    # Basic Types
    STRING = "string"
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    
    # Selection Types
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    RADIO = "radio"
    CHECKBOX_GROUP = "checkbox_group"
    
    # Special Input Types
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    COLOR = "color"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    
    # Complex Types
    JSON = "json"
    CODE = "code"
    SLIDER = "slider"
    RANGE = "range"
    FILE = "file"
    IMAGE = "image"
    
    # Custom Agent Builder Types
    API_KEY = "api_key"
    MODEL_SELECT = "model_select"
    PROMPT = "prompt"
    VARIABLE = "variable"
    PORT = "port"


class FieldOption(BaseModel):
    """Represents an option for select/radio/checkbox fields."""
    value: Any
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    disabled: bool = False
    group: Optional[str] = None  # For grouped options
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FieldValidation(BaseModel):
    """Validation rules for a field."""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern
    pattern_message: Optional[str] = None  # Error message for pattern
    custom_validator: Optional[str] = None  # Name of custom validator function
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None
    unique: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class FieldCondition(BaseModel):
    """Conditional visibility/requirement for a field."""
    field: str  # Field name to check
    operator: str  # eq, neq, gt, lt, gte, lte, in, not_in, contains, empty, not_empty
    value: Optional[Any] = None  # Value to compare against
    
    def evaluate(self, field_value: Any) -> bool:
        """Evaluate the condition against a field value."""
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "neq":
            return field_value != self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "contains":
            return self.value in field_value
        elif self.operator == "empty":
            return not field_value
        elif self.operator == "not_empty":
            return bool(field_value)
        return False


class FieldDependency(BaseModel):
    """Dependency configuration for a field."""
    depends_on: str  # Field name this depends on
    condition: FieldCondition
    action: str = "show"  # show, hide, enable, disable, require, optional
    
    def should_apply(self, all_values: Dict[str, Any]) -> bool:
        """Check if the dependency action should apply."""
        dependent_value = all_values.get(self.depends_on)
        return self.condition.evaluate(dependent_value)


class FieldGroup(BaseModel):
    """Grouping configuration for organizing fields."""
    id: str
    label: str
    description: Optional[str] = None
    collapsible: bool = False
    collapsed_by_default: bool = False
    icon: Optional[str] = None
    order: int = 0


class FieldDefinition(BaseModel):
    """
    Complete definition of a field including all metadata for frontend rendering.
    """
    # Core Properties
    name: str
    type: FieldTypeEnum
    label: str
    description: Optional[str] = None
    
    # Default Value
    default: Any = None
    
    # Validation
    validation: FieldValidation = Field(default_factory=FieldValidation)
    
    # UI Properties
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    hint: Optional[str] = None
    icon: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    
    # Layout
    width: str = "full"  # full, half, third, quarter, auto
    order: int = 0
    group: Optional[str] = None  # Group ID for organizing fields
    
    # Options (for select/radio/checkbox)
    options: List[FieldOption] = Field(default_factory=list)
    options_source: Optional[str] = None  # API endpoint or function name for dynamic options
    
    # Conditional Display
    show_when: Optional[List[FieldCondition]] = None
    hide_when: Optional[List[FieldCondition]] = None
    dependencies: List[FieldDependency] = Field(default_factory=list)
    
    # Type-specific Properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Advanced
    read_only: bool = False
    disabled: bool = False
    hidden: bool = False
    sensitive: bool = False  # For passwords, API keys, etc.
    copyable: bool = False  # Show copy button
    clearable: bool = True  # Show clear button
    
    # Custom Rendering
    component: Optional[str] = None  # Custom component name for frontend
    render_props: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for frontend."""
        schema = {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "description": self.description,
            "default": self.default,
            "validation": self.validation.to_dict(),
            "ui": {
                "placeholder": self.placeholder,
                "help_text": self.help_text,
                "hint": self.hint,
                "icon": self.icon,
                "prefix": self.prefix,
                "suffix": self.suffix,
                "width": self.width,
                "order": self.order,
                "group": self.group,
                "read_only": self.read_only,
                "disabled": self.disabled,
                "hidden": self.hidden,
                "sensitive": self.sensitive,
                "copyable": self.copyable,
                "clearable": self.clearable,
                "component": self.component,
                "render_props": self.render_props,
            },
            "options": [opt.model_dump() for opt in self.options] if self.options else None,
            "options_source": self.options_source,
            "conditions": {
                "show_when": [c.model_dump() for c in self.show_when] if self.show_when else None,
                "hide_when": [c.model_dump() for c in self.hide_when] if self.hide_when else None,
                "dependencies": [d.model_dump() for d in self.dependencies] if self.dependencies else None,
            },
            "properties": self.properties,
        }
        
        # Remove None values for cleaner output
        return self._clean_dict(schema)
    
    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove None values from dict."""
        if not isinstance(d, dict):
            return d
        return {
            k: self._clean_dict(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if v is not None and v != {} and v != []
        }


class FieldType(ABC):
    """
    Abstract base class for field types.
    
    Subclasses define specific field types with their properties,
    validation rules, and schema generation logic.
    """
    
    field_type: FieldTypeEnum
    default_properties: Dict[str, Any] = {}
    
    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        label: str,
        **kwargs
    ) -> FieldDefinition:
        """Create a field definition of this type."""
        pass
    
    @classmethod
    def get_type_schema(cls) -> Dict[str, Any]:
        """Get the JSON schema for this field type."""
        return {
            "type": cls.field_type.value,
            "properties": cls.default_properties,
            "description": cls.__doc__
        }
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate a value against this field type's rules."""
        errors = []
        validation = field.validation
        
        # Required check
        if validation.required and (value is None or value == ""):
            errors.append(f"{field.label} is required")
            return errors
        
        # Skip further validation if value is empty and not required
        if value is None or value == "":
            return errors
        
        return errors
