"""
Field Type Registry - manages available field types.

This module provides a registry for all field types, allowing for
dynamic lookup and instantiation.
"""
from typing import Dict, Type, List, Optional, Any
from .base import FieldType, FieldTypeEnum, FieldDefinition
from .types import (
    StringField, TextField, NumberField, IntegerField, BooleanField,
    SelectField, MultiSelectField, RadioField, CheckboxGroupField,
    PasswordField, EmailField, UrlField, ColorField,
    DateField, DateTimeField, TimeField,
    JsonField, CodeField, SliderField, RangeField, FileField, ImageField,
    ApiKeyField, ModelSelectField, PromptField, VariableField, PortField
)


class FieldTypeRegistry:
    """
    Registry for field types.
    
    Provides lookup and instantiation of field types by their enum value.
    """
    
    _instance: Optional["FieldTypeRegistry"] = None
    _types: Dict[FieldTypeEnum, Type[FieldType]] = {}
    
    def __new__(cls) -> "FieldTypeRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_types()
        return cls._instance
    
    def _initialize_default_types(self) -> None:
        """Register all default field types."""
        self.register(FieldTypeEnum.STRING, StringField)
        self.register(FieldTypeEnum.TEXT, TextField)
        self.register(FieldTypeEnum.NUMBER, NumberField)
        self.register(FieldTypeEnum.INTEGER, IntegerField)
        self.register(FieldTypeEnum.BOOLEAN, BooleanField)
        self.register(FieldTypeEnum.SELECT, SelectField)
        self.register(FieldTypeEnum.MULTI_SELECT, MultiSelectField)
        self.register(FieldTypeEnum.RADIO, RadioField)
        self.register(FieldTypeEnum.CHECKBOX_GROUP, CheckboxGroupField)
        self.register(FieldTypeEnum.PASSWORD, PasswordField)
        self.register(FieldTypeEnum.EMAIL, EmailField)
        self.register(FieldTypeEnum.URL, UrlField)
        self.register(FieldTypeEnum.COLOR, ColorField)
        self.register(FieldTypeEnum.DATE, DateField)
        self.register(FieldTypeEnum.DATETIME, DateTimeField)
        self.register(FieldTypeEnum.TIME, TimeField)
        self.register(FieldTypeEnum.JSON, JsonField)
        self.register(FieldTypeEnum.CODE, CodeField)
        self.register(FieldTypeEnum.SLIDER, SliderField)
        self.register(FieldTypeEnum.RANGE, RangeField)
        self.register(FieldTypeEnum.FILE, FileField)
        self.register(FieldTypeEnum.IMAGE, ImageField)
        self.register(FieldTypeEnum.API_KEY, ApiKeyField)
        self.register(FieldTypeEnum.MODEL_SELECT, ModelSelectField)
        self.register(FieldTypeEnum.PROMPT, PromptField)
        self.register(FieldTypeEnum.VARIABLE, VariableField)
        self.register(FieldTypeEnum.PORT, PortField)
    
    def register(self, field_type: FieldTypeEnum, type_class: Type[FieldType]) -> None:
        """
        Register a field type.
        
        Args:
            field_type: The field type enum value
            type_class: The field type class
        """
        self._types[field_type] = type_class
    
    def get(self, field_type: FieldTypeEnum) -> Optional[Type[FieldType]]:
        """
        Get a field type class.
        
        Args:
            field_type: The field type enum value
            
        Returns:
            The field type class or None
        """
        return self._types.get(field_type)
    
    def create_field(
        self,
        field_type: FieldTypeEnum,
        name: str,
        label: str,
        **kwargs
    ) -> Optional[FieldDefinition]:
        """
        Create a field definition.
        
        Args:
            field_type: The field type enum value
            name: Field name
            label: Field label
            **kwargs: Additional field arguments
            
        Returns:
            FieldDefinition or None
        """
        type_class = self.get(field_type)
        if type_class:
            return type_class.create(name=name, label=label, **kwargs)
        return None
    
    def list_types(self) -> List[Dict[str, Any]]:
        """
        List all registered field types with their metadata.
        
        Returns:
            List of field type information
        """
        return [
            {
                "type": field_type.value,
                "class": type_class.__name__,
                "description": type_class.__doc__,
                "default_properties": type_class.default_properties,
            }
            for field_type, type_class in self._types.items()
        ]
    
    def get_type_schema(self, field_type: FieldTypeEnum) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a specific field type.
        
        Args:
            field_type: The field type enum value
            
        Returns:
            Type schema or None
        """
        type_class = self.get(field_type)
        if type_class:
            return type_class.get_type_schema()
        return None
    
    @property
    def available_types(self) -> List[FieldTypeEnum]:
        """Get list of available field types."""
        return list(self._types.keys())


# Global registry instance
field_type_registry = FieldTypeRegistry()
