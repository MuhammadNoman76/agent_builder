"""
Schema generation for field types and components.

This module provides utilities for generating JSON schemas that can be
consumed by frontend applications to dynamically render forms.
"""
from typing import Any, Dict, List, Optional, Type
from .base import FieldDefinition, FieldGroup, FieldTypeEnum
from .types import *


class FieldSchemaGenerator:
    """Generates JSON schema for a single field."""
    
    @staticmethod
    def generate(field: FieldDefinition) -> Dict[str, Any]:
        """Generate JSON schema for a field."""
        return field.to_schema()
    
    @staticmethod
    def generate_json_schema(field: FieldDefinition) -> Dict[str, Any]:
        """Generate JSON Schema (draft-07) compatible schema."""
        schema = {
            "title": field.label,
            "description": field.description,
        }
        
        # Map field types to JSON Schema types
        type_mapping = {
            FieldTypeEnum.STRING: {"type": "string"},
            FieldTypeEnum.TEXT: {"type": "string"},
            FieldTypeEnum.PASSWORD: {"type": "string"},
            FieldTypeEnum.EMAIL: {"type": "string", "format": "email"},
            FieldTypeEnum.URL: {"type": "string", "format": "uri"},
            FieldTypeEnum.NUMBER: {"type": "number"},
            FieldTypeEnum.INTEGER: {"type": "integer"},
            FieldTypeEnum.BOOLEAN: {"type": "boolean"},
            FieldTypeEnum.SELECT: {"type": "string"},
            FieldTypeEnum.MULTI_SELECT: {"type": "array", "items": {"type": "string"}},
            FieldTypeEnum.JSON: {"type": "object"},
            FieldTypeEnum.DATE: {"type": "string", "format": "date"},
            FieldTypeEnum.DATETIME: {"type": "string", "format": "date-time"},
            FieldTypeEnum.TIME: {"type": "string", "format": "time"},
        }
        
        schema.update(type_mapping.get(field.type, {"type": "string"}))
        
        # Add validation constraints
        validation = field.validation
        
        if validation.min_length is not None:
            schema["minLength"] = validation.min_length
        if validation.max_length is not None:
            schema["maxLength"] = validation.max_length
        if validation.min_value is not None:
            schema["minimum"] = validation.min_value
        if validation.max_value is not None:
            schema["maximum"] = validation.max_value
        if validation.pattern:
            schema["pattern"] = validation.pattern
        
        # Add enum for select fields
        if field.options:
            schema["enum"] = [opt.value for opt in field.options]
        
        # Add default
        if field.default is not None:
            schema["default"] = field.default
        
        return schema


class ComponentSchemaGenerator:
    """Generates complete schema for a component's fields."""
    
    @staticmethod
    def generate(
        fields: List[FieldDefinition],
        groups: Optional[List[FieldGroup]] = None,
        component_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete component schema for frontend rendering.
        
        Args:
            fields: List of field definitions
            groups: Optional list of field groups
            component_info: Optional component metadata
            
        Returns:
            Complete schema dictionary
        """
        schema = {
            "version": "1.0",
            "component": component_info or {},
            "fields": [field.to_schema() for field in fields],
            "groups": [group.model_dump() for group in groups] if groups else [],
            "field_order": [field.name for field in sorted(fields, key=lambda f: f.order)],
        }
        
        # Add field type metadata
        schema["field_types"] = ComponentSchemaGenerator._get_field_type_metadata(fields)
        
        # Add validation schema
        schema["validation_schema"] = ComponentSchemaGenerator._generate_validation_schema(fields)
        
        return schema
    
    @staticmethod
    def _get_field_type_metadata(fields: List[FieldDefinition]) -> Dict[str, Any]:
        """Get metadata about field types used in the component."""
        types_used = set(field.type for field in fields)
        
        metadata = {}
        for field_type in types_used:
            metadata[field_type] = {
                "type": field_type,
                "input_component": ComponentSchemaGenerator._get_input_component(field_type),
                "requires_options": field_type in [
                    FieldTypeEnum.SELECT,
                    FieldTypeEnum.MULTI_SELECT,
                    FieldTypeEnum.RADIO,
                    FieldTypeEnum.CHECKBOX_GROUP,
                ],
            }
        
        return metadata
    
    @staticmethod
    def _get_input_component(field_type: FieldTypeEnum) -> str:
        """Map field type to recommended frontend component."""
        component_mapping = {
            FieldTypeEnum.STRING: "TextInput",
            FieldTypeEnum.TEXT: "TextArea",
            FieldTypeEnum.NUMBER: "NumberInput",
            FieldTypeEnum.INTEGER: "NumberInput",
            FieldTypeEnum.BOOLEAN: "Switch",
            FieldTypeEnum.SELECT: "Select",
            FieldTypeEnum.MULTI_SELECT: "MultiSelect",
            FieldTypeEnum.RADIO: "RadioGroup",
            FieldTypeEnum.CHECKBOX_GROUP: "CheckboxGroup",
            FieldTypeEnum.PASSWORD: "PasswordInput",
            FieldTypeEnum.EMAIL: "EmailInput",
            FieldTypeEnum.URL: "UrlInput",
            FieldTypeEnum.COLOR: "ColorPicker",
            FieldTypeEnum.DATE: "DatePicker",
            FieldTypeEnum.DATETIME: "DateTimePicker",
            FieldTypeEnum.TIME: "TimePicker",
            FieldTypeEnum.JSON: "JsonEditor",
            FieldTypeEnum.CODE: "CodeEditor",
            FieldTypeEnum.SLIDER: "Slider",
            FieldTypeEnum.RANGE: "RangeSlider",
            FieldTypeEnum.FILE: "FileUpload",
            FieldTypeEnum.IMAGE: "ImageUpload",
            FieldTypeEnum.API_KEY: "ApiKeyInput",
            FieldTypeEnum.MODEL_SELECT: "ModelSelect",
            FieldTypeEnum.PROMPT: "PromptEditor",
            FieldTypeEnum.VARIABLE: "VariableInput",
            FieldTypeEnum.PORT: "PortConfig",
        }
        return component_mapping.get(field_type, "TextInput")
    
    @staticmethod
    def _generate_validation_schema(fields: List[FieldDefinition]) -> Dict[str, Any]:
        """Generate validation schema for all fields."""
        properties = {}
        required = []
        
        for field in fields:
            properties[field.name] = FieldSchemaGenerator.generate_json_schema(field)
            if field.validation.required:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def generate_field_schema(field: FieldDefinition) -> Dict[str, Any]:
    """Convenience function to generate field schema."""
    return FieldSchemaGenerator.generate(field)


def generate_component_schema(
    fields: List[FieldDefinition],
    groups: Optional[List[FieldGroup]] = None,
    component_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to generate component schema."""
    return ComponentSchemaGenerator.generate(fields, groups, component_info)
