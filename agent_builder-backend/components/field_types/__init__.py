"""
Field Types Package - Centralized input field definitions for all components.

This package provides a unified system for defining input fields, their types,
validation rules, and JSON schemas for frontend rendering.
"""
from .base import (
    FieldType,
    FieldDefinition,
    FieldValidation,
    FieldOption,
    FieldGroup,
    FieldCondition,
    FieldDependency
)
from .types import (
    # Basic Types
    StringField,
    TextField,
    NumberField,
    IntegerField,
    BooleanField,
    # Selection Types
    SelectField,
    MultiSelectField,
    RadioField,
    CheckboxGroupField,
    # Special Types
    PasswordField,
    EmailField,
    UrlField,
    ColorField,
    DateField,
    DateTimeField,
    TimeField,
    # Complex Types
    JsonField,
    CodeField,
    SliderField,
    RangeField,
    FileField,
    ImageField,
    # Custom Types
    ApiKeyField,
    ModelSelectField,
    PromptField,
    VariableField,
    PortField,
    # Field Type Enum
    FieldTypeEnum
)
from .schema import (
    FieldSchemaGenerator,
    ComponentSchemaGenerator,
    generate_field_schema,
    generate_component_schema
)
from .registry import FieldTypeRegistry, field_type_registry
from .validators import (
    FieldValidator,
    validate_field,
    validate_fields,
    get_validation_errors,
    is_valid,
    ValidationResult
)

__all__ = [
    # Base Classes
    "FieldType",
    "FieldDefinition",
    "FieldValidation",
    "FieldOption",
    "FieldGroup",
    "FieldCondition",
    "FieldDependency",
    # Field Types
    "StringField",
    "TextField",
    "NumberField",
    "IntegerField",
    "BooleanField",
    "SelectField",
    "MultiSelectField",
    "RadioField",
    "CheckboxGroupField",
    "PasswordField",
    "EmailField",
    "UrlField",
    "ColorField",
    "DateField",
    "DateTimeField",
    "TimeField",
    "JsonField",
    "CodeField",
    "SliderField",
    "RangeField",
    "FileField",
    "ImageField",
    "ApiKeyField",
    "ModelSelectField",
    "PromptField",
    "VariableField",
    "PortField",
    "FieldTypeEnum",
    # Schema Generation
    "FieldSchemaGenerator",
    "ComponentSchemaGenerator",
    "generate_field_schema",
    "generate_component_schema",
    # Registry
    "FieldTypeRegistry",
    "field_type_registry",
    # Validators
    "FieldValidator",
    "validate_field",
    "validate_fields",
    "get_validation_errors",
    "is_valid",
    "ValidationResult",
]
