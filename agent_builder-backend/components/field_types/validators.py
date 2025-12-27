"""
Field validation utilities.

This module provides validation functions for field values.
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from .base import FieldDefinition, FieldTypeEnum, FieldCondition
from .registry import field_type_registry


@dataclass
class ValidationResult:
    """Result of field validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    field_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "field_name": self.field_name,
        }


class FieldValidator:
    """Validates field values against their definitions."""
    
    @staticmethod
    def validate(
        value: Any,
        field: FieldDefinition,
        all_values: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a value against a field definition.
        
        Args:
            value: The value to validate
            field: The field definition
            all_values: All field values (for conditional validation)
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check conditional requirements
        if all_values and field.dependencies:
            for dep in field.dependencies:
                if dep.should_apply(all_values):
                    if dep.action == "require" and not value:
                        errors.append(f"{field.label} is required")
        
        # Get field type class for validation
        type_class = field_type_registry.get(FieldTypeEnum(field.type))
        if type_class:
            type_errors = type_class.validate_value(value, field)
            errors.extend(type_errors)
        else:
            # Basic validation
            errors.extend(FieldValidator._basic_validate(value, field))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            field_name=field.name,
        )
    
    @staticmethod
    def _basic_validate(value: Any, field: FieldDefinition) -> List[str]:
        """Basic validation for unknown field types."""
        errors = []
        validation = field.validation
        
        # Required check
        if validation.required and (value is None or value == ""):
            errors.append(f"{field.label} is required")
            return errors
        
        # Skip further validation if empty and not required
        if value is None or value == "":
            return errors
        
        # Allowed values check
        if validation.allowed_values and value not in validation.allowed_values:
            errors.append(f"{field.label} has an invalid value")
        
        # Forbidden values check
        if validation.forbidden_values and value in validation.forbidden_values:
            errors.append(f"{field.label} contains a forbidden value")
        
        return errors


def validate_field(
    value: Any,
    field: FieldDefinition,
    all_values: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Convenience function to validate a single field.
    
    Args:
        value: The value to validate
        field: The field definition
        all_values: All field values (for conditional validation)
        
    Returns:
        ValidationResult
    """
    return FieldValidator.validate(value, field, all_values)


def validate_fields(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> Dict[str, ValidationResult]:
    """
    Validate multiple fields.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        Dictionary mapping field names to ValidationResults
    """
    results = {}
    for field in fields:
        value = values.get(field.name)
        results[field.name] = FieldValidator.validate(value, field, values)
    return results


def get_validation_errors(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> List[str]:
    """
    Get all validation errors for a set of fields.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        List of error messages
    """
    all_errors = []
    results = validate_fields(values, fields)
    
    for field_name, result in results.items():
        all_errors.extend(result.errors)
    
    return all_errors


def is_valid(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> bool:
    """
    Check if all fields are valid.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        True if all fields are valid
    """
    results = validate_fields(values, fields)
    return all(result.valid for result in results.values())
