"""
Concrete field type implementations.

This module contains all the specific field type classes that can be used
to define input fields in components.
"""
from typing import Any, Dict, List, Optional, Union
from .base import (
    FieldType,
    FieldTypeEnum,
    FieldDefinition,
    FieldValidation,
    FieldOption,
    FieldCondition,
    FieldDependency
)


# =============================================================================
# BASIC FIELD TYPES
# =============================================================================

class StringField(FieldType):
    """Single-line text input field."""
    
    field_type = FieldTypeEnum.STRING
    default_properties = {
        "max_display_length": 100,
        "auto_trim": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        pattern_message: Optional[str] = None,
        placeholder: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a string field definition."""
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                pattern_message=pattern_message,
            ),
            placeholder=placeholder or f"Enter {label.lower()}...",
            prefix=prefix,
            suffix=suffix,
            properties=cls.default_properties.copy(),
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate string value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        if not isinstance(value, str):
            errors.append(f"{field.label} must be a string")
            return errors
        
        validation = field.validation
        
        if validation.min_length and len(value) < validation.min_length:
            errors.append(f"{field.label} must be at least {validation.min_length} characters")
        
        if validation.max_length and len(value) > validation.max_length:
            errors.append(f"{field.label} must be at most {validation.max_length} characters")
        
        if validation.pattern:
            import re
            if not re.match(validation.pattern, value):
                msg = validation.pattern_message or f"{field.label} format is invalid"
                errors.append(msg)
        
        return errors


class TextField(FieldType):
    """Multi-line text area field."""
    
    field_type = FieldTypeEnum.TEXT
    default_properties = {
        "rows": 4,
        "max_rows": 20,
        "auto_resize": True,
        "show_character_count": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        rows: int = 4,
        max_rows: int = 20,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a text area field definition."""
        properties = cls.default_properties.copy()
        properties["rows"] = rows
        properties["max_rows"] = max_rows
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
            ),
            placeholder=placeholder or f"Enter {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate text value."""
        return StringField.validate_value(value, field)


class NumberField(FieldType):
    """Numeric input field for floating-point numbers."""
    
    field_type = FieldTypeEnum.NUMBER
    default_properties = {
        "step": 0.1,
        "precision": 2,
        "show_controls": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[float] = None,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 2,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a number field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        properties["precision"] = precision
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_value=min_value,
                max_value=max_value,
            ),
            prefix=prefix,
            suffix=suffix,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate number value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"{field.label} must be a number")
            return errors
        
        validation = field.validation
        
        if validation.min_value is not None and num_value < validation.min_value:
            errors.append(f"{field.label} must be at least {validation.min_value}")
        
        if validation.max_value is not None and num_value > validation.max_value:
            errors.append(f"{field.label} must be at most {validation.max_value}")
        
        return errors


class IntegerField(FieldType):
    """Integer input field."""
    
    field_type = FieldTypeEnum.INTEGER
    default_properties = {
        "step": 1,
        "show_controls": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
        required: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        step: int = 1,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an integer field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_value=min_value,
                max_value=max_value,
            ),
            prefix=prefix,
            suffix=suffix,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate integer value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        try:
            if isinstance(value, float) and not value.is_integer():
                errors.append(f"{field.label} must be a whole number")
                return errors
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append(f"{field.label} must be an integer")
            return errors
        
        validation = field.validation
        
        if validation.min_value is not None and int_value < validation.min_value:
            errors.append(f"{field.label} must be at least {int(validation.min_value)}")
        
        if validation.max_value is not None and int_value > validation.max_value:
            errors.append(f"{field.label} must be at most {int(validation.max_value)}")
        
        return errors


class BooleanField(FieldType):
    """Boolean toggle/checkbox field."""
    
    field_type = FieldTypeEnum.BOOLEAN
    default_properties = {
        "style": "switch",  # switch, checkbox
        "true_label": "Yes",
        "false_label": "No",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: bool = False,
        style: str = "switch",
        true_label: str = "Yes",
        false_label: str = "No",
        **kwargs
    ) -> FieldDefinition:
        """Create a boolean field definition."""
        properties = cls.default_properties.copy()
        properties["style"] = style
        properties["true_label"] = true_label
        properties["false_label"] = false_label
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate boolean value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is not None and not isinstance(value, bool):
            # Try to coerce
            if isinstance(value, str):
                if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    errors.append(f"{field.label} must be a boolean value")
            elif not isinstance(value, (int, float)):
                errors.append(f"{field.label} must be a boolean value")
        
        return errors


# =============================================================================
# SELECTION FIELD TYPES
# =============================================================================

class SelectField(FieldType):
    """Single-selection dropdown field."""
    
    field_type = FieldTypeEnum.SELECT
    default_properties = {
        "searchable": True,
        "clearable": True,
        "grouped": False,
        "virtual_scroll": True,  # For large option lists
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        searchable: bool = True,
        clearable: bool = True,
        placeholder: Optional[str] = None,
        options_source: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a select field definition."""
        properties = cls.default_properties.copy()
        properties["searchable"] = searchable
        properties["clearable"] = clearable
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            options=field_options,
            options_source=options_source,
            placeholder=placeholder or f"Select {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate select value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        if field.options:
            valid_values = [opt.value for opt in field.options]
            if value not in valid_values:
                errors.append(f"{field.label} has an invalid selection")
        
        return errors


class MultiSelectField(FieldType):
    """Multi-selection field allowing multiple values."""
    
    field_type = FieldTypeEnum.MULTI_SELECT
    default_properties = {
        "searchable": True,
        "clearable": True,
        "max_selections": None,
        "min_selections": None,
        "show_selected_count": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: List[Any] = None,
        required: bool = False,
        min_selections: Optional[int] = None,
        max_selections: Optional[int] = None,
        searchable: bool = True,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a multi-select field definition."""
        properties = cls.default_properties.copy()
        properties["searchable"] = searchable
        properties["min_selections"] = min_selections
        properties["max_selections"] = max_selections
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [],
            validation=FieldValidation(required=required),
            options=field_options,
            placeholder=placeholder or f"Select {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate multi-select value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None:
            value = []
        
        if not isinstance(value, list):
            errors.append(f"{field.label} must be a list")
            return errors
        
        properties = field.properties
        
        if properties.get("min_selections") and len(value) < properties["min_selections"]:
            errors.append(f"{field.label} requires at least {properties['min_selections']} selections")
        
        if properties.get("max_selections") and len(value) > properties["max_selections"]:
            errors.append(f"{field.label} allows at most {properties['max_selections']} selections")
        
        if field.options:
            valid_values = [opt.value for opt in field.options]
            for v in value:
                if v not in valid_values:
                    errors.append(f"{field.label} contains invalid selection: {v}")
        
        return errors


class RadioField(FieldType):
    """Radio button group for single selection."""
    
    field_type = FieldTypeEnum.RADIO
    default_properties = {
        "layout": "vertical",  # vertical, horizontal, grid
        "columns": 2,  # For grid layout
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        layout: str = "vertical",
        **kwargs
    ) -> FieldDefinition:
        """Create a radio field definition."""
        properties = cls.default_properties.copy()
        properties["layout"] = layout
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            options=field_options,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate radio value."""
        return SelectField.validate_value(value, field)


class CheckboxGroupField(FieldType):
    """Checkbox group for multiple selections."""
    
    field_type = FieldTypeEnum.CHECKBOX_GROUP
    default_properties = {
        "layout": "vertical",
        "columns": 2,
        "select_all": False,  # Show "Select All" option
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: List[Any] = None,
        required: bool = False,
        layout: str = "vertical",
        select_all: bool = False,
        **kwargs
    ) -> FieldDefinition:
        """Create a checkbox group field definition."""
        properties = cls.default_properties.copy()
        properties["layout"] = layout
        properties["select_all"] = select_all
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [],
            validation=FieldValidation(required=required),
            options=field_options,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate checkbox group value."""
        return MultiSelectField.validate_value(value, field)


# =============================================================================
# SPECIAL INPUT FIELD TYPES
# =============================================================================

class PasswordField(FieldType):
    """Password input field with masking."""
    
    field_type = FieldTypeEnum.PASSWORD
    default_properties = {
        "show_toggle": True,  # Show/hide password toggle
        "strength_indicator": True,
        "generate_button": False,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        show_toggle: bool = True,
        strength_indicator: bool = False,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a password field definition."""
        properties = cls.default_properties.copy()
        properties["show_toggle"] = show_toggle
        properties["strength_indicator"] = strength_indicator
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default="",
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
            ),
            placeholder=placeholder or "Enter password...",
            sensitive=True,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate password value."""
        return StringField.validate_value(value, field)


class EmailField(FieldType):
    """Email input field with validation."""
    
    field_type = FieldTypeEnum.EMAIL
    default_properties = {
        "suggest_domains": True,
        "common_domains": ["gmail.com", "outlook.com", "yahoo.com"],
    }
    
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an email field definition."""
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.EMAIL_PATTERN,
                pattern_message="Please enter a valid email address",
            ),
            placeholder=placeholder or "email@example.com",
            properties=cls.default_properties.copy(),
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate email value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.EMAIL_PATTERN, value):
            errors.append("Please enter a valid email address")
        
        return errors


class UrlField(FieldType):
    """URL input field with validation."""
    
    field_type = FieldTypeEnum.URL
    default_properties = {
        "protocols": ["http", "https"],
        "show_preview": False,
    }
    
    URL_PATTERN = r'^https?://[^\s/$.?#].[^\s]*$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        protocols: List[str] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a URL field definition."""
        properties = cls.default_properties.copy()
        if protocols:
            properties["protocols"] = protocols
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.URL_PATTERN,
                pattern_message="Please enter a valid URL",
            ),
            placeholder=placeholder or "https://example.com",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate URL value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.URL_PATTERN, value):
            errors.append("Please enter a valid URL")
        
        return errors


class ColorField(FieldType):
    """Color picker field."""
    
    field_type = FieldTypeEnum.COLOR
    default_properties = {
        "format": "hex",  # hex, rgb, hsl
        "show_alpha": False,
        "presets": [],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "#000000",
        required: bool = False,
        format: str = "hex",
        show_alpha: bool = False,
        presets: List[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a color field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["show_alpha"] = show_alpha
        if presets:
            properties["presets"] = presets
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


class DateField(FieldType):
    """Date picker field."""
    
    field_type = FieldTypeEnum.DATE
    default_properties = {
        "format": "YYYY-MM-DD",
        "min_date": None,
        "max_date": None,
        "disabled_dates": [],
        "show_today_button": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "YYYY-MM-DD",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a date field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["min_date"] = min_date
        properties["max_date"] = max_date
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select date...",
            properties=properties,
            **kwargs
        )


class DateTimeField(FieldType):
    """Date and time picker field."""
    
    field_type = FieldTypeEnum.DATETIME
    default_properties = {
        "format": "YYYY-MM-DD HH:mm",
        "time_format": "24h",  # 12h, 24h
        "min_datetime": None,
        "max_datetime": None,
        "step_minutes": 15,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "YYYY-MM-DD HH:mm",
        time_format: str = "24h",
        step_minutes: int = 15,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a datetime field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["time_format"] = time_format
        properties["step_minutes"] = step_minutes
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select date and time...",
            properties=properties,
            **kwargs
        )


class TimeField(FieldType):
    """Time picker field."""
    
    field_type = FieldTypeEnum.TIME
    default_properties = {
        "format": "HH:mm",
        "time_format": "24h",
        "step_minutes": 15,
        "min_time": None,
        "max_time": None,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "HH:mm",
        time_format: str = "24h",
        step_minutes: int = 15,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a time field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["time_format"] = time_format
        properties["step_minutes"] = step_minutes
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select time...",
            properties=properties,
            **kwargs
        )


# =============================================================================
# COMPLEX FIELD TYPES
# =============================================================================

class JsonField(FieldType):
    """JSON editor field."""
    
    field_type = FieldTypeEnum.JSON
    default_properties = {
        "mode": "code",  # code, tree, view
        "validate_json": True,
        "indent": 2,
        "line_numbers": True,
        "folding": True,
        "schema": None,  # JSON schema for validation
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        mode: str = "code",
        schema: Optional[Dict[str, Any]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a JSON field definition."""
        properties = cls.default_properties.copy()
        properties["mode"] = mode
        if schema:
            properties["schema"] = schema
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default if default is not None else {},
            validation=FieldValidation(required=required),
            placeholder=placeholder or '{}',
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate JSON value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None or value == "":
            return errors
        
        if isinstance(value, str):
            import json
            try:
                json.loads(value)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {str(e)}")
        
        return errors


class CodeField(FieldType):
    """Code editor field with syntax highlighting."""
    
    field_type = FieldTypeEnum.CODE
    default_properties = {
        "language": "python",
        "theme": "vs-dark",
        "line_numbers": True,
        "folding": True,
        "minimap": False,
        "auto_format": True,
        "tab_size": 4,
    }
    
    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "json", "yaml", "markdown",
        "html", "css", "sql", "bash", "shell", "plaintext"
    ]
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        language: str = "python",
        theme: str = "vs-dark",
        line_numbers: bool = True,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a code editor field definition."""
        properties = cls.default_properties.copy()
        properties["language"] = language
        properties["theme"] = theme
        properties["line_numbers"] = line_numbers
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or f"Enter {language} code...",
            properties=properties,
            **kwargs
        )


class SliderField(FieldType):
    """Slider input for numeric values."""
    
    field_type = FieldTypeEnum.SLIDER
    default_properties = {
        "show_value": True,
        "show_marks": True,
        "marks": None,  # Custom marks: [{value: 0, label: "Min"}, ...]
        "tooltip": "always",  # always, hover, never
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        min_value: float,
        max_value: float,
        description: Optional[str] = None,
        default: Optional[float] = None,
        step: float = 1,
        marks: Optional[List[Dict[str, Any]]] = None,
        show_value: bool = True,
        **kwargs
    ) -> FieldDefinition:
        """Create a slider field definition."""
        properties = cls.default_properties.copy()
        properties["show_value"] = show_value
        properties["step"] = step
        if marks:
            properties["marks"] = marks
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default if default is not None else min_value,
            validation=FieldValidation(
                min_value=min_value,
                max_value=max_value,
            ),
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate slider value."""
        return NumberField.validate_value(value, field)


class RangeField(FieldType):
    """Range slider for selecting a range of values."""
    
    field_type = FieldTypeEnum.RANGE
    default_properties = {
        "show_values": True,
        "show_marks": True,
        "tooltip": "always",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        min_value: float,
        max_value: float,
        description: Optional[str] = None,
        default: Optional[List[float]] = None,
        step: float = 1,
        **kwargs
    ) -> FieldDefinition:
        """Create a range field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        properties["min"] = min_value
        properties["max"] = max_value
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [min_value, max_value],
            validation=FieldValidation(
                min_value=min_value,
                max_value=max_value,
            ),
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate range value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None:
            return errors
        
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            errors.append(f"{field.label} must be a range with two values")
            return errors
        
        min_val, max_val = value
        if min_val > max_val:
            errors.append(f"{field.label} minimum cannot be greater than maximum")
        
        return errors


class FileField(FieldType):
    """File upload field."""
    
    field_type = FieldTypeEnum.FILE
    default_properties = {
        "accept": "*/*",
        "multiple": False,
        "max_size": 10 * 1024 * 1024,  # 10MB
        "show_preview": True,
        "drag_drop": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        accept: str = "*/*",
        multiple: bool = False,
        max_size: int = 10 * 1024 * 1024,
        **kwargs
    ) -> FieldDefinition:
        """Create a file upload field definition."""
        properties = cls.default_properties.copy()
        properties["accept"] = accept
        properties["multiple"] = multiple
        properties["max_size"] = max_size
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=None,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


class ImageField(FieldType):
    """Image upload field with preview."""
    
    field_type = FieldTypeEnum.IMAGE
    default_properties = {
        "accept": "image/*",
        "max_size": 5 * 1024 * 1024,  # 5MB
        "show_preview": True,
        "preview_size": {"width": 200, "height": 200},
        "crop": False,
        "aspect_ratio": None,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        max_size: int = 5 * 1024 * 1024,
        crop: bool = False,
        aspect_ratio: Optional[float] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an image upload field definition."""
        properties = cls.default_properties.copy()
        properties["max_size"] = max_size
        properties["crop"] = crop
        if aspect_ratio:
            properties["aspect_ratio"] = aspect_ratio
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=None,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


# =============================================================================
# CUSTOM AGENT BUILDER FIELD TYPES
# =============================================================================

class ApiKeyField(FieldType):
    """API key input field with provider-specific validation."""
    
    field_type = FieldTypeEnum.API_KEY
    default_properties = {
        "provider": None,
        "show_toggle": True,
        "validate_format": True,
        "test_connection": True,
    }
    
    # Provider-specific patterns - UPDATED to support newer key formats
    PROVIDER_PATTERNS = {
        # OpenAI keys: sk-, sk-proj-, sk-svcacct- followed by alphanumeric/underscore/hyphen
        "openai": r"^sk-(?:proj-|svcacct-)?[a-zA-Z0-9_-]{20,}$",
        # Anthropic keys
        "anthropic": r"^sk-ant-[a-zA-Z0-9_-]{20,}$",
        # OpenRouter keys
        "openrouter": r"^sk-or-v1-[a-zA-Z0-9_-]{20,}$",
        # Composio keys
        "composio": r"^[a-zA-Z0-9_-]{20,}$",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        provider: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = True,
        placeholder: Optional[str] = None,
        test_connection: bool = True,
        **kwargs
    ) -> FieldDefinition:
        """Create an API key field definition."""
        properties = cls.default_properties.copy()
        properties["provider"] = provider
        properties["test_connection"] = test_connection
        
        pattern = cls.PROVIDER_PATTERNS.get(provider) if provider else None
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or f"API key for {provider or 'the service'}",
            default="",
            validation=FieldValidation(
                required=required,
                pattern=pattern,
                pattern_message=f"Invalid {provider or 'API'} key format" if pattern else None,
            ),
            placeholder=placeholder or f"Enter {provider or 'API'} key...",
            sensitive=True,
            copyable=True,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate API key value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        provider = field.properties.get("provider")
        if provider and provider in cls.PROVIDER_PATTERNS:
            import re
            if not re.match(cls.PROVIDER_PATTERNS[provider], value):
                errors.append(f"Invalid {provider} API key format")
        
        return errors


class ModelSelectField(FieldType):
    """Model selection field with provider-specific options."""
    
    field_type = FieldTypeEnum.MODEL_SELECT
    default_properties = {
        "provider": None,
        "show_model_info": True,
        "allow_custom": True,
        "group_by_provider": True,
    }
    
    # Default models by provider
    PROVIDER_MODELS = {
        "openai": [
            {"value": "gpt-4o", "label": "GPT-4o (Latest)", "group": "GPT-4"},
            {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "GPT-4"},
            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "GPT-4"},
            {"value": "gpt-4", "label": "GPT-4", "group": "GPT-4"},
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "group": "GPT-3.5"},
        ],
        "anthropic": [
            {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet", "group": "Claude 3.5"},
            {"value": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku", "group": "Claude 3.5"},
            {"value": "claude-3-opus-20240229", "label": "Claude 3 Opus", "group": "Claude 3"},
            {"value": "claude-3-sonnet-20240229", "label": "Claude 3 Sonnet", "group": "Claude 3"},
            {"value": "claude-3-haiku-20240307", "label": "Claude 3 Haiku", "group": "Claude 3"},
        ],
        "openrouter": [
            {"value": "openai/gpt-4o", "label": "OpenAI GPT-4o", "group": "OpenAI"},
            {"value": "anthropic/claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "group": "Anthropic"},
            {"value": "google/gemini-pro-1.5", "label": "Gemini 1.5 Pro", "group": "Google"},
            {"value": "meta-llama/llama-3-70b-instruct", "label": "Llama 3 70B", "group": "Meta"},
            {"value": "mistralai/mistral-large", "label": "Mistral Large", "group": "Mistral"},
        ],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        provider: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = True,
        allow_custom: bool = True,
        custom_models: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a model select field definition."""
        properties = cls.default_properties.copy()
        properties["provider"] = provider
        properties["allow_custom"] = allow_custom
        
        # Get models for provider
        models = custom_models or cls.PROVIDER_MODELS.get(provider, [])
        options = [FieldOption(**m) for m in models]
        
        # Set default if not provided
        if not default and options:
            default = options[0].value
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or f"Select a {provider} model",
            default=default,
            validation=FieldValidation(required=required),
            options=options,
            placeholder=f"Select {provider} model...",
            properties=properties,
            **kwargs
        )


class PromptField(FieldType):
    """Prompt/system message input field with variable support."""
    
    field_type = FieldTypeEnum.PROMPT
    default_properties = {
        "rows": 6,
        "max_rows": 20,
        "show_variable_picker": True,
        "show_token_count": True,
        "syntax_highlighting": True,
        "available_variables": [],
        "templates": [],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        rows: int = 6,
        available_variables: Optional[List[str]] = None,
        templates: Optional[List[Dict[str, str]]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a prompt field definition."""
        properties = cls.default_properties.copy()
        properties["rows"] = rows
        if available_variables:
            properties["available_variables"] = available_variables
        if templates:
            properties["templates"] = templates
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or "Enter prompt with optional {{variables}}",
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Enter your prompt...",
            properties=properties,
            **kwargs
        )


class VariableField(FieldType):
    """Variable name input field with validation."""
    
    field_type = FieldTypeEnum.VARIABLE
    default_properties = {
        "validate_identifier": True,
        "suggest_names": True,
        "reserved_words": ["input", "output", "self", "this"],
    }
    
    IDENTIFIER_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = True,
        reserved_words: Optional[List[str]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a variable name field definition."""
        properties = cls.default_properties.copy()
        if reserved_words:
            properties["reserved_words"] = reserved_words
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or "Must be a valid identifier (letters, numbers, underscores)",
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.IDENTIFIER_PATTERN,
                pattern_message="Must start with letter/underscore, contain only letters, numbers, underscores",
                forbidden_values=properties.get("reserved_words", []),
            ),
            placeholder=placeholder or "my_variable",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate variable name."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.IDENTIFIER_PATTERN, value):
            errors.append("Variable name must be a valid identifier")
        
        reserved = field.properties.get("reserved_words", [])
        if value in reserved:
            errors.append(f"'{value}' is a reserved word and cannot be used")
        
        return errors


class PortField(FieldType):
    """Port configuration field for component connections."""
    
    field_type = FieldTypeEnum.PORT
    default_properties = {
        "port_type": "input",  # input, output
        "data_type": "any",
        "show_type_selector": True,
    }
    
    DATA_TYPES = [
        {"value": "any", "label": "Any"},
        {"value": "string", "label": "String"},
        {"value": "number", "label": "Number"},
        {"value": "boolean", "label": "Boolean"},
        {"value": "message", "label": "Message"},
        {"value": "messages", "label": "Messages"},
        {"value": "dict", "label": "Dictionary"},
        {"value": "list", "label": "List"},
        {"value": "tools", "label": "Tools"},
        {"value": "llm_model", "label": "LLM Model"},
    ]
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        port_type: str = "input",
        data_type: str = "any",
        description: Optional[str] = None,
        required: bool = False,
        **kwargs
    ) -> FieldDefinition:
        """Create a port configuration field definition."""
        properties = cls.default_properties.copy()
        properties["port_type"] = port_type
        properties["data_type"] = data_type
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default={"type": data_type, "required": required},
            validation=FieldValidation(required=required),
            options=[FieldOption(**dt) for dt in cls.DATA_TYPES],
            properties=properties,
            **kwargs
        )
