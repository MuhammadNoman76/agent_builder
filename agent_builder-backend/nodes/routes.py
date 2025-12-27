"""
Node/Component API routes - lists available components and field types.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Dict, Any, List

from components.registry import ComponentRegistry
from components.field_types import (
    field_type_registry,
    FieldTypeEnum,
    FieldDefinition,
    validate_fields,
    get_validation_errors,
)
from .schemas import (
    ComponentListResponse,
    ComponentDetailResponse,
    ComponentCategoryResponse,
    ComponentConfigSchema,
    FieldTypesListResponse,
    FieldTypeInfoSchema,
    ValidateFieldsRequest,
    ValidateFieldsResponse,
    ValidationResultSchema,
)


router = APIRouter(prefix="/components", tags=["Components"])


# Get registry instances
component_registry = ComponentRegistry()


@router.get("", response_model=ComponentListResponse)
async def list_components(category: Optional[str] = None):
    """
    List all available components for the flow builder.
    
    Optionally filter by category.
    """
    if category:
        components = component_registry.list_by_category(category)
    else:
        components = component_registry.list_components()
    
    return ComponentListResponse(
        components=[
            ComponentDetailResponse(
                component_type=comp["component_type"],
                config=ComponentConfigSchema(**comp["config"])
            )
            for comp in components
        ],
        total=len(components)
    )


@router.get("/categories", response_model=ComponentCategoryResponse)
async def list_categories():
    """
    List all component categories with their components.
    """
    categories = component_registry.get_categories()
    
    components_by_category = {}
    for category in categories:
        components = component_registry.list_by_category(category)
        components_by_category[category] = [
            ComponentDetailResponse(
                component_type=comp["component_type"],
                config=ComponentConfigSchema(**comp["config"])
            )
            for comp in components
        ]
    
    return ComponentCategoryResponse(
        categories=categories,
        components_by_category=components_by_category
    )


@router.get("/field-types", response_model=FieldTypesListResponse)
async def list_field_types():
    """
    List all available field types for component configuration.
    
    This endpoint provides frontend developers with information about
    all supported input types and their properties.
    """
    field_types = field_type_registry.list_types()
    
    # Map field types to input components
    input_component_map = {
        "string": "TextInput",
        "text": "TextArea",
        "number": "NumberInput",
        "integer": "NumberInput",
        "boolean": "Switch",
        "select": "Select",
        "multi_select": "MultiSelect",
        "radio": "RadioGroup",
        "checkbox_group": "CheckboxGroup",
        "password": "PasswordInput",
        "email": "EmailInput",
        "url": "UrlInput",
        "color": "ColorPicker",
        "date": "DatePicker",
        "datetime": "DateTimePicker",
        "time": "TimePicker",
        "json": "JsonEditor",
        "code": "CodeEditor",
        "slider": "Slider",
        "range": "RangeSlider",
        "file": "FileUpload",
        "image": "ImageUpload",
        "api_key": "ApiKeyInput",
        "model_select": "ModelSelect",
        "prompt": "PromptEditor",
        "variable": "VariableInput",
        "port": "PortConfig",
    }
    
    return FieldTypesListResponse(
        field_types=[
            FieldTypeInfoSchema(
                type=ft["type"],
                class_name=ft["class"],
                description=ft["description"],
                default_properties=ft["default_properties"],
                input_component=input_component_map.get(ft["type"], "TextInput")
            )
            for ft in field_types
        ],
        total=len(field_types)
    )


@router.get("/field-types/{field_type}")
async def get_field_type(field_type: str):
    """
    Get detailed information about a specific field type.
    """
    try:
        ft_enum = FieldTypeEnum(field_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field type '{field_type}' not found"
        )
    
    schema = field_type_registry.get_type_schema(ft_enum)
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field type '{field_type}' not found"
        )
    
    return schema


@router.get("/{component_type}", response_model=ComponentDetailResponse)
async def get_component(component_type: str):
    """
    Get details for a specific component type including all field definitions.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    config = component_class.get_config()
    
    return ComponentDetailResponse(
        component_type=component_type,
        config=ComponentConfigSchema(**config.model_dump())
    )


@router.get("/{component_type}/schema")
async def get_component_field_schema(component_type: str):
    """
    Get the complete field schema for a component.
    
    This returns a detailed schema that can be used by the frontend
    to dynamically render the component configuration form.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    # Create temporary instance to get field schema
    try:
        instance = component_class(parameters={})
    except ValueError:
        # Some components require parameters, create with empty required fields
        instance = component_class.__new__(component_class)
        instance.node_id = "temp"
        instance.parameters = {}
        instance._fields = component_class._get_fields()
        instance._field_groups = component_class._get_field_groups()
    
    return instance.get_field_schema()


@router.post("/{component_type}/validate")
async def validate_component_parameters(
    component_type: str,
    parameters: Dict[str, Any]
):
    """
    Validate parameters for a component type.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    try:
        instance = component_class(parameters=parameters)
        errors = instance.validate_parameters()
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }


@router.post("/validate-fields", response_model=ValidateFieldsResponse)
async def validate_field_values(request: ValidateFieldsRequest):
    """
    Validate field values against their definitions.
    
    This endpoint allows validation of arbitrary field configurations,
    useful for custom form validation.
    """
    from components.field_types.base import FieldDefinition
    
    # Convert field dicts to FieldDefinition objects
    field_definitions = []
    for field_dict in request.fields:
        try:
            field = FieldDefinition(**field_dict)
            field_definitions.append(field)
        except Exception as e:
            return ValidateFieldsResponse(
                valid=False,
                results={},
                all_errors=[f"Invalid field definition: {str(e)}"]
            )
    
    # Validate all fields
    results = validate_fields(request.values, field_definitions)
    
    # Convert to response schema
    result_schemas = {
        name: ValidationResultSchema(
            valid=result.valid,
            errors=result.errors,
            warnings=result.warnings,
            field_name=result.field_name
        )
        for name, result in results.items()
    }
    
    all_errors = get_validation_errors(request.values, field_definitions)
    
    return ValidateFieldsResponse(
        valid=len(all_errors) == 0,
        results=result_schemas,
        all_errors=all_errors
    )
