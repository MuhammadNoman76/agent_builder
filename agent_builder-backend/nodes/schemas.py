"""
Node/Component schemas for API responses.
Updated to use centralized field types system.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class PortSchema(BaseModel):
    """Schema for component port."""
    id: str
    name: str
    type: str
    data_type: str
    required: bool = False
    multiple: bool = False
    description: Optional[str] = None


class FieldValidationSchema(BaseModel):
    """Schema for field validation rules."""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    pattern_message: Optional[str] = None


class FieldOptionSchema(BaseModel):
    """Schema for field option."""
    value: Any
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    disabled: bool = False
    group: Optional[str] = None


class FieldUISchema(BaseModel):
    """Schema for field UI properties."""
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    hint: Optional[str] = None
    icon: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    width: str = "full"
    order: int = 0
    group: Optional[str] = None
    read_only: bool = False
    disabled: bool = False
    hidden: bool = False
    sensitive: bool = False
    copyable: bool = False
    clearable: bool = True
    component: Optional[str] = None
    render_props: Dict[str, Any] = Field(default_factory=dict)


class FieldConditionSchema(BaseModel):
    """Schema for field condition."""
    field: str
    operator: str
    value: Optional[Any] = None


class FieldDependencySchema(BaseModel):
    """Schema for field dependency."""
    depends_on: str
    condition: FieldConditionSchema
    action: str = "show"


class FieldSchema(BaseModel):
    """Complete schema for a field."""
    name: str
    type: str
    label: str
    description: Optional[str] = None
    default: Any = None
    validation: FieldValidationSchema = Field(default_factory=FieldValidationSchema)
    ui: FieldUISchema = Field(default_factory=FieldUISchema)
    options: Optional[List[FieldOptionSchema]] = None
    options_source: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class FieldGroupSchema(BaseModel):
    """Schema for field group."""
    id: str
    label: str
    description: Optional[str] = None
    collapsible: bool = False
    collapsed_by_default: bool = False
    icon: Optional[str] = None
    order: int = 0


class ComponentConfigSchema(BaseModel):
    """Schema for component configuration with fields."""
    component_type: str
    name: str
    description: str
    category: str
    icon: str
    color: str
    input_ports: List[PortSchema]
    output_ports: List[PortSchema]
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    field_groups: List[FieldGroupSchema] = Field(default_factory=list)


class ComponentDetailResponse(BaseModel):
    """Response schema for a single component."""
    component_type: str
    config: ComponentConfigSchema


class ComponentListResponse(BaseModel):
    """Response schema for listing components."""
    components: List[ComponentDetailResponse]
    total: int


class ComponentCategoryResponse(BaseModel):
    """Response schema for component categories."""
    categories: List[str]
    components_by_category: Dict[str, List[ComponentDetailResponse]]


class FieldTypeInfoSchema(BaseModel):
    """Schema for field type information."""
    type: str
    class_name: str
    description: Optional[str] = None
    default_properties: Dict[str, Any] = Field(default_factory=dict)
    input_component: str


class FieldTypesListResponse(BaseModel):
    """Response schema for listing field types."""
    field_types: List[FieldTypeInfoSchema]
    total: int


class ValidateFieldsRequest(BaseModel):
    """Request schema for validating fields."""
    fields: List[Dict[str, Any]]
    values: Dict[str, Any]


class ValidationResultSchema(BaseModel):
    """Schema for validation result."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    field_name: str


class ValidateFieldsResponse(BaseModel):
    """Response schema for field validation."""
    valid: bool
    results: Dict[str, ValidationResultSchema]
    all_errors: List[str]
