"""
Edge schemas for API validation.
"""
from pydantic import BaseModel
from typing import Optional, List


class EdgeValidationRequest(BaseModel):
    """Request schema for validating an edge connection."""
    source_component_type: str
    source_port: str
    target_component_type: str
    target_port: str


class EdgeValidationResponse(BaseModel):
    """Response schema for edge validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]


class PortCompatibilityInfo(BaseModel):
    """Information about port compatibility."""
    source_data_type: str
    target_data_type: str
    compatible: bool
    reason: Optional[str] = None
