"""
Edge API routes - validates connections between components.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any

from components.registry import ComponentRegistry
from .schemas import (
    EdgeValidationRequest,
    EdgeValidationResponse,
    PortCompatibilityInfo
)


router = APIRouter(prefix="/edges", tags=["Edges"])


# Get registry instance
registry = ComponentRegistry()


# Define data type compatibility rules
COMPATIBLE_TYPES = {
    "any": ["any", "message", "string", "dict", "messages", "tools", "llm_model", "memory"],
    "message": ["message", "any", "string"],
    "messages": ["messages", "any", "message"],
    "string": ["string", "any", "message"],
    "dict": ["dict", "any"],
    "tools": ["tools", "any"],
    "llm_model": ["llm_model", "any"],
    "memory": ["memory", "any", "messages"],
    "tool_calls": ["tool_calls", "any", "dict"],
}


def check_type_compatibility(source_type: str, target_type: str) -> tuple[bool, str]:
    """
    Check if two port data types are compatible.
    
    Args:
        source_type: Data type of the source port
        target_type: Data type of the target port
        
    Returns:
        Tuple of (is_compatible, reason)
    """
    # Any type is always compatible
    if source_type == "any" or target_type == "any":
        return True, "Compatible (any type)"
    
    # Check direct compatibility
    compatible_with = COMPATIBLE_TYPES.get(source_type, [source_type])
    if target_type in compatible_with:
        return True, f"Compatible ({source_type} -> {target_type})"
    
    # Check reverse compatibility
    target_compatible = COMPATIBLE_TYPES.get(target_type, [target_type])
    if source_type in target_compatible:
        return True, f"Compatible ({source_type} -> {target_type})"
    
    return False, f"Incompatible types: {source_type} cannot connect to {target_type}"


@router.post("/validate", response_model=EdgeValidationResponse)
async def validate_edge(request: EdgeValidationRequest):
    """
    Validate if an edge connection between two components is valid.
    
    Checks:
    - Source component exists and has the specified output port
    - Target component exists and has the specified input port
    - Data types are compatible
    """
    errors = []
    warnings = []
    
    # Get source component
    source_class = registry.get(request.source_component_type)
    if not source_class:
        errors.append(f"Source component type '{request.source_component_type}' not found")
    
    # Get target component
    target_class = registry.get(request.target_component_type)
    if not target_class:
        errors.append(f"Target component type '{request.target_component_type}' not found")
    
    if errors:
        return EdgeValidationResponse(valid=False, errors=errors, warnings=warnings)
    
    # Get component configs
    source_config = source_class.get_config()
    target_config = target_class.get_config()
    
    # Find source port
    source_port = None
    for port in source_config.output_ports:
        if port.id == request.source_port:
            source_port = port
            break
    
    if not source_port:
        errors.append(
            f"Output port '{request.source_port}' not found on component '{request.source_component_type}'"
        )
    
    # Find target port
    target_port = None
    for port in target_config.input_ports:
        if port.id == request.target_port:
            target_port = port
            break
    
    if not target_port:
        errors.append(
            f"Input port '{request.target_port}' not found on component '{request.target_component_type}'"
        )
    
    if errors:
        return EdgeValidationResponse(valid=False, errors=errors, warnings=warnings)
    
    # Check type compatibility
    compatible, reason = check_type_compatibility(
        source_port.data_type,
        target_port.data_type
    )
    
    if not compatible:
        errors.append(reason)
    
    # Add warnings for potential issues
    if source_port.data_type != target_port.data_type and compatible:
        warnings.append(
            f"Type conversion may occur: {source_port.data_type} -> {target_port.data_type}"
        )
    
    return EdgeValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


@router.get("/compatible-ports/{component_type}")
async def get_compatible_ports(component_type: str):
    """
    Get all ports and their compatible connection targets for a component.
    """
    component_class = registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    config = component_class.get_config()
    
    # Build compatibility map for input ports
    input_compatibility = {}
    for port in config.input_ports:
        compatible_types = COMPATIBLE_TYPES.get(port.data_type, [port.data_type])
        input_compatibility[port.id] = {
            "port": {
                "id": port.id,
                "name": port.name,
                "data_type": port.data_type,
                "required": port.required,
                "multiple": port.multiple
            },
            "compatible_source_types": compatible_types
        }
    
    # Build compatibility map for output ports
    output_compatibility = {}
    for port in config.output_ports:
        # Find all target types this port can connect to
        compatible_targets = []
        for target_type, sources in COMPATIBLE_TYPES.items():
            if port.data_type in sources or port.data_type == "any":
                compatible_targets.append(target_type)
        
        output_compatibility[port.id] = {
            "port": {
                "id": port.id,
                "name": port.name,
                "data_type": port.data_type
            },
            "compatible_target_types": compatible_targets
        }
    
    return {
        "component_type": component_type,
        "input_ports": input_compatibility,
        "output_ports": output_compatibility
    }


@router.post("/check-compatibility")
async def check_port_compatibility(
    source_data_type: str,
    target_data_type: str
):
    """
    Check if two data types are compatible for connection.
    """
    compatible, reason = check_type_compatibility(source_data_type, target_data_type)
    
    return PortCompatibilityInfo(
        source_data_type=source_data_type,
        target_data_type=target_data_type,
        compatible=compatible,
        reason=reason
    )
