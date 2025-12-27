"""
Input component - entry point for user-provided data.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List
from datetime import datetime

from .base import (
    BaseComponent,
    ComponentPort,
    PortType,
    FieldDefinition,
    FieldGroup,
    StringField,
    BooleanField,
)
from .field_types import JsonField


class InputComponent(BaseComponent):
    """
    Entry node that injects user input into the flow.
    
    Input nodes act as entry points, so the executor can pre-seed their
    outputs with request data before downstream components run.
    """
    
    _component_type = "input"
    _name = "Input"
    _description = "Flow entry point for user-provided data"
    _category = "io"
    _icon = "log-in"
    _color = "#22c55e"
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Input nodes have no upstream connections."""
        return []
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Outputs user input and optional metadata."""
        return [
            ComponentPort(
                id="output",
                name="Output",
                type=PortType.OUTPUT,
                data_type="message",
                required=True,
                description="User-provided value"
            ),
            ComponentPort(
                id="metadata",
                name="Metadata",
                type=PortType.OUTPUT,
                data_type="dict",
                description="Input metadata and context"
            ),
        ]
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields for the input node."""
        return [
            StringField.create(
                name="label",
                label="Label",
                description="Display label for this input source",
                default="User Input",
                required=True,
                order=1,
                group="settings",
            ),
            StringField.create(
                name="placeholder",
                label="Placeholder",
                description="Helper text shown in the UI",
                default="Enter your message",
                order=2,
                group="settings",
            ),
            JsonField.create(
                name="default_value",
                label="Default Value",
                description="Value to use when no input is provided",
                default=None,
                order=3,
                group="settings",
            ),
            BooleanField.create(
                name="include_metadata",
                label="Include Metadata",
                description="Attach timestamp and label to the output",
                default=True,
                order=4,
                group="advanced",
            ),
        ]
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Groups for organizing input configuration."""
        return [
            FieldGroup(
                id="settings",
                label="Settings",
                description="Basic input configuration",
                order=0,
            ),
            FieldGroup(
                id="advanced",
                label="Advanced",
                description="Additional metadata options",
                collapsible=True,
                collapsed_by_default=True,
                order=1,
            ),
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the incoming value or fall back to the configured default.
        """
        value = inputs.get("input")
        if value is None:
            value = inputs.get("value")
        if value is None:
            value = self.parameters.get("default_value")
        
        metadata: Dict[str, Any] = {}
        if self.parameters.get("include_metadata", True):
            metadata = {
                "label": self.parameters.get("label", "User Input"),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "user_input",
            }
        
        return {
            "output": value,
            "metadata": metadata,
        }
