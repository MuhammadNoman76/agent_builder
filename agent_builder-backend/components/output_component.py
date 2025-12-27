"""
Output component - handles agent output/response.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List
from .base import (
    BaseComponent,
    ComponentPort,
    PortType,
    FieldDefinition,
    FieldGroup,
    SelectField,
    StringField,
    BooleanField,
    IntegerField,
)


class OutputComponent(BaseComponent):
    """
    Output component that formats and returns agent responses.
    
    This component serves as the exit point for the agent workflow.
    """
    
    _component_type = "output"
    _name = "Output"
    _description = "Formats and returns the final agent response"
    _category = "io"
    _icon = "message-circle"
    _color = "#f59e0b"
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Define input ports for the output component."""
        return [
            ComponentPort(
                id="input",
                name="Input",
                type=PortType.INPUT,
                data_type="message",
                required=True,
                description="Response from agent to output"
            ),
            ComponentPort(
                id="metadata",
                name="Metadata",
                type=PortType.INPUT,
                data_type="dict",
                description="Additional metadata to include in output"
            )
        ]
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Output component has no output ports (it's the sink)."""
        return []
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields using field types."""
        return [
            SelectField.create(
                name="output_format",
                label="Output Format",
                description="Format of the output response",
                options=[
                    {"value": "text", "label": "Plain Text"},
                    {"value": "json", "label": "JSON"},
                    {"value": "markdown", "label": "Markdown"},
                    {"value": "html", "label": "HTML"},
                ],
                default="text",
                order=1,
                group="basic",
            ),
            StringField.create(
                name="variable_name",
                label="Variable Name",
                description="Name of the output variable",
                default="agent_response",
                order=2,
                group="basic",
            ),
            BooleanField.create(
                name="include_metadata",
                label="Include Metadata",
                description="Include metadata in the response",
                default=False,
                order=3,
                group="options",
            ),
            BooleanField.create(
                name="stream",
                label="Stream Response",
                description="Enable streaming for the response",
                default=False,
                order=4,
                group="options",
            ),
            IntegerField.create(
                name="max_length",
                label="Max Length",
                description="Maximum length of output (0 for unlimited)",
                default=0,
                min_value=0,
                order=5,
                group="advanced",
            ),
        ]
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organization."""
        return [
            FieldGroup(
                id="basic",
                label="Basic Settings",
                order=0,
            ),
            FieldGroup(
                id="options",
                label="Options",
                collapsible=True,
                order=1,
            ),
            FieldGroup(
                id="advanced",
                label="Advanced",
                collapsible=True,
                collapsed_by_default=True,
                order=2,
            ),
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format the final output."""
        input_data = inputs.get("input", {})
        metadata = inputs.get("metadata", {})
        
        # Extract content from input
        if isinstance(input_data, dict):
            content = input_data.get("content", str(input_data))
        else:
            content = str(input_data)
        
        # Apply max length if set
        max_length = self.get_parameter("max_length", 0)
        if max_length > 0 and len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Format output
        output_format = self.get_parameter("output_format", "text")
        formatted_output = self._format_output(content, output_format)
        
        # Build response
        response = {
            "content": formatted_output,
            "format": output_format,
            "variable_name": self.get_parameter("variable_name", "agent_response")
        }
        
        if self.get_parameter("include_metadata", False):
            response["metadata"] = metadata
        
        return response
    
    def _format_output(self, content: str, format_type: str) -> Any:
        """Format content based on output type."""
        if format_type == "json":
            import json
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return {"response": content}
        elif format_type == "html":
            import html
            return f"<p>{html.escape(content)}</p>"
        return content
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema with output-specific fields."""
        schema = super().to_schema()
        schema["role"] = "exit_point"
        schema["output_config"] = {
            "format": self.get_parameter("output_format", "text"),
            "stream": self.get_parameter("stream", False),
            "variable_name": self.get_parameter("variable_name", "agent_response")
        }
        return schema
