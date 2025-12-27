"""
Composio Tool component - integrates Composio tools into the agent.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List, Optional
from .base import (
    BaseComponent,
    ComponentPort,
    PortType,
    FieldDefinition,
    FieldGroup,
)
from .field_types import (
    ApiKeyField,
    SelectField,
    StringField,
    BooleanField,
    JsonField,
)


class ComposioToolComponent(BaseComponent):
    """
    Composio tool component for integrating external services.
    
    This component allows agents to use Composio tools like Gmail,
    Slack, GitHub, and other services through the Composio platform.
    """
    
    _component_type = "composio_tool"
    _name = "Composio Tool"
    _description = "Integrate external services via Composio"
    _category = "tools"
    _icon = "plug"
    _color = "#ec4899"
    
    # Common Composio toolkits
    _available_toolkits = [
        {"value": "GMAIL", "label": "Gmail"},
        {"value": "SLACK", "label": "Slack"},
        {"value": "GITHUB", "label": "GitHub"},
        {"value": "NOTION", "label": "Notion"},
        {"value": "GOOGLE_CALENDAR", "label": "Google Calendar"},
        {"value": "GOOGLE_DRIVE", "label": "Google Drive"},
        {"value": "GOOGLE_SHEETS", "label": "Google Sheets"},
        {"value": "TWITTER", "label": "Twitter/X"},
        {"value": "LINKEDIN", "label": "LinkedIn"},
        {"value": "DISCORD", "label": "Discord"},
        {"value": "TRELLO", "label": "Trello"},
        {"value": "ASANA", "label": "Asana"},
        {"value": "JIRA", "label": "Jira"},
        {"value": "SALESFORCE", "label": "Salesforce"},
        {"value": "HUBSPOT", "label": "HubSpot"},
    ]
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Define input ports for Composio component."""
        return [
            ComponentPort(
                id="config",
                name="Configuration",
                type=PortType.INPUT,
                data_type="dict",
                description="Tool configuration and parameters"
            )
        ]
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Define output ports for Composio component."""
        return [
            ComponentPort(
                id="tools",
                name="Tools",
                type=PortType.OUTPUT,
                data_type="tools",
                description="Composio tools for agent to use"
            ),
            ComponentPort(
                id="result",
                name="Result",
                type=PortType.OUTPUT,
                data_type="dict",
                description="Direct tool execution result"
            )
        ]
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields using field types."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="Composio API Key",
                provider="composio",
                description="Your Composio API key",
                required=True,
                placeholder="...",
                order=1,
                group="connection",
            ),
            SelectField.create(
                name="toolkit",
                label="Toolkit",
                description="Composio toolkit to use",
                options=cls._available_toolkits,
                required=True,
                order=2,
                group="toolkit",
            ),
            JsonField.create(
                name="tools",
                label="Specific Tools",
                description="List of specific tools to include (optional)",
                default=[],
                placeholder='["GMAIL_SEND_EMAIL", "GMAIL_FETCH_EMAILS"]',
                order=3,
                group="toolkit",
            ),
            StringField.create(
                name="user_id",
                label="External User ID",
                description="User ID for Composio connection",
                default="default",
                order=4,
                group="auth",
            ),
            StringField.create(
                name="auth_config_id",
                label="Auth Config ID",
                description="Authentication configuration ID (optional)",
                placeholder="<authConfigId>",
                order=5,
                group="auth",
            ),
            BooleanField.create(
                name="auto_connect",
                label="Auto Connect",
                description="Automatically initiate connection if needed",
                default=True,
                order=6,
                group="options",
            ),
        ]
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organization."""
        return [
            FieldGroup(
                id="connection",
                label="Connection",
                description="Composio API connection",
                order=0,
            ),
            FieldGroup(
                id="toolkit",
                label="Toolkit Settings",
                description="Select toolkit and tools",
                order=1,
            ),
            FieldGroup(
                id="auth",
                label="Authentication",
                description="User authentication settings",
                collapsible=True,
                order=2,
            ),
            FieldGroup(
                id="options",
                label="Options",
                collapsible=True,
                order=3,
            ),
        ]
    
    async def _initialize_composio(self) -> None:
        """Initialize Composio client and tools."""
        from composio import Composio
        from composio_langchain import LangchainProvider
        
        self._composio = Composio(
            api_key=self.get_parameter("api_key"),
            provider=LangchainProvider()
        )
        
        user_id = self.get_parameter("user_id", "default")
        toolkit = self.get_parameter("toolkit")
        specific_tools = self.get_parameter("tools", [])
        
        # Get tools from Composio
        if specific_tools:
            self._tools = self._composio.tools.get(
                user_id=user_id,
                tools=specific_tools
            )
        else:
            self._tools = self._composio.tools.get(
                user_id=user_id,
                toolkits=[toolkit]
            )
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Composio component."""
        # Initialize if needed
        if not hasattr(self, "_composio") or self._composio is None:
            await self._initialize_composio()
        
        # Check for connection
        user_id = self.get_parameter("user_id", "default")
        toolkit = self.get_parameter("toolkit")
        
        try:
            # Try to get connected account
            connected_accounts = self._composio.connected_accounts.list(
                user_id=user_id
            )
            
            toolkit_connected = any(
                acc.toolkit == toolkit for acc in connected_accounts
            )
            
            if not toolkit_connected and self.get_parameter("auto_connect", True):
                # Initiate connection
                auth_config_id = self.get_parameter("auth_config_id")
                connection_request = self._composio.connected_accounts.link(
                    user_id=user_id,
                    auth_config_id=auth_config_id,
                )
                
                return {
                    "tools": [],
                    "result": {
                        "status": "connection_required",
                        "redirect_url": connection_request.redirect_url,
                        "message": f"Please authorize {toolkit} by visiting the URL"
                    }
                }
        except Exception:
            # Continue even if check fails - tools might still work
            pass
        
        return {
            "tools": self._tools,
            "result": {
                "status": "ready",
                "toolkit": toolkit,
                "tool_count": len(self._tools)
            }
        }
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema with Composio-specific fields."""
        schema = super().to_schema()
        schema["composio_config"] = {
            "toolkit": self.get_parameter("toolkit"),
            "tools": self.get_parameter("tools", []),
            "user_id": self.get_parameter("user_id", "default"),
            "auto_connect": self.get_parameter("auto_connect", True)
        }
        return schema
