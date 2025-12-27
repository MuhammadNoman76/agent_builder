"""
Pydantic models for MongoDB documents.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


def generate_uuid() -> str:
    """Generate a unique UUID string."""
    return str(uuid.uuid4())


class ComponentType(str, Enum):
    """Enumeration of available component types."""
    INPUT = "input"
    OUTPUT = "output"
    AGENT = "agent"
    LLM_MODEL = "llm_model"
    COMPOSIO_TOOL = "composio_tool"


class ConnectionPort(str, Enum):
    """Enumeration of connection port types."""
    INPUT = "input"
    OUTPUT = "output"
    MODEL = "model"
    TOOLS = "tools"
    SYSTEM_PROMPT = "system_prompt"


class UserModel(BaseModel):
    """User document model."""
    id: str = Field(default_factory=generate_uuid, alias="_id")
    username: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True


class NodePosition(BaseModel):
    """Position of a node in the canvas."""
    x: float = 0.0
    y: float = 0.0


class NodeData(BaseModel):
    """Data associated with a node."""
    label: str
    component_type: ComponentType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class NodeModel(BaseModel):
    """Node document model."""
    node_id: str = Field(default_factory=generate_uuid)
    flow_id: str
    type: str = "custom"
    position: NodePosition = Field(default_factory=NodePosition)
    data: NodeData
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EdgeModel(BaseModel):
    """Edge document model representing connections between nodes."""
    edge_id: str = Field(default_factory=generate_uuid)
    flow_id: str
    source: str  # Source node ID
    target: str  # Target node ID
    source_handle: Optional[str] = None  # Port on source node
    target_handle: Optional[str] = None  # Port on target node
    animated: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FlowModel(BaseModel):
    """Flow document model - represents an agent workflow."""
    flow_id: str = Field(default_factory=generate_uuid)
    user_id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    agent_schema: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
