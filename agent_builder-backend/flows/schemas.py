"""
Flow schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class NodeCreate(BaseModel):
    """Schema for creating a node."""
    node_id: Optional[str] = None
    type: str = "custom"
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: Dict[str, Any]


class NodeUpdate(BaseModel):
    """Schema for updating a node."""
    position: Optional[Dict[str, float]] = None
    data: Optional[Dict[str, Any]] = None


class EdgeCreate(BaseModel):
    """Schema for creating an edge."""
    edge_id: Optional[str] = None
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    animated: bool = True


class FlowCreate(BaseModel):
    """Schema for creating a flow."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    nodes: List[NodeCreate] = Field(default_factory=list)
    edges: List[EdgeCreate] = Field(default_factory=list)


class FlowUpdate(BaseModel):
    """Schema for updating a flow."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    nodes: Optional[List[NodeCreate]] = None
    edges: Optional[List[EdgeCreate]] = None
    is_active: Optional[bool] = None


class FlowResponse(BaseModel):
    """Schema for flow response."""
    flow_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    agent_schema: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FlowListResponse(BaseModel):
    """Schema for listing flows."""
    flows: List[FlowResponse]
    total: int
    page: int
    page_size: int


class FlowExecuteRequest(BaseModel):
    """Schema for executing a flow."""
    input_data: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class FlowExecuteResponse(BaseModel):
    """Schema for flow execution response."""
    flow_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class AgentSchemaResponse(BaseModel):
    """Schema for agent schema response."""
    flow_id: str
    agent_schema: Dict[str, Any]
    validation_errors: List[str]
    is_valid: bool
