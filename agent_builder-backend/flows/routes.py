"""
Flow API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional
import json

from database import get_database
from auth.utils import get_current_user
from .schemas import (
    FlowCreate,
    FlowUpdate,
    FlowResponse,
    FlowListResponse,
    FlowExecuteRequest,
    FlowExecuteResponse,
    AgentSchemaResponse,
    NodeCreate,
    NodeUpdate,
    EdgeCreate
)
from .services import FlowService
from .executor import FlowExecutorFactory


router = APIRouter(prefix="/flows", tags=["Flows"])


# Helper to get flow service
async def get_flow_service(
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> FlowService:
    return FlowService(db)


@router.post("", response_model=FlowResponse, status_code=status.HTTP_201_CREATED)
async def create_flow(
    flow_data: FlowCreate,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Create a new flow/agent workflow."""
    flow = await service.create_flow(
        user_id=current_user["_id"],
        flow_data=flow_data.model_dump()
    )
    return FlowResponse(**flow)


@router.get("", response_model=FlowListResponse)
async def list_flows(
    page: int = 1,
    page_size: int = 10,
    is_active: Optional[bool] = None,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """List all flows for the current user."""
    result = await service.list_flows(
        user_id=current_user["_id"],
        page=page,
        page_size=page_size,
        is_active=is_active
    )
    
    return FlowListResponse(
        flows=[FlowResponse(**f) for f in result["flows"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"]
    )


@router.get("/{flow_id}", response_model=FlowResponse)
async def get_flow(
    flow_id: str,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Get a specific flow by ID."""
    flow = await service.get_flow(flow_id, current_user["_id"])
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    return FlowResponse(**flow)


@router.put("/{flow_id}", response_model=FlowResponse)
async def update_flow(
    flow_id: str,
    update_data: FlowUpdate,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Update a flow."""
    flow = await service.update_flow(
        flow_id=flow_id,
        user_id=current_user["_id"],
        update_data=update_data.model_dump(exclude_unset=True)
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    return FlowResponse(**flow)


@router.delete("/{flow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_flow(
    flow_id: str,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Delete a flow."""
    deleted = await service.delete_flow(flow_id, current_user["_id"])
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )


@router.get("/{flow_id}/schema", response_model=AgentSchemaResponse)
async def get_flow_schema(
    flow_id: str,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Get the generated agent schema for a flow."""
    schema_info = await service.get_agent_schema(flow_id, current_user["_id"])
    
    if not schema_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    return AgentSchemaResponse(**schema_info)


@router.post("/{flow_id}/execute", response_model=FlowExecuteResponse)
async def execute_flow(
    flow_id: str,
    request: FlowExecuteRequest,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Execute a flow/agent workflow."""
    flow = await service.get_flow(flow_id, current_user["_id"])
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    agent_schema = flow.get("agent_schema", {})
    
    # Check if schema is valid
    if not agent_schema.get("validation", {}).get("is_valid", False):
        return FlowExecuteResponse(
            flow_id=flow_id,
            status="error",
            error="Flow schema is not valid. Please fix validation errors before executing."
        )
    
    # Execute the flow
    try:
        result = await FlowExecutorFactory.execute_flow(
            agent_schema=agent_schema,
            input_data=request.input_data,
            stream=False
        )
        
        return FlowExecuteResponse(
            flow_id=flow_id,
            status=result.get("status", "unknown"),
            output=result.get("output"),
            error=result.get("errors", [None])[0] if result.get("errors") else None,
            execution_time=result.get("execution_time")
        )
    except Exception as e:
        return FlowExecuteResponse(
            flow_id=flow_id,
            status="error",
            error=str(e)
        )


@router.post("/{flow_id}/execute/stream")
async def execute_flow_stream(
    flow_id: str,
    request: FlowExecuteRequest,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Execute a flow with streaming response."""
    flow = await service.get_flow(flow_id, current_user["_id"])
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    agent_schema = flow.get("agent_schema", {})
    
    async def generate():
        executor = FlowExecutorFactory.create(agent_schema)
        async for event in executor.execute_stream(request.input_data):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Node operations within a flow
@router.post("/{flow_id}/nodes", response_model=FlowResponse)
async def add_node(
    flow_id: str,
    node_data: NodeCreate,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Add a node to a flow."""
    flow = await service.add_node(
        flow_id=flow_id,
        user_id=current_user["_id"],
        node_data=node_data.model_dump()
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    return FlowResponse(**flow)


@router.put("/{flow_id}/nodes/{node_id}", response_model=FlowResponse)
async def update_node(
    flow_id: str,
    node_id: str,
    update_data: NodeUpdate,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Update a node in a flow."""
    flow = await service.update_node(
        flow_id=flow_id,
        user_id=current_user["_id"],
        node_id=node_id,
        update_data=update_data.model_dump(exclude_unset=True)
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow or node not found"
        )
    
    return FlowResponse(**flow)


@router.delete("/{flow_id}/nodes/{node_id}", response_model=FlowResponse)
async def delete_node(
    flow_id: str,
    node_id: str,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Delete a node from a flow (also removes connected edges)."""
    flow = await service.delete_node(
        flow_id=flow_id,
        user_id=current_user["_id"],
        node_id=node_id
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow or node not found"
        )
    
    return FlowResponse(**flow)


# Edge operations within a flow
@router.post("/{flow_id}/edges", response_model=FlowResponse)
async def add_edge(
    flow_id: str,
    edge_data: EdgeCreate,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Add an edge (connection) to a flow."""
    flow = await service.add_edge(
        flow_id=flow_id,
        user_id=current_user["_id"],
        edge_data=edge_data.model_dump()
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow not found"
        )
    
    return FlowResponse(**flow)


@router.delete("/{flow_id}/edges/{edge_id}", response_model=FlowResponse)
async def delete_edge(
    flow_id: str,
    edge_id: str,
    current_user: dict = Depends(get_current_user),
    service: FlowService = Depends(get_flow_service)
):
    """Delete an edge from a flow."""
    flow = await service.delete_edge(
        flow_id=flow_id,
        user_id=current_user["_id"],
        edge_id=edge_id
    )
    
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flow or edge not found"
        )
    
    return FlowResponse(**flow)
