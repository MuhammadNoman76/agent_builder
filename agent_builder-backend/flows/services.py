"""
Flow service - business logic for flow operations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import uuid

from database.models import FlowModel, NodeModel, EdgeModel
from components.registry import ComponentRegistry


class FlowService:
    """
    Service class for flow operations.
    
    Handles CRUD operations, schema generation, and validation for flows.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = db
        self.registry = ComponentRegistry()
    
    async def create_flow(self, user_id: str, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new flow.
        
        Args:
            user_id: The owner's user ID
            flow_data: Flow creation data
            
        Returns:
            Created flow document
        """
        flow_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Process nodes
        nodes = []
        for node_data in flow_data.get("nodes", []):
            node = {
                "node_id": node_data.get("node_id") or str(uuid.uuid4()),
                "flow_id": flow_id,
                "type": node_data.get("type", "custom"),
                "position": node_data.get("position", {"x": 0, "y": 0}),
                "data": node_data.get("data", {}),
                "created_at": now,
                "updated_at": now
            }
            nodes.append(node)
        
        # Process edges
        edges = []
        for edge_data in flow_data.get("edges", []):
            edge = {
                "edge_id": edge_data.get("edge_id") or str(uuid.uuid4()),
                "flow_id": flow_id,
                "source": edge_data.get("source"),
                "target": edge_data.get("target"),
                "source_handle": edge_data.get("source_handle"),
                "target_handle": edge_data.get("target_handle"),
                "animated": edge_data.get("animated", True),
                "created_at": now
            }
            edges.append(edge)
        
        # Generate agent schema
        agent_schema = self._generate_agent_schema(nodes, edges)
        
        # Create flow document
        flow = {
            "flow_id": flow_id,
            "user_id": user_id,
            "name": flow_data.get("name"),
            "description": flow_data.get("description"),
            "nodes": nodes,
            "edges": edges,
            "agent_schema": agent_schema,
            "is_active": True,
            "created_at": now,
            "updated_at": now
        }
        
        await self.db.flows.insert_one(flow)
        return flow
    
    async def get_flow(self, flow_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a flow by ID.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            
        Returns:
            Flow document or None
        """
        return await self.db.flows.find_one({
            "flow_id": flow_id,
            "user_id": user_id
        })
    
    async def list_flows(
        self, 
        user_id: str, 
        page: int = 1, 
        page_size: int = 10,
        is_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        List flows for a user with pagination.
        
        Args:
            user_id: The owner's user ID
            page: Page number (1-indexed)
            page_size: Number of items per page
            is_active: Filter by active status
            
        Returns:
            Paginated list of flows
        """
        query = {"user_id": user_id}
        if is_active is not None:
            query["is_active"] = is_active
        
        total = await self.db.flows.count_documents(query)
        skip = (page - 1) * page_size
        
        cursor = self.db.flows.find(query).skip(skip).limit(page_size).sort("updated_at", -1)
        flows = await cursor.to_list(length=page_size)
        
        return {
            "flows": flows,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    async def update_flow(
        self, 
        flow_id: str, 
        user_id: str, 
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            update_data: Fields to update
            
        Returns:
            Updated flow document or None
        """
        now = datetime.utcnow()
        
        # Build update document
        update_doc = {"$set": {"updated_at": now}}
        
        if "name" in update_data:
            update_doc["$set"]["name"] = update_data["name"]
        
        if "description" in update_data:
            update_doc["$set"]["description"] = update_data["description"]
        
        if "is_active" in update_data:
            update_doc["$set"]["is_active"] = update_data["is_active"]
        
        if "nodes" in update_data:
            nodes = []
            for node_data in update_data["nodes"]:
                node = {
                    "node_id": node_data.get("node_id") or str(uuid.uuid4()),
                    "flow_id": flow_id,
                    "type": node_data.get("type", "custom"),
                    "position": node_data.get("position", {"x": 0, "y": 0}),
                    "data": node_data.get("data", {}),
                    "created_at": now,
                    "updated_at": now
                }
                nodes.append(node)
            update_doc["$set"]["nodes"] = nodes
        
        if "edges" in update_data:
            edges = []
            for edge_data in update_data["edges"]:
                edge = {
                    "edge_id": edge_data.get("edge_id") or str(uuid.uuid4()),
                    "flow_id": flow_id,
                    "source": edge_data.get("source"),
                    "target": edge_data.get("target"),
                    "source_handle": edge_data.get("source_handle"),
                    "target_handle": edge_data.get("target_handle"),
                    "animated": edge_data.get("animated", True),
                    "created_at": now
                }
                edges.append(edge)
            update_doc["$set"]["edges"] = edges
        
        # Regenerate schema if nodes or edges changed
        if "nodes" in update_data or "edges" in update_data:
            flow = await self.get_flow(flow_id, user_id)
            if flow:
                nodes_for_schema = update_data.get("nodes")
                edges_for_schema = update_data.get("edges")
                
                # If only nodes updated, get existing edges
                if nodes_for_schema and not edges_for_schema:
                    edges_for_schema = flow.get("edges", [])
                # If only edges updated, get existing nodes
                elif edges_for_schema and not nodes_for_schema:
                    nodes_for_schema = flow.get("nodes", [])
                
                # Process nodes for schema generation
                processed_nodes = []
                for node_data in nodes_for_schema:
                    if isinstance(node_data, dict) and "node_id" in node_data:
                        processed_nodes.append(node_data)
                    else:
                        processed_nodes.append({
                            "node_id": node_data.get("node_id") or str(uuid.uuid4()),
                            "flow_id": flow_id,
                            "type": node_data.get("type", "custom"),
                            "position": node_data.get("position", {"x": 0, "y": 0}),
                            "data": node_data.get("data", {}),
                        })
                
                agent_schema = self._generate_agent_schema(processed_nodes, edges_for_schema)
                update_doc["$set"]["agent_schema"] = agent_schema
        
        result = await self.db.flows.find_one_and_update(
            {"flow_id": flow_id, "user_id": user_id},
            update_doc,
            return_document=True
        )
        
        return result
    
    async def delete_flow(self, flow_id: str, user_id: str) -> bool:
        """
        Delete a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            
        Returns:
            True if deleted, False otherwise
        """
        result = await self.db.flows.delete_one({
            "flow_id": flow_id,
            "user_id": user_id
        })
        return result.deleted_count > 0
    
    async def add_node(
        self, 
        flow_id: str, 
        user_id: str, 
        node_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Add a node to a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            node_data: Node data
            
        Returns:
            Updated flow or None
        """
        now = datetime.utcnow()
        
        node = {
            "node_id": node_data.get("node_id") or str(uuid.uuid4()),
            "flow_id": flow_id,
            "type": node_data.get("type", "custom"),
            "position": node_data.get("position", {"x": 0, "y": 0}),
            "data": node_data.get("data", {}),
            "created_at": now,
            "updated_at": now
        }
        
        result = await self.db.flows.find_one_and_update(
            {"flow_id": flow_id, "user_id": user_id},
            {
                "$push": {"nodes": node},
                "$set": {"updated_at": now}
            },
            return_document=True
        )
        
        if result:
            # Regenerate schema
            await self._regenerate_schema(flow_id, user_id)
            # Fetch updated flow with new schema
            result = await self.get_flow(flow_id, user_id)
        
        return result
    
    async def update_node(
        self, 
        flow_id: str, 
        user_id: str, 
        node_id: str, 
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update a specific node in a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            node_id: The node ID
            update_data: Fields to update
            
        Returns:
            Updated flow or None
        """
        now = datetime.utcnow()
        
        # Build update document for array element
        update_fields = {"nodes.$.updated_at": now, "updated_at": now}
        
        if "position" in update_data:
            update_fields["nodes.$.position"] = update_data["position"]
        
        if "data" in update_data:
            update_fields["nodes.$.data"] = update_data["data"]
        
        result = await self.db.flows.find_one_and_update(
            {
                "flow_id": flow_id,
                "user_id": user_id,
                "nodes.node_id": node_id
            },
            {"$set": update_fields},
            return_document=True
        )
        
        if result:
            # Regenerate schema
            await self._regenerate_schema(flow_id, user_id)
            # Fetch updated flow with new schema
            result = await self.get_flow(flow_id, user_id)
        
        return result
    
    async def delete_node(
        self, 
        flow_id: str, 
        user_id: str, 
        node_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Delete a node from a flow (also removes connected edges).
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            node_id: The node ID
            
        Returns:
            Updated flow or None
        """
        now = datetime.utcnow()
        
        # First, get the flow to find edges to remove
        flow = await self.get_flow(flow_id, user_id)
        if not flow:
            return None
        
        # Find edges connected to this node
        edges_to_remove = [
            edge["edge_id"] for edge in flow.get("edges", [])
            if edge["source"] == node_id or edge["target"] == node_id
        ]
        
        # Remove the node
        result = await self.db.flows.find_one_and_update(
            {"flow_id": flow_id, "user_id": user_id},
            {
                "$pull": {"nodes": {"node_id": node_id}},
                "$set": {"updated_at": now}
            },
            return_document=True
        )
        
        # Remove connected edges
        if edges_to_remove:
            await self.db.flows.update_one(
                {"flow_id": flow_id, "user_id": user_id},
                {
                    "$pull": {"edges": {"edge_id": {"$in": edges_to_remove}}}
                }
            )
        
        if result:
            # Regenerate schema
            await self._regenerate_schema(flow_id, user_id)
            # Fetch updated flow with new schema
            result = await self.get_flow(flow_id, user_id)
        
        return result
    
    async def add_edge(
        self, 
        flow_id: str, 
        user_id: str, 
        edge_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Add an edge to a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            edge_data: Edge data
            
        Returns:
            Updated flow or None
        """
        now = datetime.utcnow()
        
        # Validate that source and target nodes exist
        flow = await self.get_flow(flow_id, user_id)
        if not flow:
            return None
        
        node_ids = {node["node_id"] for node in flow.get("nodes", [])}
        source = edge_data.get("source")
        target = edge_data.get("target")
        
        if source not in node_ids:
            raise ValueError(f"Source node '{source}' does not exist in flow")
        if target not in node_ids:
            raise ValueError(f"Target node '{target}' does not exist in flow")
        
        edge = {
            "edge_id": edge_data.get("edge_id") or str(uuid.uuid4()),
            "flow_id": flow_id,
            "source": source,
            "target": target,
            "source_handle": edge_data.get("source_handle", "output"),
            "target_handle": edge_data.get("target_handle", "input"),
            "animated": edge_data.get("animated", True),
            "created_at": now
        }
        
        result = await self.db.flows.find_one_and_update(
            {"flow_id": flow_id, "user_id": user_id},
            {
                "$push": {"edges": edge},
                "$set": {"updated_at": now}
            },
            return_document=True
        )
        
        if result:
            # Regenerate schema
            await self._regenerate_schema(flow_id, user_id)
            # Fetch updated flow with new schema
            result = await self.get_flow(flow_id, user_id)
        
        return result
    
    async def delete_edge(
        self, 
        flow_id: str, 
        user_id: str, 
        edge_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Delete an edge from a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            edge_id: The edge ID
            
        Returns:
            Updated flow or None
        """
        now = datetime.utcnow()
        
        result = await self.db.flows.find_one_and_update(
            {"flow_id": flow_id, "user_id": user_id},
            {
                "$pull": {"edges": {"edge_id": edge_id}},
                "$set": {"updated_at": now}
            },
            return_document=True
        )
        
        if result:
            # Regenerate schema
            await self._regenerate_schema(flow_id, user_id)
            # Fetch updated flow with new schema
            result = await self.get_flow(flow_id, user_id)
        
        return result
    
    async def _regenerate_schema(self, flow_id: str, user_id: str) -> None:
        """Regenerate the agent schema for a flow."""
        flow = await self.get_flow(flow_id, user_id)
        if flow:
            nodes = flow.get("nodes", [])
            edges = flow.get("edges", [])
            agent_schema = self._generate_agent_schema(nodes, edges)
            
            await self.db.flows.update_one(
                {"flow_id": flow_id, "user_id": user_id},
                {"$set": {"agent_schema": agent_schema}}
            )
    
    def _generate_agent_schema(
        self, 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate the agent schema from nodes and edges.
        
        This schema defines how the agent will be built and executed.
        
        Args:
            nodes: List of node documents
            edges: List of edge documents
            
        Returns:
            Agent schema dictionary
        """
        # Handle empty nodes case
        if not nodes:
            return {
                "version": "1.0",
                "entry_points": [],
                "exit_points": [],
                "components": [],
                "connections": {},
                "execution_order": [],
                "validation": {
                    "is_valid": False,
                    "errors": ["Flow has no nodes"]
                }
            }
        
        # Build node map
        node_map = {node["node_id"]: node for node in nodes}
        
        # Build connection graph
        connections = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            source_handle = edge.get("source_handle", "output")
            target_handle = edge.get("target_handle", "input")
            
            if source not in connections:
                connections[source] = []
            
            connections[source].append({
                "target": target,
                "source_port": source_handle,
                "target_port": target_handle
            })
        
        # Identify entry points (nodes with no incoming edges)
        targets = set(edge["target"] for edge in edges)
        sources = set(edge["source"] for edge in edges)
        entry_points = [
            node["node_id"] for node in nodes 
            if node["node_id"] not in targets
        ]
        
        # Identify exit points (nodes with no outgoing edges)
        exit_points = [
            node["node_id"] for node in nodes 
            if node["node_id"] not in sources
        ]
        
        # Build component schemas
        component_schemas = []
        for node in nodes:
            component_type = node.get("data", {}).get("component_type")
            
            if not component_type:
                component_schemas.append({
                    "node_id": node["node_id"],
                    "component_type": None,
                    "parameters": node.get("data", {}).get("parameters", {}),
                    "error": "Missing component_type in node data"
                })
                continue
            
            component_class = self.registry.get(component_type)
            
            if component_class:
                try:
                    instance = component_class(
                        node_id=node["node_id"],
                        parameters=node.get("data", {}).get("parameters", {})
                    )
                    component_schemas.append(instance.to_schema())
                except Exception as e:
                    component_schemas.append({
                        "node_id": node["node_id"],
                        "component_type": component_type,
                        "parameters": node.get("data", {}).get("parameters", {}),
                        "error": f"Failed to instantiate component: {str(e)}"
                    })
            else:
                component_schemas.append({
                    "node_id": node["node_id"],
                    "component_type": component_type,
                    "parameters": node.get("data", {}).get("parameters", {}),
                    "error": f"Unknown component type: {component_type}"
                })
        
        # Validate flow
        validation_errors = self._validate_flow(nodes, edges, connections)
        
        return {
            "version": "1.0",
            "entry_points": entry_points,
            "exit_points": exit_points,
            "components": component_schemas,
            "connections": connections,
            "execution_order": self._compute_execution_order(nodes, edges),
            "validation": {
                "is_valid": len(validation_errors) == 0,
                "errors": validation_errors
            }
        }
    
    def _compute_execution_order(
        self, 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Compute topological sort for execution order.
        
        Args:
            nodes: List of node documents
            edges: List of edge documents
            
        Returns:
            List of node IDs in execution order
        """
        if not nodes:
            return []
        
        # Build adjacency list and in-degree map
        in_degree = {node["node_id"]: 0 for node in nodes}
        adjacency = {node["node_id"]: [] for node in nodes}
        
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            
            if source in adjacency and target in in_degree:
                adjacency[source].append(target)
                in_degree[target] += 1
        
        # Kahn's algorithm for topological sort
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(nodes):
            return []  # Cycle detected
        
        return result
    
    def _validate_flow(
        self, 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]],
        connections: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """
        Validate the flow structure.
        
        Args:
            nodes: List of node documents
            edges: List of edge documents
            connections: Connection graph
            
        Returns:
            List of validation error messages
        """
        errors = []
        node_map = {node["node_id"]: node for node in nodes}
        
        # Check for empty flow
        if not nodes:
            errors.append("Flow has no nodes")
            return errors
        
        # Check for input node
        input_nodes = [
            n for n in nodes 
            if n.get("data", {}).get("component_type") == "input"
        ]
        if not input_nodes:
            errors.append("Flow requires at least one Input node")
        
        # Check for output node
        output_nodes = [
            n for n in nodes 
            if n.get("data", {}).get("component_type") == "output"
        ]
        if not output_nodes:
            errors.append("Flow requires at least one Output node")
        
        # Check for agent node
        agent_nodes = [
            n for n in nodes 
            if n.get("data", {}).get("component_type") == "agent"
        ]
        if not agent_nodes:
            errors.append("Flow requires at least one Agent node")
        
        # Check for cycles
        execution_order = self._compute_execution_order(nodes, edges)
        if not execution_order and nodes:
            errors.append("Flow contains a cycle")
        
        # Validate edges reference existing nodes
        node_ids = set(node["node_id"] for node in nodes)
        for edge in edges:
            if edge["source"] not in node_ids:
                errors.append(f"Edge references non-existent source node: {edge['source']}")
            if edge["target"] not in node_ids:
                errors.append(f"Edge references non-existent target node: {edge['target']}")
        
        # Validate required connections for agents
        for agent in agent_nodes:
            agent_id = agent["node_id"]
            incoming = [e for e in edges if e["target"] == agent_id]
            
            # Check for model connection
            model_connection = [
                e for e in incoming 
                if e.get("target_handle") == "model"
            ]
            if not model_connection:
                errors.append(f"Agent '{agent_id}' requires a model connection")
            
            # Check for input connection
            input_connection = [
                e for e in incoming 
                if e.get("target_handle") == "input"
            ]
            if not input_connection:
                errors.append(f"Agent '{agent_id}' requires an input connection")
        
        # Validate component parameters
        for node in nodes:
            component_type = node.get("data", {}).get("component_type")
            
            if not component_type:
                errors.append(f"Node '{node['node_id']}' is missing component_type")
                continue
            
            component_class = self.registry.get(component_type)
            
            if component_class:
                try:
                    instance = component_class(
                        node_id=node["node_id"],
                        parameters=node.get("data", {}).get("parameters", {})
                    )
                    param_errors = instance.validate_parameters()
                    for error in param_errors:
                        errors.append(f"Node '{node['node_id']}': {error}")
                except ValueError as e:
                    errors.append(f"Node '{node['node_id']}': {str(e)}")
                except Exception as e:
                    errors.append(f"Node '{node['node_id']}': Failed to validate - {str(e)}")
            else:
                errors.append(f"Node '{node['node_id']}': Unknown component type '{component_type}'")
        
        return errors
    
    async def get_agent_schema(self, flow_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the generated agent schema for a flow.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            
        Returns:
            Agent schema with validation info or None if flow not found
        """
        flow = await self.get_flow(flow_id, user_id)
        if not flow:
            return None
        
        return {
            "flow_id": flow_id,
            "agent_schema": flow.get("agent_schema", {}),
            "validation_errors": flow.get("agent_schema", {}).get("validation", {}).get("errors", []),
            "is_valid": flow.get("agent_schema", {}).get("validation", {}).get("is_valid", False)
        }
    
    async def duplicate_flow(self, flow_id: str, user_id: str, new_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Duplicate an existing flow.
        
        Args:
            flow_id: The flow ID to duplicate
            user_id: The owner's user ID
            new_name: Optional new name for the duplicated flow
            
        Returns:
            New flow document or None if original not found
        """
        original_flow = await self.get_flow(flow_id, user_id)
        if not original_flow:
            return None
        
        # Create new flow data
        new_flow_data = {
            "name": new_name or f"{original_flow['name']} (Copy)",
            "description": original_flow.get("description"),
            "nodes": original_flow.get("nodes", []),
            "edges": original_flow.get("edges", [])
        }
        
        # Generate new IDs for nodes and edges
        node_id_map = {}
        new_nodes = []
        for node in new_flow_data["nodes"]:
            old_id = node["node_id"]
            new_id = str(uuid.uuid4())
            node_id_map[old_id] = new_id
            
            new_node = node.copy()
            new_node["node_id"] = new_id
            new_nodes.append(new_node)
        
        new_edges = []
        for edge in new_flow_data["edges"]:
            new_edge = edge.copy()
            new_edge["edge_id"] = str(uuid.uuid4())
            new_edge["source"] = node_id_map.get(edge["source"], edge["source"])
            new_edge["target"] = node_id_map.get(edge["target"], edge["target"])
            new_edges.append(new_edge)
        
        new_flow_data["nodes"] = new_nodes
        new_flow_data["edges"] = new_edges
        
        return await self.create_flow(user_id, new_flow_data)
    
    async def export_flow(self, flow_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a flow as a portable JSON structure.
        
        Args:
            flow_id: The flow ID
            user_id: The owner's user ID
            
        Returns:
            Exportable flow data or None if not found
        """
        flow = await self.get_flow(flow_id, user_id)
        if not flow:
            return None
        
        return {
            "export_version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "flow": {
                "name": flow["name"],
                "description": flow.get("description"),
                "nodes": [
                    {
                        "node_id": node["node_id"],
                        "type": node.get("type", "custom"),
                        "position": node.get("position", {"x": 0, "y": 0}),
                        "data": node.get("data", {})
                    }
                    for node in flow.get("nodes", [])
                ],
                "edges": [
                    {
                        "edge_id": edge["edge_id"],
                        "source": edge["source"],
                        "target": edge["target"],
                        "source_handle": edge.get("source_handle"),
                        "target_handle": edge.get("target_handle"),
                        "animated": edge.get("animated", True)
                    }
                    for edge in flow.get("edges", [])
                ]
            }
        }
    
    async def import_flow(self, user_id: str, import_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a flow from exported JSON data.
        
        Args:
            user_id: The user ID to assign the imported flow to
            import_data: The exported flow data
            
        Returns:
            Created flow document
        """
        flow_data = import_data.get("flow", import_data)
        
        return await self.create_flow(user_id, {
            "name": flow_data.get("name", "Imported Flow"),
            "description": flow_data.get("description"),
            "nodes": flow_data.get("nodes", []),
            "edges": flow_data.get("edges", [])
        })
