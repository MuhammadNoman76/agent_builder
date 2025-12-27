"""
Flow executor - executes agent workflows based on the generated schema.
"""
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import asyncio
import time

from components.registry import ComponentRegistry
from components.base import BaseComponent


class FlowExecutor:
    """
    Executor for agent workflows.
    
    This class takes a flow's agent schema and executes the workflow,
    managing data flow between components and handling errors.
    """
    
    def __init__(self, agent_schema: Dict[str, Any]):
        """
        Initialize the executor with an agent schema.
        
        Args:
            agent_schema: The compiled agent schema from a flow
        """
        self.schema = agent_schema
        self.registry = ComponentRegistry()
        self.components: Dict[str, BaseComponent] = {}
        self.execution_state: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
    
    def validate_schema(self) -> List[str]:
        """
        Validate the agent schema before execution.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check schema version
        if self.schema.get("version") != "1.0":
            errors.append(f"Unsupported schema version: {self.schema.get('version')}")
        
        # Check for entry points
        if not self.schema.get("entry_points"):
            errors.append("No entry points defined in schema")
        
        # Check for exit points
        if not self.schema.get("exit_points"):
            errors.append("No exit points defined in schema")
        
        # Check execution order
        if not self.schema.get("execution_order"):
            errors.append("No execution order defined (possible cycle)")
        
        # Validate components
        for comp_schema in self.schema.get("components", []):
            if "error" in comp_schema:
                errors.append(comp_schema["error"])
        
        # Include schema validation errors
        validation = self.schema.get("validation", {})
        if not validation.get("is_valid", True):
            errors.extend(validation.get("errors", []))
        
        return errors
    
    async def initialize_components(self) -> None:
        """Initialize all components from the schema."""
        for comp_schema in self.schema.get("components", []):
            node_id = comp_schema.get("node_id")
            component_type = comp_schema.get("component_type")
            parameters = comp_schema.get("parameters", {})
            
            component = self.registry.create(
                component_type=component_type,
                node_id=node_id,
                parameters=parameters
            )
            
            if component:
                self.components[node_id] = component
            else:
                raise ValueError(f"Failed to create component: {component_type}")
    
    def get_component_inputs(self, node_id: str) -> Dict[str, Any]:
        """
        Get inputs for a component from upstream results.
        
        Args:
            node_id: The target node ID
            
        Returns:
            Dictionary of inputs mapped to target ports
        """
        inputs = {}
        connections = self.schema.get("connections", {})
        
        # Find all connections targeting this node
        for source_id, targets in connections.items():
            for target_info in targets:
                if target_info["target"] == node_id:
                    source_port = target_info["source_port"]
                    target_port = target_info["target_port"]
                    
                    # Get result from source component
                    source_result = self.results.get(source_id, {})
                    
                    # Map source output to target input
                    if source_port in source_result:
                        inputs[target_port] = source_result[source_port]
        
        return inputs
    
    async def execute_component(self, node_id: str) -> Dict[str, Any]:
        """
        Execute a single component.
        
        Args:
            node_id: The node ID to execute
            
        Returns:
            Component execution results
        """
        component = self.components.get(node_id)
        if not component:
            raise ValueError(f"Component not found: {node_id}")
        
        # Get inputs from upstream components
        inputs = self.get_component_inputs(node_id)
        
        # Execute component
        try:
            result = await component.execute(inputs)
            self.results[node_id] = result
            return result
        except Exception as e:
            error_result = {
                "error": str(e),
                "node_id": node_id,
                "component_type": component._component_type
            }
            self.results[node_id] = error_result
            raise
    
    async def execute(
        self, 
        input_data: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the entire workflow.
        
        Args:
            input_data: Input data to feed to entry points
            stream: Whether to stream the output
            
        Returns:
            Final execution results
        """
        start_time = time.time()
        
        # Validate schema
        validation_errors = self.validate_schema()
        if validation_errors:
            return {
                "status": "error",
                "errors": validation_errors,
                "execution_time": time.time() - start_time
            }
        
        # Initialize components
        try:
            await self.initialize_components()
        except Exception as e:
            return {
                "status": "error",
                "errors": [f"Failed to initialize components: {str(e)}"],
                "execution_time": time.time() - start_time
            }
        
        # Set input data for entry points
        for entry_point in self.schema.get("entry_points", []):
            if entry_point in self.components:
                # Pre-populate results with input data
                self.results[entry_point] = {
                    "output": input_data.get("value", input_data),
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "user_input"
                    }
                }
        
        # Execute in topological order
        execution_order = self.schema.get("execution_order", [])
        
        for node_id in execution_order:
            # Skip entry points that already have results
            if node_id in self.schema.get("entry_points", []) and node_id in self.results:
                # Re-execute entry point with proper input handling
                component = self.components.get(node_id)
                if component:
                    inputs = {"value": input_data.get("value", input_data)}
                    self.results[node_id] = await component.execute(inputs)
                continue
            
            try:
                await self.execute_component(node_id)
            except Exception as e:
                return {
                    "status": "error",
                    "errors": [f"Execution failed at node {node_id}: {str(e)}"],
                    "partial_results": self.results,
                    "execution_time": time.time() - start_time
                }
        
        # Collect outputs from exit points
        outputs = {}
        for exit_point in self.schema.get("exit_points", []):
            if exit_point in self.results:
                outputs[exit_point] = self.results[exit_point]
        
        return {
            "status": "success",
            "output": outputs,
            "execution_time": time.time() - start_time
        }
    
    async def execute_stream(
        self, 
        input_data: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute the workflow with streaming output.
        
        Args:
            input_data: Input data to feed to entry points
            
        Yields:
            Streaming execution updates
        """
        start_time = time.time()
        
        # Validate and initialize
        validation_errors = self.validate_schema()
        if validation_errors:
            yield {
                "type": "error",
                "errors": validation_errors
            }
            return
        
        try:
            await self.initialize_components()
        except Exception as e:
            yield {
                "type": "error",
                "errors": [f"Failed to initialize components: {str(e)}"]
            }
            return
        
        # Set input data for entry points
        for entry_point in self.schema.get("entry_points", []):
            self.results[entry_point] = {
                "output": input_data.get("value", input_data),
                "metadata": {"timestamp": datetime.utcnow().isoformat()}
            }
        
        yield {"type": "start", "message": "Execution started"}
        
        # Execute in order
        execution_order = self.schema.get("execution_order", [])
        
        for i, node_id in enumerate(execution_order):
            component = self.components.get(node_id)
            if not component:
                continue
            
            yield {
                "type": "progress",
                "node_id": node_id,
                "component_type": component._component_type,
                "step": i + 1,
                "total_steps": len(execution_order)
            }
            
            try:
                # Handle entry points
                if node_id in self.schema.get("entry_points", []):
                    inputs = {"value": input_data.get("value", input_data)}
                    result = await component.execute(inputs)
                else:
                    result = await self.execute_component(node_id)
                
                self.results[node_id] = result
                
                # Check for streaming result
                if isinstance(result.get("response"), AsyncIterator):
                    async for chunk in result["response"]:
                        yield {
                            "type": "stream",
                            "node_id": node_id,
                            "content": chunk
                        }
                
                yield {
                    "type": "node_complete",
                    "node_id": node_id,
                    "result": result
                }
                
            except Exception as e:
                yield {
                    "type": "error",
                    "node_id": node_id,
                    "error": str(e)
                }
                return
        
        # Final output
        outputs = {}
        for exit_point in self.schema.get("exit_points", []):
            if exit_point in self.results:
                outputs[exit_point] = self.results[exit_point]
        
        yield {
            "type": "complete",
            "output": outputs,
            "execution_time": time.time() - start_time
        }


class FlowExecutorFactory:
    """Factory for creating flow executors."""
    
    @staticmethod
    def create(agent_schema: Dict[str, Any]) -> FlowExecutor:
        """
        Create a new flow executor.
        
        Args:
            agent_schema: The agent schema to execute
            
        Returns:
            Configured FlowExecutor instance
        """
        return FlowExecutor(agent_schema)
    
    @staticmethod
    async def execute_flow(
        agent_schema: Dict[str, Any],
        input_data: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Convenience method to execute a flow directly.
        
        Args:
            agent_schema: The agent schema
            input_data: Input data for the flow
            stream: Whether to stream output
            
        Returns:
            Execution results
        """
        executor = FlowExecutor(agent_schema)
        return await executor.execute(input_data, stream)
