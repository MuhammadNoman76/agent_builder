"""
Agent component - orchestrates LLM and tools.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List
from .base import (
    BaseComponent,
    ComponentPort,
    PortType,
    FieldDefinition,
    FieldGroup,
    StringField,
    IntegerField,
    BooleanField,
    PromptField,
)


class AgentComponent(BaseComponent):
    """
    Agent component that orchestrates LLM models and tools.
    """
    
    _component_type = "agent"
    _name = "Agent"
    _description = "AI Agent that orchestrates LLM and tools"
    _category = "agents"
    _icon = "bot"
    _color = "#3b82f6"
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Define input ports for the agent."""
        return [
            ComponentPort(
                id="input",
                name="Input",
                type=PortType.INPUT,
                data_type="message",
                required=True,
                description="User input message"
            ),
            ComponentPort(
                id="model",
                name="Model",
                type=PortType.INPUT,
                data_type="llm_model",
                required=True,
                description="LLM model to use for reasoning"
            ),
            ComponentPort(
                id="tools",
                name="Tools",
                type=PortType.INPUT,
                data_type="tools",
                multiple=True,
                description="Tools available for the agent"
            ),
            ComponentPort(
                id="memory",
                name="Memory",
                type=PortType.INPUT,
                data_type="memory",
                description="Conversation memory/history"
            )
        ]
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Define output ports for the agent."""
        return [
            ComponentPort(
                id="output",
                name="Output",
                type=PortType.OUTPUT,
                data_type="message",
                description="Agent response"
            ),
            ComponentPort(
                id="tool_results",
                name="Tool Results",
                type=PortType.OUTPUT,
                data_type="dict",
                description="Results from tool executions"
            ),
            ComponentPort(
                id="history",
                name="History",
                type=PortType.OUTPUT,
                data_type="messages",
                description="Updated conversation history"
            )
        ]
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields using field types."""
        return [
            StringField.create(
                name="name",
                label="Agent Name",
                description="Name of the agent",
                default="Assistant",
                required=True,
                order=1,
                group="basic",
            ),
            PromptField.create(
                name="system_prompt",
                label="System Prompt",
                description="Instructions for the agent",
                default="You are a helpful assistant.",
                rows=6,
                available_variables=["user_name", "current_date", "context"],
                templates=[
                    {"name": "Helpful Assistant", "content": "You are a helpful assistant."},
                    {"name": "Code Expert", "content": "You are an expert programmer. Help users with coding questions."},
                    {"name": "Creative Writer", "content": "You are a creative writer. Help users with writing tasks."},
                ],
                order=2,
                group="basic",
            ),
            IntegerField.create(
                name="max_iterations",
                label="Max Iterations",
                description="Maximum tool call iterations",
                default=10,
                min_value=1,
                max_value=50,
                order=3,
                group="execution",
            ),
            BooleanField.create(
                name="handle_parsing_errors",
                label="Handle Parsing Errors",
                description="Gracefully handle LLM parsing errors",
                default=True,
                order=4,
                group="execution",
            ),
            BooleanField.create(
                name="return_intermediate_steps",
                label="Return Intermediate Steps",
                description="Include tool call details in output",
                default=False,
                order=5,
                group="advanced",
            ),
            BooleanField.create(
                name="verbose",
                label="Verbose Mode",
                description="Enable detailed logging",
                default=False,
                order=6,
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
                description="Agent identity and behavior",
                order=0,
            ),
            FieldGroup(
                id="execution",
                label="Execution Settings",
                description="Control agent execution behavior",
                collapsible=True,
                order=1,
            ),
            FieldGroup(
                id="advanced",
                label="Advanced",
                description="Advanced debugging options",
                collapsible=True,
                collapsed_by_default=True,
                order=2,
            ),
        ]
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the given inputs."""
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        user_input = inputs.get("input", {})
        model = inputs.get("model")
        tools = inputs.get("tools", [])
        memory = inputs.get("memory", [])
        
        if not model:
            raise ValueError("Agent requires an LLM model connection")
        
        system_prompt = self.get_parameter("system_prompt", "You are a helpful assistant.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        if tools:
            agent = create_tool_calling_agent(model._client, tools, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=self.get_parameter("max_iterations", 10),
                handle_parsing_errors=self.get_parameter("handle_parsing_errors", True),
                return_intermediate_steps=self.get_parameter("return_intermediate_steps", False),
                verbose=self.get_parameter("verbose", False)
            )
        else:
            from langchain_core.runnables import RunnablePassthrough
            executor = (
                {"input": RunnablePassthrough(), "chat_history": lambda x: memory, "agent_scratchpad": lambda x: []}
                | prompt 
                | model._client
            )
        
        if isinstance(user_input, dict):
            content = user_input.get("content", str(user_input))
        else:
            content = str(user_input)
        
        if tools:
            result = await executor.ainvoke({
                "input": content,
                "chat_history": memory
            })
            output_content = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
        else:
            result = await executor.ainvoke(content)
            output_content = result.content
            intermediate_steps = []
        
        return {
            "output": {
                "role": "assistant",
                "content": output_content
            },
            "tool_results": {
                "steps": intermediate_steps
            },
            "history": memory + [
                {"role": "user", "content": content},
                {"role": "assistant", "content": output_content}
            ]
        }
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema with agent-specific fields."""
        schema = super().to_schema()
        schema["agent_config"] = {
            "name": self.get_parameter("name", "Assistant"),
            "system_prompt": self.get_parameter("system_prompt", "You are a helpful assistant."),
            "max_iterations": self.get_parameter("max_iterations", 10),
            "handle_parsing_errors": self.get_parameter("handle_parsing_errors", True)
        }
        return schema
