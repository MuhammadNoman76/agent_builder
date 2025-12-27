# Project Code Documentation

## Project Structure

```
agent_builder-backend/
├── auth
│   ├── __init__.py
│   ├── routes.py
│   ├── schemas.py
│   └── utils.py
├── components
│   ├── field_types
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── schema.py
│   │   ├── types.py
│   │   └── validators.py
│   ├── llm
│   │   ├── __init__.py
│   │   ├── anthropic_model.py
│   │   ├── base.py
│   │   ├── openai_model.py
│   │   ├── openrouter_model.py
│   │   └── registry.py
│   ├── __init__.py
│   ├── agent_component.py
│   ├── base.py
│   ├── composio_component.py
│   ├── input_component.py
│   ├── output_component.py
│   └── registry.py
├── database
│   ├── __init__.py
│   ├── models.py
│   └── mongodb.py
├── edges
│   ├── __init__.py
│   ├── routes.py
│   └── schemas.py
├── flows
│   ├── __init__.py
│   ├── executor.py
│   ├── routes.py
│   ├── schemas.py
│   └── services.py
├── nodes
│   ├── __init__.py
│   ├── routes.py
│   └── schemas.py
├── config.py
├── document_code.py
└── main.py
```

# auth/__init__.py

```python
"""Authentication package initialization."""
from .routes import router as auth_router
from .utils import get_current_user, create_access_token

__all__ = ["auth_router", "get_current_user", "create_access_token"]

```

# auth/routes.py

```python
"""
Authentication API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime

from database import get_database
from database.models import UserModel
from .schemas import UserCreate, UserLogin, UserResponse, Token
from .utils import (
    get_password_hash, 
    verify_password, 
    create_access_token,
    get_current_user
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Register a new user."""
    # Check if email already exists
    existing_email = await db.users.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    existing_username = await db.users.find_one({"username": user_data.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user document
    user = UserModel(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )
    
    # Insert into database
    user_dict = user.model_dump(by_alias=True)
    await db.users.insert_one(user_dict)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Login and get access token."""
    # Find user by email (username field in OAuth2 form)
    user = await db.users.find_one({"email": form_data.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["_id"], "email": user["email"]}
    )
    
    return Token(access_token=access_token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user["_id"],
        username=current_user["username"],
        email=current_user["email"],
        is_active=current_user["is_active"]
    )

```

# auth/schemas.py

```python
"""
Authentication schemas for request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response (without sensitive data)."""
    id: str
    username: str
    email: EmailStr
    is_active: bool


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded token data."""
    user_id: Optional[str] = None
    email: Optional[str] = None

```

# auth/utils.py

```python
"""
Authentication utilities - JWT handling and password hashing.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from database import get_database
from .schemas import TokenData


# Password hashing context
# bcrypt has a 72 byte limit; bcrypt_sha256 pre-hashes to safely support long passwords
pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto",
)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None:
            return None
            
        return TokenData(user_id=user_id, email=email)
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> dict:
    """
    FastAPI dependency to get the current authenticated user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_access_token(token)
    if token_data is None or token_data.user_id is None:
        raise credentials_exception
    
    user = await db.users.find_one({"_id": token_data.user_id})
    if user is None:
        raise credentials_exception
    
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user

```

# components/__init__.py

```python
"""Components package initialization."""
from .base import (
    BaseComponent,
    ComponentConfig,
    ComponentPort,
    PortType,
    # Re-exported field types for convenience
    FieldDefinition,
    FieldGroup,
    FieldTypeEnum,
    FieldValidation,
    FieldOption,
)
from .input_component import InputComponent
from .output_component import OutputComponent
from .agent_component import AgentComponent
from .composio_component import ComposioToolComponent
from .registry import ComponentRegistry
from .llm import (
    BaseLLMModel,
    LLMConfig,
    LLMResponse,
    LLMProvider,
    OpenAIModel,
    AnthropicModel,
    OpenRouterModel,
    LLMRegistry,
)
from .field_types import (
    # All field types
    StringField,
    TextField,
    NumberField,
    IntegerField,
    BooleanField,
    SelectField,
    MultiSelectField,
    RadioField,
    CheckboxGroupField,
    PasswordField,
    EmailField,
    UrlField,
    ColorField,
    DateField,
    DateTimeField,
    TimeField,
    JsonField,
    CodeField,
    SliderField,
    RangeField,
    FileField,
    ImageField,
    ApiKeyField,
    ModelSelectField,
    PromptField,
    VariableField,
    PortField,
    # Schema generation
    generate_field_schema,
    generate_component_schema,
    # Registry
    field_type_registry,
    # Validation
    validate_field,
    validate_fields,
)

__all__ = [
    # Base classes
    "BaseComponent",
    "ComponentConfig",
    "ComponentPort",
    "PortType",
    "FieldDefinition",
    "FieldGroup",
    "FieldTypeEnum",
    "FieldValidation",
    "FieldOption",
    # Components
    "InputComponent",
    "OutputComponent",
    "AgentComponent",
    "ComposioToolComponent",
    "ComponentRegistry",
    # LLM
    "BaseLLMModel",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    "OpenAIModel",
    "AnthropicModel",
    "OpenRouterModel",
    "LLMRegistry",
    # Field types
    "StringField",
    "TextField",
    "NumberField",
    "IntegerField",
    "BooleanField",
    "SelectField",
    "MultiSelectField",
    "RadioField",
    "CheckboxGroupField",
    "PasswordField",
    "EmailField",
    "UrlField",
    "ColorField",
    "DateField",
    "DateTimeField",
    "TimeField",
    "JsonField",
    "CodeField",
    "SliderField",
    "RangeField",
    "FileField",
    "ImageField",
    "ApiKeyField",
    "ModelSelectField",
    "PromptField",
    "VariableField",
    "PortField",
    # Utilities
    "generate_field_schema",
    "generate_component_schema",
    "field_type_registry",
    "validate_field",
    "validate_fields",
]

```

# components/agent_component.py

```python
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

```

# components/base.py

```python
"""
Base component class - foundation for all components.
Uses centralized field types for consistent input handling.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum
import uuid

# Import from field_types package
from .field_types import (
    FieldDefinition,
    FieldGroup,
    FieldTypeEnum,
    FieldValidation,
    FieldOption,
    ComponentSchemaGenerator,
    validate_fields,
    get_validation_errors,
    # Import specific field types for convenience
    StringField,
    TextField,
    NumberField,
    IntegerField,
    BooleanField,
    SelectField,
    MultiSelectField,
    PasswordField,
    JsonField,
    SliderField,
    ApiKeyField,
    ModelSelectField,
    PromptField,
    VariableField,
)


class PortType(str, Enum):
    """Port types for component connections."""
    INPUT = "input"
    OUTPUT = "output"


class ComponentPort(BaseModel):
    """Represents a connection port on a component."""
    id: str
    name: str
    type: PortType
    data_type: str = "any"
    required: bool = False
    multiple: bool = False
    description: Optional[str] = None


class ComponentConfig(BaseModel):
    """Configuration schema for a component."""
    component_type: str
    name: str
    description: str
    category: str
    icon: str = "box"
    color: str = "#6366f1"
    input_ports: List[ComponentPort] = Field(default_factory=list)
    output_ports: List[ComponentPort] = Field(default_factory=list)
    # Updated to use FieldDefinition
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    field_groups: List[Dict[str, Any]] = Field(default_factory=list)


class BaseComponent(ABC):
    """
    Abstract base class for all components in the agent builder.
    
    Uses the centralized field types system for input definitions.
    """
    
    _component_type: str = "base"
    _name: str = "Base Component"
    _description: str = "Base component class"
    _category: str = "general"
    _icon: str = "box"
    _color: str = "#6366f1"
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Initialize the component."""
        self.node_id = node_id or str(uuid.uuid4())
        self.parameters = parameters or {}
        self._fields: List[FieldDefinition] = []
        self._field_groups: List[FieldGroup] = []
        self._initialize_fields()
        self._validate_and_set_defaults()
    
    def _initialize_fields(self) -> None:
        """Initialize field definitions. Called once during init."""
        self._fields = self._get_fields()
        self._field_groups = self._get_field_groups()
    
    def _validate_and_set_defaults(self) -> None:
        """Validate parameters and set defaults for missing values."""
        for field in self._fields:
            if field.name not in self.parameters:
                if field.default is not None:
                    self.parameters[field.name] = field.default
                elif field.validation.required:
                    raise ValueError(
                        f"Required field '{field.name}' is missing for component '{self._name}'"
                    )
    
    @classmethod
    def get_config(cls) -> ComponentConfig:
        """Get the component configuration."""
        instance = cls.__new__(cls)
        instance._fields = []
        instance._field_groups = []
        instance._initialize_fields = lambda: None
        instance.parameters = {}
        instance.node_id = "temp"
        
        # Get fields using class method
        fields = cls._get_fields()
        groups = cls._get_field_groups()
        
        return ComponentConfig(
            component_type=cls._component_type,
            name=cls._name,
            description=cls._description,
            category=cls._category,
            icon=cls._icon,
            color=cls._color,
            input_ports=cls._get_input_ports(),
            output_ports=cls._get_output_ports(),
            fields=[f.to_schema() for f in fields],
            field_groups=[g.model_dump() for g in groups],
        )
    
    @classmethod
    @abstractmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """Define input ports for the component."""
        pass
    
    @classmethod
    @abstractmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """Define output ports for the component."""
        pass
    
    @classmethod
    @abstractmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Define configurable fields for the component using field types."""
        pass
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organizing fields. Override in subclasses."""
        return []
    
    def validate_parameters(self) -> List[str]:
        """Validate component parameters using field validators."""
        return get_validation_errors(self.parameters, self._fields)
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component logic."""
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert component to JSON schema representation."""
        config = self.get_config()
        return {
            "node_id": self.node_id,
            "component_type": self._component_type,
            "name": config.name,
            "parameters": self.parameters,
            "input_ports": [port.model_dump() for port in config.input_ports],
            "output_ports": [port.model_dump() for port in config.output_ports],
            "fields": config.fields,
            "field_groups": config.field_groups,
        }
    
    def get_field_schema(self) -> Dict[str, Any]:
        """Get complete field schema for frontend rendering."""
        return ComponentSchemaGenerator.generate(
            fields=self._fields,
            groups=self._field_groups,
            component_info={
                "type": self._component_type,
                "name": self._name,
                "description": self._description,
                "category": self._category,
                "icon": self._icon,
                "color": self._color,
            }
        )
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update component parameters."""
        self.parameters.update(parameters)
        self._validate_and_set_defaults()
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a specific parameter value."""
        return self.parameters.get(name, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_id={self.node_id}, parameters={self.parameters})"


# Re-export field types for convenience
__all__ = [
    "BaseComponent",
    "ComponentConfig",
    "ComponentPort",
    "PortType",
    # Field types
    "FieldDefinition",
    "FieldGroup",
    "FieldTypeEnum",
    "FieldValidation",
    "FieldOption",
    "StringField",
    "TextField",
    "NumberField",
    "IntegerField",
    "BooleanField",
    "SelectField",
    "MultiSelectField",
    "PasswordField",
    "JsonField",
    "SliderField",
    "ApiKeyField",
    "ModelSelectField",
    "PromptField",
    "VariableField",
]

```

# components/composio_component.py

```python
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

```

# components/field_types/__init__.py

```python
"""
Field Types Package - Centralized input field definitions for all components.

This package provides a unified system for defining input fields, their types,
validation rules, and JSON schemas for frontend rendering.
"""
from .base import (
    FieldType,
    FieldDefinition,
    FieldValidation,
    FieldOption,
    FieldGroup,
    FieldCondition,
    FieldDependency
)
from .types import (
    # Basic Types
    StringField,
    TextField,
    NumberField,
    IntegerField,
    BooleanField,
    # Selection Types
    SelectField,
    MultiSelectField,
    RadioField,
    CheckboxGroupField,
    # Special Types
    PasswordField,
    EmailField,
    UrlField,
    ColorField,
    DateField,
    DateTimeField,
    TimeField,
    # Complex Types
    JsonField,
    CodeField,
    SliderField,
    RangeField,
    FileField,
    ImageField,
    # Custom Types
    ApiKeyField,
    ModelSelectField,
    PromptField,
    VariableField,
    PortField,
    # Field Type Enum
    FieldTypeEnum
)
from .schema import (
    FieldSchemaGenerator,
    ComponentSchemaGenerator,
    generate_field_schema,
    generate_component_schema
)
from .registry import FieldTypeRegistry, field_type_registry
from .validators import (
    FieldValidator,
    validate_field,
    validate_fields,
    get_validation_errors,
    is_valid,
    ValidationResult
)

__all__ = [
    # Base Classes
    "FieldType",
    "FieldDefinition",
    "FieldValidation",
    "FieldOption",
    "FieldGroup",
    "FieldCondition",
    "FieldDependency",
    # Field Types
    "StringField",
    "TextField",
    "NumberField",
    "IntegerField",
    "BooleanField",
    "SelectField",
    "MultiSelectField",
    "RadioField",
    "CheckboxGroupField",
    "PasswordField",
    "EmailField",
    "UrlField",
    "ColorField",
    "DateField",
    "DateTimeField",
    "TimeField",
    "JsonField",
    "CodeField",
    "SliderField",
    "RangeField",
    "FileField",
    "ImageField",
    "ApiKeyField",
    "ModelSelectField",
    "PromptField",
    "VariableField",
    "PortField",
    "FieldTypeEnum",
    # Schema Generation
    "FieldSchemaGenerator",
    "ComponentSchemaGenerator",
    "generate_field_schema",
    "generate_component_schema",
    # Registry
    "FieldTypeRegistry",
    "field_type_registry",
    # Validators
    "FieldValidator",
    "validate_field",
    "validate_fields",
    "get_validation_errors",
    "is_valid",
    "ValidationResult",
]

```

# components/field_types/base.py

```python
"""
Base classes for field type definitions.

This module contains the foundational classes and types used to define
input fields across all components in the agent builder.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re


class FieldTypeEnum(str, Enum):
    """Enumeration of all available field types."""
    # Basic Types
    STRING = "string"
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    
    # Selection Types
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    RADIO = "radio"
    CHECKBOX_GROUP = "checkbox_group"
    
    # Special Input Types
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    COLOR = "color"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    
    # Complex Types
    JSON = "json"
    CODE = "code"
    SLIDER = "slider"
    RANGE = "range"
    FILE = "file"
    IMAGE = "image"
    
    # Custom Agent Builder Types
    API_KEY = "api_key"
    MODEL_SELECT = "model_select"
    PROMPT = "prompt"
    VARIABLE = "variable"
    PORT = "port"


class FieldOption(BaseModel):
    """Represents an option for select/radio/checkbox fields."""
    value: Any
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    disabled: bool = False
    group: Optional[str] = None  # For grouped options
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FieldValidation(BaseModel):
    """Validation rules for a field."""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern
    pattern_message: Optional[str] = None  # Error message for pattern
    custom_validator: Optional[str] = None  # Name of custom validator function
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None
    unique: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class FieldCondition(BaseModel):
    """Conditional visibility/requirement for a field."""
    field: str  # Field name to check
    operator: str  # eq, neq, gt, lt, gte, lte, in, not_in, contains, empty, not_empty
    value: Optional[Any] = None  # Value to compare against
    
    def evaluate(self, field_value: Any) -> bool:
        """Evaluate the condition against a field value."""
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "neq":
            return field_value != self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "contains":
            return self.value in field_value
        elif self.operator == "empty":
            return not field_value
        elif self.operator == "not_empty":
            return bool(field_value)
        return False


class FieldDependency(BaseModel):
    """Dependency configuration for a field."""
    depends_on: str  # Field name this depends on
    condition: FieldCondition
    action: str = "show"  # show, hide, enable, disable, require, optional
    
    def should_apply(self, all_values: Dict[str, Any]) -> bool:
        """Check if the dependency action should apply."""
        dependent_value = all_values.get(self.depends_on)
        return self.condition.evaluate(dependent_value)


class FieldGroup(BaseModel):
    """Grouping configuration for organizing fields."""
    id: str
    label: str
    description: Optional[str] = None
    collapsible: bool = False
    collapsed_by_default: bool = False
    icon: Optional[str] = None
    order: int = 0


class FieldDefinition(BaseModel):
    """
    Complete definition of a field including all metadata for frontend rendering.
    """
    # Core Properties
    name: str
    type: FieldTypeEnum
    label: str
    description: Optional[str] = None
    
    # Default Value
    default: Any = None
    
    # Validation
    validation: FieldValidation = Field(default_factory=FieldValidation)
    
    # UI Properties
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    hint: Optional[str] = None
    icon: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    
    # Layout
    width: str = "full"  # full, half, third, quarter, auto
    order: int = 0
    group: Optional[str] = None  # Group ID for organizing fields
    
    # Options (for select/radio/checkbox)
    options: List[FieldOption] = Field(default_factory=list)
    options_source: Optional[str] = None  # API endpoint or function name for dynamic options
    
    # Conditional Display
    show_when: Optional[List[FieldCondition]] = None
    hide_when: Optional[List[FieldCondition]] = None
    dependencies: List[FieldDependency] = Field(default_factory=list)
    
    # Type-specific Properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Advanced
    read_only: bool = False
    disabled: bool = False
    hidden: bool = False
    sensitive: bool = False  # For passwords, API keys, etc.
    copyable: bool = False  # Show copy button
    clearable: bool = True  # Show clear button
    
    # Custom Rendering
    component: Optional[str] = None  # Custom component name for frontend
    render_props: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for frontend."""
        schema = {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "description": self.description,
            "default": self.default,
            "validation": self.validation.to_dict(),
            "ui": {
                "placeholder": self.placeholder,
                "help_text": self.help_text,
                "hint": self.hint,
                "icon": self.icon,
                "prefix": self.prefix,
                "suffix": self.suffix,
                "width": self.width,
                "order": self.order,
                "group": self.group,
                "read_only": self.read_only,
                "disabled": self.disabled,
                "hidden": self.hidden,
                "sensitive": self.sensitive,
                "copyable": self.copyable,
                "clearable": self.clearable,
                "component": self.component,
                "render_props": self.render_props,
            },
            "options": [opt.model_dump() for opt in self.options] if self.options else None,
            "options_source": self.options_source,
            "conditions": {
                "show_when": [c.model_dump() for c in self.show_when] if self.show_when else None,
                "hide_when": [c.model_dump() for c in self.hide_when] if self.hide_when else None,
                "dependencies": [d.model_dump() for d in self.dependencies] if self.dependencies else None,
            },
            "properties": self.properties,
        }
        
        # Remove None values for cleaner output
        return self._clean_dict(schema)
    
    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove None values from dict."""
        if not isinstance(d, dict):
            return d
        return {
            k: self._clean_dict(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if v is not None and v != {} and v != []
        }


class FieldType(ABC):
    """
    Abstract base class for field types.
    
    Subclasses define specific field types with their properties,
    validation rules, and schema generation logic.
    """
    
    field_type: FieldTypeEnum
    default_properties: Dict[str, Any] = {}
    
    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        label: str,
        **kwargs
    ) -> FieldDefinition:
        """Create a field definition of this type."""
        pass
    
    @classmethod
    def get_type_schema(cls) -> Dict[str, Any]:
        """Get the JSON schema for this field type."""
        return {
            "type": cls.field_type.value,
            "properties": cls.default_properties,
            "description": cls.__doc__
        }
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate a value against this field type's rules."""
        errors = []
        validation = field.validation
        
        # Required check
        if validation.required and (value is None or value == ""):
            errors.append(f"{field.label} is required")
            return errors
        
        # Skip further validation if value is empty and not required
        if value is None or value == "":
            return errors
        
        return errors

```

# components/field_types/registry.py

```python
"""
Field Type Registry - manages available field types.

This module provides a registry for all field types, allowing for
dynamic lookup and instantiation.
"""
from typing import Dict, Type, List, Optional, Any
from .base import FieldType, FieldTypeEnum, FieldDefinition
from .types import (
    StringField, TextField, NumberField, IntegerField, BooleanField,
    SelectField, MultiSelectField, RadioField, CheckboxGroupField,
    PasswordField, EmailField, UrlField, ColorField,
    DateField, DateTimeField, TimeField,
    JsonField, CodeField, SliderField, RangeField, FileField, ImageField,
    ApiKeyField, ModelSelectField, PromptField, VariableField, PortField
)


class FieldTypeRegistry:
    """
    Registry for field types.
    
    Provides lookup and instantiation of field types by their enum value.
    """
    
    _instance: Optional["FieldTypeRegistry"] = None
    _types: Dict[FieldTypeEnum, Type[FieldType]] = {}
    
    def __new__(cls) -> "FieldTypeRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_types()
        return cls._instance
    
    def _initialize_default_types(self) -> None:
        """Register all default field types."""
        self.register(FieldTypeEnum.STRING, StringField)
        self.register(FieldTypeEnum.TEXT, TextField)
        self.register(FieldTypeEnum.NUMBER, NumberField)
        self.register(FieldTypeEnum.INTEGER, IntegerField)
        self.register(FieldTypeEnum.BOOLEAN, BooleanField)
        self.register(FieldTypeEnum.SELECT, SelectField)
        self.register(FieldTypeEnum.MULTI_SELECT, MultiSelectField)
        self.register(FieldTypeEnum.RADIO, RadioField)
        self.register(FieldTypeEnum.CHECKBOX_GROUP, CheckboxGroupField)
        self.register(FieldTypeEnum.PASSWORD, PasswordField)
        self.register(FieldTypeEnum.EMAIL, EmailField)
        self.register(FieldTypeEnum.URL, UrlField)
        self.register(FieldTypeEnum.COLOR, ColorField)
        self.register(FieldTypeEnum.DATE, DateField)
        self.register(FieldTypeEnum.DATETIME, DateTimeField)
        self.register(FieldTypeEnum.TIME, TimeField)
        self.register(FieldTypeEnum.JSON, JsonField)
        self.register(FieldTypeEnum.CODE, CodeField)
        self.register(FieldTypeEnum.SLIDER, SliderField)
        self.register(FieldTypeEnum.RANGE, RangeField)
        self.register(FieldTypeEnum.FILE, FileField)
        self.register(FieldTypeEnum.IMAGE, ImageField)
        self.register(FieldTypeEnum.API_KEY, ApiKeyField)
        self.register(FieldTypeEnum.MODEL_SELECT, ModelSelectField)
        self.register(FieldTypeEnum.PROMPT, PromptField)
        self.register(FieldTypeEnum.VARIABLE, VariableField)
        self.register(FieldTypeEnum.PORT, PortField)
    
    def register(self, field_type: FieldTypeEnum, type_class: Type[FieldType]) -> None:
        """
        Register a field type.
        
        Args:
            field_type: The field type enum value
            type_class: The field type class
        """
        self._types[field_type] = type_class
    
    def get(self, field_type: FieldTypeEnum) -> Optional[Type[FieldType]]:
        """
        Get a field type class.
        
        Args:
            field_type: The field type enum value
            
        Returns:
            The field type class or None
        """
        return self._types.get(field_type)
    
    def create_field(
        self,
        field_type: FieldTypeEnum,
        name: str,
        label: str,
        **kwargs
    ) -> Optional[FieldDefinition]:
        """
        Create a field definition.
        
        Args:
            field_type: The field type enum value
            name: Field name
            label: Field label
            **kwargs: Additional field arguments
            
        Returns:
            FieldDefinition or None
        """
        type_class = self.get(field_type)
        if type_class:
            return type_class.create(name=name, label=label, **kwargs)
        return None
    
    def list_types(self) -> List[Dict[str, Any]]:
        """
        List all registered field types with their metadata.
        
        Returns:
            List of field type information
        """
        return [
            {
                "type": field_type.value,
                "class": type_class.__name__,
                "description": type_class.__doc__,
                "default_properties": type_class.default_properties,
            }
            for field_type, type_class in self._types.items()
        ]
    
    def get_type_schema(self, field_type: FieldTypeEnum) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a specific field type.
        
        Args:
            field_type: The field type enum value
            
        Returns:
            Type schema or None
        """
        type_class = self.get(field_type)
        if type_class:
            return type_class.get_type_schema()
        return None
    
    @property
    def available_types(self) -> List[FieldTypeEnum]:
        """Get list of available field types."""
        return list(self._types.keys())


# Global registry instance
field_type_registry = FieldTypeRegistry()

```

# components/field_types/schema.py

```python
"""
Schema generation for field types and components.

This module provides utilities for generating JSON schemas that can be
consumed by frontend applications to dynamically render forms.
"""
from typing import Any, Dict, List, Optional, Type
from .base import FieldDefinition, FieldGroup, FieldTypeEnum
from .types import *


class FieldSchemaGenerator:
    """Generates JSON schema for a single field."""
    
    @staticmethod
    def generate(field: FieldDefinition) -> Dict[str, Any]:
        """Generate JSON schema for a field."""
        return field.to_schema()
    
    @staticmethod
    def generate_json_schema(field: FieldDefinition) -> Dict[str, Any]:
        """Generate JSON Schema (draft-07) compatible schema."""
        schema = {
            "title": field.label,
            "description": field.description,
        }
        
        # Map field types to JSON Schema types
        type_mapping = {
            FieldTypeEnum.STRING: {"type": "string"},
            FieldTypeEnum.TEXT: {"type": "string"},
            FieldTypeEnum.PASSWORD: {"type": "string"},
            FieldTypeEnum.EMAIL: {"type": "string", "format": "email"},
            FieldTypeEnum.URL: {"type": "string", "format": "uri"},
            FieldTypeEnum.NUMBER: {"type": "number"},
            FieldTypeEnum.INTEGER: {"type": "integer"},
            FieldTypeEnum.BOOLEAN: {"type": "boolean"},
            FieldTypeEnum.SELECT: {"type": "string"},
            FieldTypeEnum.MULTI_SELECT: {"type": "array", "items": {"type": "string"}},
            FieldTypeEnum.JSON: {"type": "object"},
            FieldTypeEnum.DATE: {"type": "string", "format": "date"},
            FieldTypeEnum.DATETIME: {"type": "string", "format": "date-time"},
            FieldTypeEnum.TIME: {"type": "string", "format": "time"},
        }
        
        schema.update(type_mapping.get(field.type, {"type": "string"}))
        
        # Add validation constraints
        validation = field.validation
        
        if validation.min_length is not None:
            schema["minLength"] = validation.min_length
        if validation.max_length is not None:
            schema["maxLength"] = validation.max_length
        if validation.min_value is not None:
            schema["minimum"] = validation.min_value
        if validation.max_value is not None:
            schema["maximum"] = validation.max_value
        if validation.pattern:
            schema["pattern"] = validation.pattern
        
        # Add enum for select fields
        if field.options:
            schema["enum"] = [opt.value for opt in field.options]
        
        # Add default
        if field.default is not None:
            schema["default"] = field.default
        
        return schema


class ComponentSchemaGenerator:
    """Generates complete schema for a component's fields."""
    
    @staticmethod
    def generate(
        fields: List[FieldDefinition],
        groups: Optional[List[FieldGroup]] = None,
        component_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete component schema for frontend rendering.
        
        Args:
            fields: List of field definitions
            groups: Optional list of field groups
            component_info: Optional component metadata
            
        Returns:
            Complete schema dictionary
        """
        schema = {
            "version": "1.0",
            "component": component_info or {},
            "fields": [field.to_schema() for field in fields],
            "groups": [group.model_dump() for group in groups] if groups else [],
            "field_order": [field.name for field in sorted(fields, key=lambda f: f.order)],
        }
        
        # Add field type metadata
        schema["field_types"] = ComponentSchemaGenerator._get_field_type_metadata(fields)
        
        # Add validation schema
        schema["validation_schema"] = ComponentSchemaGenerator._generate_validation_schema(fields)
        
        return schema
    
    @staticmethod
    def _get_field_type_metadata(fields: List[FieldDefinition]) -> Dict[str, Any]:
        """Get metadata about field types used in the component."""
        types_used = set(field.type for field in fields)
        
        metadata = {}
        for field_type in types_used:
            metadata[field_type] = {
                "type": field_type,
                "input_component": ComponentSchemaGenerator._get_input_component(field_type),
                "requires_options": field_type in [
                    FieldTypeEnum.SELECT,
                    FieldTypeEnum.MULTI_SELECT,
                    FieldTypeEnum.RADIO,
                    FieldTypeEnum.CHECKBOX_GROUP,
                ],
            }
        
        return metadata
    
    @staticmethod
    def _get_input_component(field_type: FieldTypeEnum) -> str:
        """Map field type to recommended frontend component."""
        component_mapping = {
            FieldTypeEnum.STRING: "TextInput",
            FieldTypeEnum.TEXT: "TextArea",
            FieldTypeEnum.NUMBER: "NumberInput",
            FieldTypeEnum.INTEGER: "NumberInput",
            FieldTypeEnum.BOOLEAN: "Switch",
            FieldTypeEnum.SELECT: "Select",
            FieldTypeEnum.MULTI_SELECT: "MultiSelect",
            FieldTypeEnum.RADIO: "RadioGroup",
            FieldTypeEnum.CHECKBOX_GROUP: "CheckboxGroup",
            FieldTypeEnum.PASSWORD: "PasswordInput",
            FieldTypeEnum.EMAIL: "EmailInput",
            FieldTypeEnum.URL: "UrlInput",
            FieldTypeEnum.COLOR: "ColorPicker",
            FieldTypeEnum.DATE: "DatePicker",
            FieldTypeEnum.DATETIME: "DateTimePicker",
            FieldTypeEnum.TIME: "TimePicker",
            FieldTypeEnum.JSON: "JsonEditor",
            FieldTypeEnum.CODE: "CodeEditor",
            FieldTypeEnum.SLIDER: "Slider",
            FieldTypeEnum.RANGE: "RangeSlider",
            FieldTypeEnum.FILE: "FileUpload",
            FieldTypeEnum.IMAGE: "ImageUpload",
            FieldTypeEnum.API_KEY: "ApiKeyInput",
            FieldTypeEnum.MODEL_SELECT: "ModelSelect",
            FieldTypeEnum.PROMPT: "PromptEditor",
            FieldTypeEnum.VARIABLE: "VariableInput",
            FieldTypeEnum.PORT: "PortConfig",
        }
        return component_mapping.get(field_type, "TextInput")
    
    @staticmethod
    def _generate_validation_schema(fields: List[FieldDefinition]) -> Dict[str, Any]:
        """Generate validation schema for all fields."""
        properties = {}
        required = []
        
        for field in fields:
            properties[field.name] = FieldSchemaGenerator.generate_json_schema(field)
            if field.validation.required:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def generate_field_schema(field: FieldDefinition) -> Dict[str, Any]:
    """Convenience function to generate field schema."""
    return FieldSchemaGenerator.generate(field)


def generate_component_schema(
    fields: List[FieldDefinition],
    groups: Optional[List[FieldGroup]] = None,
    component_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to generate component schema."""
    return ComponentSchemaGenerator.generate(fields, groups, component_info)

```

# components/field_types/types.py

```python
"""
Concrete field type implementations.

This module contains all the specific field type classes that can be used
to define input fields in components.
"""
from typing import Any, Dict, List, Optional, Union
from .base import (
    FieldType,
    FieldTypeEnum,
    FieldDefinition,
    FieldValidation,
    FieldOption,
    FieldCondition,
    FieldDependency
)


# =============================================================================
# BASIC FIELD TYPES
# =============================================================================

class StringField(FieldType):
    """Single-line text input field."""
    
    field_type = FieldTypeEnum.STRING
    default_properties = {
        "max_display_length": 100,
        "auto_trim": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        pattern_message: Optional[str] = None,
        placeholder: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a string field definition."""
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                pattern_message=pattern_message,
            ),
            placeholder=placeholder or f"Enter {label.lower()}...",
            prefix=prefix,
            suffix=suffix,
            properties=cls.default_properties.copy(),
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate string value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        if not isinstance(value, str):
            errors.append(f"{field.label} must be a string")
            return errors
        
        validation = field.validation
        
        if validation.min_length and len(value) < validation.min_length:
            errors.append(f"{field.label} must be at least {validation.min_length} characters")
        
        if validation.max_length and len(value) > validation.max_length:
            errors.append(f"{field.label} must be at most {validation.max_length} characters")
        
        if validation.pattern:
            import re
            if not re.match(validation.pattern, value):
                msg = validation.pattern_message or f"{field.label} format is invalid"
                errors.append(msg)
        
        return errors


class TextField(FieldType):
    """Multi-line text area field."""
    
    field_type = FieldTypeEnum.TEXT
    default_properties = {
        "rows": 4,
        "max_rows": 20,
        "auto_resize": True,
        "show_character_count": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        rows: int = 4,
        max_rows: int = 20,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a text area field definition."""
        properties = cls.default_properties.copy()
        properties["rows"] = rows
        properties["max_rows"] = max_rows
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
            ),
            placeholder=placeholder or f"Enter {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate text value."""
        return StringField.validate_value(value, field)


class NumberField(FieldType):
    """Numeric input field for floating-point numbers."""
    
    field_type = FieldTypeEnum.NUMBER
    default_properties = {
        "step": 0.1,
        "precision": 2,
        "show_controls": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[float] = None,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 2,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a number field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        properties["precision"] = precision
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_value=min_value,
                max_value=max_value,
            ),
            prefix=prefix,
            suffix=suffix,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate number value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"{field.label} must be a number")
            return errors
        
        validation = field.validation
        
        if validation.min_value is not None and num_value < validation.min_value:
            errors.append(f"{field.label} must be at least {validation.min_value}")
        
        if validation.max_value is not None and num_value > validation.max_value:
            errors.append(f"{field.label} must be at most {validation.max_value}")
        
        return errors


class IntegerField(FieldType):
    """Integer input field."""
    
    field_type = FieldTypeEnum.INTEGER
    default_properties = {
        "step": 1,
        "show_controls": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
        required: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        step: int = 1,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an integer field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                min_value=min_value,
                max_value=max_value,
            ),
            prefix=prefix,
            suffix=suffix,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate integer value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        try:
            if isinstance(value, float) and not value.is_integer():
                errors.append(f"{field.label} must be a whole number")
                return errors
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append(f"{field.label} must be an integer")
            return errors
        
        validation = field.validation
        
        if validation.min_value is not None and int_value < validation.min_value:
            errors.append(f"{field.label} must be at least {int(validation.min_value)}")
        
        if validation.max_value is not None and int_value > validation.max_value:
            errors.append(f"{field.label} must be at most {int(validation.max_value)}")
        
        return errors


class BooleanField(FieldType):
    """Boolean toggle/checkbox field."""
    
    field_type = FieldTypeEnum.BOOLEAN
    default_properties = {
        "style": "switch",  # switch, checkbox
        "true_label": "Yes",
        "false_label": "No",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: bool = False,
        style: str = "switch",
        true_label: str = "Yes",
        false_label: str = "No",
        **kwargs
    ) -> FieldDefinition:
        """Create a boolean field definition."""
        properties = cls.default_properties.copy()
        properties["style"] = style
        properties["true_label"] = true_label
        properties["false_label"] = false_label
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate boolean value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is not None and not isinstance(value, bool):
            # Try to coerce
            if isinstance(value, str):
                if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    errors.append(f"{field.label} must be a boolean value")
            elif not isinstance(value, (int, float)):
                errors.append(f"{field.label} must be a boolean value")
        
        return errors


# =============================================================================
# SELECTION FIELD TYPES
# =============================================================================

class SelectField(FieldType):
    """Single-selection dropdown field."""
    
    field_type = FieldTypeEnum.SELECT
    default_properties = {
        "searchable": True,
        "clearable": True,
        "grouped": False,
        "virtual_scroll": True,  # For large option lists
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        searchable: bool = True,
        clearable: bool = True,
        placeholder: Optional[str] = None,
        options_source: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a select field definition."""
        properties = cls.default_properties.copy()
        properties["searchable"] = searchable
        properties["clearable"] = clearable
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            options=field_options,
            options_source=options_source,
            placeholder=placeholder or f"Select {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate select value."""
        errors = super().validate_value(value, field)
        if errors or value is None:
            return errors
        
        if field.options:
            valid_values = [opt.value for opt in field.options]
            if value not in valid_values:
                errors.append(f"{field.label} has an invalid selection")
        
        return errors


class MultiSelectField(FieldType):
    """Multi-selection field allowing multiple values."""
    
    field_type = FieldTypeEnum.MULTI_SELECT
    default_properties = {
        "searchable": True,
        "clearable": True,
        "max_selections": None,
        "min_selections": None,
        "show_selected_count": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: List[Any] = None,
        required: bool = False,
        min_selections: Optional[int] = None,
        max_selections: Optional[int] = None,
        searchable: bool = True,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a multi-select field definition."""
        properties = cls.default_properties.copy()
        properties["searchable"] = searchable
        properties["min_selections"] = min_selections
        properties["max_selections"] = max_selections
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [],
            validation=FieldValidation(required=required),
            options=field_options,
            placeholder=placeholder or f"Select {label.lower()}...",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate multi-select value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None:
            value = []
        
        if not isinstance(value, list):
            errors.append(f"{field.label} must be a list")
            return errors
        
        properties = field.properties
        
        if properties.get("min_selections") and len(value) < properties["min_selections"]:
            errors.append(f"{field.label} requires at least {properties['min_selections']} selections")
        
        if properties.get("max_selections") and len(value) > properties["max_selections"]:
            errors.append(f"{field.label} allows at most {properties['max_selections']} selections")
        
        if field.options:
            valid_values = [opt.value for opt in field.options]
            for v in value:
                if v not in valid_values:
                    errors.append(f"{field.label} contains invalid selection: {v}")
        
        return errors


class RadioField(FieldType):
    """Radio button group for single selection."""
    
    field_type = FieldTypeEnum.RADIO
    default_properties = {
        "layout": "vertical",  # vertical, horizontal, grid
        "columns": 2,  # For grid layout
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        layout: str = "vertical",
        **kwargs
    ) -> FieldDefinition:
        """Create a radio field definition."""
        properties = cls.default_properties.copy()
        properties["layout"] = layout
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            options=field_options,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate radio value."""
        return SelectField.validate_value(value, field)


class CheckboxGroupField(FieldType):
    """Checkbox group for multiple selections."""
    
    field_type = FieldTypeEnum.CHECKBOX_GROUP
    default_properties = {
        "layout": "vertical",
        "columns": 2,
        "select_all": False,  # Show "Select All" option
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        options: List[Union[FieldOption, Dict[str, Any]]],
        description: Optional[str] = None,
        default: List[Any] = None,
        required: bool = False,
        layout: str = "vertical",
        select_all: bool = False,
        **kwargs
    ) -> FieldDefinition:
        """Create a checkbox group field definition."""
        properties = cls.default_properties.copy()
        properties["layout"] = layout
        properties["select_all"] = select_all
        
        # Convert dict options to FieldOption
        field_options = []
        for opt in options:
            if isinstance(opt, dict):
                field_options.append(FieldOption(**opt))
            else:
                field_options.append(opt)
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [],
            validation=FieldValidation(required=required),
            options=field_options,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate checkbox group value."""
        return MultiSelectField.validate_value(value, field)


# =============================================================================
# SPECIAL INPUT FIELD TYPES
# =============================================================================

class PasswordField(FieldType):
    """Password input field with masking."""
    
    field_type = FieldTypeEnum.PASSWORD
    default_properties = {
        "show_toggle": True,  # Show/hide password toggle
        "strength_indicator": True,
        "generate_button": False,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        show_toggle: bool = True,
        strength_indicator: bool = False,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a password field definition."""
        properties = cls.default_properties.copy()
        properties["show_toggle"] = show_toggle
        properties["strength_indicator"] = strength_indicator
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default="",
            validation=FieldValidation(
                required=required,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
            ),
            placeholder=placeholder or "Enter password...",
            sensitive=True,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate password value."""
        return StringField.validate_value(value, field)


class EmailField(FieldType):
    """Email input field with validation."""
    
    field_type = FieldTypeEnum.EMAIL
    default_properties = {
        "suggest_domains": True,
        "common_domains": ["gmail.com", "outlook.com", "yahoo.com"],
    }
    
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an email field definition."""
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.EMAIL_PATTERN,
                pattern_message="Please enter a valid email address",
            ),
            placeholder=placeholder or "email@example.com",
            properties=cls.default_properties.copy(),
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate email value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.EMAIL_PATTERN, value):
            errors.append("Please enter a valid email address")
        
        return errors


class UrlField(FieldType):
    """URL input field with validation."""
    
    field_type = FieldTypeEnum.URL
    default_properties = {
        "protocols": ["http", "https"],
        "show_preview": False,
    }
    
    URL_PATTERN = r'^https?://[^\s/$.?#].[^\s]*$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        protocols: List[str] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a URL field definition."""
        properties = cls.default_properties.copy()
        if protocols:
            properties["protocols"] = protocols
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.URL_PATTERN,
                pattern_message="Please enter a valid URL",
            ),
            placeholder=placeholder or "https://example.com",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate URL value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.URL_PATTERN, value):
            errors.append("Please enter a valid URL")
        
        return errors


class ColorField(FieldType):
    """Color picker field."""
    
    field_type = FieldTypeEnum.COLOR
    default_properties = {
        "format": "hex",  # hex, rgb, hsl
        "show_alpha": False,
        "presets": [],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "#000000",
        required: bool = False,
        format: str = "hex",
        show_alpha: bool = False,
        presets: List[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a color field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["show_alpha"] = show_alpha
        if presets:
            properties["presets"] = presets
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


class DateField(FieldType):
    """Date picker field."""
    
    field_type = FieldTypeEnum.DATE
    default_properties = {
        "format": "YYYY-MM-DD",
        "min_date": None,
        "max_date": None,
        "disabled_dates": [],
        "show_today_button": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "YYYY-MM-DD",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a date field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["min_date"] = min_date
        properties["max_date"] = max_date
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select date...",
            properties=properties,
            **kwargs
        )


class DateTimeField(FieldType):
    """Date and time picker field."""
    
    field_type = FieldTypeEnum.DATETIME
    default_properties = {
        "format": "YYYY-MM-DD HH:mm",
        "time_format": "24h",  # 12h, 24h
        "min_datetime": None,
        "max_datetime": None,
        "step_minutes": 15,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "YYYY-MM-DD HH:mm",
        time_format: str = "24h",
        step_minutes: int = 15,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a datetime field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["time_format"] = time_format
        properties["step_minutes"] = step_minutes
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select date and time...",
            properties=properties,
            **kwargs
        )


class TimeField(FieldType):
    """Time picker field."""
    
    field_type = FieldTypeEnum.TIME
    default_properties = {
        "format": "HH:mm",
        "time_format": "24h",
        "step_minutes": 15,
        "min_time": None,
        "max_time": None,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        format: str = "HH:mm",
        time_format: str = "24h",
        step_minutes: int = 15,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a time field definition."""
        properties = cls.default_properties.copy()
        properties["format"] = format
        properties["time_format"] = time_format
        properties["step_minutes"] = step_minutes
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Select time...",
            properties=properties,
            **kwargs
        )


# =============================================================================
# COMPLEX FIELD TYPES
# =============================================================================

class JsonField(FieldType):
    """JSON editor field."""
    
    field_type = FieldTypeEnum.JSON
    default_properties = {
        "mode": "code",  # code, tree, view
        "validate_json": True,
        "indent": 2,
        "line_numbers": True,
        "folding": True,
        "schema": None,  # JSON schema for validation
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: Any = None,
        required: bool = False,
        mode: str = "code",
        schema: Optional[Dict[str, Any]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a JSON field definition."""
        properties = cls.default_properties.copy()
        properties["mode"] = mode
        if schema:
            properties["schema"] = schema
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default if default is not None else {},
            validation=FieldValidation(required=required),
            placeholder=placeholder or '{}',
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate JSON value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None or value == "":
            return errors
        
        if isinstance(value, str):
            import json
            try:
                json.loads(value)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {str(e)}")
        
        return errors


class CodeField(FieldType):
    """Code editor field with syntax highlighting."""
    
    field_type = FieldTypeEnum.CODE
    default_properties = {
        "language": "python",
        "theme": "vs-dark",
        "line_numbers": True,
        "folding": True,
        "minimap": False,
        "auto_format": True,
        "tab_size": 4,
    }
    
    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "json", "yaml", "markdown",
        "html", "css", "sql", "bash", "shell", "plaintext"
    ]
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        language: str = "python",
        theme: str = "vs-dark",
        line_numbers: bool = True,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a code editor field definition."""
        properties = cls.default_properties.copy()
        properties["language"] = language
        properties["theme"] = theme
        properties["line_numbers"] = line_numbers
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or f"Enter {language} code...",
            properties=properties,
            **kwargs
        )


class SliderField(FieldType):
    """Slider input for numeric values."""
    
    field_type = FieldTypeEnum.SLIDER
    default_properties = {
        "show_value": True,
        "show_marks": True,
        "marks": None,  # Custom marks: [{value: 0, label: "Min"}, ...]
        "tooltip": "always",  # always, hover, never
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        min_value: float,
        max_value: float,
        description: Optional[str] = None,
        default: Optional[float] = None,
        step: float = 1,
        marks: Optional[List[Dict[str, Any]]] = None,
        show_value: bool = True,
        **kwargs
    ) -> FieldDefinition:
        """Create a slider field definition."""
        properties = cls.default_properties.copy()
        properties["show_value"] = show_value
        properties["step"] = step
        if marks:
            properties["marks"] = marks
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default if default is not None else min_value,
            validation=FieldValidation(
                min_value=min_value,
                max_value=max_value,
            ),
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate slider value."""
        return NumberField.validate_value(value, field)


class RangeField(FieldType):
    """Range slider for selecting a range of values."""
    
    field_type = FieldTypeEnum.RANGE
    default_properties = {
        "show_values": True,
        "show_marks": True,
        "tooltip": "always",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        min_value: float,
        max_value: float,
        description: Optional[str] = None,
        default: Optional[List[float]] = None,
        step: float = 1,
        **kwargs
    ) -> FieldDefinition:
        """Create a range field definition."""
        properties = cls.default_properties.copy()
        properties["step"] = step
        properties["min"] = min_value
        properties["max"] = max_value
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=default or [min_value, max_value],
            validation=FieldValidation(
                min_value=min_value,
                max_value=max_value,
            ),
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate range value."""
        errors = super().validate_value(value, field)
        if errors:
            return errors
        
        if value is None:
            return errors
        
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            errors.append(f"{field.label} must be a range with two values")
            return errors
        
        min_val, max_val = value
        if min_val > max_val:
            errors.append(f"{field.label} minimum cannot be greater than maximum")
        
        return errors


class FileField(FieldType):
    """File upload field."""
    
    field_type = FieldTypeEnum.FILE
    default_properties = {
        "accept": "*/*",
        "multiple": False,
        "max_size": 10 * 1024 * 1024,  # 10MB
        "show_preview": True,
        "drag_drop": True,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        accept: str = "*/*",
        multiple: bool = False,
        max_size: int = 10 * 1024 * 1024,
        **kwargs
    ) -> FieldDefinition:
        """Create a file upload field definition."""
        properties = cls.default_properties.copy()
        properties["accept"] = accept
        properties["multiple"] = multiple
        properties["max_size"] = max_size
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=None,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


class ImageField(FieldType):
    """Image upload field with preview."""
    
    field_type = FieldTypeEnum.IMAGE
    default_properties = {
        "accept": "image/*",
        "max_size": 5 * 1024 * 1024,  # 5MB
        "show_preview": True,
        "preview_size": {"width": 200, "height": 200},
        "crop": False,
        "aspect_ratio": None,
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        required: bool = False,
        max_size: int = 5 * 1024 * 1024,
        crop: bool = False,
        aspect_ratio: Optional[float] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create an image upload field definition."""
        properties = cls.default_properties.copy()
        properties["max_size"] = max_size
        properties["crop"] = crop
        if aspect_ratio:
            properties["aspect_ratio"] = aspect_ratio
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default=None,
            validation=FieldValidation(required=required),
            properties=properties,
            **kwargs
        )


# =============================================================================
# CUSTOM AGENT BUILDER FIELD TYPES
# =============================================================================

class ApiKeyField(FieldType):
    """API key input field with provider-specific validation."""
    
    field_type = FieldTypeEnum.API_KEY
    default_properties = {
        "provider": None,
        "show_toggle": True,
        "validate_format": True,
        "test_connection": True,
    }
    
    # Provider-specific patterns
    PROVIDER_PATTERNS = {
        "openai": r"^sk-[a-zA-Z0-9]{32,}$",
        "anthropic": r"^sk-ant-[a-zA-Z0-9-]{32,}$",
        "openrouter": r"^sk-or-v1-[a-zA-Z0-9]{32,}$",
        "composio": r"^[a-zA-Z0-9]{20,}$",
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        provider: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = True,
        placeholder: Optional[str] = None,
        test_connection: bool = True,
        **kwargs
    ) -> FieldDefinition:
        """Create an API key field definition."""
        properties = cls.default_properties.copy()
        properties["provider"] = provider
        properties["test_connection"] = test_connection
        
        pattern = cls.PROVIDER_PATTERNS.get(provider) if provider else None
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or f"API key for {provider or 'the service'}",
            default="",
            validation=FieldValidation(
                required=required,
                pattern=pattern,
                pattern_message=f"Invalid {provider or 'API'} key format" if pattern else None,
            ),
            placeholder=placeholder or f"Enter {provider or 'API'} key...",
            sensitive=True,
            copyable=True,
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate API key value."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        provider = field.properties.get("provider")
        if provider and provider in cls.PROVIDER_PATTERNS:
            import re
            if not re.match(cls.PROVIDER_PATTERNS[provider], value):
                errors.append(f"Invalid {provider} API key format")
        
        return errors


class ModelSelectField(FieldType):
    """Model selection field with provider-specific options."""
    
    field_type = FieldTypeEnum.MODEL_SELECT
    default_properties = {
        "provider": None,
        "show_model_info": True,
        "allow_custom": True,
        "group_by_provider": True,
    }
    
    # Default models by provider
    PROVIDER_MODELS = {
        "openai": [
            {"value": "gpt-4o", "label": "GPT-4o (Latest)", "group": "GPT-4"},
            {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "GPT-4"},
            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "GPT-4"},
            {"value": "gpt-4", "label": "GPT-4", "group": "GPT-4"},
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "group": "GPT-3.5"},
        ],
        "anthropic": [
            {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet", "group": "Claude 3.5"},
            {"value": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku", "group": "Claude 3.5"},
            {"value": "claude-3-opus-20240229", "label": "Claude 3 Opus", "group": "Claude 3"},
            {"value": "claude-3-sonnet-20240229", "label": "Claude 3 Sonnet", "group": "Claude 3"},
            {"value": "claude-3-haiku-20240307", "label": "Claude 3 Haiku", "group": "Claude 3"},
        ],
        "openrouter": [
            {"value": "openai/gpt-4o", "label": "OpenAI GPT-4o", "group": "OpenAI"},
            {"value": "anthropic/claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "group": "Anthropic"},
            {"value": "google/gemini-pro-1.5", "label": "Gemini 1.5 Pro", "group": "Google"},
            {"value": "meta-llama/llama-3-70b-instruct", "label": "Llama 3 70B", "group": "Meta"},
            {"value": "mistralai/mistral-large", "label": "Mistral Large", "group": "Mistral"},
        ],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        provider: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = True,
        allow_custom: bool = True,
        custom_models: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a model select field definition."""
        properties = cls.default_properties.copy()
        properties["provider"] = provider
        properties["allow_custom"] = allow_custom
        
        # Get models for provider
        models = custom_models or cls.PROVIDER_MODELS.get(provider, [])
        options = [FieldOption(**m) for m in models]
        
        # Set default if not provided
        if not default and options:
            default = options[0].value
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or f"Select a {provider} model",
            default=default,
            validation=FieldValidation(required=required),
            options=options,
            placeholder=f"Select {provider} model...",
            properties=properties,
            **kwargs
        )


class PromptField(FieldType):
    """Prompt/system message input field with variable support."""
    
    field_type = FieldTypeEnum.PROMPT
    default_properties = {
        "rows": 6,
        "max_rows": 20,
        "show_variable_picker": True,
        "show_token_count": True,
        "syntax_highlighting": True,
        "available_variables": [],
        "templates": [],
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = False,
        rows: int = 6,
        available_variables: Optional[List[str]] = None,
        templates: Optional[List[Dict[str, str]]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a prompt field definition."""
        properties = cls.default_properties.copy()
        properties["rows"] = rows
        if available_variables:
            properties["available_variables"] = available_variables
        if templates:
            properties["templates"] = templates
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or "Enter prompt with optional {{variables}}",
            default=default,
            validation=FieldValidation(required=required),
            placeholder=placeholder or "Enter your prompt...",
            properties=properties,
            **kwargs
        )


class VariableField(FieldType):
    """Variable name input field with validation."""
    
    field_type = FieldTypeEnum.VARIABLE
    default_properties = {
        "validate_identifier": True,
        "suggest_names": True,
        "reserved_words": ["input", "output", "self", "this"],
    }
    
    IDENTIFIER_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        description: Optional[str] = None,
        default: str = "",
        required: bool = True,
        reserved_words: Optional[List[str]] = None,
        placeholder: Optional[str] = None,
        **kwargs
    ) -> FieldDefinition:
        """Create a variable name field definition."""
        properties = cls.default_properties.copy()
        if reserved_words:
            properties["reserved_words"] = reserved_words
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description or "Must be a valid identifier (letters, numbers, underscores)",
            default=default,
            validation=FieldValidation(
                required=required,
                pattern=cls.IDENTIFIER_PATTERN,
                pattern_message="Must start with letter/underscore, contain only letters, numbers, underscores",
                forbidden_values=properties.get("reserved_words", []),
            ),
            placeholder=placeholder or "my_variable",
            properties=properties,
            **kwargs
        )
    
    @classmethod
    def validate_value(cls, value: Any, field: FieldDefinition) -> List[str]:
        """Validate variable name."""
        errors = super().validate_value(value, field)
        if errors or not value:
            return errors
        
        import re
        if not re.match(cls.IDENTIFIER_PATTERN, value):
            errors.append("Variable name must be a valid identifier")
        
        reserved = field.properties.get("reserved_words", [])
        if value in reserved:
            errors.append(f"'{value}' is a reserved word and cannot be used")
        
        return errors


class PortField(FieldType):
    """Port configuration field for component connections."""
    
    field_type = FieldTypeEnum.PORT
    default_properties = {
        "port_type": "input",  # input, output
        "data_type": "any",
        "show_type_selector": True,
    }
    
    DATA_TYPES = [
        {"value": "any", "label": "Any"},
        {"value": "string", "label": "String"},
        {"value": "number", "label": "Number"},
        {"value": "boolean", "label": "Boolean"},
        {"value": "message", "label": "Message"},
        {"value": "messages", "label": "Messages"},
        {"value": "dict", "label": "Dictionary"},
        {"value": "list", "label": "List"},
        {"value": "tools", "label": "Tools"},
        {"value": "llm_model", "label": "LLM Model"},
    ]
    
    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        port_type: str = "input",
        data_type: str = "any",
        description: Optional[str] = None,
        required: bool = False,
        **kwargs
    ) -> FieldDefinition:
        """Create a port configuration field definition."""
        properties = cls.default_properties.copy()
        properties["port_type"] = port_type
        properties["data_type"] = data_type
        
        return FieldDefinition(
            name=name,
            type=cls.field_type,
            label=label,
            description=description,
            default={"type": data_type, "required": required},
            validation=FieldValidation(required=required),
            options=[FieldOption(**dt) for dt in cls.DATA_TYPES],
            properties=properties,
            **kwargs
        )

```

# components/field_types/validators.py

```python
"""
Field validation utilities.

This module provides validation functions for field values.
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from .base import FieldDefinition, FieldTypeEnum, FieldCondition
from .registry import field_type_registry


@dataclass
class ValidationResult:
    """Result of field validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    field_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "field_name": self.field_name,
        }


class FieldValidator:
    """Validates field values against their definitions."""
    
    @staticmethod
    def validate(
        value: Any,
        field: FieldDefinition,
        all_values: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a value against a field definition.
        
        Args:
            value: The value to validate
            field: The field definition
            all_values: All field values (for conditional validation)
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check conditional requirements
        if all_values and field.dependencies:
            for dep in field.dependencies:
                if dep.should_apply(all_values):
                    if dep.action == "require" and not value:
                        errors.append(f"{field.label} is required")
        
        # Get field type class for validation
        type_class = field_type_registry.get(FieldTypeEnum(field.type))
        if type_class:
            type_errors = type_class.validate_value(value, field)
            errors.extend(type_errors)
        else:
            # Basic validation
            errors.extend(FieldValidator._basic_validate(value, field))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            field_name=field.name,
        )
    
    @staticmethod
    def _basic_validate(value: Any, field: FieldDefinition) -> List[str]:
        """Basic validation for unknown field types."""
        errors = []
        validation = field.validation
        
        # Required check
        if validation.required and (value is None or value == ""):
            errors.append(f"{field.label} is required")
            return errors
        
        # Skip further validation if empty and not required
        if value is None or value == "":
            return errors
        
        # Allowed values check
        if validation.allowed_values and value not in validation.allowed_values:
            errors.append(f"{field.label} has an invalid value")
        
        # Forbidden values check
        if validation.forbidden_values and value in validation.forbidden_values:
            errors.append(f"{field.label} contains a forbidden value")
        
        return errors


def validate_field(
    value: Any,
    field: FieldDefinition,
    all_values: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Convenience function to validate a single field.
    
    Args:
        value: The value to validate
        field: The field definition
        all_values: All field values (for conditional validation)
        
    Returns:
        ValidationResult
    """
    return FieldValidator.validate(value, field, all_values)


def validate_fields(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> Dict[str, ValidationResult]:
    """
    Validate multiple fields.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        Dictionary mapping field names to ValidationResults
    """
    results = {}
    for field in fields:
        value = values.get(field.name)
        results[field.name] = FieldValidator.validate(value, field, values)
    return results


def get_validation_errors(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> List[str]:
    """
    Get all validation errors for a set of fields.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        List of error messages
    """
    all_errors = []
    results = validate_fields(values, fields)
    
    for field_name, result in results.items():
        all_errors.extend(result.errors)
    
    return all_errors


def is_valid(
    values: Dict[str, Any],
    fields: List[FieldDefinition]
) -> bool:
    """
    Check if all fields are valid.
    
    Args:
        values: Dictionary of field values
        fields: List of field definitions
        
    Returns:
        True if all fields are valid
    """
    results = validate_fields(values, fields)
    return all(result.valid for result in results.values())

```

# components/input_component.py

```python
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

```

# components/llm/__init__.py

```python
"""LLM models package initialization."""
from .base import BaseLLMModel, LLMConfig, LLMResponse, LLMProvider
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .openrouter_model import OpenRouterModel
from .registry import LLMRegistry

__all__ = [
    "BaseLLMModel",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    "OpenAIModel",
    "AnthropicModel",
    "OpenRouterModel",
    "LLMRegistry",
]

```

# components/llm/anthropic_model.py

```python
"""
Anthropic LLM model component.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List, Optional, AsyncIterator
from .base import BaseLLMModel, LLMProvider, LLMResponse
from components.field_types import (
    FieldDefinition,
    ApiKeyField,
    ModelSelectField,
    StringField,
)


class AnthropicModel(BaseLLMModel):
    """
    Anthropic model component supporting Claude models.
    """
    
    _component_type = "llm_anthropic"
    _name = "Anthropic Model"
    _description = "Anthropic Claude language models"
    _category = "models"
    _icon = "brain"
    _color = "#d97757"
    _provider = LLMProvider.ANTHROPIC
    _supported_models = [
        {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet (Latest)", "group": "Claude 3.5"},
        {"value": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku", "group": "Claude 3.5"},
        {"value": "claude-3-opus-20240229", "label": "Claude 3 Opus", "group": "Claude 3"},
        {"value": "claude-3-sonnet-20240229", "label": "Claude 3 Sonnet", "group": "Claude 3"},
        {"value": "claude-3-haiku-20240307", "label": "Claude 3 Haiku", "group": "Claude 3"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get Anthropic-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="anthropic",
                description="Your Anthropic API key",
                required=True,
                placeholder="sk-ant-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="anthropic",
                description="Claude model to use",
                default="claude-3-5-sonnet-20241022",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="Custom API base URL (optional)",
                default="https://api.anthropic.com",
                placeholder="https://api.anthropic.com",
                order=3,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        from langchain_anthropic import ChatAnthropic
        
        self._client = ChatAnthropic(
            api_key=self.get_parameter("api_key"),
            model=self.get_parameter("model", "claude-3-5-sonnet-20241022"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
        )
    
    async def _generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """Generate a response using Anthropic."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        # Bind tools if provided
        if tools:
            model_with_tools = self._client.bind_tools(tools)
            response = await model_with_tools.ainvoke(lc_messages)
        else:
            response = await self._client.ainvoke(lc_messages)
        
        # Extract tool calls if any
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
        
        return LLMResponse(
            content=response.content,
            model=self.get_parameter("model", "claude-3-5-sonnet-20241022"),
            provider="anthropic",
            usage={
                "input_tokens": response.response_metadata.get("usage", {}).get("input_tokens", 0),
                "output_tokens": response.response_metadata.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=response.response_metadata.get("stop_reason"),
            metadata={"tool_calls": tool_calls}
        )
    
    async def _stream_generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Stream a response using Anthropic."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        async for chunk in self._client.astream(lc_messages):
            if chunk.content:
                yield chunk.content

```

# components/llm/base.py

```python
"""
Base LLM model class - foundation for all LLM providers.
Uses centralized field types for configuration.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from pydantic import BaseModel, Field
from enum import Enum

from components.base import BaseComponent, ComponentPort, PortType, FieldDefinition, FieldGroup
from components.field_types import (
    NumberField,
    IntegerField,
    BooleanField,
    ApiKeyField,
    ModelSelectField,
    StringField,
    SliderField,
)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class LLMConfig(BaseModel):
    """Configuration for LLM model."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: List[str] = Field(default_factory=list)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM model."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMModel(BaseComponent, ABC):
    """
    Abstract base class for all LLM model components.
    """
    
    _component_type = "llm_model"
    _name = "LLM Model"
    _description = "Base LLM model component"
    _category = "models"
    _icon = "brain"
    _color = "#8b5cf6"
    _provider: LLMProvider = LLMProvider.CUSTOM
    _supported_models: List[Dict[str, str]] = []
    
    def __init__(
        self, 
        node_id: Optional[str] = None, 
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None
    ):
        """Initialize the LLM model component."""
        super().__init__(node_id, parameters)
        self.config = config
        self._client = None
    
    @classmethod
    def _get_input_ports(cls) -> List[ComponentPort]:
        """LLM models receive prompts/messages."""
        return [
            ComponentPort(
                id="messages",
                name="Messages",
                type=PortType.INPUT,
                data_type="messages",
                required=True,
                description="Chat messages to send to the model"
            ),
            ComponentPort(
                id="system_prompt",
                name="System Prompt",
                type=PortType.INPUT,
                data_type="string",
                description="System prompt to configure model behavior"
            ),
            ComponentPort(
                id="tools",
                name="Tools",
                type=PortType.INPUT,
                data_type="tools",
                multiple=True,
                description="Tools available for the model to use"
            )
        ]
    
    @classmethod
    def _get_output_ports(cls) -> List[ComponentPort]:
        """LLM models output responses."""
        return [
            ComponentPort(
                id="response",
                name="Response",
                type=PortType.OUTPUT,
                data_type="message",
                description="Model response message"
            ),
            ComponentPort(
                id="tool_calls",
                name="Tool Calls",
                type=PortType.OUTPUT,
                data_type="tool_calls",
                description="Tool calls requested by the model"
            )
        ]
    
    @classmethod
    def _get_base_fields(cls) -> List[FieldDefinition]:
        """Get base fields common to all LLM models."""
        return [
            SliderField.create(
                name="temperature",
                label="Temperature",
                description="Controls randomness in output (0 = deterministic, 2 = creative)",
                min_value=0.0,
                max_value=2.0,
                default=0.7,
                step=0.1,
                order=10,
                group="generation",
            ),
            IntegerField.create(
                name="max_tokens",
                label="Max Tokens",
                description="Maximum tokens in response",
                default=4096,
                min_value=1,
                max_value=128000,
                order=11,
                group="generation",
            ),
            SliderField.create(
                name="top_p",
                label="Top P",
                description="Nucleus sampling parameter (alternative to temperature)",
                min_value=0.0,
                max_value=1.0,
                default=1.0,
                step=0.05,
                order=12,
                group="advanced",
            ),
            SliderField.create(
                name="frequency_penalty",
                label="Frequency Penalty",
                description="Penalize frequent tokens (-2 to 2)",
                min_value=-2.0,
                max_value=2.0,
                default=0.0,
                step=0.1,
                order=13,
                group="advanced",
            ),
            SliderField.create(
                name="presence_penalty",
                label="Presence Penalty",
                description="Penalize tokens already present (-2 to 2)",
                min_value=-2.0,
                max_value=2.0,
                default=0.0,
                step=0.1,
                order=14,
                group="advanced",
            ),
            BooleanField.create(
                name="stream",
                label="Stream Response",
                description="Enable response streaming",
                default=False,
                order=15,
                group="options",
            ),
        ]
    
    @classmethod
    @abstractmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get provider-specific fields (to be implemented by subclasses)."""
        pass
    
    @classmethod
    def _get_fields(cls) -> List[FieldDefinition]:
        """Combine base and provider-specific fields."""
        return cls._get_provider_fields() + cls._get_base_fields()
    
    @classmethod
    def _get_field_groups(cls) -> List[FieldGroup]:
        """Define field groups for organization."""
        return [
            FieldGroup(
                id="connection",
                label="Connection",
                description="API connection settings",
                order=0,
            ),
            FieldGroup(
                id="model",
                label="Model",
                description="Model selection",
                order=1,
            ),
            FieldGroup(
                id="generation",
                label="Generation Settings",
                description="Control output generation",
                collapsible=True,
                order=2,
            ),
            FieldGroup(
                id="options",
                label="Options",
                collapsible=True,
                order=3,
            ),
            FieldGroup(
                id="advanced",
                label="Advanced",
                description="Advanced model parameters",
                collapsible=True,
                collapsed_by_default=True,
                order=4,
            ),
        ]
    
    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the LLM client (provider-specific)."""
        pass
    
    @abstractmethod
    async def _generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """Generate a response from the model (provider-specific)."""
        pass
    
    @abstractmethod
    async def _stream_generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Stream a response from the model (provider-specific)."""
        pass
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM model with the given inputs."""
        if self._client is None:
            await self._initialize_client()
        
        messages = inputs.get("messages", [])
        system_prompt = inputs.get("system_prompt")
        tools = inputs.get("tools")
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        if self.get_parameter("stream", False):
            return {
                "response": self._stream_generate(messages, tools),
                "streaming": True
            }
        else:
            response = await self._generate(messages, tools)
            return {
                "response": {
                    "role": "assistant",
                    "content": response.content
                },
                "tool_calls": response.metadata.get("tool_calls"),
                "usage": response.usage,
                "model": response.model
            }
    
    def get_llm_config(self) -> LLMConfig:
        """Build LLMConfig from parameters."""
        return LLMConfig(
            provider=self._provider,
            model=self.get_parameter("model", ""),
            api_key=self.get_parameter("api_key"),
            base_url=self.get_parameter("base_url"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            top_p=self.get_parameter("top_p", 1.0),
            frequency_penalty=self.get_parameter("frequency_penalty", 0.0),
            presence_penalty=self.get_parameter("presence_penalty", 0.0)
        )
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema with LLM-specific fields."""
        schema = super().to_schema()
        schema["provider"] = self._provider.value
        schema["model_config"] = self.get_llm_config().model_dump()
        return schema

```

# components/llm/openai_model.py

```python
"""
OpenAI LLM model component.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List, Optional, AsyncIterator
from .base import BaseLLMModel, LLMProvider, LLMResponse
from components.field_types import (
    FieldDefinition,
    ApiKeyField,
    ModelSelectField,
    StringField,
)


class OpenAIModel(BaseLLMModel):
    """
    OpenAI model component supporting GPT-4, GPT-3.5, and other OpenAI models.
    """
    
    _component_type = "llm_openai"
    _name = "OpenAI Model"
    _description = "OpenAI language models (GPT-4, GPT-3.5, etc.)"
    _category = "models"
    _icon = "brain"
    _color = "#10a37f"
    _provider = LLMProvider.OPENAI
    _supported_models = [
        {"value": "gpt-4o", "label": "GPT-4o (Latest)", "group": "GPT-4"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "GPT-4"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "GPT-4"},
        {"value": "gpt-4", "label": "GPT-4", "group": "GPT-4"},
        {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "group": "GPT-3.5"},
        {"value": "gpt-3.5-turbo-16k", "label": "GPT-3.5 Turbo 16K", "group": "GPT-3.5"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get OpenAI-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="openai",
                description="Your OpenAI API key",
                required=True,
                placeholder="sk-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="openai",
                description="OpenAI model to use",
                default="gpt-4o",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="Custom API base URL (optional)",
                default="https://api.openai.com/v1",
                placeholder="https://api.openai.com/v1",
                order=3,
                group="connection",
            ),
            StringField.create(
                name="organization",
                label="Organization ID",
                description="OpenAI organization ID (optional)",
                placeholder="org-...",
                order=4,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        from langchain_openai import ChatOpenAI
        
        self._client = ChatOpenAI(
            api_key=self.get_parameter("api_key"),
            model=self.get_parameter("model", "gpt-4o"),
            base_url=self.get_parameter("base_url"),
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            model_kwargs={
                "top_p": self.get_parameter("top_p", 1.0),
                "frequency_penalty": self.get_parameter("frequency_penalty", 0.0),
                "presence_penalty": self.get_parameter("presence_penalty", 0.0),
            }
        )
    
    async def _generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        # Bind tools if provided
        if tools:
            model_with_tools = self._client.bind_tools(tools)
            response = await model_with_tools.ainvoke(lc_messages)
        else:
            response = await self._client.ainvoke(lc_messages)
        
        # Extract tool calls if any
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
        
        return LLMResponse(
            content=response.content,
            model=self.get_parameter("model", "gpt-4o"),
            provider="openai",
            usage={
                "prompt_tokens": response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            },
            finish_reason=response.response_metadata.get("finish_reason"),
            metadata={"tool_calls": tool_calls}
        )
    
    async def _stream_generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Stream a response using OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        async for chunk in self._client.astream(lc_messages):
            if chunk.content:
                yield chunk.content

```

# components/llm/openrouter_model.py

```python
"""
OpenRouter LLM model component - access multiple models through one API.
Uses centralized field types for configuration.
"""
from typing import Dict, Any, List, Optional, AsyncIterator
from .base import BaseLLMModel, LLMProvider, LLMResponse
from components.field_types import (
    FieldDefinition,
    ApiKeyField,
    ModelSelectField,
    StringField,
)


class OpenRouterModel(BaseLLMModel):
    """
    OpenRouter model component - provides access to multiple LLM providers
    through a single API endpoint.
    """
    
    _component_type = "llm_openrouter"
    _name = "OpenRouter Model"
    _description = "Access multiple LLM providers through OpenRouter"
    _category = "models"
    _icon = "globe"
    _color = "#6366f1"
    _provider = LLMProvider.OPENROUTER
    _supported_models = [
        # OpenAI Models
        {"value": "openai/gpt-4o", "label": "OpenAI GPT-4o", "group": "OpenAI"},
        {"value": "openai/gpt-4-turbo", "label": "OpenAI GPT-4 Turbo", "group": "OpenAI"},
        {"value": "openai/gpt-3.5-turbo", "label": "OpenAI GPT-3.5 Turbo", "group": "OpenAI"},
        # Anthropic Models
        {"value": "anthropic/claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "group": "Anthropic"},
        {"value": "anthropic/claude-3-opus", "label": "Claude 3 Opus", "group": "Anthropic"},
        {"value": "anthropic/claude-3-haiku", "label": "Claude 3 Haiku", "group": "Anthropic"},
        # Google Models
        {"value": "google/gemini-pro", "label": "Google Gemini Pro", "group": "Google"},
        {"value": "google/gemini-pro-1.5", "label": "Google Gemini 1.5 Pro", "group": "Google"},
        # Meta Models
        {"value": "meta-llama/llama-3-70b-instruct", "label": "Llama 3 70B", "group": "Meta"},
        {"value": "meta-llama/llama-3-8b-instruct", "label": "Llama 3 8B", "group": "Meta"},
        # Mistral Models
        {"value": "mistralai/mistral-large", "label": "Mistral Large", "group": "Mistral"},
        {"value": "mistralai/mixtral-8x7b-instruct", "label": "Mixtral 8x7B", "group": "Mistral"},
        # Free Models
        {"value": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo:free", "label": "Nous Hermes 2 (Free)", "group": "Free"},
    ]
    
    @classmethod
    def _get_provider_fields(cls) -> List[FieldDefinition]:
        """Get OpenRouter-specific fields."""
        return [
            ApiKeyField.create(
                name="api_key",
                label="API Key",
                provider="openrouter",
                description="Your OpenRouter API key",
                required=True,
                placeholder="sk-or-v1-...",
                order=1,
                group="connection",
            ),
            ModelSelectField.create(
                name="model",
                label="Model",
                provider="openrouter",
                description="Model to use via OpenRouter",
                default="openai/gpt-4o",
                required=True,
                custom_models=cls._supported_models,
                order=2,
                group="model",
            ),
            StringField.create(
                name="custom_model",
                label="Custom Model ID",
                description="Enter a custom model ID if not in the list",
                placeholder="provider/model-name",
                order=3,
                group="model",
            ),
            StringField.create(
                name="base_url",
                label="Base URL",
                description="OpenRouter API base URL",
                default="https://openrouter.ai/api/v1",
                order=4,
                group="connection",
            ),
            StringField.create(
                name="site_url",
                label="Site URL",
                description="Your site URL for OpenRouter ranking",
                placeholder="https://yoursite.com",
                order=5,
                group="connection",
            ),
            StringField.create(
                name="site_name",
                label="Site Name",
                description="Your site name for OpenRouter ranking",
                placeholder="Your App Name",
                order=6,
                group="connection",
            ),
        ]
    
    async def _initialize_client(self) -> None:
        """Initialize the OpenRouter client using LangChain OpenAI."""
        from langchain_openai import ChatOpenAI
        
        # Use custom model if provided, otherwise use selected model
        model = self.get_parameter("custom_model") or self.get_parameter("model", "openai/gpt-4o")
        
        # Build extra headers for OpenRouter
        extra_headers = {}
        if site_url := self.get_parameter("site_url"):
            extra_headers["HTTP-Referer"] = site_url
        if site_name := self.get_parameter("site_name"):
            extra_headers["X-Title"] = site_name
        
        self._client = ChatOpenAI(
            api_key=self.get_parameter("api_key"),
            base_url=self.get_parameter("base_url", "https://openrouter.ai/api/v1"),
            model=model,
            temperature=self.get_parameter("temperature", 0.7),
            max_tokens=self.get_parameter("max_tokens", 4096),
            default_headers=extra_headers if extra_headers else None,
            model_kwargs={
                "top_p": self.get_parameter("top_p", 1.0),
                "frequency_penalty": self.get_parameter("frequency_penalty", 0.0),
                "presence_penalty": self.get_parameter("presence_penalty", 0.0),
            }
        )
    
    async def _generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """Generate a response using OpenRouter."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        # Bind tools if provided
        if tools:
            model_with_tools = self._client.bind_tools(tools)
            response = await model_with_tools.ainvoke(lc_messages)
        else:
            response = await self._client.ainvoke(lc_messages)
        
        # Extract tool calls if any
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
        
        model = self.get_parameter("custom_model") or self.get_parameter("model", "openai/gpt-4o")
        
        return LLMResponse(
            content=response.content,
            model=model,
            provider="openrouter",
            usage={
                "prompt_tokens": response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            },
            finish_reason=response.response_metadata.get("finish_reason"),
            metadata={"tool_calls": tool_calls}
        )
    
    async def _stream_generate(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Stream a response using OpenRouter."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        async for chunk in self._client.astream(lc_messages):
            if chunk.content:
                yield chunk.content

```

# components/llm/registry.py

```python
"""
LLM Model Registry - manages available LLM providers and models.
"""
from typing import Dict, Type, List, Optional, Any
from .base import BaseLLMModel, LLMProvider
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .openrouter_model import OpenRouterModel


class LLMRegistry:
    """
    Registry for LLM model components.
    
    This class manages the registration and retrieval of LLM model
    components, allowing for dynamic addition of new providers.
    """
    
    _instance: Optional["LLMRegistry"] = None
    _models: Dict[str, Type[BaseLLMModel]] = {}
    
    def __new__(cls) -> "LLMRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_models()
        return cls._instance
    
    def _initialize_default_models(self) -> None:
        """Register default LLM models."""
        self.register(OpenAIModel)
        self.register(AnthropicModel)
        self.register(OpenRouterModel)
    
    def register(self, model_class: Type[BaseLLMModel]) -> None:
        """
        Register a new LLM model class.
        
        Args:
            model_class: The model class to register
        """
        self._models[model_class._component_type] = model_class
    
    def unregister(self, component_type: str) -> bool:
        """
        Unregister an LLM model.
        
        Args:
            component_type: The component type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if component_type in self._models:
            del self._models[component_type]
            return True
        return False
    
    def get(self, component_type: str) -> Optional[Type[BaseLLMModel]]:
        """
        Get an LLM model class by component type.
        
        Args:
            component_type: The component type identifier
            
        Returns:
            The model class or None if not found
        """
        return self._models.get(component_type)
    
    def create(
        self, 
        component_type: str, 
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseLLMModel]:
        """
        Create an instance of an LLM model.
        
        Args:
            component_type: The component type identifier
            node_id: Optional node ID
            parameters: Component parameters
            
        Returns:
            Model instance or None if not found
        """
        model_class = self.get(component_type)
        if model_class:
            return model_class(node_id=node_id, parameters=parameters)
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered LLM models with their configurations.
        
        Returns:
            List of model configurations
        """
        return [
            {
                "component_type": component_type,
                "config": model_class.get_config().model_dump()
            }
            for component_type, model_class in self._models.items()
        ]
    
    def get_by_provider(self, provider: LLMProvider) -> List[Type[BaseLLMModel]]:
        """
        Get all models for a specific provider.
        
        Args:
            provider: The LLM provider
            
        Returns:
            List of model classes for the provider
        """
        return [
            model_class 
            for model_class in self._models.values() 
            if model_class._provider == provider
        ]
    
    @property
    def available_types(self) -> List[str]:
        """Get list of available component types."""
        return list(self._models.keys())


# Global registry instance
llm_registry = LLMRegistry()

```

# components/output_component.py

```python
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

```

# components/registry.py

```python
"""
Component Registry - manages all available components.
"""
from typing import Dict, Type, List, Optional, Any
from .base import BaseComponent, ComponentConfig
from .input_component import InputComponent
from .output_component import OutputComponent
from .agent_component import AgentComponent
from .composio_component import ComposioToolComponent
from .llm import LLMRegistry, OpenAIModel, AnthropicModel, OpenRouterModel


class ComponentRegistry:
    """
    Central registry for all components in the agent builder.
    
    This class manages component registration, retrieval, and instantiation,
    providing a unified interface for the flow builder.
    """
    
    _instance: Optional["ComponentRegistry"] = None
    _components: Dict[str, Type[BaseComponent]] = {}
    
    def __new__(cls) -> "ComponentRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_default_components()
        return cls._instance
    
    def _initialize_default_components(self) -> None:
        """Register all default components."""
        # Core components
        self.register(InputComponent)
        self.register(OutputComponent)
        self.register(AgentComponent)
        self.register(ComposioToolComponent)
        
        # LLM models
        self.register(OpenAIModel)
        self.register(AnthropicModel)
        self.register(OpenRouterModel)
    
    def register(self, component_class: Type[BaseComponent]) -> None:
        """
        Register a new component class.
        
        Args:
            component_class: The component class to register
        """
        self._components[component_class._component_type] = component_class
    
    def unregister(self, component_type: str) -> bool:
        """
        Unregister a component.
        
        Args:
            component_type: The component type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if component_type in self._components:
            del self._components[component_type]
            return True
        return False
    
    def get(self, component_type: str) -> Optional[Type[BaseComponent]]:
        """
        Get a component class by type.
        
        Args:
            component_type: The component type identifier
            
        Returns:
            The component class or None if not found
        """
        return self._components.get(component_type)
    
    def create(
        self, 
        component_type: str, 
        node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseComponent]:
        """
        Create an instance of a component.
        
        Args:
            component_type: The component type identifier
            node_id: Optional node ID
            parameters: Component parameters
            
        Returns:
            Component instance or None if not found
        """
        component_class = self.get(component_type)
        if component_class:
            return component_class(node_id=node_id, parameters=parameters)
        return None
    
    def list_components(self) -> List[Dict[str, Any]]:
        """
        List all registered components with their configurations.
        
        Returns:
            List of component configurations
        """
        return [
            {
                "component_type": component_type,
                "config": component_class.get_config().model_dump()
            }
            for component_type, component_class in self._components.items()
        ]
    
    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        List components filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of component configurations in the category
        """
        return [
            {
                "component_type": component_type,
                "config": component_class.get_config().model_dump()
            }
            for component_type, component_class in self._components.items()
            if component_class._category == category
        ]
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        categories = set()
        for component_class in self._components.values():
            categories.add(component_class._category)
        return sorted(list(categories))
    
    @property
    def available_types(self) -> List[str]:
        """Get list of available component types."""
        return list(self._components.keys())


# Global registry instance
component_registry = ComponentRegistry()

```

# config.py

```python
"""
Configuration settings for the Agent Builder application.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Agent Builder API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "agent_builder"
    
    # JWT Authentication
    JWT_SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Keys (optional defaults)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    COMPOSIO_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

```

# database/__init__.py

```python
"""Database package initialization."""
from .mongodb import MongoDB, get_database
from .models import UserModel, FlowModel, NodeModel, EdgeModel

__all__ = [
    "MongoDB",
    "get_database",
    "UserModel",
    "FlowModel",
    "NodeModel",
    "EdgeModel",
]

```

# database/models.py

```python
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

```

# database/mongodb.py

```python
"""
MongoDB connection and database management.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from config import settings


class MongoDB:
    """MongoDB connection manager using singleton pattern."""
    
    _instance: Optional["MongoDB"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls) -> "MongoDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._client is None:
            self._client = AsyncIOMotorClient(settings.MONGODB_URL)
            self._database = self._client[settings.MONGODB_DB_NAME]
            
            # Create indexes
            await self._create_indexes()
            
            print(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for collections."""
        # Users collection indexes
        await self._database.users.create_index("email", unique=True)
        await self._database.users.create_index("username", unique=True)
        
        # Flows collection indexes
        await self._database.flows.create_index("user_id")
        await self._database.flows.create_index("flow_id", unique=True)
        
        # Nodes collection indexes
        await self._database.nodes.create_index("flow_id")
        await self._database.nodes.create_index([("flow_id", 1), ("node_id", 1)], unique=True)
        
        # Edges collection indexes
        await self._database.edges.create_index("flow_id")
        await self._database.edges.create_index([("flow_id", 1), ("edge_id", 1)], unique=True)
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            print("Disconnected from MongoDB")
    
    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self._database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """Get client instance."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client


# Dependency for FastAPI
async def get_database() -> AsyncIOMotorDatabase:
    """FastAPI dependency to get database instance."""
    mongodb = MongoDB()
    return mongodb.database

```

# document_code.py

```python
import os

# Folders and files to skip
SKIP_DIRS = {
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'env',
    '.env',
    '.mypy_cache',
    '.pytest_cache',
    '.vscode',
    '.idea',
    'node_modules',
    'dist',
    'build',
    '.github',
    '.gitignore',  # not a dir but often mistaken
}

SKIP_EXTENSIONS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.egg',
    '.whl',
    '.md',
    '.txt',
    '.log',
    '.json',
    '.yaml',
    '.yml',
    '.toml',
    '.lock',
    '.env',
}

def get_tree_output():
    """Manually generate a clean tree structure without using shell `tree`."""
    tree_lines = []
    root_dir = os.getcwd()
    tree_lines.append(f"{os.path.basename(root_dir)}/")

    def walk_dir(current_path, prefix=""):
        try:
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return
        files = []
        dirs = []
        for item in items:
            if item in SKIP_DIRS or item.startswith('.') and item not in ('.', '..'):
                # Allow hidden files like .env but skip hidden dirs (like .git, .venv)
                if os.path.isdir(os.path.join(current_path, item)):
                    continue
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                if os.path.basename(full_path) not in SKIP_DIRS:
                    dirs.append(item)
            else:
                # Skip unwanted extensions and hidden files (except .env, .gitignore)
                _, ext = os.path.splitext(item)
                if ext.lower() in SKIP_EXTENSIONS and not item.startswith('.'):
                    continue
                if item in SKIP_DIRS:
                    continue
                files.append(item)

        entries = dirs + files
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{entry}")
            if entry in dirs:
                extension = "    " if is_last else "│   "
                walk_dir(os.path.join(current_path, entry), prefix + extension)

    walk_dir(root_dir)
    return "\n".join(tree_lines)

def get_python_files(root_dir):
    """Recursively collect all .py files, skipping unwanted directories."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip unwanted folders
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.') or d in ('.', '..')]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Skip pycache-like files just in case
                if '__pycache__' in full_path:
                    continue
                py_files.append(full_path)
    py_files.sort()
    return py_files

def make_markdown_heading(path, root_dir):
    rel_path = os.path.relpath(path, root_dir).replace(os.sep, '/')
    return f"# {rel_path}"

def main():
    root_dir = os.getcwd()
    output_file = os.path.join(root_dir, "code_documentation.md")

    # Generate clean tree
    tree_output = get_tree_output()

    # Get filtered Python files
    py_files = get_python_files(root_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Project Code Documentation\n\n")
        f.write("## Project Structure\n\n")
        f.write("```\n")
        f.write(tree_output)
        f.write("\n```\n\n")

        for file_path in py_files:
            heading = make_markdown_heading(file_path, root_dir)
            f.write(f"{heading}\n\n")
            f.write("```python\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as pf:
                    f.write(pf.read())
            except Exception as e:
                f.write(f"# ERROR reading file: {e}")
            f.write("\n```\n\n")

    print(f"✅ Clean documentation generated at: {output_file}")

if __name__ == "__main__":
    main()
```

# edges/__init__.py

```python
"""Edges package initialization."""
from .routes import router as edges_router
from .schemas import EdgeValidationRequest, EdgeValidationResponse

__all__ = [
    "edges_router",
    "EdgeValidationRequest",
    "EdgeValidationResponse",
]

```

# edges/routes.py

```python
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

```

# edges/schemas.py

```python
"""
Edge schemas for API validation.
"""
from pydantic import BaseModel
from typing import Optional, List


class EdgeValidationRequest(BaseModel):
    """Request schema for validating an edge connection."""
    source_component_type: str
    source_port: str
    target_component_type: str
    target_port: str


class EdgeValidationResponse(BaseModel):
    """Response schema for edge validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]


class PortCompatibilityInfo(BaseModel):
    """Information about port compatibility."""
    source_data_type: str
    target_data_type: str
    compatible: bool
    reason: Optional[str] = None

```

# flows/__init__.py

```python
"""Flows package initialization."""
from .routes import router as flows_router
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
from .executor import FlowExecutor, FlowExecutorFactory

__all__ = [
    "flows_router",
    "FlowCreate",
    "FlowUpdate",
    "FlowResponse",
    "FlowListResponse",
    "FlowExecuteRequest",
    "FlowExecuteResponse",
    "AgentSchemaResponse",
    "NodeCreate",
    "NodeUpdate",
    "EdgeCreate",
    "FlowService",
    "FlowExecutor",
    "FlowExecutorFactory",
]

```

# flows/executor.py

```python
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

```

# flows/routes.py

```python
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

```

# flows/schemas.py

```python
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

```

# flows/services.py

```python
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

```

# main.py

```python
"""
Agent Builder API - Main Application Entry Point

A FastAPI application for building AI agents through a visual drag-and-drop interface.
Supports multiple LLM providers, Composio tools, and custom workflows.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging

from config import settings
from database import MongoDB
from auth.routes import router as auth_router
from flows.routes import router as flows_router
from nodes.routes import router as nodes_router
from edges.routes import router as edges_router


# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Agent Builder API...")
    
    # Connect to MongoDB
    mongodb = MongoDB()
    await mongodb.connect()
    logger.info("Connected to MongoDB")
    
    yield
    
    logger.info("Shutting down Agent Builder API...")
    await mongodb.disconnect()
    logger.info("Disconnected from MongoDB")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Agent Builder API
    
    Build AI agents through a visual drag-and-drop interface.
    
    ### Features
    - **Visual Flow Builder**: Create agent workflows by connecting components
    - **Multiple LLM Providers**: Support for OpenAI, Anthropic, OpenRouter, and more
    - **Composio Integration**: Connect to 500+ external services
    - **Real-time Execution**: Execute agents with streaming responses
    - **Centralized Field Types**: Consistent input handling across all components
    
    ### New Field Types System
    The API now includes a centralized field types system that provides:
    - 25+ predefined input types (string, number, select, API key, etc.)
    - Consistent validation and schema generation
    - Frontend-friendly JSON schemas for dynamic form rendering
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc) if settings.DEBUG else "Internal server error"
        }
    )


# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(flows_router, prefix="/api/v1")
app.include_router(nodes_router, prefix="/api/v1")
app.include_router(edges_router, prefix="/api/v1")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic information."""
    return {
        "message": "Welcome to Agent Builder API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# API info endpoint
@app.get("/api/v1", tags=["API Info"])
async def api_info():
    """Get API information and available endpoints."""
    return {
        "version": "v1",
        "endpoints": {
            "auth": {
                "register": "POST /api/v1/auth/register",
                "login": "POST /api/v1/auth/login",
                "me": "GET /api/v1/auth/me"
            },
            "flows": {
                "list": "GET /api/v1/flows",
                "create": "POST /api/v1/flows",
                "get": "GET /api/v1/flows/{flow_id}",
                "update": "PUT /api/v1/flows/{flow_id}",
                "delete": "DELETE /api/v1/flows/{flow_id}",
                "execute": "POST /api/v1/flows/{flow_id}/execute",
                "stream": "POST /api/v1/flows/{flow_id}/execute/stream",
                "schema": "GET /api/v1/flows/{flow_id}/schema"
            },
            "components": {
                "list": "GET /api/v1/components",
                "categories": "GET /api/v1/components/categories",
                "get": "GET /api/v1/components/{component_type}",
                "schema": "GET /api/v1/components/{component_type}/schema",
                "validate": "POST /api/v1/components/{component_type}/validate",
                "field_types": "GET /api/v1/components/field-types",
                "validate_fields": "POST /api/v1/components/validate-fields"
            },
            "edges": {
                "validate": "POST /api/v1/edges/validate",
                "compatible_ports": "GET /api/v1/edges/compatible-ports/{component_type}"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )

```

# nodes/__init__.py

```python
"""Nodes package initialization."""
from .routes import router as nodes_router
from .schemas import (
    ComponentListResponse,
    ComponentDetailResponse,
    ComponentCategoryResponse,
    ComponentConfigSchema,
    FieldTypesListResponse,
    FieldTypeInfoSchema,
    FieldSchema,
    FieldGroupSchema,
    FieldValidationSchema,
    FieldOptionSchema,
    FieldUISchema,
    ValidateFieldsRequest,
    ValidateFieldsResponse,
    ValidationResultSchema,
)

__all__ = [
    "nodes_router",
    "ComponentListResponse",
    "ComponentDetailResponse",
    "ComponentCategoryResponse",
    "ComponentConfigSchema",
    "FieldTypesListResponse",
    "FieldTypeInfoSchema",
    "FieldSchema",
    "FieldGroupSchema",
    "FieldValidationSchema",
    "FieldOptionSchema",
    "FieldUISchema",
    "ValidateFieldsRequest",
    "ValidateFieldsResponse",
    "ValidationResultSchema",
]

```

# nodes/routes.py

```python
"""
Node/Component API routes - lists available components and field types.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Dict, Any, List

from components.registry import ComponentRegistry
from components.field_types import (
    field_type_registry,
    FieldTypeEnum,
    FieldDefinition,
    validate_fields,
    get_validation_errors,
)
from .schemas import (
    ComponentListResponse,
    ComponentDetailResponse,
    ComponentCategoryResponse,
    ComponentConfigSchema,
    FieldTypesListResponse,
    FieldTypeInfoSchema,
    ValidateFieldsRequest,
    ValidateFieldsResponse,
    ValidationResultSchema,
)


router = APIRouter(prefix="/components", tags=["Components"])


# Get registry instances
component_registry = ComponentRegistry()


@router.get("", response_model=ComponentListResponse)
async def list_components(category: Optional[str] = None):
    """
    List all available components for the flow builder.
    
    Optionally filter by category.
    """
    if category:
        components = component_registry.list_by_category(category)
    else:
        components = component_registry.list_components()
    
    return ComponentListResponse(
        components=[
            ComponentDetailResponse(
                component_type=comp["component_type"],
                config=ComponentConfigSchema(**comp["config"])
            )
            for comp in components
        ],
        total=len(components)
    )


@router.get("/categories", response_model=ComponentCategoryResponse)
async def list_categories():
    """
    List all component categories with their components.
    """
    categories = component_registry.get_categories()
    
    components_by_category = {}
    for category in categories:
        components = component_registry.list_by_category(category)
        components_by_category[category] = [
            ComponentDetailResponse(
                component_type=comp["component_type"],
                config=ComponentConfigSchema(**comp["config"])
            )
            for comp in components
        ]
    
    return ComponentCategoryResponse(
        categories=categories,
        components_by_category=components_by_category
    )


@router.get("/field-types", response_model=FieldTypesListResponse)
async def list_field_types():
    """
    List all available field types for component configuration.
    
    This endpoint provides frontend developers with information about
    all supported input types and their properties.
    """
    field_types = field_type_registry.list_types()
    
    # Map field types to input components
    input_component_map = {
        "string": "TextInput",
        "text": "TextArea",
        "number": "NumberInput",
        "integer": "NumberInput",
        "boolean": "Switch",
        "select": "Select",
        "multi_select": "MultiSelect",
        "radio": "RadioGroup",
        "checkbox_group": "CheckboxGroup",
        "password": "PasswordInput",
        "email": "EmailInput",
        "url": "UrlInput",
        "color": "ColorPicker",
        "date": "DatePicker",
        "datetime": "DateTimePicker",
        "time": "TimePicker",
        "json": "JsonEditor",
        "code": "CodeEditor",
        "slider": "Slider",
        "range": "RangeSlider",
        "file": "FileUpload",
        "image": "ImageUpload",
        "api_key": "ApiKeyInput",
        "model_select": "ModelSelect",
        "prompt": "PromptEditor",
        "variable": "VariableInput",
        "port": "PortConfig",
    }
    
    return FieldTypesListResponse(
        field_types=[
            FieldTypeInfoSchema(
                type=ft["type"],
                class_name=ft["class"],
                description=ft["description"],
                default_properties=ft["default_properties"],
                input_component=input_component_map.get(ft["type"], "TextInput")
            )
            for ft in field_types
        ],
        total=len(field_types)
    )


@router.get("/field-types/{field_type}")
async def get_field_type(field_type: str):
    """
    Get detailed information about a specific field type.
    """
    try:
        ft_enum = FieldTypeEnum(field_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field type '{field_type}' not found"
        )
    
    schema = field_type_registry.get_type_schema(ft_enum)
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field type '{field_type}' not found"
        )
    
    return schema


@router.get("/{component_type}", response_model=ComponentDetailResponse)
async def get_component(component_type: str):
    """
    Get details for a specific component type including all field definitions.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    config = component_class.get_config()
    
    return ComponentDetailResponse(
        component_type=component_type,
        config=ComponentConfigSchema(**config.model_dump())
    )


@router.get("/{component_type}/schema")
async def get_component_field_schema(component_type: str):
    """
    Get the complete field schema for a component.
    
    This returns a detailed schema that can be used by the frontend
    to dynamically render the component configuration form.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    # Create temporary instance to get field schema
    try:
        instance = component_class(parameters={})
    except ValueError:
        # Some components require parameters, create with empty required fields
        instance = component_class.__new__(component_class)
        instance.node_id = "temp"
        instance.parameters = {}
        instance._fields = component_class._get_fields()
        instance._field_groups = component_class._get_field_groups()
    
    return instance.get_field_schema()


@router.post("/{component_type}/validate")
async def validate_component_parameters(
    component_type: str,
    parameters: Dict[str, Any]
):
    """
    Validate parameters for a component type.
    """
    component_class = component_registry.get(component_type)
    
    if not component_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component type '{component_type}' not found"
        )
    
    try:
        instance = component_class(parameters=parameters)
        errors = instance.validate_parameters()
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }


@router.post("/validate-fields", response_model=ValidateFieldsResponse)
async def validate_field_values(request: ValidateFieldsRequest):
    """
    Validate field values against their definitions.
    
    This endpoint allows validation of arbitrary field configurations,
    useful for custom form validation.
    """
    from components.field_types.base import FieldDefinition
    
    # Convert field dicts to FieldDefinition objects
    field_definitions = []
    for field_dict in request.fields:
        try:
            field = FieldDefinition(**field_dict)
            field_definitions.append(field)
        except Exception as e:
            return ValidateFieldsResponse(
                valid=False,
                results={},
                all_errors=[f"Invalid field definition: {str(e)}"]
            )
    
    # Validate all fields
    results = validate_fields(request.values, field_definitions)
    
    # Convert to response schema
    result_schemas = {
        name: ValidationResultSchema(
            valid=result.valid,
            errors=result.errors,
            warnings=result.warnings,
            field_name=result.field_name
        )
        for name, result in results.items()
    }
    
    all_errors = get_validation_errors(request.values, field_definitions)
    
    return ValidateFieldsResponse(
        valid=len(all_errors) == 0,
        results=result_schemas,
        all_errors=all_errors
    )

```

# nodes/schemas.py

```python
"""
Node/Component schemas for API responses.
Updated to use centralized field types system.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class PortSchema(BaseModel):
    """Schema for component port."""
    id: str
    name: str
    type: str
    data_type: str
    required: bool = False
    multiple: bool = False
    description: Optional[str] = None


class FieldValidationSchema(BaseModel):
    """Schema for field validation rules."""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    pattern_message: Optional[str] = None


class FieldOptionSchema(BaseModel):
    """Schema for field option."""
    value: Any
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    disabled: bool = False
    group: Optional[str] = None


class FieldUISchema(BaseModel):
    """Schema for field UI properties."""
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    hint: Optional[str] = None
    icon: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    width: str = "full"
    order: int = 0
    group: Optional[str] = None
    read_only: bool = False
    disabled: bool = False
    hidden: bool = False
    sensitive: bool = False
    copyable: bool = False
    clearable: bool = True
    component: Optional[str] = None
    render_props: Dict[str, Any] = Field(default_factory=dict)


class FieldConditionSchema(BaseModel):
    """Schema for field condition."""
    field: str
    operator: str
    value: Optional[Any] = None


class FieldDependencySchema(BaseModel):
    """Schema for field dependency."""
    depends_on: str
    condition: FieldConditionSchema
    action: str = "show"


class FieldSchema(BaseModel):
    """Complete schema for a field."""
    name: str
    type: str
    label: str
    description: Optional[str] = None
    default: Any = None
    validation: FieldValidationSchema = Field(default_factory=FieldValidationSchema)
    ui: FieldUISchema = Field(default_factory=FieldUISchema)
    options: Optional[List[FieldOptionSchema]] = None
    options_source: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class FieldGroupSchema(BaseModel):
    """Schema for field group."""
    id: str
    label: str
    description: Optional[str] = None
    collapsible: bool = False
    collapsed_by_default: bool = False
    icon: Optional[str] = None
    order: int = 0


class ComponentConfigSchema(BaseModel):
    """Schema for component configuration with fields."""
    component_type: str
    name: str
    description: str
    category: str
    icon: str
    color: str
    input_ports: List[PortSchema]
    output_ports: List[PortSchema]
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    field_groups: List[FieldGroupSchema] = Field(default_factory=list)


class ComponentDetailResponse(BaseModel):
    """Response schema for a single component."""
    component_type: str
    config: ComponentConfigSchema


class ComponentListResponse(BaseModel):
    """Response schema for listing components."""
    components: List[ComponentDetailResponse]
    total: int


class ComponentCategoryResponse(BaseModel):
    """Response schema for component categories."""
    categories: List[str]
    components_by_category: Dict[str, List[ComponentDetailResponse]]


class FieldTypeInfoSchema(BaseModel):
    """Schema for field type information."""
    type: str
    class_name: str
    description: Optional[str] = None
    default_properties: Dict[str, Any] = Field(default_factory=dict)
    input_component: str


class FieldTypesListResponse(BaseModel):
    """Response schema for listing field types."""
    field_types: List[FieldTypeInfoSchema]
    total: int


class ValidateFieldsRequest(BaseModel):
    """Request schema for validating fields."""
    fields: List[Dict[str, Any]]
    values: Dict[str, Any]


class ValidationResultSchema(BaseModel):
    """Schema for validation result."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    field_name: str


class ValidateFieldsResponse(BaseModel):
    """Response schema for field validation."""
    valid: bool
    results: Dict[str, ValidationResultSchema]
    all_errors: List[str]

```

