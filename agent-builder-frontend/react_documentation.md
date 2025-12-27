# React Application Documentation

## Project Structure

```text
agent-builder-frontend/
├── public
│   └── index.html
├── src
│   ├── api
│   │   └── index.ts
│   ├── components
│   │   ├── auth
│   │   │   ├── LoginForm.tsx
│   │   │   └── SignupForm.tsx
│   │   ├── builder
│   │   │   ├── Canvas.tsx
│   │   │   ├── ComponentPanel.tsx
│   │   │   ├── ConfigPanel.tsx
│   │   │   ├── CustomNode.tsx
│   │   │   └── Toolbar.tsx
│   │   ├── dashboard
│   │   │   ├── FlowCard.tsx
│   │   │   └── FlowGrid.tsx
│   │   ├── layout
│   │   │   ├── Header.tsx
│   │   │   └── Layout.tsx
│   │   └── ui
│   │       ├── Button.tsx
│   │       ├── Input.tsx
│   │       ├── Modal.tsx
│   │       └── Toast.tsx
│   ├── contexts
│   │   ├── AuthContext.tsx
│   │   └── FlowContext.tsx
│   ├── hooks
│   │   ├── useAuth.ts
│   │   └── useFlow.ts
│   ├── pages
│   │   ├── Auth.tsx
│   │   ├── Builder.tsx
│   │   └── Dashboard.tsx
│   ├── types
│   │   └── index.ts
│   ├── utils
│   │   ├── defaultComponentConfig.ts
│   │   └── helpers.ts
│   ├── App.tsx
│   ├── index.css
│   └── index.tsx
├── index.html
├── package-lock.json
├── package.json
├── postcss.config.js
├── react_documentation.md
├── tailwind.config.js
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

# index.html

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Agent Builder - Build AI Agents Visually</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/index.tsx"></script>
  </body>
</html>

```
# src/App.tsx

```tsx
// src/App.tsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ToastContainer } from './components/ui/Toast';
import { Auth } from './pages/Auth';
import { Dashboard } from './pages/Dashboard';
import { Builder } from './pages/Builder';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-dark-300 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />;
  }

  return <>{children}</>;
};

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/auth" element={<Auth />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/builder/:flowId"
        element={
          <ProtectedRoute>
            <Builder />
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
        <ToastContainer />
      </AuthProvider>
    </BrowserRouter>
  );
};

export default App;

```

# src/api/index.ts

```typescript
// src/api/index.ts
import axios, { AxiosInstance, AxiosError } from 'axios';
import { 
  User, 
  Flow, 
  AuthTokens, 
  ComponentConfig,
  ApiError 
} from '../types';

const API_BASE_URL = '/api/v1';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for auth token
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          window.location.href = '/auth';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth endpoints
  async register(username: string, email: string, password: string): Promise<User> {
    const response = await this.client.post('/auth/register', {
      username,
      email,
      password,
    });
    return response.data;
  }

  async login(email: string, password: string): Promise<AuthTokens> {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const response = await this.client.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get('/auth/me');
    return response.data;
  }

  // Flow endpoints
  async getFlows(page = 1, pageSize = 10): Promise<{
    flows: Flow[];
    total: number;
    page: number;
    page_size: number;
  }> {
    const response = await this.client.get('/flows', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  }

  async getFlow(flowId: string): Promise<Flow> {
    const response = await this.client.get(`/flows/${flowId}`);
    return response.data;
  }

  async createFlow(data: {
    name: string;
    description?: string;
    nodes?: any[];
    edges?: any[];
  }): Promise<Flow> {
    const response = await this.client.post('/flows', data);
    return response.data;
  }

  async updateFlow(flowId: string, data: Partial<{
    name: string;
    description: string;
    nodes: any[];
    edges: any[];
    is_active: boolean;
  }>): Promise<Flow> {
    const response = await this.client.put(`/flows/${flowId}`, data);
    return response.data;
  }

  async deleteFlow(flowId: string): Promise<void> {
    await this.client.delete(`/flows/${flowId}`);
  }

  async executeFlow(flowId: string, inputData: Record<string, any>): Promise<{
    flow_id: string;
    status: string;
    output?: any;
    error?: string;
    execution_time?: number;
  }> {
    const response = await this.client.post(`/flows/${flowId}/execute`, {
      input_data: inputData,
    });
    return response.data;
  }

  async getFlowSchema(flowId: string): Promise<{
    flow_id: string;
    agent_schema: any;
    validation_errors: string[];
    is_valid: boolean;
  }> {
    const response = await this.client.get(`/flows/${flowId}/schema`);
    return response.data;
  }

  // Export flow as JSON
  async exportFlow(flowId: string): Promise<{
    flow: Flow;
    export_date: string;
    version: string;
  }> {
    const response = await this.client.get(`/flows/${flowId}/export`);
    return response.data;
  }

  // Import flow from JSON
  async importFlow(flowData: any): Promise<Flow> {
    const response = await this.client.post('/flows/import', flowData);
    return response.data;
  }

  // Component endpoints
  async getComponents(category?: string): Promise<{
    components: Array<{
      component_type: string;
      config: ComponentConfig;
    }>;
    total: number;
  }> {
    const response = await this.client.get('/components', {
      params: category ? { category } : {},
    });
    return response.data;
  }

  async getComponentCategories(): Promise<{
    categories: string[];
    components_by_category: Record<string, Array<{
      component_type: string;
      config: ComponentConfig;
    }>>;
  }> {
    const response = await this.client.get('/components/categories');
    return response.data;
  }

  async getComponent(componentType: string): Promise<{
    component_type: string;
    config: ComponentConfig;
  }> {
    const response = await this.client.get(`/components/${componentType}`);
    return response.data;
  }

  async getComponentSchema(componentType: string): Promise<any> {
    const response = await this.client.get(`/components/${componentType}/schema`);
    return response.data;
  }

  async validateEdge(data: {
    source_component_type: string;
    source_port: string;
    target_component_type: string;
    target_port: string;
  }): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const response = await this.client.post('/edges/validate', data);
    return response.data;
  }

  // Node operations
  async addNode(flowId: string, nodeData: any): Promise<Flow> {
    const response = await this.client.post(`/flows/${flowId}/nodes`, nodeData);
    return response.data;
  }

  async updateNode(flowId: string, nodeId: string, updateData: any): Promise<Flow> {
    const response = await this.client.put(`/flows/${flowId}/nodes/${nodeId}`, updateData);
    return response.data;
  }

  async deleteNode(flowId: string, nodeId: string): Promise<Flow> {
    const response = await this.client.delete(`/flows/${flowId}/nodes/${nodeId}`);
    return response.data;
  }

  // Edge operations
  async addEdge(flowId: string, edgeData: any): Promise<Flow> {
    const response = await this.client.post(`/flows/${flowId}/edges`, edgeData);
    return response.data;
  }

  async deleteEdge(flowId: string, edgeId: string): Promise<Flow> {
    const response = await this.client.delete(`/flows/${flowId}/edges/${edgeId}`);
    return response.data;
  }
}

export const api = new ApiClient();
export default api;

```

# src/components/auth/LoginForm.tsx

```tsx
// src/components/auth/LoginForm.tsx
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { Mail, Lock } from 'lucide-react';
import toast from 'react-hot-toast';

interface LoginFormProps {
  onSwitchToSignup: () => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSwitchToSignup }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});

  const validate = () => {
    const newErrors: { email?: string; password?: string } = {};
    
    if (!email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Invalid email address';
    }
    
    if (!password) {
      newErrors.password = 'Password is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsLoading(true);
    try {
      await login(email, password);
      toast.success('Welcome back!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Welcome Back</h2>
        <p className="text-gray-400">Sign in to continue building agents</p>
      </div>

      <Input
        label="Email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
        icon={<Mail className="w-5 h-5 text-gray-400" />}
        error={errors.email}
      />

      <Input
        label="Password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Enter your password"
        icon={<Lock className="w-5 h-5 text-gray-400" />}
        error={errors.password}
      />

      <Button
        type="submit"
        className="w-full"
        size="lg"
        isLoading={isLoading}
      >
        Sign In
      </Button>

      <p className="text-center text-gray-400">
        Don't have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToSignup}
          className="text-indigo-400 hover:text-indigo-300 font-medium"
        >
          Sign up
        </button>
      </p>
    </form>
  );
};

```

# src/components/auth/SignupForm.tsx

```tsx
// src/components/auth/SignupForm.tsx
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { User, Mail, Lock } from 'lucide-react';
import toast from 'react-hot-toast';

interface SignupFormProps {
  onSwitchToLogin: () => void;
}

export const SignupForm: React.FC<SignupFormProps> = ({ onSwitchToLogin }) => {
  const { register } = useAuth();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{
    username?: string;
    email?: string;
    password?: string;
    confirmPassword?: string;
  }>({});

  const validate = () => {
    const newErrors: typeof errors = {};
    
    if (!username) {
      newErrors.username = 'Username is required';
    } else if (username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }
    
    if (!email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Invalid email address';
    }
    
    if (!password) {
      newErrors.password = 'Password is required';
    } else if (password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }
    
    if (password !== confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsLoading(true);
    try {
      await register(username, email, password);
      toast.success('Account created successfully!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Create Account</h2>
        <p className="text-gray-400">Start building AI agents today</p>
      </div>

      <Input
        label="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="johndoe"
        icon={<User className="w-5 h-5 text-gray-400" />}
        error={errors.username}
      />

      <Input
        label="Email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
        icon={<Mail className="w-5 h-5 text-gray-400" />}
        error={errors.email}
      />

      <Input
        label="Password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Create a password"
        icon={<Lock className="w-5 h-5 text-gray-400" />}
        error={errors.password}
      />

      <Input
        label="Confirm Password"
        type="password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
        placeholder="Confirm your password"
        icon={<Lock className="w-5 h-5 text-gray-400" />}
        error={errors.confirmPassword}
      />

      <Button
        type="submit"
        className="w-full"
        size="lg"
        isLoading={isLoading}
      >
        Create Account
      </Button>

      <p className="text-center text-gray-400">
        Already have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToLogin}
          className="text-indigo-400 hover:text-indigo-300 font-medium"
        >
          Sign in
        </button>
      </p>
    </form>
  );
};

```

# src/components/builder/Canvas.tsx

```tsx
// src/components/builder/Canvas.tsx
import React, { useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Node,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useFlowStore } from '../../contexts/FlowContext';
import CustomNode from './CustomNode';
import { ComponentPanel } from './ComponentPanel';
import { ConfigPanel } from './ConfigPanel';
import { Toolbar } from './Toolbar';

const nodeTypes = {
  custom: CustomNode,
};

interface CanvasProps {
  flowId: string;
}

const FlowCanvas: React.FC<CanvasProps> = ({ flowId }) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { 
    nodes, 
    edges, 
    currentFlow,
    selectedNode,
    onNodesChange, 
    onEdgesChange, 
    onConnect,
    addNode,
    setSelectedNode,
    loadFlow,
    loadComponents,
  } = useFlowStore();
  
  const { project, zoomIn, zoomOut, fitView } = useReactFlow();

  useEffect(() => {
    const init = async () => {
      await loadComponents();
      await loadFlow(flowId);
    };
    init();
  }, [flowId, loadComponents, loadFlow]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const componentType = event.dataTransfer.getData('application/reactflow');
      if (!componentType || !reactFlowWrapper.current) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      addNode(componentType, position);
    },
    [project, addNode]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node);
    },
    [setSelectedNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  return (
    <div className="flex flex-col h-screen">
      <Toolbar
        flowName={currentFlow?.name || 'Untitled Flow'}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onFitView={fitView}
      />
      
      <div className="flex flex-1 overflow-hidden">
        <ComponentPanel />
        
        <div ref={reactFlowWrapper} className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[15, 15]}
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: '#6366f1', strokeWidth: 2 },
            }}
          >
            <Background color="#374151" gap={20} />
            <Controls className="!bg-dark-200 !border-gray-700" />
            <MiniMap
              nodeColor={(node) => {
                const componentType = node.data?.component_type;
                switch (componentType) {
                  case 'input':
                    return '#10b981';
                  case 'output':
                    return '#f59e0b';
                  case 'agent':
                    return '#3b82f6';
                  default:
                    return '#6366f1';
                }
              }}
              maskColor="rgba(0, 0, 0, 0.8)"
            />
          </ReactFlow>
        </div>

        {selectedNode && (
          <ConfigPanel
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>
    </div>
  );
};

export const Canvas: React.FC<CanvasProps> = ({ flowId }) => {
  return (
    <ReactFlowProvider>
      <FlowCanvas flowId={flowId} />
    </ReactFlowProvider>
  );
};

```

# src/components/builder/ComponentPanel.tsx

```tsx
// src/components/builder/ComponentPanel.tsx
import React, { useState } from 'react';
import { useFlowStore } from '../../contexts/FlowContext';
import { 
  Search, 
  ChevronDown, 
  ChevronRight,
  MessageSquare,
  Bot,
  Brain,
  Plug,
  Layers
} from 'lucide-react';

const getCategoryIcon = (category: string) => {
  switch (category) {
    case 'io':
      return <MessageSquare className="w-4 h-4" />;
    case 'agents':
      return <Bot className="w-4 h-4" />;
    case 'models':
      return <Brain className="w-4 h-4" />;
    case 'tools':
      return <Plug className="w-4 h-4" />;
    default:
      return <Layers className="w-4 h-4" />;
  }
};

const getCategoryLabel = (category: string) => {
  switch (category) {
    case 'io':
      return 'Input/Output';
    case 'agents':
      return 'Agents';
    case 'models':
      return 'LLM Models';
    case 'tools':
      return 'Tools';
    default:
      return category.charAt(0).toUpperCase() + category.slice(1);
  }
};

export const ComponentPanel: React.FC = () => {
  const { categories, componentsByCategory } = useFlowStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(categories)
  );

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const onDragStart = (event: React.DragEvent, componentType: string) => {
    event.dataTransfer.setData('application/reactflow', componentType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const filteredCategories = categories.filter((category) => {
    if (!searchQuery) return true;
    
    const components = componentsByCategory[category] || [];
    return components.some(
      (comp) =>
        comp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        comp.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  return (
    <div className="w-64 bg-dark-200 border-r border-gray-700 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white mb-3">Components</h2>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search components..."
            className="w-full pl-9 pr-3 py-2 bg-dark-300 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      </div>

      {/* Component List */}
      <div className="flex-1 overflow-y-auto p-2">
        {filteredCategories.map((category) => {
          const components = componentsByCategory[category] || [];
          const isExpanded = expandedCategories.has(category);
          
          const filteredComponents = searchQuery
            ? components.filter(
                (comp) =>
                  comp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                  comp.description.toLowerCase().includes(searchQuery.toLowerCase())
              )
            : components;

          if (filteredComponents.length === 0) return null;

          return (
            <div key={category} className="mb-2">
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-300 hover:bg-white/5 rounded-lg transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                {getCategoryIcon(category)}
                <span>{getCategoryLabel(category)}</span>
                <span className="ml-auto text-xs text-gray-500">
                  {filteredComponents.length}
                </span>
              </button>

              {/* Components */}
              {isExpanded && (
                <div className="mt-1 space-y-1 ml-2">
                  {filteredComponents.map((component) => (
                    <div
                      key={component.component_type}
                      draggable
                      onDragStart={(e) => onDragStart(e, component.component_type)}
                      className="flex items-center gap-3 px-3 py-2 bg-dark-300 rounded-lg cursor-grab hover:bg-dark-100 transition-colors group"
                    >
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ backgroundColor: component.color + '20' }}
                      >
                        {getCategoryIcon(component.category)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white truncate">
                          {component.name}
                        </p>
                        <p className="text-xs text-gray-500 truncate">
                          {component.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <p className="text-xs text-gray-500 text-center">
          Drag components to canvas
        </p>
      </div>
    </div>
  );
};

```

# src/components/builder/ConfigPanel.tsx

```tsx
// src/components/builder/ConfigPanel.tsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Node } from 'reactflow';
import { useFlowStore } from '../../contexts/FlowContext';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { 
  X, 
  Save, 
  Trash2, 
  ChevronDown, 
  ChevronRight,
  Eye,
  EyeOff,
  Copy,
  Check,
  AlertCircle,
  Info,
  RotateCcw,
  Code,
  Sliders,
} from 'lucide-react';
import { FieldDefinition, ComponentConfig } from '../../types';
import { copyToClipboard } from '../../utils/helpers';
import toast from 'react-hot-toast';

interface ConfigPanelProps {
  node: Node;
  onClose: () => void;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ node, onClose }) => {
  const { updateNodeData, deleteNode } = useFlowStore();
  const [parameters, setParameters] = useState<Record<string, any>>(
    node.data.parameters || {}
  );
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [showPasswords, setShowPasswords] = useState<Set<string>>(new Set());
  const [copiedFields, setCopiedFields] = useState<Set<string>>(new Set());
  const [isDirty, setIsDirty] = useState(false);
  const [activeTab, setActiveTab] = useState<'settings' | 'json'>('settings');
  const [jsonValue, setJsonValue] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);

  const config: ComponentConfig | undefined = node.data.config;

  // Initialize expanded groups
  useEffect(() => {
    if (config?.field_groups) {
      const defaultExpanded = config.field_groups
        .filter(g => !g.collapsed_by_default)
        .map(g => g.id);
      setExpandedGroups(new Set(defaultExpanded));
    }
  }, [config]);

  // Sync parameters with node data
  useEffect(() => {
    setParameters(node.data.parameters || {});
    setIsDirty(false);
  }, [node.id, node.data.parameters]);

  // Update JSON view
  useEffect(() => {
    setJsonValue(JSON.stringify(parameters, null, 2));
    setJsonError(null);
  }, [parameters]);

  const toggleGroup = (groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  };

  const togglePasswordVisibility = (fieldName: string) => {
    setShowPasswords((prev) => {
      const next = new Set(prev);
      if (next.has(fieldName)) {
        next.delete(fieldName);
      } else {
        next.add(fieldName);
      }
      return next;
    });
  };

  const handleCopy = async (fieldName: string, value: string) => {
    const success = await copyToClipboard(value);
    if (success) {
      setCopiedFields(prev => new Set(prev).add(fieldName));
      setTimeout(() => {
        setCopiedFields(prev => {
          const next = new Set(prev);
          next.delete(fieldName);
          return next;
        });
      }, 2000);
      toast.success('Copied to clipboard');
    }
  };

  const handleChange = useCallback((name: string, value: any) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
    setIsDirty(true);
  }, []);

  const handleSave = () => {
    updateNodeData(node.id, { parameters });
    setIsDirty(false);
    toast.success('Settings saved');
  };

  const handleReset = () => {
    setParameters(node.data.parameters || {});
    setIsDirty(false);
    toast.success('Settings reset');
  };

  const handleDelete = () => {
    if (window.confirm('Are you sure you want to delete this node?')) {
      deleteNode(node.id);
      onClose();
      toast.success('Node deleted');
    }
  };

  const handleJsonChange = (value: string) => {
    setJsonValue(value);
    try {
      const parsed = JSON.parse(value);
      setParameters(parsed);
      setJsonError(null);
      setIsDirty(true);
    } catch (e) {
      setJsonError('Invalid JSON');
    }
  };

  // Render individual field based on type
  const renderField = (field: FieldDefinition) => {
    const value = parameters[field.name] ?? field.default ?? '';
    const isCopied = copiedFields.has(field.name);

    const fieldLabel = (
      <div className="flex items-center justify-between mb-1.5">
        <label className="block text-sm font-medium text-gray-300">
          {field.label}
          {field.validation?.required && <span className="text-red-400 ml-1">*</span>}
        </label>
        {field.ui?.copyable && value && (
          <button
            type="button"
            onClick={() => handleCopy(field.name, String(value))}
            className="text-gray-400 hover:text-gray-300 p-1"
          >
            {isCopied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>
    );

    const helpText = field.ui?.help_text && (
      <p className="mt-1 text-xs text-gray-500 flex items-start gap-1">
        <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
        {field.ui.help_text}
      </p>
    );

    switch (field.type) {
      case 'string':
      case 'email':
      case 'url':
        return (
          <div>
            {fieldLabel}
            <div className="relative">
              <Input
                type={field.type === 'email' ? 'email' : field.type === 'url' ? 'url' : 'text'}
                value={value}
                onChange={(e) => handleChange(field.name, e.target.value)}
                placeholder={field.ui?.placeholder}
              />
            </div>
            {helpText}
          </div>
        );

      case 'password':
      case 'api_key':
        return (
          <div>
            {fieldLabel}
            <div className="relative">
              <input
                type={showPasswords.has(field.name) ? 'text' : 'password'}
                value={value}
                onChange={(e) => handleChange(field.name, e.target.value)}
                placeholder={field.ui?.placeholder}
                className="w-full px-4 py-2.5 pr-20 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-sm"
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                {value && (
                  <button
                    type="button"
                    onClick={() => handleCopy(field.name, String(value))}
                    className="p-1.5 text-gray-400 hover:text-gray-300"
                  >
                    {isCopied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => togglePasswordVisibility(field.name)}
                  className="p-1.5 text-gray-400 hover:text-gray-300"
                >
                  {showPasswords.has(field.name) ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
            {helpText}
          </div>
        );

      case 'text':
      case 'prompt':
        return (
          <div>
            {fieldLabel}
            <textarea
              value={value}
              onChange={(e) => handleChange(field.name, e.target.value)}
              placeholder={field.ui?.placeholder}
              rows={field.properties?.rows || 4}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
            />
            {helpText}
          </div>
        );

      case 'number':
      case 'integer':
        return (
          <div>
            {fieldLabel}
            <input
              type="number"
              value={value}
              onChange={(e) => handleChange(field.name, field.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))}
              placeholder={field.ui?.placeholder}
              min={field.validation?.min_value}
              max={field.validation?.max_value}
              step={field.type === 'integer' ? 1 : 0.1}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            {helpText}
          </div>
        );

      case 'boolean':
        return (
          <div className="flex items-center justify-between py-2">
            <div>
              <label className="text-sm font-medium text-gray-300">
                {field.label}
              </label>
              {field.description && (
                <p className="text-xs text-gray-500">{field.description}</p>
              )}
            </div>
            <button
              type="button"
              onClick={() => handleChange(field.name, !value)}
              className={`
                relative w-11 h-6 rounded-full transition-colors
                ${value ? 'bg-indigo-600' : 'bg-gray-700'}
              `}
            >
              <span
                className={`
                  absolute top-1 w-4 h-4 rounded-full bg-white transition-transform
                  ${value ? 'translate-x-6' : 'translate-x-1'}
                `}
              />
            </button>
          </div>
        );

      case 'select':
      case 'model_select':
        return (
          <div>
            {fieldLabel}
            <select
              value={value}
              onChange={(e) => handleChange(field.name, e.target.value)}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 appearance-none cursor-pointer"
            >
              <option value="">Select {field.label}</option>
              {field.options?.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {helpText}
          </div>
        );

      case 'slider':
        const min = field.validation?.min_value ?? 0;
        const max = field.validation?.max_value ?? 100;
        const step = field.properties?.step ?? 1;
        return (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-sm font-medium text-gray-300">
                {field.label}
              </label>
              <span className="text-sm text-indigo-400 font-mono">{value}</span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={value || min}
              onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{min}</span>
              <span>{max}</span>
            </div>
            {helpText}
          </div>
        );

      case 'json':
        return (
          <div>
            {fieldLabel}
            <textarea
              value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  handleChange(field.name, parsed);
                } catch {
                  handleChange(field.name, e.target.value);
                }
              }}
              placeholder={field.ui?.placeholder || '{}'}
              rows={4}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-sm resize-none"
            />
            {helpText}
          </div>
        );

      default:
        return (
          <div>
            {fieldLabel}
            <Input
              value={value}
              onChange={(e) => handleChange(field.name, e.target.value)}
              placeholder={field.ui?.placeholder}
            />
            {helpText}
          </div>
        );
    }
  };

  // Group fields by their group
  const { groupedFields, ungroupedFields } = useMemo(() => {
    const grouped: Record<string, FieldDefinition[]> = {};
    const ungrouped: FieldDefinition[] = [];

    config?.fields.forEach((fieldData) => {
      // Convert field data to FieldDefinition if needed
      const field = fieldData as any;
      if (field.ui?.group) {
        if (!grouped[field.ui.group]) {
          grouped[field.ui.group] = [];
        }
        grouped[field.ui.group].push(field);
      } else if (field.group) {
        if (!grouped[field.group]) {
          grouped[field.group] = [];
        }
        grouped[field.group].push(field);
      } else {
        ungrouped.push(field);
      }
    });

    return { groupedFields: grouped, ungroupedFields: ungrouped };
  }, [config?.fields]);

  const fieldGroups = config?.field_groups || [];

  return (
    <div className="w-96 bg-dark-200 border-l border-gray-700 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-white truncate">
              {node.data.label}
            </h3>
            <p className="text-xs text-gray-500 font-mono">
              {node.data.component_type}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {config?.description && (
          <p className="text-sm text-gray-400 mt-2">{config.description}</p>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg transition-colors ${
              activeTab === 'settings'
                ? 'bg-indigo-500/20 text-indigo-400'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <Sliders className="w-4 h-4" />
            Settings
          </button>
          <button
            onClick={() => setActiveTab('json')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg transition-colors ${
              activeTab === 'json'
                ? 'bg-indigo-500/20 text-indigo-400'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <Code className="w-4 h-4" />
            JSON
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'settings' ? (
          <div className="p-4 space-y-4">
            {/* Ungrouped fields first */}
            {ungroupedFields.map((field: any) => (
              <div key={field.name}>{renderField(field)}</div>
            ))}

            {/* Grouped fields */}
            {fieldGroups
              .sort((a, b) => a.order - b.order)
              .map((group) => {
                const fields = groupedFields[group.id] || [];
                if (fields.length === 0) return null;

                const isExpanded = expandedGroups.has(group.id);

                return (
                  <div 
                    key={group.id} 
                    className="border border-gray-700 rounded-lg overflow-hidden"
                  >
                    <button
                      onClick={() => group.collapsible && toggleGroup(group.id)}
                      className={`w-full flex items-center justify-between px-4 py-3 bg-dark-300 text-left ${
                        group.collapsible ? 'cursor-pointer hover:bg-dark-100' : 'cursor-default'
                      }`}
                    >
                      <div>
                        <span className="text-sm font-medium text-white">
                          {group.label}
                        </span>
                        {group.description && (
                          <p className="text-xs text-gray-500">{group.description}</p>
                        )}
                      </div>
                      {group.collapsible && (
                        isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        )
                      )}
                    </button>

                    {(!group.collapsible || isExpanded) && (
                      <div className="p-4 space-y-4 border-t border-gray-700">
                        {fields
                          .sort((a: any, b: any) => (a.ui?.order || 0) - (b.ui?.order || 0))
                          .map((field: any) => (
                            <div key={field.name}>{renderField(field)}</div>
                          ))}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        ) : (
          <div className="p-4">
            <textarea
              value={jsonValue}
              onChange={(e) => handleJsonChange(e.target.value)}
              className={`w-full h-80 px-4 py-3 bg-dark-300 border rounded-lg text-white font-mono text-sm resize-none focus:outline-none focus:ring-2 ${
                jsonError 
                  ? 'border-red-500 focus:ring-red-500' 
                  : 'border-gray-700 focus:ring-indigo-500'
              }`}
            />
            {jsonError && (
              <p className="mt-2 text-sm text-red-400 flex items-center gap-1">
                <AlertCircle className="w-4 h-4" />
                {jsonError}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700 space-y-2">
        {isDirty && (
          <div className="flex items-center gap-2 text-xs text-amber-400 mb-2">
            <AlertCircle className="w-4 h-4" />
            You have unsaved changes
          </div>
        )}
        
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            disabled={!isDirty}
            icon={<RotateCcw className="w-4 h-4" />}
          >
            Reset
          </Button>
          <Button
            variant="danger"
            size="sm"
            onClick={handleDelete}
            icon={<Trash2 className="w-4 h-4" />}
          >
            Delete
          </Button>
          <Button
            variant="primary"
            size="sm"
            className="flex-1"
            onClick={handleSave}
            disabled={!isDirty || !!jsonError}
            icon={<Save className="w-4 h-4" />}
          >
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ConfigPanel;

```

# src/components/builder/CustomNode.tsx

```tsx
// src/components/builder/CustomNode.tsx
import React, { memo, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  MessageSquare,
  Bot,
  Brain,
  Plug,
  ArrowRightCircle,
  Settings,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import { ComponentConfig } from '../../types';
import { getDefaultPortsForType } from '../../utils/defaultComponentConfig';

interface CustomNodeData {
  label: string;
  component_type: string;
  config?: ComponentConfig;
  parameters: Record<string, any>;
  isValid?: boolean;
  errors?: string[];
}

const getNodeIcon = (componentType: string): React.ReactNode => {
  const iconMap: Record<string, React.ReactNode> = {
    input: <MessageSquare className="w-5 h-5" />,
    output: <ArrowRightCircle className="w-5 h-5" />,
    agent: <Bot className="w-5 h-5" />,
    llm_openai: <Brain className="w-5 h-5" />,
    llm_anthropic: <Brain className="w-5 h-5" />,
    llm_openrouter: <Brain className="w-5 h-5" />,
    composio_tool: <Plug className="w-5 h-5" />,
  };
  return iconMap[componentType] || <Settings className="w-5 h-5" />;
};

const getNodeColors = (componentType: string) => {
  const colorMap: Record<string, { bg: string; border: string; text: string; glow: string }> = {
    input: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500',
      text: 'text-emerald-400',
      glow: 'shadow-emerald-500/20',
    },
    output: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500',
      text: 'text-amber-400',
      glow: 'shadow-amber-500/20',
    },
    agent: {
      bg: 'bg-blue-500/10',
      border: 'border-blue-500',
      text: 'text-blue-400',
      glow: 'shadow-blue-500/20',
    },
    llm_openai: {
      bg: 'bg-green-500/10',
      border: 'border-green-500',
      text: 'text-green-400',
      glow: 'shadow-green-500/20',
    },
    llm_anthropic: {
      bg: 'bg-orange-500/10',
      border: 'border-orange-500',
      text: 'text-orange-400',
      glow: 'shadow-orange-500/20',
    },
    llm_openrouter: {
      bg: 'bg-indigo-500/10',
      border: 'border-indigo-500',
      text: 'text-indigo-400',
      glow: 'shadow-indigo-500/20',
    },
    composio_tool: {
      bg: 'bg-pink-500/10',
      border: 'border-pink-500',
      text: 'text-pink-400',
      glow: 'shadow-pink-500/20',
    },
  };
  return (
    colorMap[componentType] || {
      bg: 'bg-gray-500/10',
      border: 'border-gray-500',
      text: 'text-gray-400',
      glow: 'shadow-gray-500/20',
    }
  );
};

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data, selected, isConnectable }) => {
  const colors = useMemo(() => getNodeColors(data.component_type), [data.component_type]);
  const config = data.config;
  const isValid = data.isValid !== false;
  const errors = data.errors || [];

  const defaultPorts = useMemo(() => getDefaultPortsForType(data.component_type), [data.component_type]);
  const inputPorts = config?.input_ports?.length ? config.input_ports : defaultPorts.input_ports;
  const outputPorts = config?.output_ports?.length ? config.output_ports : defaultPorts.output_ports;

  // Get parameter preview
  const parameterPreview = useMemo(() => {
    const params = data.parameters || {};
    const entries = Object.entries(params).slice(0, 3);
    return entries.map(([key, value]) => {
      let displayValue = value;
      if (typeof value === 'string' && value.length > 20) {
        displayValue = value.substring(0, 20) + '...';
      } else if (typeof value === 'object') {
        displayValue = JSON.stringify(value).substring(0, 20) + '...';
      }
      return { key, value: displayValue };
    });
  }, [data.parameters]);

  const handleStyle = {
    width: 12,
    height: 12,
    border: '2px solid white',
    background: '#6366f1',
  };

  return (
    <div
      className={`
        min-w-[220px] max-w-[280px] bg-dark-200 rounded-xl border-2 transition-all duration-200
        ${selected ? `${colors.border} shadow-lg ${colors.glow}` : 'border-gray-700 hover:border-gray-600'}
      `}
    >
      {/* Input Handles */}
      {inputPorts.map((port, index) => (
        <Handle
          key={`input-${port.id}`}
          type="target"
          position={Position.Left}
          id={port.id}
          isConnectable={isConnectable}
          style={{
            ...handleStyle,
            top: `${((index + 1) / (inputPorts.length + 1)) * 100}%`,
          }}
          className="!bg-indigo-500 hover:!bg-indigo-400 transition-colors"
        />
      ))}

      {/* Node Header */}
      <div className={`px-4 py-3 ${colors.bg} rounded-t-[10px] border-b border-gray-700/50`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <div className={`${colors.text} flex-shrink-0`}>
              {getNodeIcon(data.component_type)}
            </div>
            <span className="font-medium text-white truncate">{data.label}</span>
          </div>
          
          {/* Status indicator */}
          <div className="flex-shrink-0 ml-2">
            {isValid ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-500" />
            )}
          </div>
        </div>
      </div>

      {/* Node Body */}
      <div className="px-4 py-3">
        {/* Description */}
        <p className="text-xs text-gray-400 mb-3 line-clamp-2">
          {config?.description || 'Configure this component'}
        </p>
        
        {/* Parameter Preview */}
        {parameterPreview.length > 0 && (
          <div className="space-y-1.5 mb-2">
            {parameterPreview.map(({ key, value }) => (
              <div 
                key={key} 
                className="flex items-center justify-between text-xs bg-dark-300/50 rounded px-2 py-1"
              >
                <span className="text-gray-500 truncate mr-2">{key}</span>
                <span className="text-gray-300 truncate max-w-[100px] font-mono text-[10px]">
                  {String(value)}
                </span>
              </div>
            ))}
            {Object.keys(data.parameters || {}).length > 3 && (
              <span className="text-xs text-gray-500 italic">
                +{Object.keys(data.parameters).length - 3} more parameters
              </span>
            )}
          </div>
        )}

        {/* Errors */}
        {errors.length > 0 && (
          <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400">
            {errors[0]}
            {errors.length > 1 && ` (+${errors.length - 1} more)`}
          </div>
        )}
      </div>

      {/* Port Labels */}
      {(inputPorts.length > 0 || outputPorts.length > 0) && (
        <div className="px-4 pb-3 flex justify-between text-[10px] text-gray-500 font-mono">
          <div className="space-y-0.5">
            {inputPorts.map((port: any) => (
              <div key={port.id} className="flex items-center gap-1">
                <div className={`w-1.5 h-1.5 rounded-full ${port.required ? 'bg-red-400' : 'bg-gray-500'}`} />
                <span>{port.name}</span>
              </div>
            ))}
          </div>
          <div className="space-y-0.5 text-right">
            {outputPorts.map((port: any) => (
              <div key={port.id} className="flex items-center justify-end gap-1">
                <span>{port.name}</span>
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Component Type Badge */}
      <div className="absolute -bottom-2 left-1/2 -translate-x-1/2">
        <span className={`
          px-2 py-0.5 text-[9px] font-medium rounded-full
          ${colors.bg} ${colors.text} border ${colors.border}
        `}>
          {data.component_type}
        </span>
      </div>

      {/* Output Handles */}
      {outputPorts.map((port, index) => (
        <Handle
          key={`output-${port.id}`}
          type="source"
          position={Position.Right}
          id={port.id}
          isConnectable={isConnectable}
          style={{
            ...handleStyle,
            top: `${((index + 1) / (outputPorts.length + 1)) * 100}%`,
          }}
          className="!bg-indigo-500 hover:!bg-indigo-400 transition-colors"
        />
      ))}
    </div>
  );
};

export default memo(CustomNode);

```

# src/components/builder/Toolbar.tsx

```tsx
// src/components/builder/Toolbar.tsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useFlowStore } from '../../contexts/FlowContext';
import { Button } from '../ui/Button';
import { 
  ArrowLeft, 
  Save, 
  Play, 
  ZoomIn, 
  ZoomOut,
  Maximize,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import toast from 'react-hot-toast';

interface ToolbarProps {
  flowName: string;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  flowName,
  onZoomIn,
  onZoomOut,
  onFitView,
}) => {
  const navigate = useNavigate();
  const { currentFlow, isDirty, isSaving, saveFlow } = useFlowStore();

  const isValid = currentFlow?.agent_schema?.validation?.is_valid ?? false;
  const validationErrors = currentFlow?.agent_schema?.validation?.errors ?? [];

  const handleSave = async () => {
    try {
      await saveFlow();
      toast.success('Flow saved successfully');
    } catch (error) {
      toast.error('Failed to save flow');
    }
  };

  const handleRun = () => {
    if (!isValid) {
      toast.error('Please fix validation errors before running');
      return;
    }
    // TODO: Implement run modal
    toast.success('Flow execution started');
  };

  return (
    <div className="h-14 bg-dark-200 border-b border-gray-700 flex items-center justify-between px-4">
      {/* Left section */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/dashboard')}
          icon={<ArrowLeft className="w-4 h-4" />}
        >
          Back
        </Button>

        <div className="h-6 w-px bg-gray-700" />

        <div className="flex items-center gap-2">
          <h1 className="text-lg font-semibold text-white">{flowName}</h1>
          {isDirty && (
            <span className="text-xs text-amber-400">(unsaved)</span>
          )}
        </div>
      </div>

      {/* Center section - Zoom controls */}
      <div className="flex items-center gap-1 bg-dark-300 rounded-lg p-1">
        <button
          onClick={onZoomOut}
          className="p-1.5 text-gray-400 hover:text-white rounded hover:bg-white/10 transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={onFitView}
          className="p-1.5 text-gray-400 hover:text-white rounded hover:bg-white/10 transition-colors"
          title="Fit View"
        >
          <Maximize className="w-4 h-4" />
        </button>
        <button
          onClick={onZoomIn}
          className="p-1.5 text-gray-400 hover:text-white rounded hover:bg-white/10 transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-3">
        {/* Validation status */}
        <div className="flex items-center gap-2">
          {isValid ? (
            <div className="flex items-center gap-1.5 text-sm text-green-400">
              <CheckCircle className="w-4 h-4" />
              <span>Valid</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-sm text-amber-400">
              <AlertCircle className="w-4 h-4" />
              <span>{validationErrors.length} issues</span>
            </div>
          )}
        </div>

        <div className="h-6 w-px bg-gray-700" />

        <Button
          variant="secondary"
          size="sm"
          onClick={handleSave}
          isLoading={isSaving}
          icon={<Save className="w-4 h-4" />}
        >
          Save
        </Button>

        <Button
          variant="primary"
          size="sm"
          onClick={handleRun}
          disabled={!isValid}
          icon={<Play className="w-4 h-4" />}
        >
          Run
        </Button>
      </div>
    </div>
  );
};

```

# src/components/dashboard/FlowCard.tsx

```tsx
// src/components/dashboard/FlowCard.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Flow } from '../../types';
import { Button } from '../ui/Button';
import { Modal } from '../ui/Modal';
import { 
  Play, 
  Edit3, 
  Trash2, 
  MoreVertical,
  Clock,
  CheckCircle,
  XCircle,
  GitBranch
} from 'lucide-react';
import api from '../../api';
import toast from 'react-hot-toast';

interface FlowCardProps {
  flow: Flow;
  onDelete: (flowId: string) => void;
}

export const FlowCard: React.FC<FlowCardProps> = ({ flow, onDelete }) => {
  const navigate = useNavigate();
  const [showMenu, setShowMenu] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const isValid = flow.agent_schema?.validation?.is_valid ?? false;
  const nodeCount = flow.nodes?.length ?? 0;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await api.deleteFlow(flow.flow_id);
      toast.success('Flow deleted successfully');
      onDelete(flow.flow_id);
    } catch (error) {
      toast.error('Failed to delete flow');
    } finally {
      setIsDeleting(false);
      setShowDeleteModal(false);
    }
  };

  return (
    <>
      <div className="group relative bg-dark-200 rounded-xl border border-gray-700 hover:border-indigo-500/50 transition-all duration-300 overflow-hidden">
        {/* Card Header */}
        <div className="p-5">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-white truncate">
                {flow.name}
              </h3>
              {flow.description && (
                <p className="text-sm text-gray-400 mt-1 line-clamp-2">
                  {flow.description}
                </p>
              )}
            </div>
            
            {/* Menu */}
            <div className="relative ml-2">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors"
              >
                <MoreVertical className="w-5 h-5" />
              </button>
              
              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  />
                  <div className="absolute right-0 mt-1 w-36 bg-dark-100 rounded-lg shadow-xl border border-gray-700 py-1 z-20 animate-fadeIn">
                    <button
                      onClick={() => {
                        setShowMenu(false);
                        navigate(`/builder/${flow.flow_id}`);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-white/5"
                    >
                      <Edit3 className="w-4 h-4" />
                      Edit
                    </button>
                    <button
                      onClick={() => {
                        setShowMenu(false);
                        setShowDeleteModal(true);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-white/5"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <div className="flex items-center gap-1.5">
              <GitBranch className="w-4 h-4" />
              <span>{nodeCount} nodes</span>
            </div>
            <div className="flex items-center gap-1.5">
              {isValid ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-amber-500" />
              )}
              <span>{isValid ? 'Valid' : 'Invalid'}</span>
            </div>
          </div>
        </div>

        {/* Card Footer */}
        <div className="px-5 py-3 bg-dark-300/50 border-t border-gray-700/50 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <Clock className="w-3.5 h-3.5" />
            <span>Updated {formatDate(flow.updated_at)}</span>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(`/builder/${flow.flow_id}`)}
              icon={<Edit3 className="w-4 h-4" />}
            >
              Edit
            </Button>
            {isValid && (
              <Button
                variant="primary"
                size="sm"
                icon={<Play className="w-4 h-4" />}
              >
                Run
              </Button>
            )}
          </div>
        </div>

        {/* Hover gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/0 via-indigo-500/5 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Flow"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-gray-300">
            Are you sure you want to delete <strong>"{flow.name}"</strong>? 
            This action cannot be undone.
          </p>
          <div className="flex gap-3 justify-end">
            <Button
              variant="secondary"
              onClick={() => setShowDeleteModal(false)}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleDelete}
              isLoading={isDeleting}
            >
              Delete
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
};

```

# src/components/dashboard/FlowGrid.tsx

```tsx
// src/components/dashboard/FlowGrid.tsx
import React from 'react';
import { Flow } from '../../types';
import { FlowCard } from './FlowCard';
import { Plus, FolderOpen } from 'lucide-react';
import { Button } from '../ui/Button';

interface FlowGridProps {
  flows: Flow[];
  isLoading: boolean;
  onCreateNew: () => void;
  onDelete: (flowId: string) => void;
}

export const FlowGrid: React.FC<FlowGridProps> = ({
  flows,
  isLoading,
  onCreateNew,
  onDelete,
}) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="h-48 bg-dark-200 rounded-xl animate-pulse"
          />
        ))}
      </div>
    );
  }

  if (flows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="w-20 h-20 bg-dark-200 rounded-2xl flex items-center justify-center mb-6">
          <FolderOpen className="w-10 h-10 text-gray-500" />
        </div>
        <h3 className="text-xl font-semibold text-white mb-2">
          No flows yet
        </h3>
        <p className="text-gray-400 mb-6 text-center max-w-md">
          Create your first AI agent flow to get started. 
          Drag and drop components to build powerful workflows.
        </p>
        <Button onClick={onCreateNew} icon={<Plus className="w-5 h-5" />}>
          Create Your First Flow
        </Button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {flows.map((flow) => (
        <FlowCard key={flow.flow_id} flow={flow} onDelete={onDelete} />
      ))}
    </div>
  );
};

```

# src/components/layout/Header.tsx

```tsx
// src/components/layout/Header.tsx
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { Button } from '../ui/Button';
import { 
  Bot, 
  LogOut, 
  User, 
  LayoutDashboard,
  ChevronDown 
} from 'lucide-react';

export const Header: React.FC = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = React.useState(false);

  const handleLogout = () => {
    logout();
    navigate('/auth');
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-dark-200/80 backdrop-blur-lg border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-xl">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">Agent Builder</span>
          </Link>

          {/* Navigation */}
          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-6">
              <Link
                to="/dashboard"
                className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
              >
                <LayoutDashboard className="w-4 h-4" />
                Dashboard
              </Link>
            </nav>
          )}

          {/* User Menu */}
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-sm font-medium text-white">
                    {user?.username}
                  </span>
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </button>

                {showDropdown && (
                  <>
                    <div
                      className="fixed inset-0"
                      onClick={() => setShowDropdown(false)}
                    />
                    <div className="absolute right-0 mt-2 w-48 bg-dark-100 rounded-xl shadow-xl border border-gray-700 py-1 animate-fadeIn">
                      <div className="px-4 py-2 border-b border-gray-700">
                        <p className="text-sm text-gray-400">Signed in as</p>
                        <p className="text-sm font-medium text-white truncate">
                          {user?.email}
                        </p>
                      </div>
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-white/5 transition-colors"
                      >
                        <LogOut className="w-4 h-4" />
                        Sign out
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <Button
                onClick={() => navigate('/auth')}
                variant="primary"
              >
                Sign In
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

```

# src/components/layout/Layout.tsx

```tsx
// src/components/layout/Layout.tsx
import React from 'react';
import { Header } from './Header';

interface LayoutProps {
  children: React.ReactNode;
  showHeader?: boolean;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  showHeader = true 
}) => {
  return (
    <div className="min-h-screen bg-dark-300">
      {showHeader && <Header />}
      <main className={showHeader ? 'pt-16' : ''}>
        {children}
      </main>
    </div>
  );
};

```

# src/components/ui/Button.tsx

```tsx
// src/components/ui/Button.tsx
import React from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-300 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-indigo-600 hover:bg-indigo-700 text-white focus:ring-indigo-500',
    secondary: 'bg-dark-100 hover:bg-dark-200 text-white border border-gray-700 focus:ring-gray-500',
    ghost: 'bg-transparent hover:bg-white/10 text-gray-300 focus:ring-gray-500',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2',
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : icon ? (
        icon
      ) : null}
      {children}
    </button>
  );
};

```

# src/components/ui/Input.tsx

```tsx
// src/components/ui/Input.tsx
import React, { forwardRef } from 'react';
import { Eye, EyeOff } from 'lucide-react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, icon, className = '', type = 'text', ...props }, ref) => {
    const [showPassword, setShowPassword] = React.useState(false);
    const isPassword = type === 'password';

    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-gray-300 mb-1.5">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              {icon}
            </div>
          )}
          <input
            ref={ref}
            type={isPassword ? (showPassword ? 'text' : 'password') : type}
            className={`
              w-full px-4 py-2.5 bg-dark-200 border border-gray-700 rounded-lg
              text-white placeholder-gray-500
              focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
              transition-all duration-200
              ${icon ? 'pl-10' : ''}
              ${isPassword ? 'pr-10' : ''}
              ${error ? 'border-red-500 focus:ring-red-500' : ''}
              ${className}
            `}
            {...props}
          />
          {isPassword && (
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-300"
            >
              {showPassword ? (
                <EyeOff className="w-5 h-5" />
              ) : (
                <Eye className="w-5 h-5" />
              )}
            </button>
          )}
        </div>
        {error && (
          <p className="mt-1.5 text-sm text-red-500">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

```

# src/components/ui/Modal.tsx

```tsx
// src/components/ui/Modal.tsx
import React, { useEffect } from 'react';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
}) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sizes = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div
        className={`
          relative w-full ${sizes[size]} mx-4
          bg-dark-200 rounded-xl shadow-2xl
          animate-slideUp
        `}
      >
        {title && (
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <button
              onClick={onClose}
              className="p-1 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
};

```

# src/components/ui/Toast.tsx

```tsx
// src/components/ui/Toast.tsx
import { Toaster } from 'react-hot-toast';

export const ToastContainer = () => {
  return (
    <Toaster
      position="top-right"
      toastOptions={{
        duration: 4000,
        style: {
          background: '#1e1e2e',
          color: '#fff',
          border: '1px solid #374151',
          borderRadius: '12px',
          padding: '16px',
        },
        success: {
          iconTheme: {
            primary: '#10b981',
            secondary: '#fff',
          },
        },
        error: {
          iconTheme: {
            primary: '#ef4444',
            secondary: '#fff',
          },
        },
      }}
    />
  );
};

```

# src/contexts/AuthContext.tsx

```tsx
// src/contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User } from '../types';
import api from '../api';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

// Export the context so it can be used in hooks
export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      setUser(null);
      setIsLoading(false);
      return;
    }

    try {
      const userData = await api.getCurrentUser();
      setUser(userData);
    } catch (error) {
      localStorage.removeItem('access_token');
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (email: string, password: string) => {
    const tokens = await api.login(email, password);
    localStorage.setItem('access_token', tokens.access_token);
    await refreshUser();
  };

  const register = async (username: string, email: string, password: string) => {
    await api.register(username, email, password);
    await login(email, password);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

```

# src/contexts/FlowContext.tsx

```tsx
// src/contexts/FlowContext.tsx
import { create } from 'zustand';
import {
  Node,
  Edge,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
  Connection,
} from 'reactflow';
import { Flow, ComponentConfig } from '../types';
import api from '../api';
import { getDefaultPortsForType, getFallbackComponentConfig } from '../utils/defaultComponentConfig';

interface FlowState {
  // Current flow data
  currentFlow: Flow | null;
  nodes: Node[];
  edges: Edge[];
  
  // Component data
  components: ComponentConfig[];
  componentsByCategory: Record<string, ComponentConfig[]>;
  categories: string[];
  
  // UI state
  selectedNode: Node | null;
  isDirty: boolean;
  isSaving: boolean;
  isLoading: boolean;
  
  // Actions
  setCurrentFlow: (flow: Flow | null) => void;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (componentType: string, position: { x: number; y: number }) => void;
  updateNodeData: (nodeId: string, data: any) => void;
  deleteNode: (nodeId: string) => void;
  setSelectedNode: (node: Node | null) => void;
  
  // API actions
  loadComponents: () => Promise<void>;
  loadFlow: (flowId: string) => Promise<void>;
  saveFlow: () => Promise<void>;
  createNewFlow: (name: string, description?: string) => Promise<Flow>;
  resetStore: () => void;
}

const generateNodeId = () => `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
const generateEdgeId = () => `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const withDefaultPorts = (config: ComponentConfig): ComponentConfig => {
  const defaults = getDefaultPortsForType(config.component_type);
  const mergePorts = (fallback: any[], provided?: any[]) => {
    if (!provided || provided.length === 0) return fallback;
    const byId = new Map<string, any>();
    fallback.forEach((p) => byId.set(p.id, p));
    provided.forEach((p) => byId.set(p.id, { ...byId.get(p.id), ...p }));
    return Array.from(byId.values());
  };

  return {
    ...config,
    input_ports: mergePorts(defaults.input_ports, config.input_ports),
    output_ports: mergePorts(defaults.output_ports, config.output_ports),
  };
};

const normalizeConfig = (componentType: string, config?: ComponentConfig, name?: string) =>
  withDefaultPorts(config ?? getFallbackComponentConfig(componentType, name));

const initialState = {
  currentFlow: null,
  nodes: [],
  edges: [],
  components: [],
  componentsByCategory: {},
  categories: [],
  selectedNode: null,
  isDirty: false,
  isSaving: false,
  isLoading: false,
};

export const useFlowStore = create<FlowState>((set, get) => ({
  ...initialState,

  setCurrentFlow: (flow) => set({ currentFlow: flow, isDirty: false }),
  
  setNodes: (nodes) => set({ nodes, isDirty: true }),
  
  setEdges: (edges) => set({ edges, isDirty: true }),

  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
      isDirty: true,
    });
  },

  onEdgesChange: (changes) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
      isDirty: true,
    });
  },

  onConnect: (connection) => {
    const newEdge = {
      ...connection,
      id: generateEdgeId(),
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
    };
    set({
      edges: addEdge(newEdge, get().edges),
      isDirty: true,
    });
  },

  addNode: (componentType, position) => {
    const components = get().components;
    const componentConfig = components.find((c) => c.component_type === componentType);
    const resolvedConfig = normalizeConfig(componentType, componentConfig);

    // Build default parameters from field definitions
    const defaultParams: Record<string, any> = {};
    resolvedConfig.fields?.forEach((field) => {
      if (field.default !== undefined) {
        defaultParams[field.name] = field.default;
      }
    });

    const newNode: Node = {
      id: generateNodeId(),
      type: 'custom',
      position,
      data: {
        label: resolvedConfig.name,
        component_type: componentType,
        config: resolvedConfig,
        parameters: defaultParams,
      },
    };

    set({
      nodes: [...get().nodes, newNode],
      isDirty: true,
    });
  },

  updateNodeData: (nodeId, data) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      ),
      isDirty: true,
    });
  },

  deleteNode: (nodeId) => {
    set({
      nodes: get().nodes.filter((node) => node.id !== nodeId),
      edges: get().edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId
      ),
      selectedNode: get().selectedNode?.id === nodeId ? null : get().selectedNode,
      isDirty: true,
    });
  },

  setSelectedNode: (node) => set({ selectedNode: node }),

  loadComponents: async () => {
    try {
      const data = await api.getComponentCategories();
      const allComponents: ComponentConfig[] = [];

      const componentsByCategory: Record<string, ComponentConfig[]> = {};
      Object.entries(data.components_by_category).forEach(([category, components]) => {
        const normalized = components.map((c: any) =>
          normalizeConfig(c.component_type, c.config, c.config?.name || c.component_type)
        );
        componentsByCategory[category] = normalized;
        normalized.forEach((cfg) => allComponents.push(cfg));
      });

      set({
        components: allComponents,
        componentsByCategory,
        categories: data.categories,
      });
    } catch (error) {
      console.error('Failed to load components:', error);
      throw error;
    }
  },

  loadFlow: async (flowId) => {
    set({ isLoading: true });
    try {
      const flow = await api.getFlow(flowId);
      const components = get().components;
      
      // Convert flow nodes to React Flow nodes
      const nodes: Node[] = (flow.nodes || []).map((node) => {
        const componentConfig = components.find(
          (c) => c.component_type === node.data.component_type
        );
        const resolvedConfig = normalizeConfig(
          node.data.component_type,
          componentConfig,
          node.data.label
        );
        
        return {
          id: node.node_id,
          type: 'custom',
          position: node.position,
          data: {
            label: node.data.label,
            component_type: node.data.component_type,
            config: resolvedConfig,
            parameters: node.data.parameters || {},
          },
        };
      });

      // Convert flow edges to React Flow edges
      const edges: Edge[] = (flow.edges || []).map((edge) => ({
        id: edge.edge_id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.source_handle,
        targetHandle: edge.target_handle,
        animated: edge.animated,
        style: { stroke: '#6366f1', strokeWidth: 2 },
      }));

      set({
        currentFlow: flow,
        nodes,
        edges,
        isDirty: false,
        isLoading: false,
      });
    } catch (error) {
      console.error('Failed to load flow:', error);
      set({ isLoading: false });
      throw error;
    }
  },

  saveFlow: async () => {
    const { currentFlow, nodes, edges } = get();
    if (!currentFlow) return;

    set({ isSaving: true });
    try {
      // Convert React Flow nodes back to flow nodes
      const flowNodes = nodes.map((node) => ({
        node_id: node.id,
        type: node.type || 'custom',
        position: node.position,
        data: {
          label: node.data.label,
          component_type: node.data.component_type,
          parameters: node.data.parameters || {},
        },
      }));

      // Convert React Flow edges back to flow edges
      const flowEdges = edges.map((edge) => ({
        edge_id: edge.id,
        source: edge.source,
        target: edge.target,
        source_handle: edge.sourceHandle || 'output',
        target_handle: edge.targetHandle || 'input',
        animated: edge.animated ?? true,
      }));

      const updatedFlow = await api.updateFlow(currentFlow.flow_id, {
        nodes: flowNodes,
        edges: flowEdges,
      });

      set({
        currentFlow: updatedFlow,
        isDirty: false,
        isSaving: false,
      });
    } catch (error) {
      console.error('Failed to save flow:', error);
      set({ isSaving: false });
      throw error;
    }
  },

  createNewFlow: async (name, description) => {
    const flow = await api.createFlow({ name, description });
    set({
      currentFlow: flow,
      nodes: [],
      edges: [],
      isDirty: false,
    });
    return flow;
  },

  resetStore: () => set(initialState),
}));

export default useFlowStore;

```

# src/hooks/useAuth.ts

```typescript
// src/hooks/useAuth.ts
import { useContext, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

/**
 * Custom hook for authentication functionality
 * Extends the base AuthContext with additional utilities
 */
export const useAuthHook = () => {
  const context = useContext(AuthContext);
  const navigate = useNavigate();

  if (context === undefined) {
    throw new Error('useAuthHook must be used within an AuthProvider');
  }

  const {
    user,
    isLoading,
    isAuthenticated,
    login: contextLogin,
    register: contextRegister,
    logout: contextLogout,
    refreshUser,
  } = context;

  /**
   * Login with error handling and navigation
   */
  const loginWithRedirect = useCallback(
    async (email: string, password: string, redirectTo: string = '/dashboard') => {
      try {
        await contextLogin(email, password);
        toast.success('Welcome back!');
        navigate(redirectTo);
        return { success: true };
      } catch (error: any) {
        const message = error.response?.data?.detail || 'Login failed. Please try again.';
        toast.error(message);
        return { success: false, error: message };
      }
    },
    [contextLogin, navigate]
  );

  /**
   * Register with error handling and navigation
   */
  const registerWithRedirect = useCallback(
    async (
      username: string,
      email: string,
      password: string,
      redirectTo: string = '/dashboard'
    ) => {
      try {
        await contextRegister(username, email, password);
        toast.success('Account created successfully!');
        navigate(redirectTo);
        return { success: true };
      } catch (error: any) {
        const message =
          error.response?.data?.detail || 'Registration failed. Please try again.';
        toast.error(message);
        return { success: false, error: message };
      }
    },
    [contextRegister, navigate]
  );

  /**
   * Logout with navigation
   */
  const logoutWithRedirect = useCallback(
    (redirectTo: string = '/auth') => {
      contextLogout();
      toast.success('Logged out successfully');
      navigate(redirectTo);
    },
    [contextLogout, navigate]
  );

  /**
   * Check if user has a specific role (for future role-based access)
   */
  const hasRole = useCallback(
    (_role: string): boolean => {
      // Implement role checking logic when roles are added
      return isAuthenticated;
    },
    [isAuthenticated]
  );

  /**
   * Get user's display name
   */
  const displayName = useMemo(() => {
    if (!user) return '';
    return user.username || user.email.split('@')[0];
  }, [user]);

  /**
   * Get user's initials for avatar
   */
  const initials = useMemo(() => {
    if (!user) return '';
    const name = user.username || user.email;
    return name
      .split(/[\s@]/)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() || '')
      .join('');
  }, [user]);

  return {
    // State
    user,
    isLoading,
    isAuthenticated,
    displayName,
    initials,

    // Original actions
    login: contextLogin,
    register: contextRegister,
    logout: contextLogout,
    refreshUser,

    // Extended actions
    loginWithRedirect,
    registerWithRedirect,
    logoutWithRedirect,
    hasRole,
  };
};

export default useAuthHook;

```

# src/hooks/useFlow.ts

```typescript
// src/hooks/useFlow.ts
import { useCallback, useMemo } from 'react';
import { useFlowStore } from '../contexts/FlowContext';
import { Node, useReactFlow, XYPosition } from 'reactflow';
import api from '../api';
import toast from 'react-hot-toast';
import { generateId } from '../utils/helpers';

/**
 * Custom hook for flow operations
 * Provides convenient methods for working with the flow builder
 */
export const useFlow = () => {
  const store = useFlowStore();
  const reactFlow = useReactFlow();

  /**
   * Add a new node to the canvas
   */
  const addNodeAtPosition = useCallback(
    (componentType: string, position: XYPosition) => {
      store.addNode(componentType, position);
    },
    [store]
  );

  /**
   * Add a node at the center of the viewport
   */
  const addNodeAtCenter = useCallback(
    (componentType: string) => {
      const { x, y, zoom } = reactFlow.getViewport();
      const centerX = (-x + window.innerWidth / 2) / zoom;
      const centerY = (-y + window.innerHeight / 2) / zoom;
      store.addNode(componentType, { x: centerX - 100, y: centerY - 50 });
    },
    [store, reactFlow]
  );

  /**
   * Duplicate a node
   */
  const duplicateNode = useCallback(
    (nodeId: string) => {
      const node = store.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      const newNode: Node = {
        ...node,
        id: generateId('node'),
        position: {
          x: node.position.x + 50,
          y: node.position.y + 50,
        },
        data: { ...node.data },
        selected: false,
      };

      store.setNodes([...store.nodes, newNode]);
      toast.success('Node duplicated');
    },
    [store]
  );

  /**
   * Delete selected nodes
   */
  const deleteSelectedNodes = useCallback(() => {
    const selectedNodes = store.nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    selectedNodes.forEach((node) => {
      store.deleteNode(node.id);
    });

    toast.success(`Deleted ${selectedNodes.length} node(s)`);
  }, [store]);

  /**
   * Select all nodes
   */
  const selectAllNodes = useCallback(() => {
    const updatedNodes = store.nodes.map((node) => ({
      ...node,
      selected: true,
    }));
    store.setNodes(updatedNodes);
  }, [store]);

  /**
   * Deselect all nodes
   */
  const deselectAllNodes = useCallback(() => {
    const updatedNodes = store.nodes.map((node) => ({
      ...node,
      selected: false,
    }));
    store.setNodes(updatedNodes);
    store.setSelectedNode(null);
  }, [store]);

  /**
   * Center view on a specific node
   */
  const centerOnNode = useCallback(
    (nodeId: string) => {
      const node = store.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      reactFlow.setCenter(node.position.x + 100, node.position.y + 50, {
        duration: 500,
        zoom: 1,
      });
    },
    [store.nodes, reactFlow]
  );

  /**
   * Fit all nodes in view
   */
  const fitToView = useCallback(() => {
    reactFlow.fitView({ padding: 0.2, duration: 500 });
  }, [reactFlow]);

  /**
   * Zoom to specific level
   */
  const zoomTo = useCallback(
    (level: number) => {
      reactFlow.zoomTo(level, { duration: 300 });
    },
    [reactFlow]
  );

  /**
   * Validate the current flow
   */
  const validateFlow = useCallback(async (): Promise<{
    isValid: boolean;
    errors: string[];
  }> => {
    const { nodes, edges } = store;
    const errors: string[] = [];

    // Check for minimum nodes
    if (nodes.length === 0) {
      errors.push('Flow must have at least one node');
    }

    // Check for input node
    const hasInput = nodes.some((n) => n.data.component_type === 'input');
    if (!hasInput) {
      errors.push('Flow requires an Input node');
    }

    // Check for output node
    const hasOutput = nodes.some((n) => n.data.component_type === 'output');
    if (!hasOutput) {
      errors.push('Flow requires an Output node');
    }

    // Check for agent node
    const hasAgent = nodes.some((n) => n.data.component_type === 'agent');
    if (!hasAgent) {
      errors.push('Flow requires an Agent node');
    }

    // Check for disconnected nodes
    const connectedNodeIds = new Set<string>();
    edges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    if (nodes.length > 1) {
      const disconnectedNodes = nodes.filter((n) => !connectedNodeIds.has(n.id));
      if (disconnectedNodes.length > 0) {
        errors.push(
          `${disconnectedNodes.length} node(s) are not connected to the flow`
        );
      }
    }

    // Validate agent connections
    const agentNodes = nodes.filter((n) => n.data.component_type === 'agent');
    agentNodes.forEach((agent) => {
      const incomingEdges = edges.filter((e) => e.target === agent.id);
      const hasModelConnection = incomingEdges.some(
        (e) => e.targetHandle === 'model'
      );
      const hasInputConnection = incomingEdges.some(
        (e) => e.targetHandle === 'input'
      );

      if (!hasModelConnection) {
        errors.push(`Agent "${agent.data.label}" requires a model connection`);
      }
      if (!hasInputConnection) {
        errors.push(`Agent "${agent.data.label}" requires an input connection`);
      }
    });

    return {
      isValid: errors.length === 0,
      errors,
    };
  }, [store]);

  /**
   * Export flow as JSON
   */
  const exportFlow = useCallback(async () => {
    const { currentFlow } = store;
    if (!currentFlow) {
      toast.error('No flow to export');
      return null;
    }

    try {
      const exportData = await api.exportFlow(currentFlow.flow_id);
      return exportData;
    } catch (error) {
      toast.error('Failed to export flow');
      return null;
    }
  }, [store]);

  /**
   * Auto-layout nodes
   */
  const autoLayout = useCallback(() => {
    const { nodes, edges } = store;
    if (nodes.length === 0) return;

    // Simple left-to-right layout
    const nodeWidth = 250;
    const nodeHeight = 150;
    const horizontalGap = 100;
    const verticalGap = 50;

    // Find entry points (nodes with no incoming edges)
    const targetIds = new Set(edges.map((e) => e.target));
    const entryPoints = nodes.filter((n) => !targetIds.has(n.id));

    // BFS to layout nodes
    const visited = new Set<string>();
    const positions = new Map<string, XYPosition>();
    const queue: { nodeId: string; level: number }[] = [];

    entryPoints.forEach((node) => {
      queue.push({ nodeId: node.id, level: 0 });
    });

    const levelCounts = new Map<number, number>();

    while (queue.length > 0) {
      const { nodeId, level } = queue.shift()!;

      if (visited.has(nodeId)) continue;
      visited.add(nodeId);

      const levelIndex = levelCounts.get(level) || 0;
      levelCounts.set(level, levelIndex + 1);

      positions.set(nodeId, {
        x: level * (nodeWidth + horizontalGap) + 50,
        y: levelIndex * (nodeHeight + verticalGap) + 50,
      });

      // Find connected nodes
      const outgoingEdges = edges.filter((e) => e.source === nodeId);
      outgoingEdges.forEach((edge) => {
        if (!visited.has(edge.target)) {
          queue.push({ nodeId: edge.target, level: level + 1 });
        }
      });
    }

    // Update node positions
    const updatedNodes = nodes.map((node) => ({
      ...node,
      position: positions.get(node.id) || node.position,
    }));

    store.setNodes(updatedNodes);
    setTimeout(() => fitToView(), 100);
    toast.success('Layout applied');
  }, [store, fitToView]);

  /**
   * Get statistics about the current flow
   */
  const flowStats = useMemo(() => {
    const { nodes, edges } = store;

    const nodesByType: Record<string, number> = {};
    nodes.forEach((node) => {
      const type = node.data.component_type;
      nodesByType[type] = (nodesByType[type] || 0) + 1;
    });

    return {
      totalNodes: nodes.length,
      totalEdges: edges.length,
      nodesByType,
      hasInput: nodes.some((n) => n.data.component_type === 'input'),
      hasOutput: nodes.some((n) => n.data.component_type === 'output'),
      hasAgent: nodes.some((n) => n.data.component_type === 'agent'),
    };
  }, [store.nodes, store.edges]);

  /**
   * Check if flow has unsaved changes
   */
  const hasUnsavedChanges = useMemo(() => {
    return store.isDirty;
  }, [store.isDirty]);

  /**
   * Undo last action (placeholder for undo functionality)
   */
  const undo = useCallback(() => {
    // TODO: Implement undo stack
    toast('Undo not yet implemented');
  }, []);

  /**
   * Redo last undone action (placeholder for redo functionality)
   */
  const redo = useCallback(() => {
    // TODO: Implement redo stack
    toast('Redo not yet implemented');
  }, []);

  return {
    // State from store
    ...store,

    // Computed
    flowStats,
    hasUnsavedChanges,

    // Node operations
    addNodeAtPosition,
    addNodeAtCenter,
    duplicateNode,
    deleteSelectedNodes,
    selectAllNodes,
    deselectAllNodes,

    // View operations
    centerOnNode,
    fitToView,
    zoomTo,

    // Flow operations
    validateFlow,
    exportFlow,
    autoLayout,
    undo,
    redo,
  };
};

export default useFlow;

```

# src/index.css

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg-primary: #0f0f1a;
  --bg-secondary: #1a1a2e;
  --bg-tertiary: #252542;
  --text-primary: #ffffff;
  --text-secondary: #a0a0b0;
  --accent-primary: #6366f1;
  --accent-secondary: #8b5cf6;
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
  background: var(--bg-tertiary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #3a3a5c;
}

/* React Flow customizations */
.react-flow__node {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.react-flow__handle {
  width: 12px;
  height: 12px;
  border: 2px solid #fff;
}

.react-flow__handle-left {
  left: -6px;
}

.react-flow__handle-right {
  right: -6px;
}

.react-flow__edge-path {
  stroke-width: 2;
}

.react-flow__background {
  background-color: var(--bg-primary) !important;
}

.react-flow__minimap {
  background-color: var(--bg-secondary) !important;
  border-radius: 8px;
  overflow: hidden;
}

.react-flow__controls {
  background-color: var(--bg-secondary) !important;
  border-radius: 8px;
  overflow: hidden;
}

.react-flow__controls-button {
  background-color: var(--bg-tertiary) !important;
  border-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

.react-flow__controls-button:hover {
  background-color: var(--accent-primary) !important;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.animate-fadeIn {
  animation: fadeIn 0.3s ease-out;
}

.animate-slideUp {
  animation: slideUp 0.4s ease-out;
}

.animate-slideIn {
  animation: slideIn 0.3s ease-out;
}

/* Glass effect */
.glass {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Gradient text */
.gradient-text {
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Node colors by type */
.node-input { border-color: #10b981 !important; }
.node-output { border-color: #f59e0b !important; }
.node-agent { border-color: #3b82f6 !important; }
.node-llm { border-color: #8b5cf6 !important; }
.node-tool { border-color: #ec4899 !important; }

```

# src/index.tsx

```tsx
// src/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

```

# src/pages/Auth.tsx

```tsx
// src/pages/Auth.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LoginForm } from '../components/auth/LoginForm';
import { SignupForm } from '../components/auth/SignupForm';
import { Bot, Sparkles, Zap, Shield } from 'lucide-react';

export const Auth: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, isLoading, navigate]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-dark-300 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500" />
      </div>
    );
  }

  const features = [
    {
      icon: <Sparkles className="w-5 h-5" />,
      title: 'Visual Builder',
      description: 'Drag and drop components to build agents',
    },
    {
      icon: <Zap className="w-5 h-5" />,
      title: 'Multiple LLMs',
      description: 'Connect to OpenAI, Anthropic, and more',
    },
    {
      icon: <Shield className="w-5 h-5" />,
      title: '500+ Integrations',
      description: 'Use Composio tools for endless possibilities',
    },
  ];

  return (
    <div className="min-h-screen bg-dark-300 flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-indigo-900/50 to-purple-900/50 p-12 flex-col justify-between relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-20 left-20 w-72 h-72 bg-indigo-500 rounded-full blur-[100px]" />
          <div className="absolute bottom-20 right-20 w-72 h-72 bg-purple-500 rounded-full blur-[100px]" />
        </div>

        {/* Content */}
        <div className="relative">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-white/10 rounded-xl backdrop-blur-sm">
              <Bot className="w-8 h-8 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">Agent Builder</span>
          </div>
          
          <h1 className="text-4xl font-bold text-white mb-4">
            Build AI Agents
            <br />
            <span className="gradient-text">Visually</span>
          </h1>
          <p className="text-lg text-gray-300 max-w-md">
            Create powerful AI workflows by connecting components. 
            No coding required, but fully customizable.
          </p>
        </div>

        {/* Features */}
        <div className="relative space-y-4">
          {features.map((feature, index) => (
            <div
              key={index}
              className="flex items-center gap-4 p-4 bg-white/5 rounded-xl backdrop-blur-sm"
            >
              <div className="p-2 bg-indigo-500/20 rounded-lg text-indigo-400">
                {feature.icon}
              </div>
              <div>
                <h3 className="font-medium text-white">{feature.title}</h3>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right side - Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
            <div className="p-2 bg-indigo-600 rounded-xl">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">Agent Builder</span>
          </div>

          {/* Form */}
          <div className="bg-dark-200 rounded-2xl p-8 border border-gray-800 animate-fadeIn">
            {isLogin ? (
              <LoginForm onSwitchToSignup={() => setIsLogin(false)} />
            ) : (
              <SignupForm onSwitchToLogin={() => setIsLogin(true)} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

```

# src/pages/Builder.tsx

```tsx
// src/pages/Builder.tsx
import React from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { Canvas } from '../components/builder/Canvas';

export const Builder: React.FC = () => {
  const { flowId } = useParams<{ flowId: string }>();

  if (!flowId) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="h-screen overflow-hidden">
      <Canvas flowId={flowId} />
    </div>
  );
};

```

# src/pages/Dashboard.tsx

```tsx
// src/pages/Dashboard.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { FlowGrid } from '../components/dashboard/FlowGrid';
import { Modal } from '../components/ui/Modal';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { Flow } from '../types';
import api from '../api';
import toast from 'react-hot-toast';
import { Plus, Search, RefreshCw } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [flows, setFlows] = useState<Flow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newFlowName, setNewFlowName] = useState('');
  const [newFlowDescription, setNewFlowDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const loadFlows = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await api.getFlows(1, 50);
      setFlows(data.flows);
    } catch (error) {
      toast.error('Failed to load flows');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFlows();
  }, [loadFlows]);

  const handleCreateFlow = async () => {
    if (!newFlowName.trim()) {
      toast.error('Please enter a flow name');
      return;
    }

    setIsCreating(true);
    try {
      const flow = await api.createFlow({
        name: newFlowName,
        description: newFlowDescription || undefined,
      });
      toast.success('Flow created successfully');
      setShowCreateModal(false);
      setNewFlowName('');
      setNewFlowDescription('');
      navigate(`/builder/${flow.flow_id}`);
    } catch (error) {
      toast.error('Failed to create flow');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteFlow = (flowId: string) => {
    setFlows((prev) => prev.filter((f) => f.flow_id !== flowId));
  };

  const filteredFlows = flows.filter(
    (flow) =>
      flow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      flow.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">My Flows</h1>
            <p className="text-gray-400 mt-1">
              Create and manage your AI agent workflows
            </p>
          </div>
          <Button
            onClick={() => setShowCreateModal(true)}
            icon={<Plus className="w-5 h-5" />}
          >
            New Flow
          </Button>
        </div>

        {/* Search and filters */}
        <div className="flex items-center gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search flows..."
              className="w-full pl-10 pr-4 py-2.5 bg-dark-200 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <Button
            variant="secondary"
            onClick={loadFlows}
            icon={<RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />}
          >
            Refresh
          </Button>
        </div>

        {/* Flow grid */}
        <FlowGrid
          flows={filteredFlows}
          isLoading={isLoading}
          onCreateNew={() => setShowCreateModal(true)}
          onDelete={handleDeleteFlow}
        />
      </div>

      {/* Create Flow Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Flow"
      >
        <div className="space-y-4">
          <Input
            label="Flow Name"
            value={newFlowName}
            onChange={(e) => setNewFlowName(e.target.value)}
            placeholder="My Agent Flow"
          />
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">
              Description (optional)
            </label>
            <textarea
              value={newFlowDescription}
              onChange={(e) => setNewFlowDescription(e.target.value)}
              placeholder="Describe what this flow does..."
              rows={3}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
            />
          </div>
          <div className="flex gap-3 pt-2">
            <Button
              variant="secondary"
              className="flex-1"
              onClick={() => setShowCreateModal(false)}
            >
              Cancel
            </Button>
            <Button
              className="flex-1"
              onClick={handleCreateFlow}
              isLoading={isCreating}
            >
              Create Flow
            </Button>
          </div>
        </div>
      </Modal>
    </Layout>
  );
};

```

# src/types/index.ts

```typescript
// src/types/index.ts
export interface User {
  id: string;
  username: string;
  email: string;
  is_active: boolean;
}

export interface Flow {
  flow_id: string;
  user_id: string;
  name: string;
  description?: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
  agent_schema: AgentSchema;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface FlowNode {
  node_id: string;
  flow_id: string;
  type: string;
  position: { x: number; y: number };
  data: NodeData;
  created_at?: string;
  updated_at?: string;
}

export interface NodeData {
  label: string;
  component_type: string;
  parameters: Record<string, any>;
}

export interface FlowEdge {
  edge_id: string;
  flow_id: string;
  source: string;
  target: string;
  source_handle?: string;
  target_handle?: string;
  animated: boolean;
  created_at?: string;
}

export interface AgentSchema {
  version: string;
  entry_points: string[];
  exit_points: string[];
  components: ComponentSchema[];
  connections: Record<string, ConnectionTarget[]>;
  execution_order: string[];
  validation: {
    is_valid: boolean;
    errors: string[];
  };
}

export interface ComponentSchema {
  node_id: string;
  component_type: string;
  parameters: Record<string, any>;
  error?: string;
}

export interface ConnectionTarget {
  target: string;
  source_port: string;
  target_port: string;
}

export interface ComponentConfig {
  component_type: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  color: string;
  input_ports: Port[];
  output_ports: Port[];
  fields: FieldDefinition[];
  field_groups: FieldGroup[];
}

export interface Port {
  id: string;
  name: string;
  type: 'input' | 'output';
  data_type: string;
  required: boolean;
  multiple: boolean;
  description?: string;
}

export interface FieldDefinition {
  name: string;
  type: string;
  label: string;
  description?: string;
  default?: any;
  validation: FieldValidation;
  ui: FieldUI;
  options?: FieldOption[];
  properties: Record<string, any>;
}

export interface FieldValidation {
  required: boolean;
  min_length?: number;
  max_length?: number;
  min_value?: number;
  max_value?: number;
  pattern?: string;
  pattern_message?: string;
}

export interface FieldUI {
  placeholder?: string;
  help_text?: string;
  width: string;
  order: number;
  group?: string;
  sensitive: boolean;
  copyable?: boolean;
}

export interface FieldOption {
  value: any;
  label: string;
  description?: string;
  group?: string;
}

export interface FieldGroup {
  id: string;
  label: string;
  description?: string;
  collapsible: boolean;
  collapsed_by_default: boolean;
  order: number;
}

export interface AuthTokens {
  access_token: string;
  token_type: string;
}

export interface ApiError {
  detail: string;
  errors?: Array<{
    field: string;
    message: string;
    type: string;
  }>;
}

```

# src/utils/defaultComponentConfig.ts

```typescript
import { ComponentConfig, Port } from '../types';

const defaultPorts: Record<string, { input_ports?: Port[]; output_ports?: Port[] }> = {
  agent: {
    input_ports: [
      { id: 'input', name: 'Input', type: 'input', data_type: 'any', required: false, multiple: false },
      { id: 'model', name: 'Model', type: 'input', data_type: 'llm', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
      { id: 'memory', name: 'Memory', type: 'input', data_type: 'memory', required: false, multiple: false },
    ],
    output_ports: [
      { id: 'output', name: 'Output', type: 'output', data_type: 'any', required: false, multiple: false },
      { id: 'tool_results', name: 'Tool Results', type: 'output', data_type: 'tool_result', required: false, multiple: true },
      { id: 'history', name: 'History', type: 'output', data_type: 'memory', required: false, multiple: false },
    ],
  },
  agent_default: {
    input_ports: [
      { id: 'input', name: 'Input', type: 'input', data_type: 'any', required: false, multiple: false },
      { id: 'model', name: 'Model', type: 'input', data_type: 'llm', required: false, multiple: false },
    ],
    output_ports: [
      { id: 'output', name: 'Output', type: 'output', data_type: 'any', required: false, multiple: false },
    ],
  },
  llm_openai: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  openai_model: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  llm_anthropic: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  anthropic_model: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  llm_openrouter: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  openrouter_model: {
    input_ports: [
      { id: 'messages', name: 'Messages', type: 'input', data_type: 'messages', required: true, multiple: false },
      { id: 'system_prompt', name: 'System Prompt', type: 'input', data_type: 'prompt', required: false, multiple: false },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tool', required: false, multiple: true },
    ],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false },
      { id: 'response', name: 'Response', type: 'output', data_type: 'text', required: true, multiple: false },
      { id: 'tool_calls', name: 'Tool Calls', type: 'output', data_type: 'tool_call', required: false, multiple: true },
    ],
  },
  input: {
    output_ports: [
      { id: 'output', name: 'Output', type: 'output', data_type: 'any', required: true, multiple: false },
      { id: 'metadata', name: 'Metadata', type: 'output', data_type: 'metadata', required: false, multiple: false },
    ],
  },
  output: {
    input_ports: [
      { id: 'input', name: 'Input', type: 'input', data_type: 'any', required: true, multiple: false },
      { id: 'metadata', name: 'Metadata', type: 'input', data_type: 'metadata', required: false, multiple: false },
    ],
  },
  composio_tool: {
    input_ports: [
      { id: 'tool_input', name: 'Tool Input', type: 'input', data_type: 'any', required: false, multiple: false },
    ],
    output_ports: [
      { id: 'tool_output', name: 'Tool Output', type: 'output', data_type: 'any', required: false, multiple: false },
    ],
  },
};

const basePorts: { input_ports: Port[]; output_ports: Port[] } = {
  input_ports: [{ id: 'input', name: 'Input', type: 'input', data_type: 'any', required: false, multiple: false }],
  output_ports: [{ id: 'output', name: 'Output', type: 'output', data_type: 'any', required: false, multiple: false }],
};

const normalizeType = (componentType: string): string => {
  const type = componentType.toLowerCase();
  if (type.includes('openrouter')) return 'openrouter_model';
  if (type.includes('openai')) return 'openai_model';
  if (type.includes('anthropic')) return 'anthropic_model';
  if (type === 'agent') return 'agent';
  if (type.startsWith('llm_')) return type;
  return componentType;
};

export const getDefaultPortsForType = (componentType: string): { input_ports: Port[]; output_ports: Port[] } => {
  const normalized = normalizeType(componentType);
  const ports = defaultPorts[normalized] || defaultPorts[componentType] || defaultPorts.agent_default;
  return {
    input_ports: ports?.input_ports?.length ? ports.input_ports : basePorts.input_ports,
    output_ports: ports?.output_ports?.length ? ports.output_ports : basePorts.output_ports,
  };
};

export const getFallbackComponentConfig = (componentType: string, name?: string): ComponentConfig => {
  const ports = getDefaultPortsForType(componentType);
  return {
    component_type: componentType,
    name: name || componentType,
    description: '',
    category: 'default',
    icon: '',
    color: '#6366f1',
    fields: [],
    field_groups: [],
    ...ports,
  };
};

```

# src/utils/helpers.ts

```typescript
// src/utils/helpers.ts

/**
 * Generate a unique ID with optional prefix
 */
export const generateId = (prefix: string = 'id'): string => {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Debounce function for performance optimization
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * Throttle function for rate limiting
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

/**
 * Deep clone an object
 */
export const deepClone = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj));
};

/**
 * Check if value is a plain object
 */
export const isObject = (item: any): item is Record<string, any> => {
  return item && typeof item === 'object' && !Array.isArray(item);
};

/**
 * Format date to human-readable string
 */
export const formatDate = (
  dateString: string | Date,
  options?: Intl.DateTimeFormatOptions
): string => {
  const date = typeof dateString === 'string' ? new Date(dateString) : dateString;
  const defaultOptions: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    ...options,
  };
  return date.toLocaleDateString('en-US', defaultOptions);
};

/**
 * Format date to relative time (e.g., "2 hours ago")
 */
export const formatRelativeTime = (dateString: string | Date): string => {
  const date = typeof dateString === 'string' ? new Date(dateString) : dateString;
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return 'just now';
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  } else if (diffInSeconds < 604800) {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days} day${days > 1 ? 's' : ''} ago`;
  } else {
    return formatDate(date);
  }
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
};

/**
 * Capitalize first letter of string
 */
export const capitalize = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

/**
 * Convert camelCase to Title Case
 */
export const camelToTitle = (str: string): string => {
  return str
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (char) => char.toUpperCase())
    .trim();
};

/**
 * Convert snake_case to Title Case
 */
export const snakeToTitle = (str: string): string => {
  return str
    .split('_')
    .map((word) => capitalize(word))
    .join(' ');
};

/**
 * Validate email format
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

/**
 * Validate URL format
 */
export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Get color for component type
 */
export const getComponentColor = (componentType: string): string => {
  const colorMap: Record<string, string> = {
    input: '#10b981',
    output: '#f59e0b',
    agent: '#3b82f6',
    llm_openai: '#10a37f',
    llm_anthropic: '#d97757',
    llm_openrouter: '#6366f1',
    composio_tool: '#ec4899',
  };
  return colorMap[componentType] || '#6366f1';
};

/**
 * Copy text to clipboard
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy text:', err);
    return false;
  }
};

/**
 * Parse JSON safely
 */
export const safeJsonParse = <T>(json: string, fallback: T): T => {
  try {
    return JSON.parse(json);
  } catch {
    return fallback;
  }
};

/**
 * Sleep for specified milliseconds
 */
export const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Download JSON as file
 */
export const downloadJson = (data: any, filename: string): void => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Class names utility (like clsx/classnames)
 */
export const cn = (...classes: (string | undefined | null | boolean)[]): string => {
  return classes.filter(Boolean).join(' ');
};

export default {
  generateId,
  debounce,
  throttle,
  deepClone,
  isObject,
  formatDate,
  formatRelativeTime,
  truncateText,
  capitalize,
  camelToTitle,
  snakeToTitle,
  isValidEmail,
  isValidUrl,
  getComponentColor,
  copyToClipboard,
  safeJsonParse,
  sleep,
  downloadJson,
  cn,
};

```

# tailwind.config.js

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        dark: {
          100: '#1e1e2e',
          200: '#181825',
          300: '#11111b',
          400: '#0a0a0f',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}

```

# tsconfig.json

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}

```

# tsconfig.node.json

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true,
    "types": ["node"]
  },
  "include": ["vite.config.ts"]
}

```

# vite.config.ts

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

```

