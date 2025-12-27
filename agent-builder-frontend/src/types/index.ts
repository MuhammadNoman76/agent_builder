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
