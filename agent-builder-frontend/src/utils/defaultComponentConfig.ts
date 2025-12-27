// src/utils/defaultComponentConfig.ts
import { ComponentConfig, Port } from '../types';

/**
 * Default port configurations for each component type.
 * 
 * Connection pattern:
 * - Input (output) → Agent (input)
 * - LLM Model (model output) → Agent (model input)
 * - Agent (output) → Output (input)
 */
const defaultPorts: Record<string, { input_ports?: Port[]; output_ports?: Port[] }> = {
  // Agent has multiple input ports and one output port
  agent: {
    input_ports: [
      { id: 'input', name: 'Input', type: 'input', data_type: 'message', required: true, multiple: false, description: 'User input message' },
      { id: 'model', name: 'Model', type: 'input', data_type: 'llm_model', required: true, multiple: false, description: 'LLM model to use' },
      { id: 'tools', name: 'Tools', type: 'input', data_type: 'tools', required: false, multiple: true, description: 'Tools for the agent' },
      { id: 'memory', name: 'Memory', type: 'input', data_type: 'memory', required: false, multiple: false, description: 'Conversation memory' },
    ],
    output_ports: [
      { id: 'output', name: 'Output', type: 'output', data_type: 'message', required: true, multiple: false, description: 'Agent response' },
    ],
  },
  
  // Input component - entry point, only has output
  input: {
    input_ports: [],
    output_ports: [
      { id: 'output', name: 'Output', type: 'output', data_type: 'message', required: true, multiple: false, description: 'User input value' },
    ],
  },
  
  // Output component - exit point, only has input
  output: {
    input_ports: [
      { id: 'input', name: 'Input', type: 'input', data_type: 'message', required: true, multiple: false, description: 'Response to output' },
    ],
    output_ports: [],
  },
  
  // LLM Models - NO input ports, single "model" output port
  llm_openai: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'OpenAI model instance' },
    ],
  },
  openai_model: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'OpenAI model instance' },
    ],
  },
  
  llm_anthropic: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'Anthropic model instance' },
    ],
  },
  anthropic_model: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'Anthropic model instance' },
    ],
  },
  
  llm_openrouter: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'OpenRouter model instance' },
    ],
  },
  openrouter_model: {
    input_ports: [],
    output_ports: [
      { id: 'model', name: 'Model', type: 'output', data_type: 'llm_model', required: true, multiple: false, description: 'OpenRouter model instance' },
    ],
  },
  
  // Composio tool component
  composio_tool: {
    input_ports: [
      { id: 'tool_call', name: 'Tool Call', type: 'input', data_type: 'tool_call', required: true, multiple: false, description: 'Incoming tool call from agent' },
      { id: 'context', name: 'Context', type: 'input', data_type: 'messages', required: false, multiple: false, description: 'Optional conversation context' },
    ],
    output_ports: [
      { id: 'tool_result', name: 'Tool Result', type: 'output', data_type: 'tool_result', required: true, multiple: false, description: 'Tool execution result' },
      { id: 'logs', name: 'Logs', type: 'output', data_type: 'text', required: false, multiple: false, description: 'Optional execution logs' },
    ],
  },
};

// Base fallback ports
const basePorts: { input_ports: Port[]; output_ports: Port[] } = {
  input_ports: [{ id: 'input', name: 'Input', type: 'input', data_type: 'any', required: false, multiple: false }],
  output_ports: [{ id: 'output', name: 'Output', type: 'output', data_type: 'any', required: false, multiple: false }],
};

/**
 * Normalize component type to match port configuration keys
 */
const normalizeType = (componentType: string): string => {
  const type = componentType.toLowerCase();
  
  // Map various LLM type names to their port configs
  if (type.includes('openrouter') || type === 'llm_openrouter') return 'llm_openrouter';
  if (type.includes('openai') || type === 'llm_openai') return 'llm_openai';
  if (type.includes('anthropic') || type === 'llm_anthropic') return 'llm_anthropic';
  if (type === 'agent') return 'agent';
  if (type === 'input') return 'input';
  if (type === 'output') return 'output';
  if (type === 'composio_tool') return 'composio_tool';
  
  return componentType;
};

/**
 * Get default ports for a component type
 */
export const getDefaultPortsForType = (componentType: string): { input_ports: Port[]; output_ports: Port[] } => {
  const normalized = normalizeType(componentType);
  const ports = defaultPorts[normalized] || defaultPorts[componentType];
  
  if (ports) {
    return {
      input_ports: ports.input_ports || [],
      output_ports: ports.output_ports || [],
    };
  }
  
  // Return base ports as fallback
  return basePorts;
};

/**
 * Get a fallback component config when API doesn't return one
 */
export const getFallbackComponentConfig = (componentType: string, name?: string): ComponentConfig => {
  const ports = getDefaultPortsForType(componentType);
  
  // Determine category based on component type
  let category = 'general';
  if (componentType.startsWith('llm_') || componentType.includes('model')) {
    category = 'models';
  } else if (componentType === 'agent') {
    category = 'agents';
  } else if (componentType === 'input' || componentType === 'output') {
    category = 'io';
  } else if (componentType === 'composio_tool') {
    category = 'tools';
  }
  
  const baseConfig: ComponentConfig = {
    component_type: componentType,
    name: name || componentType,
    description: '',
    category,
    icon: '',
    color: '#6366f1',
    fields: [],
    field_groups: [],
    input_ports: ports.input_ports,
    output_ports: ports.output_ports,
  };

  // Inject sensible defaults for Composio if backend doesn't provide them
  if (componentType === 'composio_tool') {
    baseConfig.description =
      baseConfig.description || 'Connect Composio toolkits and expose selected tools to the agent.';
    baseConfig.fields = [
      {
        name: 'api_key',
        type: 'password',
        label: 'Composio API Key',
        description: 'API key used to fetch toolkits and execute actions.',
        default: '',
        validation: { required: true, min_length: 10 },
        ui: { placeholder: 'sk-...', width: 'full', order: 1, group: 'connection', sensitive: true, copyable: true },
        properties: {},
      },
      {
        name: 'toolkit',
        type: 'string',
        label: 'Toolkit',
        description: 'Toolkit slug to connect (e.g., gmail, slack, notion).',
        default: '',
        validation: { required: true },
        ui: { placeholder: 'gmail', width: 'full', order: 2, group: 'connection', sensitive: false, copyable: false },
        properties: {},
      },
      {
        name: 'actions',
        type: 'json',
        label: 'Actions (optional)',
        description: 'List of actions to expose from the selected toolkit. Leave empty to allow all.',
        default: [],
        validation: { required: false },
        ui: {
          placeholder: '["send_email", "list_messages"]',
          width: 'full',
          order: 3,
          group: 'connection',
          sensitive: false,
          copyable: false,
        },
        properties: {},
      },
      {
        name: 'connection_status',
        type: 'string',
        label: 'Connection Status',
        description: 'Read-only indicator of connection state.',
        default: 'disconnected',
        validation: { required: false },
        ui: {
          placeholder: 'disconnected/connected',
          width: 'full',
          order: 4,
          group: 'connection',
          sensitive: false,
          copyable: false,
          },
        properties: {},
      },
      {
        name: 'allow_tool_selection',
        type: 'boolean',
        label: 'Allow Tool Selection',
        description: 'Toggle to enable selecting specific tools from the toolkit.',
        default: true,
        validation: { required: false },
        ui: { width: 'full', order: 5, group: 'connection', sensitive: false, copyable: false },
        properties: {},
      },
    ];
    baseConfig.field_groups = [
      {
        id: 'connection',
        label: 'Connection',
        description: 'Connect to Composio and select toolkit/tools.',
        collapsible: false,
        collapsed_by_default: false,
        order: 0,
      },
    ];
  }
  
  return baseConfig;
};

export default {
  getDefaultPortsForType,
  getFallbackComponentConfig,
};
