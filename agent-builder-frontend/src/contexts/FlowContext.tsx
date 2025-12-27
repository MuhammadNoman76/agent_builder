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


const relaxLlmInputs = (config: ComponentConfig): ComponentConfig => {
  const type = config.component_type.toLowerCase();
  const isLlm =
    type.includes('llm') || type.includes('openai') || type.includes('openrouter') || type.includes('anthropic');

  if (!isLlm) return config;

  return {
    ...config,
    input_ports: (config.input_ports || []).map((p) =>
      ['messages', 'system_prompt', 'tools'].includes(p.id) ? { ...p, required: false } : p
    ),
  };
};

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
    const { nodes } = get();
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);

    const sourceType = sourceNode?.data?.component_type || '';
    const targetType = targetNode?.data?.component_type || '';
    const isLlmSource =
      sourceType.includes('llm') || sourceType.includes('openai') || sourceType.includes('openrouter') || sourceType.includes('anthropic');

    if (isLlmSource) {
      const targetHandle = connection.targetHandle || 'model';
      if (targetType !== 'agent' || targetHandle !== 'model') {
        return;
      }
    }

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
    const resolvedConfig = relaxLlmInputs(normalizeConfig(componentType, componentConfig));

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
      const categories = [...data.categories];
      Object.values(data.components_by_category).forEach((components) => {
        components.forEach((c) => allComponents.push(c.config));
      });

      const componentsByCategory: Record<string, ComponentConfig[]> = {};
      Object.entries(data.components_by_category).forEach(([category, components]) => {
        componentsByCategory[category] = components.map((c) => c.config);
      });

      // Inject fallback Composio tool if backend does not provide it
      const hasComposio = allComponents.some((c) => c.component_type === 'composio_tool');
      if (!hasComposio) {
        const fallback = getFallbackComponentConfig('composio_tool', 'Composio Tool');
        fallback.description = fallback.description || 'Expose Composio tools to the agent';
        fallback.category = 'tools';
        allComponents.push(fallback);
        componentsByCategory['tools'] = [...(componentsByCategory['tools'] || []), fallback];
        if (!categories.includes('tools')) {
          categories.push('tools');
        }
      }

      set({
        components: allComponents,
        componentsByCategory,
        categories,
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
        const resolvedConfig = relaxLlmInputs(
          normalizeConfig(
            node.data.component_type,
            componentConfig,
            node.data.label
          )
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
