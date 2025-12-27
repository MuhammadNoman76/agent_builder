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
