// src/components/builder/Canvas.tsx
import React, { useCallback, useRef, useEffect, useState } from 'react';
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
import { ChatModal, ChatMessage } from './ChatModal';
import api from '../../api';
import toast from 'react-hot-toast';

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

  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

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

  const handleOpenChat = useCallback(() => {
    setIsChatOpen(true);
  }, []);

  const handleCloseChat = useCallback(() => {
    setIsChatOpen(false);
  }, []);

  const handleResetChat = useCallback(() => {
    setChatMessages([]);
    setChatInput('');
  }, []);

  const handleSendChat = useCallback(async () => {
    const trimmed = chatInput.trim();
    if (!trimmed || !currentFlow) return;

    const nextMessages = [...chatMessages, { role: 'user', content: trimmed } as ChatMessage];
    setChatMessages(nextMessages);
    setChatInput('');
    setIsSending(true);

    try {
      const payload = {
        messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
      };
      const resp = await api.executeFlow(currentFlow.flow_id, payload);
      const output = (resp as any)?.output ?? (resp as any)?.response ?? resp;
      const content =
        typeof output === 'string'
          ? output
          : JSON.stringify(output, null, 2);
      setChatMessages([...nextMessages, { role: 'assistant', content }]);
    } catch (error: any) {
      const msg = error?.response?.data?.detail || 'Failed to execute flow';
      toast.error(msg);
      setChatMessages([...nextMessages, { role: 'error', content: msg }]);
    } finally {
      setIsSending(false);
    }
  }, [chatInput, chatMessages, currentFlow]);

  const isValid = currentFlow?.agent_schema?.validation?.is_valid ?? false;

  return (
    <div className="flex flex-col h-screen">
      <Toolbar
        flowName={currentFlow?.name || 'Untitled Flow'}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onFitView={fitView}
        onRun={handleOpenChat}
        onOpenPlayground={handleOpenChat}
        isRunDisabled={!isValid}
        isRunning={isSending}
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

      <ChatModal
        isOpen={isChatOpen}
        onClose={handleCloseChat}
        messages={chatMessages}
        input={chatInput}
        onChangeInput={setChatInput}
        onSend={handleSendChat}
        onReset={handleResetChat}
        isSending={isSending}
      />
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
