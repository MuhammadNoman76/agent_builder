// src/components/builder/CustomNode.tsx
import React, { memo, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Bot, Brain, Plug, ArrowRightCircle, Settings, AlertCircle, CheckCircle, LogIn } from 'lucide-react';
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
  const type = componentType.toLowerCase();
  
  if (type === 'input') return <LogIn className="w-5 h-5" />;
  if (type === 'output') return <ArrowRightCircle className="w-5 h-5" />;
  if (type === 'agent') return <Bot className="w-5 h-5" />;
  if (type.includes('llm') || type.includes('openai') || type.includes('anthropic') || type.includes('openrouter')) {
    return <Brain className="w-5 h-5" />;
  }
  if (type === 'composio_tool') return <Plug className="w-5 h-5" />;
  
  return <Settings className="w-5 h-5" />;
};

const getNodeColors = (componentType: string) => {
  const type = componentType.toLowerCase();
  
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
    composio_tool: {
      bg: 'bg-pink-500/10',
      border: 'border-pink-500',
      text: 'text-pink-400',
      glow: 'shadow-pink-500/20',
    },
  };
  
  // Check for LLM models
  if (type.includes('openai') || type === 'llm_openai') {
    return {
      bg: 'bg-green-500/10',
      border: 'border-green-500',
      text: 'text-green-400',
      glow: 'shadow-green-500/20',
    };
  }
  if (type.includes('anthropic') || type === 'llm_anthropic') {
    return {
      bg: 'bg-orange-500/10',
      border: 'border-orange-500',
      text: 'text-orange-400',
      glow: 'shadow-orange-500/20',
    };
  }
  if (type.includes('openrouter') || type === 'llm_openrouter') {
    return {
      bg: 'bg-indigo-500/10',
      border: 'border-indigo-500',
      text: 'text-indigo-400',
      glow: 'shadow-indigo-500/20',
    };
  }
  
  return colorMap[type] || {
    bg: 'bg-gray-500/10',
    border: 'border-gray-500',
    text: 'text-gray-400',
    glow: 'shadow-gray-500/20',
  };
};

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data, selected, isConnectable }) => {
  const colors = useMemo(() => getNodeColors(data.component_type), [data.component_type]);
  const config = data.config;
  const isValid = data.isValid !== false;
  const errors = data.errors || [];

  // Get ports from config or use defaults
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
