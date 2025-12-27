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
