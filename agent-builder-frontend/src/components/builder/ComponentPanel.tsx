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
