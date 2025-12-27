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
