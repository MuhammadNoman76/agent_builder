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
  CheckCircle,
  MessagesSquare as Chat
} from 'lucide-react';
import toast from 'react-hot-toast';

interface ToolbarProps {
  flowName: string;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
  onRun: () => void;
  onOpenPlayground: () => void;
  isRunDisabled?: boolean;
  isRunning?: boolean;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  flowName,
  onZoomIn,
  onZoomOut,
  onFitView,
  onRun,
  onOpenPlayground,
  isRunDisabled = false,
  isRunning = false,
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
    onRun();
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
          variant="secondary"
          size="sm"
          onClick={onOpenPlayground}
          icon={<Chat className="w-4 h-4" />}
        >
          Playground
        </Button>

        <Button
          variant="primary"
          size="sm"
          onClick={handleRun}
          disabled={!isValid || isRunDisabled || isRunning}
          isLoading={isRunning}
          icon={<Play className="w-4 h-4" />}
        >
          Run
        </Button>
      </div>
    </div>
  );
};
