// src/components/dashboard/FlowCard.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Flow } from '../../types';
import { Button } from '../ui/Button';
import { Modal } from '../ui/Modal';
import { 
  Play, 
  Edit3, 
  Trash2, 
  MoreVertical,
  Clock,
  CheckCircle,
  XCircle,
  GitBranch
} from 'lucide-react';
import api from '../../api';
import toast from 'react-hot-toast';

interface FlowCardProps {
  flow: Flow;
  onDelete: (flowId: string) => void;
}

export const FlowCard: React.FC<FlowCardProps> = ({ flow, onDelete }) => {
  const navigate = useNavigate();
  const [showMenu, setShowMenu] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const isValid = flow.agent_schema?.validation?.is_valid ?? false;
  const nodeCount = flow.nodes?.length ?? 0;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await api.deleteFlow(flow.flow_id);
      toast.success('Flow deleted successfully');
      onDelete(flow.flow_id);
    } catch (error) {
      toast.error('Failed to delete flow');
    } finally {
      setIsDeleting(false);
      setShowDeleteModal(false);
    }
  };

  return (
    <>
      <div className="group relative bg-dark-200 rounded-xl border border-gray-700 hover:border-indigo-500/50 transition-all duration-300 overflow-hidden">
        {/* Card Header */}
        <div className="p-5">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-white truncate">
                {flow.name}
              </h3>
              {flow.description && (
                <p className="text-sm text-gray-400 mt-1 line-clamp-2">
                  {flow.description}
                </p>
              )}
            </div>
            
            {/* Menu */}
            <div className="relative ml-2">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors"
              >
                <MoreVertical className="w-5 h-5" />
              </button>
              
              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  />
                  <div className="absolute right-0 mt-1 w-36 bg-dark-100 rounded-lg shadow-xl border border-gray-700 py-1 z-20 animate-fadeIn">
                    <button
                      onClick={() => {
                        setShowMenu(false);
                        navigate(`/builder/${flow.flow_id}`);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-white/5"
                    >
                      <Edit3 className="w-4 h-4" />
                      Edit
                    </button>
                    <button
                      onClick={() => {
                        setShowMenu(false);
                        setShowDeleteModal(true);
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-white/5"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <div className="flex items-center gap-1.5">
              <GitBranch className="w-4 h-4" />
              <span>{nodeCount} nodes</span>
            </div>
            <div className="flex items-center gap-1.5">
              {isValid ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-amber-500" />
              )}
              <span>{isValid ? 'Valid' : 'Invalid'}</span>
            </div>
          </div>
        </div>

        {/* Card Footer */}
        <div className="px-5 py-3 bg-dark-300/50 border-t border-gray-700/50 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <Clock className="w-3.5 h-3.5" />
            <span>Updated {formatDate(flow.updated_at)}</span>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(`/builder/${flow.flow_id}`)}
              icon={<Edit3 className="w-4 h-4" />}
            >
              Edit
            </Button>
            {isValid && (
              <Button
                variant="primary"
                size="sm"
                icon={<Play className="w-4 h-4" />}
              >
                Run
              </Button>
            )}
          </div>
        </div>

        {/* Hover gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/0 via-indigo-500/5 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Flow"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-gray-300">
            Are you sure you want to delete <strong>"{flow.name}"</strong>? 
            This action cannot be undone.
          </p>
          <div className="flex gap-3 justify-end">
            <Button
              variant="secondary"
              onClick={() => setShowDeleteModal(false)}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleDelete}
              isLoading={isDeleting}
            >
              Delete
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
};
