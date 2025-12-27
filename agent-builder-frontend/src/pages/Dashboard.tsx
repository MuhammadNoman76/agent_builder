// src/pages/Dashboard.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { FlowGrid } from '../components/dashboard/FlowGrid';
import { Modal } from '../components/ui/Modal';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { Flow } from '../types';
import api from '../api';
import toast from 'react-hot-toast';
import { Plus, Search, RefreshCw } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [flows, setFlows] = useState<Flow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newFlowName, setNewFlowName] = useState('');
  const [newFlowDescription, setNewFlowDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const loadFlows = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await api.getFlows(1, 50);
      setFlows(data.flows);
    } catch (error) {
      toast.error('Failed to load flows');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFlows();
  }, [loadFlows]);

  const handleCreateFlow = async () => {
    if (!newFlowName.trim()) {
      toast.error('Please enter a flow name');
      return;
    }

    setIsCreating(true);
    try {
      const flow = await api.createFlow({
        name: newFlowName,
        description: newFlowDescription || undefined,
      });
      toast.success('Flow created successfully');
      setShowCreateModal(false);
      setNewFlowName('');
      setNewFlowDescription('');
      navigate(`/builder/${flow.flow_id}`);
    } catch (error) {
      toast.error('Failed to create flow');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteFlow = (flowId: string) => {
    setFlows((prev) => prev.filter((f) => f.flow_id !== flowId));
  };

  const filteredFlows = flows.filter(
    (flow) =>
      flow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      flow.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">My Flows</h1>
            <p className="text-gray-400 mt-1">
              Create and manage your AI agent workflows
            </p>
          </div>
          <Button
            onClick={() => setShowCreateModal(true)}
            icon={<Plus className="w-5 h-5" />}
          >
            New Flow
          </Button>
        </div>

        {/* Search and filters */}
        <div className="flex items-center gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search flows..."
              className="w-full pl-10 pr-4 py-2.5 bg-dark-200 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <Button
            variant="secondary"
            onClick={loadFlows}
            icon={<RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />}
          >
            Refresh
          </Button>
        </div>

        {/* Flow grid */}
        <FlowGrid
          flows={filteredFlows}
          isLoading={isLoading}
          onCreateNew={() => setShowCreateModal(true)}
          onDelete={handleDeleteFlow}
        />
      </div>

      {/* Create Flow Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Flow"
      >
        <div className="space-y-4">
          <Input
            label="Flow Name"
            value={newFlowName}
            onChange={(e) => setNewFlowName(e.target.value)}
            placeholder="My Agent Flow"
          />
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">
              Description (optional)
            </label>
            <textarea
              value={newFlowDescription}
              onChange={(e) => setNewFlowDescription(e.target.value)}
              placeholder="Describe what this flow does..."
              rows={3}
              className="w-full px-4 py-2.5 bg-dark-300 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
            />
          </div>
          <div className="flex gap-3 pt-2">
            <Button
              variant="secondary"
              className="flex-1"
              onClick={() => setShowCreateModal(false)}
            >
              Cancel
            </Button>
            <Button
              className="flex-1"
              onClick={handleCreateFlow}
              isLoading={isCreating}
            >
              Create Flow
            </Button>
          </div>
        </div>
      </Modal>
    </Layout>
  );
};
