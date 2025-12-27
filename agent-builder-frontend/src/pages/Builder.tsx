// src/pages/Builder.tsx
import React from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { Canvas } from '../components/builder/Canvas';

export const Builder: React.FC = () => {
  const { flowId } = useParams<{ flowId: string }>();

  if (!flowId) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="h-screen overflow-hidden">
      <Canvas flowId={flowId} />
    </div>
  );
};
