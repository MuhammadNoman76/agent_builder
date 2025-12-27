// src/pages/Auth.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LoginForm } from '../components/auth/LoginForm';
import { SignupForm } from '../components/auth/SignupForm';
import { Bot, Sparkles, Zap, Shield } from 'lucide-react';

export const Auth: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, isLoading, navigate]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-dark-300 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500" />
      </div>
    );
  }

  const features = [
    {
      icon: <Sparkles className="w-5 h-5" />,
      title: 'Visual Builder',
      description: 'Drag and drop components to build agents',
    },
    {
      icon: <Zap className="w-5 h-5" />,
      title: 'Multiple LLMs',
      description: 'Connect to OpenAI, Anthropic, and more',
    },
    {
      icon: <Shield className="w-5 h-5" />,
      title: '500+ Integrations',
      description: 'Use Composio tools for endless possibilities',
    },
  ];

  return (
    <div className="min-h-screen bg-dark-300 flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-indigo-900/50 to-purple-900/50 p-12 flex-col justify-between relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-20 left-20 w-72 h-72 bg-indigo-500 rounded-full blur-[100px]" />
          <div className="absolute bottom-20 right-20 w-72 h-72 bg-purple-500 rounded-full blur-[100px]" />
        </div>

        {/* Content */}
        <div className="relative">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-white/10 rounded-xl backdrop-blur-sm">
              <Bot className="w-8 h-8 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">Agent Builder</span>
          </div>
          
          <h1 className="text-4xl font-bold text-white mb-4">
            Build AI Agents
            <br />
            <span className="gradient-text">Visually</span>
          </h1>
          <p className="text-lg text-gray-300 max-w-md">
            Create powerful AI workflows by connecting components. 
            No coding required, but fully customizable.
          </p>
        </div>

        {/* Features */}
        <div className="relative space-y-4">
          {features.map((feature, index) => (
            <div
              key={index}
              className="flex items-center gap-4 p-4 bg-white/5 rounded-xl backdrop-blur-sm"
            >
              <div className="p-2 bg-indigo-500/20 rounded-lg text-indigo-400">
                {feature.icon}
              </div>
              <div>
                <h3 className="font-medium text-white">{feature.title}</h3>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right side - Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
            <div className="p-2 bg-indigo-600 rounded-xl">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">Agent Builder</span>
          </div>

          {/* Form */}
          <div className="bg-dark-200 rounded-2xl p-8 border border-gray-800 animate-fadeIn">
            {isLogin ? (
              <LoginForm onSwitchToSignup={() => setIsLogin(false)} />
            ) : (
              <SignupForm onSwitchToLogin={() => setIsLogin(true)} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
