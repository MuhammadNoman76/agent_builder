// src/components/builder/ChatModal.tsx
import React, { useRef, useEffect } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Send, X, RefreshCw } from 'lucide-react';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'error';
  content: string;
}

interface ChatModalProps {
  isOpen: boolean;
  onClose: () => void;
  messages: ChatMessage[];
  input: string;
  onChangeInput: (val: string) => void;
  onSend: () => void;
  onReset: () => void;
  isSending: boolean;
}

export const ChatModal: React.FC<ChatModalProps> = ({
  isOpen,
  onClose,
  messages,
  input,
  onChangeInput,
  onSend,
  onReset,
  isSending,
}) => {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Playground" size="xl">
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Chat with flow</h3>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onReset} icon={<RefreshCw className="w-4 h-4" />}>
              Reset
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose} icon={<X className="w-4 h-4" />}>
              Close
            </Button>
          </div>
        </div>

        <div className="h-64 overflow-y-auto rounded-lg border border-gray-700 bg-dark-200 p-3 space-y-3">
          {messages.length === 0 && (
            <p className="text-sm text-gray-500">Start the conversation by typing below.</p>
          )}
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'assistant' ? 'justify-start' : 'justify-end'}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap border
                  ${msg.role === 'assistant' ? 'bg-dark-300 border-gray-700 text-white' : ''}
                  ${msg.role === 'user' ? 'bg-indigo-600/20 border-indigo-500/50 text-white' : ''}
                  ${msg.role === 'error' ? 'bg-red-600/10 border-red-500/40 text-red-200' : ''}
                `}
              >
                <div className="text-xs uppercase tracking-wide text-gray-400 mb-1">{msg.role}</div>
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="flex items-center gap-2">
          <textarea
            value={input}
            onChange={(e) => onChangeInput(e.target.value)}
            rows={2}
            placeholder="Type your message..."
            className="flex-1 px-3 py-2 bg-dark-200 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
          />
          <Button
            variant="primary"
            size="sm"
            onClick={onSend}
            disabled={!input.trim()}
            isLoading={isSending}
            icon={<Send className="w-4 h-4" />}
          >
            Send
          </Button>
        </div>
      </div>
    </Modal>
  );
};
