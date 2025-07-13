'use client';
import { useState } from 'react';
import axios from 'axios';

interface QuestionAnswerProps {
  documentId: number;
  documentTitle: string;
}

interface QAPair {
  question: string;
  answer: string;
  timestamp: Date;
}

export default function QuestionAnswer({ documentId, documentTitle }: QuestionAnswerProps) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qaPairs, setQaPairs] = useState<QAPair[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    try {
      const response = await axios.post(
        `http://localhost:8000/api/documents/${documentId}/ask_question/`,
        { question }
      );

      const newQAPair: QAPair = {
        question,
        answer: response.data.answer,
        timestamp: new Date()
      };

      setQaPairs([newQAPair, ...qaPairs]);
      setQuestion('');
    } catch (error) {
      console.error('Error asking question:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-medium mb-4">Ask Questions about "{documentTitle}"</h3>
      
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex space-x-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about this document..."
            className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50"
          >
            {isLoading ? 'Asking...' : 'Ask'}
          </button>
        </div>
      </form>

      <div className="space-y-4 max-h-96 overflow-y-auto">
        {qaPairs.map((pair, index) => (
          <div key={index} className="border-l-4 border-indigo-500 pl-4 py-2">
            <div className="font-medium text-gray-900 mb-2">
              Q: {pair.question}
            </div>
            <div className="text-gray-700 bg-gray-50 p-3 rounded">
              A: {pair.answer}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {pair.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}