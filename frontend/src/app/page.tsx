'use client';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function Home() {
  const { token, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading) {
      if (token) {
        router.push('/dashboard');
      } else {
        router.push('/login');
      }
    }
  }, [token, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Smart Document Q&A
          </h1>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Smart Document Q&A
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Ask questions about your documents using AI
        </p>
        <div className="space-x-4">
          <button
            onClick={() => router.push('/login')}
            className="bg-indigo-600 text-white px-6 py-3 rounded-md hover:bg-indigo-700"
          >
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}