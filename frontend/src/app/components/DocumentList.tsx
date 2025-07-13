'use client';
import { useState, useEffect } from 'react';
import axios from 'axios';
import QuestionAnswer from './QuestionAnswer';

interface Document {
  id: number;
  title: string;
  file: string;
  uploaded_at: string;
  processed: boolean;
}

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/documents/');
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDelete = async (documentId: number) => {
    if (window.confirm('Are you sure you want to delete this document?')) {
      try {
        await axios.delete(`http://localhost:8000/api/documents/${documentId}/`);
        setDocuments(documents.filter(doc => doc.id !== documentId));
        if (selectedDocument?.id === documentId) {
          setSelectedDocument(null);
        }
      } catch (error) {
        console.error('Error deleting document:', error);
      }
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading documents...</div>;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-medium mb-4">Your Documents</h3>
        
        {documents.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No documents uploaded yet. Upload your first document to get started!
          </p>
        ) : (
          <div className="space-y-3">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                  selectedDocument?.id === doc.id
                    ? 'border-indigo-500 bg-indigo-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedDocument(doc)}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-gray-900">{doc.title}</h4>
                    <p className="text-sm text-gray-500">
                      Uploaded: {new Date(doc.uploaded_at).toLocaleDateString()}
                    </p>
                    <span
                      className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                        doc.processed
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}
                    >
                      {doc.processed ? 'Processed' : 'Processing...'}
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(doc.id);
                    }}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        {selectedDocument ? (
          selectedDocument.processed ? (
            <QuestionAnswer
              documentId={selectedDocument.id}
              documentTitle={selectedDocument.title}
            />
          ) : (
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Document Processing</h3>
              <p className="text-gray-600">
                "{selectedDocument.title}" is still being processed. Please wait a moment before asking questions.
              </p>
            </div>
          )
        ) : (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-medium mb-4">Select a Document</h3>
            <p className="text-gray-600">
              Choose a document from the list to start asking questions about it.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}