from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Document
from .serializers import DocumentSerializer
from .services import DocumentProcessor
from functools import lru_cache
import threading
import logging

logger = logging.getLogger(__name__)

class DocumentViewSet(viewsets.ModelViewSet):
    """
    API endpoints for documents
    
    This provides these URLs automatically:
    - GET /api/documents/ - List all user's documents
    - POST /api/documents/ - Upload a new document
    - GET /api/documents/123/ - Get specific document
    - PUT /api/documents/123/ - Update document
    - DELETE /api/documents/123/ - Delete document
    - POST /api/documents/123/ask_question/ - Ask question about document
    """
    
    serializer_class = DocumentSerializer
    
    @lru_cache(maxsize=1)
    def get_processor(self):
        """Get cached processor instance"""
        return DocumentProcessor()
    
    def get_queryset(self):
        """
        Only show documents that belong to the current user
        This is important for security - users shouldn't see each other's documents
        """
        return Document.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """
        This is called when a user uploads a new document
        
        Steps:
        1. Save the document to database
        2. Start processing it in the background
        """
        document = serializer.save(user=self.request.user)
        logger.info(f"Document created: {document.title} (ID: {document.id})")
        
        # Process document in background thread
        def process_document():
            try:
                # Use cached processor
                processor = self.get_processor()
                processor.process_document(document)
                logger.info(f"Document processing completed: {document.title}")
            except Exception as e:
                logger.error(f"Error processing document {document.id}: {e}")
        
        thread = threading.Thread(target=process_document)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Background processing started for document: {document.title}")
    
    @action(detail=True, methods=['post'])
    def ask_question(self, request, pk=None):
        """
        Custom endpoint to ask questions about a document
        
        URL: POST /api/documents/123/ask_question/
        Body: {"question": "What is this document about?"}
        
        Returns: {"answer": "This document is about...", "question": "..."}
        """
        document = self.get_object()
        question = request.data.get('question')
        
        logger.info(f"Question asked for document {document.id}: {question}")
        
        if not question:
            return Response({
                'error': 'Question is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not document.processed:
            logger.warning(f"Question asked for unprocessed document {document.id}")
            return Response({
                'error': 'Document is still being processed. Please wait a moment and try again.',
                'document_processed': False
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Use cached processor
            processor = self.get_processor()
            answer = processor.answer_question(question, document.id)
            
            logger.info(f"Question answered successfully for document {document.id}")
            
            return Response({
                'answer': answer,
                'question': question,
                'document_title': document.title,
                'document_id': document.id
            })
            
        except Exception as e:
            logger.error(f"Error answering question for document {document.id}: {e}")
            return Response({
                'error': 'An error occurred while processing your question. Please try again.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """
        Check if a document is finished processing
        
        URL: GET /api/documents/123/status/
        Returns: {"processed": true, "chunk_count": 5}
        """
        document = self.get_object()
        
        return Response({
            'document_id': document.id,
            'title': document.title,
            'processed': document.processed,
            'uploaded_at': document.uploaded_at,
            'chunk_count': document.chunks.count()
        })

# Simple test view to check if API service is working
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_api(request):
    """
    Test endpoint to check if our API service is working
    
    URL: POST /api/test/
    Body: {"question": "What is AI?", "context": "AI is artificial intelligence"}
    """
    try:
        question = request.data.get('question', 'What is this about?')
        context = request.data.get('context', 'This is a test document about technology.')
        
        # Test our API service
        from .huggingface_api_service import HuggingFaceAPIService
        service = HuggingFaceAPIService()
        
        # Test both functions
        embeddings = service.get_embeddings([context])
        answer = service.answer_question(question, context)
        
        return Response({
            'question': question,
            'context': context,
            'answer': answer,
            'embeddings_count': len(embeddings[0]) if embeddings else 0,
            'status': 'success'
        })
        
    except Exception as e:
        return Response({
            'error': str(e),
            'status': 'error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)