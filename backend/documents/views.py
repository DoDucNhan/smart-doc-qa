from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Document
from .serializers import DocumentSerializer
from .services import DocumentProcessor
from django.shortcuts import get_object_or_404
import threading

class DocumentViewSet(viewsets.ModelViewSet):
    serializer_class = DocumentSerializer
    
    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        document = serializer.save(user=self.request.user)
        
        # Process document in background thread
        def process_document():
            try:
                processor = DocumentProcessor()
                processor.process_document(document)
            except Exception as e:
                print(f"Error processing document: {e}")
                # In production, log this error and possibly update document status
        
        thread = threading.Thread(target=process_document)
        thread.start()
    
    @action(detail=True, methods=['post'])
    def ask_question(self, request, pk=None):
        document = self.get_object()
        question = request.data.get('question')
        
        if not question:
            return Response({
                'error': 'Question is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not document.processed:
            return Response({
                'error': 'Document is still being processed'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            processor = DocumentProcessor()
            answer = processor.answer_question(question, document.id)
            
            return Response({
                'answer': answer,
                'question': question,
                'document_title': document.title
            })
        except Exception as e:
            return Response({
                'error': 'An error occurred while processing your question'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)