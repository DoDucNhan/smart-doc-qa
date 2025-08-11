import os
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from rest_framework.authtoken.models import Token
from documents.models import Document, DocumentChunk
from documents.serializers import DocumentSerializer
from documents.services import DocumentProcessor, ProcessorConfig
from documents.huggingface_api_service import HuggingFaceAPIService, HuggingFaceService
from documents.databricks_service import LocalVectorStore
import json


class DocumentModelTests(TestCase):
    """Test Document and DocumentChunk models"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        
    def test_document_creation(self):
        """Test basic document creation"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        self.assertEqual(document.user, self.user)
        self.assertEqual(document.title, 'Test Document')
        self.assertEqual(str(document), 'Test Document')
        self.assertFalse(document.processed)
        self.assertIsNotNone(document.uploaded_at)
        
    def test_document_cascade_delete(self):
        """Test that documents are deleted when user is deleted"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        self.assertEqual(Document.objects.count(), 1)
        self.user.delete()
        self.assertEqual(Document.objects.count(), 0)
        
    def test_document_chunk_creation(self):
        """Test DocumentChunk creation and relationships"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        chunk = DocumentChunk.objects.create(
            document=document,
            content='This is a test chunk content.',
            chunk_index=0,
            embedding_id='doc_1_chunk_0'
        )
        
        self.assertEqual(chunk.document, document)
        self.assertEqual(chunk.content, 'This is a test chunk content.')
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.embedding_id, 'doc_1_chunk_0')
        
        # Test reverse relationship
        self.assertEqual(document.chunks.count(), 1)
        self.assertEqual(document.chunks.first(), chunk)
        
    def test_document_chunk_unique_constraint(self):
        """Test unique constraint on document and chunk_index"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        # Create first chunk
        DocumentChunk.objects.create(
            document=document,
            content='First chunk',
            chunk_index=0
        )
        
        # Try to create another chunk with same index
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            DocumentChunk.objects.create(
                document=document,
                content='Duplicate chunk',
                chunk_index=0
            )
            
    def test_document_chunk_cascade_delete(self):
        """Test that chunks are deleted when document is deleted"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        DocumentChunk.objects.create(
            document=document,
            content='Test chunk',
            chunk_index=0
        )
        
        self.assertEqual(DocumentChunk.objects.count(), 1)
        document.delete()
        self.assertEqual(DocumentChunk.objects.count(), 0)


class DocumentSerializerTests(TestCase):
    """Test DocumentSerializer"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
    def test_serializer_fields(self):
        """Test serializer includes correct fields"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        serializer = DocumentSerializer(document)
        expected_fields = ['id', 'title', 'file', 'uploaded_at', 'processed']
        
        self.assertEqual(set(serializer.data.keys()), set(expected_fields))
        
    def test_serializer_read_only_fields(self):
        """Test that read-only fields cannot be updated"""
        data = {
            'title': 'New Document',
            'uploaded_at': '2024-01-01T00:00:00Z',
            'processed': True
        }
        
        serializer = DocumentSerializer(data=data)
        self.assertTrue(serializer.is_valid())
        
        # uploaded_at and processed should not be in validated_data
        self.assertNotIn('uploaded_at', serializer.validated_data)
        self.assertNotIn('processed', serializer.validated_data)
        self.assertIn('title', serializer.validated_data)
        
    def test_serializer_validation(self):
        """Test serializer validation"""
        # Valid data
        valid_data = {'title': 'Valid Document'}
        serializer = DocumentSerializer(data=valid_data)
        self.assertTrue(serializer.is_valid())
        
        # Invalid data (missing title)
        invalid_data = {}
        serializer = DocumentSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('title', serializer.errors)


class HuggingFaceServiceTests(TestCase):
    """Test HuggingFace API service"""
    
    @patch('documents.huggingface_api_service.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        self.service = HuggingFaceAPIService()
        
    def test_get_embeddings(self):
        """Test embedding generation"""
        texts = ["This is a test sentence"]
        embeddings = self.service.get_embeddings(texts)
        
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 3)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        
    @patch('documents.huggingface_api_service.requests.post')
    def test_generate_answer_success(self, mock_post):
        """Test successful answer generation"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "The answer is machine learning."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        question = "What is this about?"
        context = "This document discusses machine learning concepts."
        answer = self.service.generate_answer(question, context)
        
        self.assertEqual(answer, "The answer is machine learning.")
        mock_post.assert_called_once()
        
    @patch('documents.huggingface_api_service.requests.post')
    def test_generate_answer_rate_limit(self, mock_post):
        """Test rate limit handling"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        question = "What is this about?"
        context = "This document discusses machine learning."
        answer = self.service.generate_answer(question, context)
        
        self.assertIn("temporarily busy", answer)
        
    @patch('documents.huggingface_api_service.requests.post')
    def test_generate_answer_auth_error(self, mock_post):
        """Test authentication error handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        question = "What is this about?"
        context = "This document discusses machine learning."
        answer = self.service.generate_answer(question, context)
        
        self.assertIn("Authentication error", answer)
        
    @patch('documents.huggingface_api_service.requests.post')
    def test_generate_answer_fallback(self, mock_post):
        """Test fallback when API fails"""
        mock_post.side_effect = Exception("Network error")
        
        question = "What is machine learning?"
        context = "Machine learning is a subset of AI. It enables computers to learn."
        answer = self.service.generate_answer(question, context)
        
        self.assertIn("Based on the document", answer)
        self.assertIn("machine learning", answer.lower())


class LocalVectorStoreTests(TestCase):
    """Test LocalVectorStore functionality"""
    
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = LocalVectorStore()
        
    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up vector store files
        if os.path.exists('vector_index.faiss'):
            os.remove('vector_index.faiss')
        if os.path.exists('vector_metadata.pkl'):
            os.remove('vector_metadata.pkl')
            
    def test_add_and_search_embeddings(self):
        """Test adding embeddings and searching"""
        # Test data
        embeddings = [
            [0.1, 0.2, 0.3] + [0.0] * 1533,  # Pad to 1536 dimensions
            [0.4, 0.5, 0.6] + [0.0] * 1533,
        ]
        metadata = [
            {'document_id': 1, 'content': 'First document'},
            {'document_id': 2, 'content': 'Second document'},
        ]
        
        # Add embeddings
        self.vector_store.add_embeddings(embeddings, metadata)
        
        # Search for similar embedding
        query_embedding = [0.1, 0.2, 0.3] + [0.0] * 1533
        results = self.vector_store.search(query_embedding, top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['metadata']['document_id'], 1)
        self.assertGreater(results[0]['score'], 0.9)  # Should be very similar
        
    def test_empty_search(self):
        """Test search with empty index"""
        query_embedding = [0.1, 0.2, 0.3] + [0.0] * 1533
        results = self.vector_store.search(query_embedding, top_k=5)
        
        self.assertEqual(len(results), 0)


@override_settings(MEDIA_ROOT=tempfile.gettempdir())
class DocumentProcessorTests(TestCase):
    """Test DocumentProcessor functionality"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create a simple test PDF content
        self.test_file_content = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
        
    def tearDown(self):
        # Clean up any created files
        if os.path.exists('vector_index.faiss'):
            os.remove('vector_index.faiss')
        if os.path.exists('vector_metadata.pkl'):
            os.remove('vector_metadata.pkl')
            
    @patch('documents.services.HuggingFaceService')
    @patch('documents.services.PyPDFLoader')
    def test_process_document_success(self, mock_pdf_loader, mock_hf_service):
        """Test successful document processing"""
        # Mock PDF loader
        mock_doc = Mock()
        mock_doc.page_content = "This is test content about machine learning."
        mock_pdf_loader.return_value.load.return_value = [mock_doc]
        
        # Mock HuggingFace service
        mock_service_instance = Mock()
        mock_service_instance.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_hf_service.return_value = mock_service_instance
        
        # Create test document
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(self.test_file_content)
            f.flush()
            
            document = Document.objects.create(
                user=self.user,
                title='Test Document',
                file=f.name
            )
            
        try:
            processor = DocumentProcessor(use_api=False)
            chunks = processor.process_document(document)
            
            # Refresh document from database
            document.refresh_from_db()
            
            # Assertions
            self.assertTrue(document.processed)
            self.assertGreater(len(chunks), 0)
            self.assertEqual(DocumentChunk.objects.filter(document=document).count(), len(chunks))
            
        finally:
            os.unlink(f.name)
            
    @patch('documents.services.HuggingFaceService')
    def test_process_document_unsupported_file(self, mock_hf_service):
        """Test processing unsupported file type"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.unsupported'
        )
        
        processor = DocumentProcessor(use_api=False)
        
        with self.assertRaises(ValueError) as context:
            processor.process_document(document)
            
        self.assertIn("Unsupported file type", str(context.exception))
        
    @patch('documents.services.HuggingFaceService')
    @patch('documents.services.LocalVectorStore')
    def test_answer_question_success(self, mock_vector_store, mock_hf_service):
        """Test successful question answering"""
        # Mock vector store
        mock_store_instance = Mock()
        mock_store_instance.search.return_value = [
            {
                'metadata': {
                    'document_id': 1,
                    'content': 'Machine learning is a subset of AI.'
                },
                'score': 0.9
            }
        ]
        mock_vector_store.return_value = mock_store_instance
        
        # Mock HuggingFace service
        mock_service_instance = Mock()
        mock_service_instance.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_service_instance.generate_answer.return_value = "Machine learning is a subset of AI."
        mock_hf_service.return_value = mock_service_instance
        
        processor = DocumentProcessor(use_api=False)
        answer = processor.answer_question("What is machine learning?", document_id=1)
        
        self.assertEqual(answer, "Machine learning is a subset of AI.")
        mock_service_instance.get_embeddings.assert_called_once()
        mock_service_instance.generate_answer.assert_called_once()
        
    @patch('documents.services.HuggingFaceService')
    @patch('documents.services.LocalVectorStore')
    def test_answer_question_no_results(self, mock_vector_store, mock_hf_service):
        """Test question answering with no relevant results"""
        # Mock vector store with no results
        mock_store_instance = Mock()
        mock_store_instance.search.return_value = []
        mock_vector_store.return_value = mock_store_instance
        
        # Mock HuggingFace service
        mock_service_instance = Mock()
        mock_service_instance.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_hf_service.return_value = mock_service_instance
        
        processor = DocumentProcessor(use_api=False)
        answer = processor.answer_question("What is quantum computing?")
        
        self.assertIn("couldn't find relevant information", answer)


class DocumentViewSetTests(APITestCase):
    """Test DocumentViewSet API endpoints"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        
        # Create another user to test isolation
        self.other_user = User.objects.create_user(
            username='otheruser',
            password='testpass123'
        )
        
    def test_list_documents_user_isolation(self):
        """Test that users only see their own documents"""
        # Create documents for both users
        Document.objects.create(
            user=self.user,
            title='My Document',
            file='documents/my_doc.pdf'
        )
        Document.objects.create(
            user=self.other_user,
            title='Other Document',
            file='documents/other_doc.pdf'
        )
        
        url = reverse('documents-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['title'], 'My Document')
        
    def test_create_document(self):
        """Test document creation"""
        file_content = b'This is a test file content'
        uploaded_file = SimpleUploadedFile(
            "test.txt", 
            file_content,
            content_type="text/plain"
        )
        
        data = {
            'title': 'New Document',
            'file': uploaded_file
        }
        
        url = reverse('documents-list')
        
        with patch('documents.views.ProcessorConfig.create_processor') as mock_processor:
            mock_processor.return_value.process_document = Mock()
            response = self.client.post(url, data, format='multipart')
            
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Document.objects.filter(user=self.user).count(), 1)
        
        document = Document.objects.get(user=self.user)
        self.assertEqual(document.title, 'New Document')
        self.assertEqual(document.user, self.user)
        
    def test_retrieve_document(self):
        """Test retrieving specific document"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf'
        )
        
        url = reverse('documents-detail', kwargs={'pk': document.pk})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Document')
        
    def test_retrieve_other_user_document_404(self):
        """Test that users cannot retrieve other users' documents"""
        document = Document.objects.create(
            user=self.other_user,
            title='Other Document',
            file='documents/other.pdf'
        )
        
        url = reverse('documents-detail', kwargs={'pk': document.pk})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        
    @patch('documents.views.ProcessorConfig.create_processor')
    def test_ask_question_success(self, mock_processor):
        """Test successful question asking"""
        # Create processed document
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf',
            processed=True
        )
        
        # Mock processor
        mock_processor_instance = Mock()
        mock_processor_instance.answer_question.return_value = "This is the answer."
        mock_processor.return_value = mock_processor_instance
        
        url = reverse('documents-ask-question', kwargs={'pk': document.pk})
        data = {'question': 'What is this about?'}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['answer'], 'This is the answer.')
        self.assertEqual(response.data['question'], 'What is this about?')
        self.assertEqual(response.data['document_title'], 'Test Document')
        
    def test_ask_question_unprocessed_document(self):
        """Test asking question on unprocessed document"""
        document = Document.objects.create(
            user=self.user,
            title='Unprocessed Document',
            file='documents/test.pdf',
            processed=False
        )
        
        url = reverse('documents-ask-question', kwargs={'pk': document.pk})
        data = {'question': 'What is this about?'}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('still being processed', response.data['error'])
        
    def test_ask_question_missing_question(self):
        """Test asking without providing question"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf',
            processed=True
        )
        
        url = reverse('documents-ask-question', kwargs={'pk': document.pk})
        data = {}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Question is required', response.data['error'])
        
    def test_processing_status(self):
        """Test processing status endpoint"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf',
            processed=True
        )
        
        # Create some chunks
        DocumentChunk.objects.create(
            document=document,
            content='Test chunk',
            chunk_index=0
        )
        
        url = reverse('documents-processing-status', kwargs={'pk': document.pk})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['document_id'], document.id)
        self.assertEqual(response.data['title'], 'Test Document')
        self.assertTrue(response.data['processed'])
        self.assertEqual(response.data['chunk_count'], 1)
        
    def test_processed_count(self):
        """Test processed count endpoint"""
        # Create mix of processed and unprocessed documents
        Document.objects.create(
            user=self.user,
            title='Processed 1',
            file='documents/test1.pdf',
            processed=True
        )
        Document.objects.create(
            user=self.user,
            title='Processed 2',
            file='documents/test2.pdf',
            processed=True
        )
        Document.objects.create(
            user=self.user,
            title='Unprocessed',
            file='documents/test3.pdf',
            processed=False
        )
        
        url = reverse('documents-processed-count')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total'], 3)
        self.assertEqual(response.data['processed'], 2)
        self.assertEqual(response.data['processing'], 1)
        
    def test_unauthorized_access(self):
        """Test that unauthorized requests are rejected"""
        self.client.credentials()  # Remove authentication
        
        url = reverse('documents-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class ProcessorConfigTests(TestCase):
    """Test ProcessorConfig functionality"""
    
    @patch.dict(os.environ, {'HF_TOKEN': 'test_token'})
    def test_create_processor_with_token(self):
        """Test processor creation with HF token"""
        processor = ProcessorConfig.create_processor()
        
        # Should use API service when token is available
        self.assertIsInstance(processor, DocumentProcessor)
        
    @patch.dict(os.environ, {}, clear=True)
    def test_create_processor_without_token(self):
        """Test processor creation without HF token"""
        processor = ProcessorConfig.create_processor()
        
        # Should fall back to local service
        self.assertIsInstance(processor, DocumentProcessor)
        
    @patch.dict(os.environ, {'HF_TOKEN': 'test_token'})
    def test_create_processor_force_local(self):
        """Test forcing local processor even with token"""
        processor = ProcessorConfig.create_processor(force_local=True)
        
        # Should use local service even with token available
        self.assertIsInstance(processor, DocumentProcessor)


# Test utilities and helpers
class TestDataHelper:
    """Helper class for creating test data"""
    
    @staticmethod
    def create_test_pdf_content():
        """Create minimal valid PDF content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\nxref\n0 3\n0000000000 65535 f \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n9\n%%EOF'
    
    @staticmethod
    def create_test_text_content():
        """Create test text content"""
        return "This is a test document about machine learning and artificial intelligence. " \
               "Machine learning is a subset of AI that enables computers to learn without " \
               "being explicitly programmed. It uses algorithms to analyze data and make predictions."


# Run tests with: python manage.py test documents.tests
# Run specific test: python manage.py test documents.tests.DocumentModelTests.test_document_creation
# Run with coverage: coverage run --source='.' manage.py test documents.tests && coverage report

# Integration test (optional - tests multiple components together)
@override_settings(MEDIA_ROOT=tempfile.gettempdir())
class DocumentIntegrationTests(APITestCase):
    """Integration tests for complete document workflow"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        
    def tearDown(self):
        # Clean up test files
        for filename in ['vector_index.faiss', 'vector_metadata.pkl']:
            if os.path.exists(filename):
                os.remove(filename)
                
    @patch('documents.services.HuggingFaceService')
    @patch('documents.services.PyPDFLoader')
    def test_complete_document_workflow(self, mock_pdf_loader, mock_hf_service):
        """Test complete workflow: upload -> process -> ask question"""
        # Mock dependencies
        mock_doc = Mock()
        mock_doc.page_content = TestDataHelper.create_test_text_content()
        mock_pdf_loader.return_value.load.return_value = [mock_doc]
        
        mock_service_instance = Mock()
        mock_service_instance.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_service_instance.generate_answer.return_value = "Machine learning is a subset of AI."
        mock_hf_service.return_value = mock_service_instance
        
        # 1. Upload document
        file_content = TestDataHelper.create_test_pdf_content()
        uploaded_file = SimpleUploadedFile(
            "test.pdf", 
            file_content,
            content_type="application/pdf"
        )
        
        data = {
            'title': 'ML Document',
            'file': uploaded_file
        }
        
        upload_url = reverse('documents-list')
        response = self.client.post(upload_url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        document_id = response.data['id']
        
        # 2. Wait for processing (simulate by marking as processed)
        document = Document.objects.get(id=document_id)
        document.processed = True
        document.save()
        
        # 3. Ask question
        question_url = reverse('documents-ask-question', kwargs={'pk': document_id})
        question_data = {'question': 'What is machine learning?'}
        response = self.client.post(question_url, question_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('Machine learning', response.data['answer'])
        
        # 4. Check document status
        status_url = reverse('documents-processing-status', kwargs={'pk': document_id})
        response = self.client.get(status_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['processed'])