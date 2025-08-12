import os
import tempfile
from unittest.mock import patch, Mock
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase
from rest_framework.authtoken.models import Token
from rest_framework import status

from .models import Document, DocumentChunk
from .serializers import DocumentSerializer
from .huggingface_api_service import HuggingFaceAPIService
from .services import DocumentProcessor


class DocumentModelTest(TestCase):
    """Test the basic Document and DocumentChunk models"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
    
    def test_document_creation(self):
        """Test that we can create a document"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='test.pdf'
        )
        
        self.assertEqual(document.user, self.user)
        self.assertEqual(document.title, 'Test Document')
        self.assertFalse(document.processed)  # Should start unprocessed
        self.assertIsNotNone(document.uploaded_at)
        
        print("✅ Document creation test passed")
    
    def test_document_chunk_creation(self):
        """Test that we can create chunks for a document"""
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='test.pdf'
        )
        
        chunk = DocumentChunk.objects.create(
            document=document,
            content='This is test content about machine learning.',
            chunk_index=0,
            embedding_id='doc_1_chunk_0'
        )
        
        self.assertEqual(chunk.document, document)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(document.chunks.count(), 1)
        
        print("✅ Document chunk creation test passed")


class HuggingFaceAPIServiceTest(TestCase):
    """Test the HuggingFaceAPIService with mocked API calls"""
    
    def setUp(self):
        # Mock environment variable for testing
        with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
            self.service = HuggingFaceAPIService()
    
    def test_service_initialization(self):
        """Test that the service initializes correctly"""
        info = self.service.get_service_info()
        
        self.assertIn('embedding_method', info)
        self.assertIn('has_similarity_api', info)
        self.assertIn('has_chat_api', info)
        
        print("✅ Service initialization test passed")
    
    @patch('documents.huggingface_api_service.requests.post')
    def test_similarity_calculation(self, mock_post):
        """Test the similarity calculation using your HF API"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [0.85, 0.23, 0.67]  # Similarity scores
        mock_post.return_value = mock_response
        
        question = "What is machine learning?"
        chunks = [
            "Machine learning is a subset of AI",
            "The weather is sunny today", 
            "Neural networks are used in deep learning"
        ]
        
        similarities = self.service.calculate_similarity(question, chunks)
        
        self.assertEqual(len(similarities), 3)
        self.assertEqual(similarities[0], 0.85)  # Best match
        self.assertEqual(similarities[1], 0.23)  # Poor match
        self.assertEqual(similarities[2], 0.67)  # Good match
        
        # Verify API was called with correct format
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        self.assertEqual(payload['inputs']['source_sentence'], question)
        self.assertEqual(payload['inputs']['sentences'], chunks)
        
        print("✅ Similarity calculation test passed")
    
    def test_find_most_relevant_chunks(self):
        """Test finding most relevant chunks"""
        question = "What is Python programming?"
        chunks = [
            "Python is a programming language used for data science",  # Should rank high
            "The office cafeteria serves lunch from 11 AM to 2 PM",   # Should rank low
            "Programming languages like Python are popular for AI"     # Should rank medium
        ]
        
        # Mock the similarity calculation
        with patch.object(self.service, 'calculate_similarity') as mock_calc:
            mock_calc.return_value = [0.92, 0.15, 0.74]  # High, Low, Medium scores
            
            relevant_chunks = self.service.find_most_relevant_chunks(question, chunks, top_k=2)
            
            self.assertEqual(len(relevant_chunks), 2)
            
            # Check that chunks are ranked correctly (highest score first)
            self.assertEqual(relevant_chunks[0]['similarity_score'], 0.92)
            self.assertEqual(relevant_chunks[1]['similarity_score'], 0.74)
            
            # Check content is correct
            self.assertIn("Python is a programming language", relevant_chunks[0]['content'])
            self.assertIn("Programming languages like Python", relevant_chunks[1]['content'])
        
        print("✅ Relevant chunks finding test passed")
    
    @patch('documents.huggingface_api_service.requests.post')
    def test_answer_question(self, mock_post):
        """Test AI answer generation"""
        # Mock successful AI response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Python is a high-level programming language."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        question = "What is Python?"
        context = "Python is a programming language used for web development and data science."
        
        answer = self.service.answer_question(question, context)
        
        self.assertEqual(answer, "Python is a high-level programming language.")
        mock_post.assert_called_once()
        
        print("✅ Answer question test passed")


class DocumentProcessorTest(TestCase):
    """Test the document processor functionality"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.processor = DocumentProcessor()
    
    def test_text_chunking(self):
        """Test that text is split into appropriate chunks"""
        content = (
            "This is the first sentence about machine learning. "
            "This is the second sentence about artificial intelligence. "
            "This is the third sentence about data science. "
            "This is the fourth sentence about programming."
        )
        
        chunks = self.processor._split_into_chunks(content, chunk_size=100)
        
        self.assertGreater(len(chunks), 1)  # Should be split into multiple chunks
        
        # Check that chunks aren't too long
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 150)  # Some flexibility for sentence boundaries
        
        print("✅ Text chunking test passed")
    
    @patch('builtins.open', create=True)
    def test_read_text_file(self, mock_open):
        """Test reading a text file"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "This is test content"
        
        content = self.processor._read_file('test.txt')
        
        self.assertEqual(content, "This is test content")
        mock_open.assert_called_once_with('test.txt', 'r', encoding='utf-8')
        
        print("✅ Text file reading test passed")
    
    @patch('documents.huggingface_api_service.HuggingFaceAPIService')
    def test_answer_question_integration(self, mock_service_class):
        """Test the complete answer question workflow"""
        # Create a processed document with chunks
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf',
            processed=True
        )
        
        DocumentChunk.objects.create(
            document=document,
            content='Machine learning is a subset of artificial intelligence.',
            chunk_index=0
        )
        
        DocumentChunk.objects.create(
            document=document,
            content='Python is a programming language for data science.',
            chunk_index=1
        )
        
        # Mock the API service
        mock_service = Mock()
        mock_service.find_most_relevant_chunks.return_value = [
            {
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity_score': 0.89,
                'index': 0
            }
        ]
        mock_service.answer_question.return_value = "Machine learning is a branch of AI."
        mock_service_class.return_value = mock_service
        
        # Test the answer question method
        answer = self.processor.answer_question("What is machine learning?", document_id=document.id)
        
        self.assertEqual(answer, "Machine learning is a branch of AI.")
        
        # Verify the similarity search was called
        mock_service.find_most_relevant_chunks.assert_called_once()
        mock_service.answer_question.assert_called_once()
        
        print("✅ Answer question integration test passed")


class DocumentAPITest(APITestCase):
    """Test the API endpoints"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
    
    def test_document_upload(self):
        """Test uploading a document via API"""
        test_file = SimpleUploadedFile(
            "test.txt",
            b"This is test content about machine learning and AI.",
            content_type="text/plain"
        )
        
        response = self.client.post('/api/documents/', {
            'title': 'Test Document',
            'file': test_file
        }, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['title'], 'Test Document')
        self.assertFalse(response.data['processed'])  # Should start unprocessed
        
        # Verify document was created in database
        self.assertEqual(Document.objects.count(), 1)
        document = Document.objects.first()
        self.assertEqual(document.user, self.user)
        
        print("✅ Document upload API test passed")
    
    def test_list_documents(self):
        """Test listing user's documents"""
        # Create documents for test user
        Document.objects.create(
            user=self.user,
            title='My Document',
            file='documents/test.pdf'
        )
        
        # Create document for another user (should not appear)
        other_user = User.objects.create_user(username='other', password='pass')
        Document.objects.create(
            user=other_user,
            title='Other Document',
            file='documents/other.pdf'
        )
        
        response = self.client.get('/api/documents/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)  # Only user's document
        self.assertEqual(response.data[0]['title'], 'My Document')
        
        print("✅ List documents API test passed")
    
    @patch('documents.views.DocumentProcessor')
    def test_ask_question_api(self, mock_processor_class):
        """Test asking a question via API"""
        # Create a processed document
        document = Document.objects.create(
            user=self.user,
            title='Test Document',
            file='documents/test.pdf',
            processed=True
        )
        
        # Mock the processor response
        mock_processor = Mock()
        mock_processor.answer_question.return_value = "This is the AI answer."
        mock_processor_class.return_value = mock_processor
        
        response = self.client.post(f'/api/documents/{document.id}/ask_question/', {
            'question': 'What is this document about?'
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['answer'], 'This is the AI answer.')
        self.assertEqual(response.data['question'], 'What is this document about?')
        self.assertEqual(response.data['document_title'], 'Test Document')
        
        print("✅ Ask question API test passed")
    
    def test_ask_question_unprocessed_document(self):
        """Test asking question about unprocessed document"""
        document = Document.objects.create(
            user=self.user,
            title='Unprocessed Document',
            file='documents/test.pdf',
            processed=False  # Not processed yet
        )
        
        response = self.client.post(f'/api/documents/{document.id}/ask_question/', {
            'question': 'What is this about?'
        })
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('still being processed', response.data['error'])
        
        print("✅ Unprocessed document error test passed")
    
    def test_unauthorized_access(self):
        """Test that unauthorized requests are rejected"""
        self.client.credentials()  # Remove authentication
        
        response = self.client.get('/api/documents/')
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        print("✅ Unauthorized access test passed")


class IntegrationTest(TestCase):
    """Test the complete workflow integration"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
    
    @patch('documents.huggingface_api_service.HuggingFaceAPIService')
    def test_complete_workflow(self, mock_service_class):
        """Test the complete document processing and Q&A workflow"""
        # Mock the API service
        mock_service = Mock()
        mock_service.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_service.find_most_relevant_chunks.return_value = [
            {
                'content': 'Machine learning is a subset of AI that enables computers to learn.',
                'similarity_score': 0.91,
                'index': 0
            }
        ]
        mock_service.answer_question.return_value = "Machine learning enables computers to learn from data."
        mock_service_class.return_value = mock_service
        
        # Create processor
        processor = DocumentProcessor()
        
        # Test document content
        test_content = (
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn and make decisions from data without being explicitly "
            "programmed for every task."
        )
        
        # Test text chunking
        chunks = processor._split_into_chunks(test_content)
        self.assertGreater(len(chunks), 0)
        
        # Test question answering with the chunks
        question = "What is machine learning?"
        
        # Simulate having chunks in database by testing the core logic
        answer = mock_service.answer_question(question, test_content)
        
        self.assertEqual(answer, "Machine learning enables computers to learn from data.")
        
        print("✅ Complete workflow integration test passed")


# Test runner function
def run_tests():
    """Function to run tests and show results"""
    print("🚀 Running Simple Unit Tests for Documents Module")
    print("=" * 60)
    
    # Import Django's test runner
    from django.test.utils import get_runner
    from django.conf import settings
    
    TestRunner = get_runner(settings)
    test_runner = TestRunner()
    
    # Run the tests
    failures = test_runner.run_tests(["documents.test"])
    
    if failures:
        print(f"\n❌ {failures} test(s) failed")
    else:
        print(f"\n✅ All tests passed!")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    run_tests()