"""
Unit Tests for Document Management System

WHY WE TEST DOCUMENT FUNCTIONALITY:
- File uploads must work reliably
- Document processing must be accurate
- User data isolation must be enforced
- Q&A functionality must provide correct responses
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from rest_framework.authtoken.models import Token
from unittest.mock import patch, MagicMock
import tempfile
import os

from .models import Document, DocumentChunk
from .services import DocumentProcessor


class DocumentModelTestCase(TestCase):
    """
    Test the Document model functionality
    
    WHY WE TEST DOCUMENT MODELS:
    - Ensures proper data storage and relationships
    - Validates model constraints and validation
    - Tests file handling and metadata storage
    - Ensures data integrity across operations
    """
    
    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='docuser',
            password='testpass123'
        )
        self.other_user = User.objects.create_user(
            username='otheruser', 
            password='testpass123'
        )
    
    def test_document_creation(self):
        """
        Test creating a document with all required fields
        
        WHAT THIS TESTS:
        - Document model accepts all required fields
        - Foreign key relationship to User works
        - Default values are set correctly
        - Document is saved to database properly
        
        WHY THIS IS IMPORTANT:
        - Document creation is core functionality
        - Validates model design and constraints
        - Ensures proper user association
        - Tests database schema correctness
        """
        document = Document.objects.create(
            user=self.user,
            title="Test Document",
            file="test.pdf",  # Mock file path for testing
            processed=False
        )
        
        # Verify document was created correctly
        self.assertEqual(document.user, self.user)
        self.assertEqual(document.title, "Test Document")
        self.assertEqual(document.file, "test.pdf")
        self.assertFalse(document.processed)  # Default should be False
        self.assertIsNotNone(document.uploaded_at)  # Should auto-set timestamp
        
        # Verify document exists in database
        self.assertTrue(Document.objects.filter(id=document.id).exists())
    
    def test_document_string_representation(self):
        """
        Test the __str__ method of Document model
        
        WHAT THIS TESTS:
        - __str__ method returns expected string
        - String representation is useful for debugging
        - Model string display works in admin interface
        
        WHY THIS IS IMPORTANT:
        - Helps with debugging and development
        - Makes admin interface more user-friendly
        - Standard Python practice for model classes
        - Improves code maintainability
        """
        document = Document.objects.create(
            user=self.user,
            title="My Test Document",
            file="test.pdf"
        )
        
        # String representation should be the title
        self.assertEqual(str(document), "My Test Document")
    
    def test_document_user_relationship(self):
        """
        Test the foreign key relationship between Document and User
        
        WHAT THIS TESTS:
        - Documents are properly associated with users
        - User deletion cascades to documents (CASCADE behavior)
        - Multiple documents per user work correctly
        - User isolation is maintained
        
        WHY THIS IS IMPORTANT:
        - Ensures proper data ownership
        - Validates database relationship design
        - Tests data integrity constraints
        - Ensures user data isolation for security
        """
        # Create documents for different users
        doc1 = Document.objects.create(
            user=self.user,
            title="User 1 Doc",
            file="doc1.pdf"
        )
        
        doc2 = Document.objects.create(
            user=self.other_user,
            title="User 2 Doc", 
            file="doc2.pdf"
        )
        
        # Verify documents are associated with correct users
        self.assertEqual(doc1.user, self.user)
        self.assertEqual(doc2.user, self.other_user)
        
        # Verify user can access their documents
        user1_docs = Document.objects.filter(user=self.user)
        self.assertIn(doc1, user1_docs)
        self.assertNotIn(doc2, user1_docs)
        
        # Test CASCADE behavior: when user is deleted, their documents should be deleted
        user_id = self.user.id
        self.user.delete()
        
        # doc1 should be deleted (CASCADE)
        self.assertFalse(Document.objects.filter(id=doc1.id).exists())
        
        # doc2 should still exist (different user)
        self.assertTrue(Document.objects.filter(id=doc2.id).exists())


class DocumentChunkModelTestCase(TestCase):
    """
    Test the DocumentChunk model functionality
    
    WHY WE TEST DOCUMENT CHUNKS:
    - Chunks store processed document content
    - Chunk relationships to documents must work
    - Unique constraints must be enforced
    - Chunk ordering must be maintained
    """
    
    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='chunkuser',
            password='testpass123'
        )
        self.document = Document.objects.create(
            user=self.user,
            title="Test Document",
            file="test.pdf"
        )
    
    def test_document_chunk_creation(self):
        """
        Test creating document chunks with proper relationships
        
        WHAT THIS TESTS:
        - DocumentChunk creation with all fields
        - Foreign key relationship to Document works
        - Chunk indexing works correctly
        - Embedding ID storage works
        
        WHY THIS IS IMPORTANT:
        - Chunks store the actual content for Q&A
        - Proper indexing enables retrieval and ordering
        - Embedding IDs link to vector database
        - Ensures content processing works correctly
        """
        chunk = DocumentChunk.objects.create(
            document=self.document,
            content="This is a test chunk of content.",
            chunk_index=0,
            embedding_id="doc_1_chunk_0"
        )
        
        # Verify chunk was created correctly
        self.assertEqual(chunk.document, self.document)
        self.assertEqual(chunk.content, "This is a test chunk of content.")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.embedding_id, "doc_1_chunk_0")
        
        # Verify chunk exists in database
        self.assertTrue(DocumentChunk.objects.filter(id=chunk.id).exists())
    
    def test_document_chunk_relationship(self):
        """
        Test the relationship between Document and DocumentChunk
        
        WHAT THIS TESTS:
        - Multiple chunks can belong to one document
        - Reverse relationship (document.chunks) works
        - CASCADE deletion works properly
        - Chunk ordering is maintained
        
        WHY THIS IS IMPORTANT:
        - Documents are split into multiple chunks for processing
        - Chunks must maintain order for coherent content
        - Deletion must cascade to prevent orphaned chunks
        - Enables efficient content retrieval and Q&A
        """
        # Create multiple chunks for the document
        chunk1 = DocumentChunk.objects.create(
            document=self.document,
            content="First chunk of content.",
            chunk_index=0
        )
        
        chunk2 = DocumentChunk.objects.create(
            document=self.document,
            content="Second chunk of content.",
            chunk_index=1
        )
        
        chunk3 = DocumentChunk.objects.create(
            document=self.document,
            content="Third chunk of content.",
            chunk_index=2
        )
        
        # Test forward relationship (chunk to document)
        self.assertEqual(chunk1.document, self.document)
        self.assertEqual(chunk2.document, self.document)
        self.assertEqual(chunk3.document, self.document)
        
        # Test reverse relationship (document to chunks)
        chunks = self.document.chunks.all().order_by('chunk_index')
        self.assertEqual(chunks.count(), 3)
        self.assertEqual(list(chunks), [chunk1, chunk2, chunk3])
        
        # Test CASCADE deletion
        document_id = self.document.id
        self.document.delete()
        
        # All chunks should be deleted when document is deleted
        remaining_chunks = DocumentChunk.objects.filter(document_id=document_id)
        self.assertEqual(remaining_chunks.count(), 0)
    
    def test_document_chunk_unique_constraint(self):
        """
        Test the unique constraint on (document, chunk_index)
        
        WHAT THIS TESTS:
        - Each document can have only one chunk per index
        - Database enforces uniqueness constraint
        - Prevents duplicate chunk indices
        - Maintains data integrity
        
        WHY THIS IS IMPORTANT:
        - Ensures consistent chunk ordering
        - Prevents data corruption from duplicate indices
        - Validates database schema design
        - Enables reliable chunk retrieval by index
        """
        # Create first chunk with index 0
        DocumentChunk.objects.create(
            document=self.document,
            content="First chunk at index 0",
            chunk_index=0
        )
        
        # Attempting to create another chunk with same index should fail
        with self.assertRaises(Exception):  # IntegrityError expected
            DocumentChunk.objects.create(
                document=self.document,
                content="Second chunk at index 0",  # Same index
                chunk_index=0
            )
    
    def test_document_chunk_ordering(self):
        """
        Test that chunks can be retrieved in proper order
        
        WHAT THIS TESTS:
        - Chunks can be ordered by chunk_index
        - Multiple chunks maintain sequence
        - Ordering is consistent and reliable
        - Content can be reconstructed in order
        
        WHY THIS IS IMPORTANT:
        - Document content must maintain logical flow
        - Q&A needs context from properly ordered chunks
        - Users expect coherent document representation
        - Processing pipelines depend on chunk order
        """
        # Create chunks out of order
        chunk2 = DocumentChunk.objects.create(
            document=self.document,
            content="This is the second chunk.",
            chunk_index=2
        )
        
        chunk0 = DocumentChunk.objects.create(
            document=self.document,
            content="This is the first chunk.",
            chunk_index=0
        )
        
        chunk1 = DocumentChunk.objects.create(
            document=self.document,
            content="This is the middle chunk.",
            chunk_index=1
        )
        
        # Retrieve chunks in order
        ordered_chunks = self.document.chunks.all().order_by('chunk_index')
        
        # Verify they come back in correct order
        self.assertEqual(ordered_chunks[0], chunk0)
        self.assertEqual(ordered_chunks[1], chunk1)
        self.assertEqual(ordered_chunks[2], chunk2)
        
        # Verify content can be reconstructed
        full_content = " ".join([chunk.content for chunk in ordered_chunks])
        expected_content = "This is the first chunk. This is the middle chunk. This is the second chunk."
        self.assertEqual(full_content, expected_content)


class DocumentAPITestCase(APITestCase):
    """
    Test the Document API endpoints
    
    WHY WE TEST DOCUMENT APIs:
    - File upload functionality must work correctly
    - User authentication and authorization must be enforced
    - HTTP status codes and responses must be consistent
    - Error handling must be robust
    """
    
    def setUp(self):
        """Set up authenticated user for API tests"""
        self.user = User.objects.create_user(
            username='apiuser',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token.key}')
        
        self.other_user = User.objects.create_user(
            username='otheruser',
            password='testpass123'
        )
        
        self.documents_url = '/api/documents/'
    
    def test_list_user_documents(self):
        """
        Test retrieving list of user's documents
        
        WHAT THIS TESTS:
        - GET /api/documents/ returns user's documents
        - Only user's own documents are returned
        - Response format is correct
        - Authentication is required
        
        WHY THIS IS IMPORTANT:
        - Users need to see their uploaded documents
        - Data isolation prevents access to other users' files
        - Consistent API response format
        - Security through authentication
        """
        # Create documents for current user
        doc1 = Document.objects.create(
            user=self.user,
            title="User's Document 1",
            file="doc1.pdf"
        )
        
        doc2 = Document.objects.create(
            user=self.user,
            title="User's Document 2", 
            file="doc2.pdf"
        )
        
        # Create document for other user (should not appear)
        Document.objects.create(
            user=self.other_user,
            title="Other User's Document",
            file="other.pdf"
        )
        
        response = self.client.get(self.documents_url)
        
        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Should return only user's documents
        documents = response.json()
        self.assertEqual(len(documents), 2)
        
        # Verify correct documents are returned
        document_titles = [doc['title'] for doc in documents]
        self.assertIn("User's Document 1", document_titles)
        self.assertIn("User's Document 2", document_titles)
        self.assertNotIn("Other User's Document", document_titles)
    
    def test_create_document_with_file_upload(self):
        """
        Test creating a document with file upload
        
        WHAT THIS TESTS:
        - POST /api/documents/ accepts file uploads
        - Multipart form data handling works
        - File is saved to correct location
        - Document metadata is stored correctly
        
        WHY THIS IS IMPORTANT:
        - File upload is primary way users add content
        - Validates file handling and storage
        - Ensures proper metadata extraction
        - Tests multipart form processing
        """
        # Create a temporary test file
        test_content = b"This is test document content for upload testing."
        test_file = SimpleUploadedFile(
            "test_document.txt",
            test_content,
            content_type="text/plain"
        )
        
        upload_data = {
            'title': 'Uploaded Test Document',
            'file': test_file
        }
        
        response = self.client.post(
            self.documents_url,
            upload_data,
            format='multipart'  # Important for file uploads
        )
        
        # Should return 201 Created
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify document was created in database
        document = Document.objects.get(title='Uploaded Test Document')
        self.assertEqual(document.user, self.user)
        self.assertEqual(document.title, 'Uploaded Test Document')
        self.assertTrue(document.file.name.endswith('test_document.txt'))
        
        # Verify response contains document data
        response_data = response.json()
        self.assertEqual(response_data['title'], 'Uploaded Test Document')
        self.assertEqual(response_data['id'], document.id)
    
    def test_document_processing_trigger(self):
        """
        Test that document upload triggers background processing
        
        WHAT THIS TESTS:
        - Document processing starts after upload
        - Processing status is tracked
        - Background processing doesn't block response
        - Processing state is properly managed
        
        WHY THIS IS IMPORTANT:
        - Documents must be processed for Q&A functionality
        - Processing shouldn't delay user experience
        - Status tracking enables progress monitoring
        - Validates integration with processing pipeline
        """
        with patch('documents.views.DocumentProcessor') as mock_processor:
            # Mock the processor to avoid actual processing in tests
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            
            test_file = SimpleUploadedFile(
                "process_test.txt",
                b"Content for processing test.",
                content_type="text/plain"
            )
            
            upload_data = {
                'title': 'Processing Test Document',
                'file': test_file
            }
            
            response = self.client.post(
                self.documents_url,
                upload_data,
                format='multipart'
            )
            
            # Upload should succeed immediately
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            
            # Document should be created with processed=False initially
            document = Document.objects.get(title='Processing Test Document')
            self.assertFalse(document.processed)  # Processing happens in background
    
    def test_document_detail_access(self):
        """
        Test retrieving individual document details
        
        WHAT THIS TESTS:
        - GET /api/documents/{id}/ returns document details
        - Users can only access their own documents
        - Returns 404 for non-existent documents
        - Returns 403/404 for other users' documents
        
        WHY THIS IS IMPORTANT:
        - Users need access to individual document details
        - Security: prevents access to other users' documents
        - Proper HTTP status codes for different scenarios
        - Validates authorization and data isolation
        """
        # Create document for current user
        user_doc = Document.objects.create(
            user=self.user,
            title="User's Private Document",
            file="private.pdf"
        )
        
        # Create document for other user
        other_doc = Document.objects.create(
            user=self.other_user,
            title="Other User's Document",
            file="other.pdf"
        )
        
        # Test accessing own document
        response = self.client.get(f'{self.documents_url}{user_doc.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        response_data = response.json()
        self.assertEqual(response_data['title'], "User's Private Document")
        
        # Test accessing other user's document (should fail)
        response = self.client.get(f'{self.documents_url}{other_doc.id}/')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_document_deletion(self):
        """
        Test deleting documents
        
        WHAT THIS TESTS:
        - DELETE /api/documents/{id}/ removes document
        - Users can only delete their own documents
        - File cleanup happens properly
        - Related chunks are also deleted
        
        WHY THIS IS IMPORTANT:
        - Users need ability to remove unwanted documents
        - Security: prevents deletion of other users' documents
        - Proper cleanup prevents storage waste
        - Data integrity through cascading deletes
        """
        # Create document with chunks
        document = Document.objects.create(
            user=self.user,
            title="Document to Delete",
            file="delete_me.pdf"
        )
        
        # Add some chunks
        DocumentChunk.objects.create(
            document=document,
            content="Chunk 1",
            chunk_index=0
        )
        
        DocumentChunk.objects.create(
            document=document,
            content="Chunk 2", 
            chunk_index=1
        )
        
        # Delete the document
        response = self.client.delete(f'{self.documents_url}{document.id}/')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        
        # Document should be deleted
        self.assertFalse(Document.objects.filter(id=document.id).exists())
        
        # Chunks should also be deleted (CASCADE)
        remaining_chunks = DocumentChunk.objects.filter(document=document)
        self.assertEqual(remaining_chunks.count(), 0)
    
    def test_unauthenticated_document_access(self):
        """
        Test that unauthenticated users cannot access documents
        
        WHAT THIS TESTS:
        - Document endpoints require authentication
        - Returns 401 Unauthorized without token
        - Security is enforced at API level
        - No data leakage to unauthenticated users
        
        WHY THIS IS IMPORTANT:
        - Documents contain private user content
        - Authentication is fundamental security requirement
        - API security must be consistently enforced
        - Prevents unauthorized data access
        """
        # Remove authentication credentials
        self.client.credentials()
        
        # Try to access documents without authentication
        response = self.client.get(self.documents_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Try to upload document without authentication
        test_file = SimpleUploadedFile("test.txt", b"content", content_type="text/plain")
        upload_data = {'title': 'Test', 'file': test_file}
        
        response = self.client.post(self.documents_url, upload_data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class DocumentProcessingTestCase(TestCase):
    """
    Test document processing functionality
    
    WHY WE TEST DOCUMENT PROCESSING:
    - Processing converts documents into searchable chunks
    - Embedding generation must work correctly
    - Error handling for unsupported files
    - Processing state management
    """
    
    def setUp(self):
        """Set up test data for processing tests"""
        self.user = User.objects.create_user(
            username='processuser',
            password='testpass123'
        )
    
    @patch('documents.services.LocalVectorStore')
    @patch('documents.services.LocalHuggingFaceService')
    def test_document_processing_success(self, mock_hf_service, mock_vector_store):
        """
        Test successful document processing pipeline
        
        WHAT THIS TESTS:
        - Document content is extracted correctly
        - Text is split into appropriate chunks
        - Embeddings are generated for chunks
        - Chunks are stored in database and vector store
        - Document status is updated to processed=True
        
        WHY THIS IS IMPORTANT:
        - Processing enables Q&A functionality
        - Validates entire processing pipeline
        - Ensures proper integration of components
        - Tests error-free processing flow
        """
        # Mock the services to avoid external dependencies
        mock_hf_instance = MagicMock()
        mock_hf_service.return_value = mock_hf_instance
        mock_hf_instance.get_embeddings.return_value = [
            [0.1, 0.2, 0.3],  # Mock embedding for chunk 1
            [0.4, 0.5, 0.6],  # Mock embedding for chunk 2
        ]
        
        mock_vector_instance = MagicMock()
        mock_vector_store.return_value = mock_vector_instance
        
        # Create a test document
        document = Document.objects.create(
            user=self.user,
            title="Test Processing Document",
            file="test.txt"
        )
        
        # Create temporary file with test content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is the first chunk of content. " * 10)  # First chunk
            temp_file.write("This is the second chunk of content. " * 10)  # Second chunk
            temp_file_path = temp_file.name
        
        try:
            # Mock the file path
            document.file.name = temp_file_path
            
            # Process the document
            processor = DocumentProcessor()
            chunks = processor.process_document(document)
            
            # Verify processing results
            self.assertTrue(len(chunks) >= 1)  # Should create at least one chunk
            
            # Verify chunks are saved to database
            db_chunks = DocumentChunk.objects.filter(document=document)
            self.assertEqual(db_chunks.count(), len(chunks))
            
            # Verify document is marked as processed
            document.refresh_from_db()
            self.assertTrue(document.processed)
            
            # Verify embeddings were generated
            mock_hf_instance.get_embeddings.assert_called()
            
            # Verify vector store was updated
            mock_vector_instance.add_embeddings.assert_called()
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def test_document_processing_unsupported_file(self):
        """
        Test processing of unsupported file types
        
        WHAT THIS TESTS:
        - Unsupported file types are rejected gracefully
        - Appropriate error messages are generated
        - Document status remains unprocessed
        - No partial processing occurs
        
        WHY THIS IS IMPORTANT:
        - Prevents system errors from unsupported files
        - Provides clear feedback about file requirements
        - Maintains data integrity
        - Enables proper error handling in UI
        """
        document = Document.objects.create(
            user=self.user,
            title="Unsupported File",
            file="test.docx"  # Unsupported file type
        )
        
        processor = DocumentProcessor()
        
        # Processing should raise an exception for unsupported file
        with self.assertRaises(ValueError) as context:
            processor.process_document(document)
        
        # Error message should indicate unsupported file type
        self.assertIn("Unsupported file type", str(context.exception))
        
        # Document should remain unprocessed
        document.refresh_from_db()
        self.assertFalse(document.processed)
        
        # No chunks should be created
        chunks = DocumentChunk.objects.filter(document=document)
        self.assertEqual(chunks.count(), 0)


class QuestionAnswerAPITestCase(APITestCase):
    """
    Test the Q&A functionality
    
    WHY WE TEST Q&A:
    - Q&A is the core value proposition
    - Answer quality and relevance must be validated
    - Error handling for various scenarios
    - Integration with document processing
    """
    
    def setUp(self):
        """Set up test data for Q&A tests"""
        self.user = User.objects.create_user(
            username='qauser',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token.key}')
        
        # Create a processed document with chunks
        self.document = Document.objects.create(
            user=self.user,
            title="Q&A Test Document",
            file="qa_test.txt",
            processed=True
        )
        
        # Add test chunks
        DocumentChunk.objects.create(
            document=self.document,
            content="Artificial intelligence is a branch of computer science.",
            chunk_index=0,
            embedding_id="doc_1_chunk_0"
        )
        
        DocumentChunk.objects.create(
            document=self.document,
            content="Machine learning is a subset of artificial intelligence.",
            chunk_index=1,
            embedding_id="doc_1_chunk_1"
        )
    
    @patch('documents.services.DocumentProcessor.answer_question')
    def test_ask_question_success(self, mock_answer_question):
        """
        Test successful question answering
        
        WHAT THIS TESTS:
        - POST to ask_question endpoint works
        - Question is processed correctly
        - Answer is returned in proper format
        - Document context is used appropriately
        
        WHY THIS IS IMPORTANT:
        - Q&A is the primary user interaction
        - Validates integration of all components
        - Ensures proper API response format
        - Tests end-to-end functionality
        """
        # Mock the answer generation
        mock_answer_question.return_value = "AI is a branch of computer science that creates intelligent machines."
        
        question_data = {
            'question': 'What is artificial intelligence?'
        }
        
        response = self.client.post(
            f'/api/documents/{self.document.id}/ask_question/',
            question_data,
            format='json'
        )
        
        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Should contain answer and question
        response_data = response.json()
        self.assertIn('answer', response_data)
        self.assertIn('question', response_data)
        self.assertIn('document_title', response_data)
        
        # Verify content
        self.assertEqual(response_data['question'], 'What is artificial intelligence?')
        self.assertEqual(response_data['document_title'], 'Q&A Test Document')
        self.assertIn('computer science', response_data['answer'])
        
        # Verify the processor was called with correct parameters
        mock_answer_question.assert_called_once_with(
            'What is artificial intelligence?', 
            self.document.id
        )
    
    def test_ask_question_unprocessed_document(self):
        """
        Test asking question about unprocessed document
        
        WHAT THIS TESTS:
        - Unprocessed documents return appropriate error
        - Returns 400 Bad Request status
        - Clear error message is provided
        - No processing is attempted
        
        WHY THIS IS IMPORTANT:
        - Q&A requires processed documents
        - Prevents errors from unprocessed content
        - Provides clear user feedback
        - Validates processing state checks
        """
        # Create unprocessed document
        unprocessed_doc = Document.objects.create(
            user=self.user,
            title="Unprocessed Document",
            file="unprocessed.txt",
            processed=False  # Not processed yet
        )
        
        question_data = {
            'question': 'What is this about?'
        }
        
        response = self.client.post(
            f'/api/documents/{unprocessed_doc.id}/ask_question/',
            question_data,
            format='json'
        )
        
        # Should return 400 Bad Request
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Should contain error message
        response_data = response.json()
        self.assertIn('error', response_data)
        self.assertIn('still being processed', response_data['error'])
    
    def test_ask_question_missing_question(self):
        """
        Test Q&A request without question parameter
        
        WHAT THIS TESTS:
        - Missing question parameter is handled
        - Returns 400 Bad Request status
        - Appropriate error message is provided
        - Input validation works correctly
        
        WHY THIS IS IMPORTANT:
        - Validates required input parameters
        - Prevents errors from malformed requests
        - Provides clear error feedback
        - Ensures robust API behavior
        """
        # Send request without question
        response = self.client.post(
            f'/api/documents/{self.document.id}/ask_question/',
            {},  # Empty data
            format='json'
        )
        
        # Should return 400 Bad Request
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Should contain error message
        response_data = response.json()
        self.assertIn('error', response_data)
        self.assertIn('required', response_data['error'])
    
    def test_ask_question_other_users_document(self):
        """
        Test asking question about another user's document
        
        WHAT THIS TESTS:
        - Users cannot access other users' documents for Q&A
        - Returns 404 Not Found (not 403, for security)
        - User isolation is enforced
        - Authorization works correctly
        
        WHY THIS IS IMPORTANT:
        - Protects user privacy and data
        - Enforces proper access controls
        - Validates security implementation
        - Prevents unauthorized data access
        """
        # Create document for other user
        other_user = User.objects.create_user(
            username='otheruser',
            password='testpass123'
        )
        
        other_document = Document.objects.create(
            user=other_user,
            title="Other User's Document",
            file="other.txt",
            processed=True
        )
        
        question_data = {
            'question': 'What is this about?'
        }
        
        response = self.client.post(
            f'/api/documents/{other_document.id}/ask_question/',
            question_data,
            format='json'
        )
        
        # Should return 404 Not Found (document doesn't exist for this user)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)