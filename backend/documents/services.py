import os
import logging
from typing import List
from .models import Document, DocumentChunk
from .simple_api_service import HuggingFaceAPIService

logger = logging.getLogger(__name__)


class SimpleDocumentProcessor:
    """
    Simple document processor that now uses the enhanced HuggingFaceAPIService
    with integrated similarity search capabilities
    """
    
    def __init__(self):
        # Create our enhanced API service
        self.api_service = HuggingFaceAPIService()
        
        # Check what capabilities we have
        info = self.api_service.get_service_info()
        logger.info(f"DocumentProcessor initialized:")
        logger.info(f"  Embeddings: {info['embedding_method']}")
        logger.info(f"  Similarity API: {'Available' if info['has_similarity_api'] else 'Not available'}")
        logger.info(f"  Production ready: {'Yes' if info['ready_for_production'] else 'No'}")
    
    def process_document(self, document: Document):
        """
        Process a document: read it, break it into chunks, and prepare for Q&A
        
        Args:
            document: A Document model instance from the database
        
        Returns:
            List of DocumentChunk objects created
        """
        logger.info(f"Starting to process document: {document.title}")
        
        try:
            # Step 1: Read the file content
            file_path = document.file.path
            content = self._read_file(file_path)
            logger.info(f"Read {len(content)} characters from file")
            
            # Step 2: Break content into smaller chunks
            chunks = self._split_into_chunks(content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Step 3: Generate embeddings (optional, for future use)
            try:
                embeddings = self.api_service.get_embeddings(chunks)
                logger.info(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                # Continue without embeddings - similarity search will still work
                embeddings = [None] * len(chunks)
            
            # Step 4: Save chunks to database
            created_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Create a DocumentChunk in the database
                chunk = DocumentChunk.objects.create(
                    document=document,
                    content=chunk_text,
                    chunk_index=i,
                    embedding_id=f"{document.id}_{i}"
                )
                created_chunks.append(chunk)
                logger.info(f"Saved chunk {i+1}/{len(chunks)}")
            
            # Step 5: Mark document as processed
            document.processed = True
            document.save()
            
            logger.info(f"✅ Successfully processed document: {document.title}")
            return created_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            # Mark as not processed if something went wrong
            document.processed = False
            document.save()
            raise e
    
    def _read_file(self, file_path):
        """
        Read content from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            String with file content
        """
        logger.info(f"Reading file: {os.path.basename(file_path)}")
        
        # Check file type and read accordingly
        if file_path.endswith('.txt'):
            # For text files, just read the content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        
        elif file_path.endswith('.pdf'):
            # For PDF files, we need a PDF reader
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                
                # Extract text from all pages
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.info(f"Extracted text from page {page_num + 1}")
                
                return text
                
            except ImportError:
                raise Exception("PyPDF2 not installed. Run: pip install PyPDF2")
            except Exception as e:
                raise Exception(f"Error reading PDF: {e}")
        
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    def _split_into_chunks(self, content, chunk_size=800):
        """
        Split content into smaller chunks for better processing
        
        Args:
            content: The full text content
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split content into sentences first
        sentences = content.split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would make chunk too big, start a new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
            else:
                current_chunk += sentence + "."
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks (less than 50 characters)
        meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        logger.info(f"Created {len(meaningful_chunks)} meaningful chunks")
        return meaningful_chunks
    
    def answer_question(self, question, document_id=None):
        """
        Answer a question using the enhanced similarity search
        
        This is the NEW IMPROVED version that uses HuggingFace similarity API!
        
        Args:
            question: The question to answer
            document_id: If provided, only search in this document
            
        Returns:
            String with the answer
        """
        logger.info(f"Answering question: '{question}'")
        
        try:
            # Step 1: Get relevant document chunks from database
            if document_id:
                # Search only in specific document
                chunks = DocumentChunk.objects.filter(
                    document_id=document_id,
                    document__processed=True
                ).order_by('chunk_index')
                logger.info(f"Searching in document {document_id}")
            else:
                # Search in all processed documents
                chunks = DocumentChunk.objects.filter(
                    document__processed=True
                ).order_by('document_id', 'chunk_index')
                logger.info("Searching across all documents")
            
            if not chunks.exists():
                return "No processed documents found to answer your question."
            
            # Step 2: Get chunk contents for similarity search
            chunk_contents = [chunk.content for chunk in chunks]
            logger.info(f"Comparing question against {len(chunk_contents)} chunks")
            
            # Step 3: Use the enhanced API service to find most relevant chunks
            try:
                # This is where the magic happens! Uses your HF similarity API
                relevant_chunks = self.api_service.find_most_relevant_chunks(
                    question, 
                    chunk_contents, 
                    top_k=3
                )
                
                if not relevant_chunks:
                    return "I couldn't find any relevant content to answer your question."
                
                # Log what we found for debugging
                logger.info("Found relevant chunks:")
                for i, chunk in enumerate(relevant_chunks, 1):
                    score = chunk['similarity_score']
                    preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                    logger.info(f"  {i}. Score: {score:.3f} - {preview}")
                
                # Filter chunks with good similarity scores
                good_chunks = [chunk for chunk in relevant_chunks if chunk['similarity_score'] > 0.3]
                
                if not good_chunks:
                    return "I found some content but it doesn't seem very relevant to your question. Try rephrasing your question or check if the document contains information about this topic."
                
            except Exception as e:
                logger.warning(f"Similarity search failed: {e}")
                # Fallback to simple keyword matching
                good_chunks = self._simple_keyword_fallback(question, chunk_contents)
            
            # Step 4: Prepare context from the best chunks
            context_parts = []
            for chunk in good_chunks:
                context_parts.append(chunk['content'])
            
            context = "\n\n".join(context_parts)
            
            # Limit context length for API efficiency
            max_context_length = 2000
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
                logger.info(f"Truncated context to {max_context_length} characters")
            
            # Step 5: Generate answer using AI
            logger.info("Generating AI answer...")
            answer = self.api_service.answer_question(question, context)
            
            # Add similarity info for debugging (optional)
            if good_chunks:
                best_score = good_chunks[0]['similarity_score']
                if best_score > 0.8:
                    logger.info(f"Excellent match found (score: {best_score:.3f})")
                elif best_score > 0.6:
                    logger.info(f"Good match found (score: {best_score:.3f})")
                else:
                    logger.info(f"Moderate match found (score: {best_score:.3f})")
            
            logger.info("✅ Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Sorry, I encountered an error while trying to answer your question. Please try again."
    
    def _simple_keyword_fallback(self, question, chunk_contents):
        """
        Fallback search method when similarity API fails
        """
        logger.info("Using keyword fallback search")
        
        question_words = set(word.lower().strip('.,!?') for word in question.split() if len(word) > 2)
        
        scored_chunks = []
        for i, content in enumerate(chunk_contents):
            content_words = set(word.lower().strip('.,!?') for word in content.split())
            overlap = len(question_words.intersection(content_words))
            
            if overlap > 0:
                score = overlap / len(question_words)  # Normalize by question length
                scored_chunks.append({
                    'content': content,
                    'similarity_score': score,
                    'index': i
                })
        
        # Sort by score and return top 3
        scored_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_chunks[:3]
    
    def get_document_summary(self, document_id):
        """
        Get a summary of what's in a document using similarity search
        """
        try:
            document = Document.objects.get(id=document_id)
            chunks = document.chunks.all()
            
            if not chunks.exists():
                return "Document not found or not processed yet."
            
            # Get all chunk contents
            chunk_contents = [chunk.content for chunk in chunks]
            
            # Find chunks that are similar to "summary" or "overview"
            summary_queries = ["summary of this document", "what is this document about", "main topics"]
            
            all_relevant_chunks = []
            for query in summary_queries:
                try:
                    relevant = self.api_service.find_most_relevant_chunks(query, chunk_contents, top_k=2)
                    all_relevant_chunks.extend(relevant)
                except:
                    pass  # Skip if similarity search fails
            
            # Remove duplicates and get best chunks
            seen_content = set()
            unique_chunks = []
            for chunk in all_relevant_chunks:
                if chunk['content'] not in seen_content:
                    seen_content.add(chunk['content'])
                    unique_chunks.append(chunk)
            
            # Sort by score and take best ones
            unique_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            best_chunks = unique_chunks[:3]
            
            if best_chunks:
                summary_context = "\n\n".join([chunk['content'] for chunk in best_chunks])
                summary = self.api_service.answer_question(
                    "Provide a brief summary of this document", 
                    summary_context
                )
                return summary
            else:
                # Fallback: return first chunk
                return f"Document summary: {chunk_contents[0][:200]}..."
                
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return "Could not generate document summary."


# Test function to demonstrate the complete workflow
def test_complete_workflow():
    """
    Test the complete document processing and Q&A workflow
    """
    print("=" * 60)
    print("TESTING COMPLETE DOCUMENT Q&A WORKFLOW")
    print("=" * 60)
    
    # Create processor
    processor = SimpleDocumentProcessor()
    
    # Test with simulated document content
    sample_content = """
    Machine Learning Overview
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Types of Machine Learning
    
    There are several types of machine learning algorithms:
    
    1. Supervised Learning: Uses labeled training data to learn a mapping function from input variables to output variables.
    
    2. Unsupervised Learning: Finds hidden patterns or structures in data without labeled examples.
    
    3. Reinforcement Learning: Learns through interaction with an environment to maximize cumulative reward.
    
    Applications
    
    Machine learning has many practical applications including image recognition, natural language processing, recommendation systems, fraud detection, and autonomous vehicles.
    
    Python Programming
    
    Python is a high-level programming language that's widely used for machine learning because of libraries like scikit-learn, TensorFlow, and PyTorch. Python's simple syntax makes it accessible for beginners while being powerful enough for complex applications.
    """
    
    # Test chunking
    print("\n1. Testing document chunking...")
    chunks = processor._split_into_chunks(sample_content)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"  {i}. {preview}")
    
    # Test questions with similarity search
    test_questions = [
        "What is machine learning?",
        "What types of machine learning are there?", 
        "What programming language is good for ML?",
        "What are some applications of machine learning?",
        "How does supervised learning work?"
    ]
    
    print(f"\n2. Testing similarity-based Q&A...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        try:
            # Find relevant chunks using similarity
            relevant_chunks = processor.api_service.find_most_relevant_chunks(
                question, chunks, top_k=2
            )
            
            print("Most relevant chunks:")
            for j, chunk in enumerate(relevant_chunks, 1):
                score = chunk['similarity_score']
                content_preview = chunk['content'][:100] + "..."
                print(f"  {j}. Score: {score:.3f} - {content_preview}")
            
            # Generate answer
            if relevant_chunks:
                context = relevant_chunks[0]['content']  # Use best match
                answer = processor.api_service.answer_question(question, context)
                print(f"\n🤖 Answer: {answer}")
                
                # Show quality assessment
                best_score = relevant_chunks[0]['similarity_score']
                if best_score > 0.7:
                    print("✅ Excellent match - high quality answer expected")
                elif best_score > 0.5:
                    print("✓ Good match - relevant answer expected")
                else:
                    print("⚠️ Moderate match - answer may be less precise")
        
        except Exception as e:
            print(f"❌ Question {i} failed: {e}")
    
    print(f"\n" + "="*60)
    print("✅ Complete workflow test finished!")
    print("="*60)
    
    # Show service status
    info = processor.api_service.get_service_info()
    print(f"\nService Status Summary:")
    print(f"  Embedding method: {info['embedding_method']}")
    print(f"  Similarity API available: {'Yes' if info['has_similarity_api'] else 'No'}")
    print(f"  Ready for production: {'Yes' if info['ready_for_production'] else 'No'}")


if __name__ == "__main__":
    test_complete_workflow()