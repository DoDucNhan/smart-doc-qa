import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document as LangchainDocument
from .models import Document, DocumentChunk
from .databricks_service import LocalVectorStore
from .huggingface_service import LocalHuggingFaceService, HuggingFaceService
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, use_local_models=True):
        # Choose between local models (completely free) or HF API (free with limits)
        if use_local_models:
            self.hf_service = LocalHuggingFaceService()
        else:
            self.hf_service = HuggingFaceService()
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = LocalVectorStore()
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """Process uploaded document into chunks with embeddings"""
        try:
            file_path = document.file.path
            
            # Load document based on file type
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Load and split document
            docs = loader.load()
            chunks = self.text_splitter.split_documents(docs)
            
            # Generate embeddings using Hugging Face
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self.hf_service.get_embeddings(chunk_texts)
            
            # Create DocumentChunk objects and store in vector database
            document_chunks = []
            metadata_list = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_chunk = DocumentChunk.objects.create(
                    document=document,
                    content=chunk.page_content,
                    chunk_index=i,
                    embedding_id=f"{document.id}_{i}"
                )
                document_chunks.append(doc_chunk)
                
                # Prepare metadata for vector store
                metadata_list.append({
                    'document_id': document.id,
                    'chunk_id': doc_chunk.id,
                    'chunk_index': i,
                    'document_title': document.title,
                    'content': chunk.page_content
                })
            
            # Store embeddings in vector database
            self.vector_store.add_embeddings(embeddings, metadata_list)
            
            # Mark document as processed
            document.processed = True
            document.save()
            
            logger.info(f"Successfully processed document: {document.title}")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            document.processed = False
            document.save()
            raise
    
    def answer_question(self, question: str, document_id: int = None) -> str:
        """Answer question using RAG with Hugging Face models"""
        try:
            # Generate embedding for the question
            question_embedding = self.hf_service.get_embeddings([question])[0]
            
            # Search for relevant chunks
            results = self.vector_store.search(question_embedding, top_k=3)
            
            # Filter by document if specified
            if document_id:
                results = [r for r in results if r['metadata']['document_id'] == document_id]
            
            if not results:
                return "I couldn't find relevant information to answer your question."
            
            # Prepare context from top results
            context_parts = []
            for result in results:
                context_parts.append(result['metadata']['content'])
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using Hugging Face
            answer = self.hf_service.generate_answer(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "I encountered an error while processing your question. Please try again."