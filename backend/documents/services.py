import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document as LangchainDocument
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from .models import Document, DocumentChunk
from .databricks_service import LocalVectorStore

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = LocalVectorStore()
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """Process uploaded document into chunks with embeddings"""
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
        
        # Generate embeddings
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(chunk_texts)
        
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
        
        return document_chunks
    
    def answer_question(self, question: str, document_id: int = None) -> str:
        """Answer question using RAG with LLM"""
        # Generate embedding for the question
        question_embedding = self.embeddings.embed_query(question)
        
        # Search for relevant chunks
        results = self.vector_store.search(question_embedding, top_k=3)
        
        # Filter by document if specified
        if document_id:
            results = [r for r in results if r['metadata']['document_id'] == document_id]
        
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context from top results
        context_docs = []
        for result in results:
            context_docs.append(LangchainDocument(
                page_content=result['metadata']['content'],
                metadata={'source': result['metadata']['document_title']}
            ))
        
        # Use LLM to generate answer
        qa_chain = load_qa_chain(self.llm, chain_type="stuff")
        response = qa_chain.run(input_documents=context_docs, question=question)
        
        return response