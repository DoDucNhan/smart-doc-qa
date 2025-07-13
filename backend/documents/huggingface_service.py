import os
import requests
from typing import List, Dict, Any
from decouple import config
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class HuggingFaceService:
    def __init__(self):
        self.api_key = config('HUGGINGFACE_API_KEY', default='')
        self.api_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # For embeddings, we'll use sentence-transformers locally (free)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded local embedding model successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers model"""
        if not self.embedding_model:
            raise Exception("Embedding model not loaded")
        
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Hugging Face API"""
        # Use a free text generation model
        model_name = "microsoft/DialoGPT-medium"
        
        # Format the prompt for better context understanding
        prompt = f"""Context: {context}

Question: {question}

Answer: """

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/{model_name}",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    return generated_text if generated_text else "I couldn't generate a proper answer."
                else:
                    return "I couldn't generate an answer from the context."
            else:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return self._fallback_answer(question, context)
                
        except requests.exceptions.Timeout:
            logger.error("HuggingFace API timeout")
            return self._fallback_answer(question, context)
        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {e}")
            return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question: str, context: str) -> str:
        """Fallback method when API fails - simple keyword matching"""
        question_words = question.lower().split()
        context_sentences = context.split('.')
        
        # Find sentences that contain question keywords
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words if len(word) > 2):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return f"Based on the document: {' '.join(relevant_sentences[:2])}"
        else:
            return "I found relevant content but couldn't generate a specific answer to your question."

# Alternative: Using Hugging Face Transformers locally (completely free)
class LocalHuggingFaceService:
    def __init__(self):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModel
            
            # Load embedding model locally
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load a small question-answering model locally
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
            
            logger.info("Loaded local models successfully")
        except Exception as e:
            logger.error(f"Failed to load local models: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using local QA model"""
        try:
            # Truncate context if too long (BERT models have token limits)
            max_context_length = 512
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            result = self.qa_pipeline(question=question, context=context)
            
            confidence = result.get('score', 0)
            answer = result.get('answer', '')
            
            if confidence > 0.1 and answer:
                return f"{answer}"
            else:
                return self._extract_relevant_text(question, context)
                
        except Exception as e:
            logger.error(f"Error in local QA: {e}")
            return self._extract_relevant_text(question, context)
    
    def _extract_relevant_text(self, question: str, context: str) -> str:
        """Extract relevant text when QA model confidence is low"""
        question_words = set(question.lower().split())
        sentences = context.split('.')
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((overlap, sentence.strip()))
        
        if scored_sentences:
            scored_sentences.sort(reverse=True)
            best_sentences = [s[1] for s in scored_sentences[:2] if s[1]]
            return f"Based on the document: {' '.join(best_sentences)}"
        else:
            return "I couldn't find specific information to answer your question in the document."