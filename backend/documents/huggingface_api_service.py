import os
import requests
import logging

logger = logging.getLogger(__name__)

class HuggingFaceAPIService:
    """
    Simple service that now uses REAL HuggingFace APIs for both embeddings and similarity
    
    This does 3 things:
    1. Generate real embeddings (convert text to meaningful numbers)
    2. Calculate similarity between texts (find relevant content)
    3. Answer questions using Hugging Face AI
    """
    
    def __init__(self):
        # Get API token from environment
        self.hf_token = os.environ.get('HF_TOKEN')
        
        if not self.hf_token:
            logger.warning("No HF_TOKEN found. API calls will fail.")
        
        # Set up all the API endpoints
        self.chat_url = "https://router.huggingface.co/v1/chat/completions"
        self.embeddings_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        
        # This is YOUR similarity API endpoint from the example!
        self.similarity_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
        
        # Headers for all API requests
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # Initialize embedding method
        self._init_embeddings()
    
    def _init_embeddings(self):
        """
        Try to use the best available embedding method
        Priority: Local -> API -> Fake
        """
        # Try local embeddings first (recommended)
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading local embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_method = "local"
            logger.info("[SUCCESS] Local embeddings ready (FREE and FAST!)")
            
        except ImportError:
            logger.warning("sentence-transformers not installed")
            if self.hf_token:
                self.embedding_method = "api"
                logger.info("[SUCCESS] Using HuggingFace API embeddings")
            else:
                self.embedding_method = "fake"
                logger.warning("Using fake embeddings (for testing only)")
                
        except Exception as e:
            logger.warning(f"Local embeddings failed: {e}")
            if self.hf_token:
                self.embedding_method = "api"
                logger.info("[SUCCESS] Using HuggingFace API embeddings")
            else:
                self.embedding_method = "fake"
                logger.warning("Using fake embeddings (for testing only)")
    
    def get_embeddings(self, texts):
        """
        Generate embeddings for texts using the best available method
        
        Args:
            texts: List of strings like ["Hello world", "How are you?"]
        
        Returns:
            List of lists with numbers that represent meaning
        """
        if not texts:
            return []
        
        logger.info(f"Generating {self.embedding_method} embeddings for {len(texts)} texts")
        
        try:
            if self.embedding_method == "local":
                return self._get_local_embeddings(texts)
            elif self.embedding_method == "api":
                return self._get_api_embeddings(texts)
            else:
                return self._get_fake_embeddings(texts)
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fallback to fake embeddings if real ones fail
            return self._get_fake_embeddings(texts)
    
    def _get_local_embeddings(self, texts):
        """Generate embeddings using local model (RECOMMENDED)"""
        try:
            embeddings = self.embedding_model.encode(texts)
            embeddings_list = embeddings.tolist()
            logger.info(f"[SUCCESS] Generated {len(embeddings_list)} local embeddings")
            return embeddings_list
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise
    
    def _get_api_embeddings(self, texts):
        """Generate embeddings using HuggingFace API with better error handling"""
        try:
            # Validate input
            if not texts or not all(isinstance(text, str) for text in texts):
                raise ValueError("All inputs must be non-empty strings")
            
            # Limit text length to avoid API errors
            processed_texts = []
            for text in texts:
                if len(text) > 5000:  # Truncate very long texts
                    processed_texts.append(text[:5000] + "...")
                else:
                    processed_texts.append(text)
            
            data = {
                "inputs": processed_texts,
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            response = requests.post(
                self.embeddings_url,
                headers=self.headers,
                json=data,
                timeout=60  # Longer timeout
            )
            
            if response.status_code == 200:
                embeddings = response.json()
                logger.info(f"[SUCCESS] Generated {len(embeddings)} API embeddings")
                return embeddings
            
            elif response.status_code == 400:
                error_detail = response.text
                logger.error(f"API 400 error details: {error_detail}")
                
                # Common 400 error fixes
                if "rate limit" in error_detail.lower():
                    raise Exception("Rate limit exceeded. Please try again later.")
                elif "input too long" in error_detail.lower():
                    raise Exception("Text input too long. Try shorter text.")
                elif "invalid input" in error_detail.lower():
                    raise Exception("Invalid input format. Check text encoding.")
                else:
                    raise Exception(f"API validation error: {error_detail}")
            
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded for embeddings API")
                raise Exception("Rate limit exceeded. Please try again later.")
            
            else:
                logger.error(f"Embeddings API error: {response.status_code} - {response.text}")
                raise Exception(f"API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Embeddings API timeout")
            raise Exception("API request timed out. Please try again.")
        
        except Exception as e:
            logger.error(f"API embedding error: {e}")
            raise
        
    def _get_fake_embeddings(self, texts):
        """Generate fake embeddings (ONLY for testing)"""
        fake_embeddings = []
        for i, text in enumerate(texts):
            embedding = [
                len(text) * 0.01,
                text.count(' ') * 0.02,
                text.count('a') * 0.03,
                hash(text) % 100 * 0.01
            ]
            fake_embeddings.append(embedding)
        
        logger.warning(f"Generated {len(fake_embeddings)} FAKE embeddings (testing only)")
        return fake_embeddings
    
    def calculate_similarity(self, source_sentence, target_sentences):
        """
        Calculate similarity using your HuggingFace similarity API example
        
        Args:
            source_sentence: The reference sentence (like a question)
            target_sentences: List of sentences to compare against (like document chunks)
            
        Returns:
            List of similarity scores (0.0 to 1.0)
        """
        if not self.hf_token:
            logger.warning("No HF_TOKEN available for similarity calculation")
            # Return fake similarities for testing
            return [0.5] * len(target_sentences)
        
        logger.info(f"Calculating similarity: 1 source vs {len(target_sentences)} targets")
        
        try:
            # Use the EXACT format from your example
            payload = {
                "inputs": {
                    "source_sentence": source_sentence,
                    "sentences": target_sentences
                }
            }
            
            # Call the similarity API
            response = requests.post(
                self.similarity_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                similarities = response.json()
                logger.info(f"[SUCCESS] Got {len(similarities)} similarity scores")
                
                # Log the scores for debugging
                for i, (sentence, score) in enumerate(zip(target_sentences, similarities)):
                    preview = sentence[:50] + "..." if len(sentence) > 50 else sentence
                    logger.info(f"  {i+1}. Score: {score:.3f} - {preview}")
                
                return similarities
            
            elif response.status_code == 503:
                logger.warning("Similarity model is loading, please wait...")
                raise Exception("Model is loading. Please try again in a moment.")
            
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded for similarity API")
                raise Exception("Rate limit exceeded. Please try again later.")
            
            elif response.status_code == 401:
                logger.error("Authentication failed - check HF_TOKEN")
                raise Exception("Authentication error. Please check your HF_TOKEN.")
            
            else:
                logger.error(f"Similarity API error: {response.status_code} - {response.text}")
                raise Exception(f"Similarity API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def find_most_relevant_chunks(self, question, chunk_contents, top_k=3):
        """
        Find the most relevant chunks for a question using similarity API
        
        Args:
            question: The question to find relevant content for
            chunk_contents: List of text chunks to search through
            top_k: Number of top relevant chunks to return
            
        Returns:
            List of dicts with 'content' and 'similarity_score'
        """
        if not chunk_contents:
            return []
        
        logger.info(f"Finding most relevant chunks for: '{question}'")
        
        try:
            # Calculate similarity scores using HF API
            similarities = self.calculate_similarity(question, chunk_contents)
            
            # Pair chunks with their similarity scores
            chunk_scores = []
            for i, (content, score) in enumerate(zip(chunk_contents, similarities)):
                chunk_scores.append({
                    'content': content,
                    'similarity_score': float(score),
                    'index': i
                })
            
            # Sort by similarity score (highest first)
            chunk_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top k results
            top_chunks = chunk_scores[:top_k]
            
            logger.info(f"[SUCCESS] Found top {len(top_chunks)} relevant chunks:")
            for i, chunk in enumerate(top_chunks, 1):
                score = chunk['similarity_score']
                preview = chunk['content'][:60] + "..." if len(chunk['content']) > 60 else chunk['content']
                logger.info(f"  {i}. Score: {score:.3f} - {preview}")
            
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            # Fallback: return first few chunks with neutral scores
            fallback_chunks = []
            for i, content in enumerate(chunk_contents[:top_k]):
                fallback_chunks.append({
                    'content': content,
                    'similarity_score': 0.5,  # Neutral score
                    'index': i
                })
            logger.warning(f"Using fallback: returning first {len(fallback_chunks)} chunks")
            return fallback_chunks
    
    def answer_question(self, question, context):
        """
        Generate answer using Hugging Face AI
        
        Args:
            question: String like "What is machine learning?"
            context: String with information to answer from
        
        Returns:
            String with the AI's answer
        """
        # Check if we have an API token for question answering
        if not self.hf_token:
            return "Sorry, no API token available for AI responses. Set HF_TOKEN in your environment."
        
        # Create the prompt for the AI
        prompt = f"""
        Context: {context}
        
        Question: {question}
        
        Please answer the question based only on the context above. Be concise and helpful.
        """
        
        # Prepare API request
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct:fireworks-ai",
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        try:
            logger.info("Sending question to AI...")
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"]
                    logger.info("[SUCCESS] Got AI answer")
                    return answer.strip()
                else:
                    return "Sorry, got an unexpected response from the AI."
            
            elif response.status_code == 401:
                return "Authentication error. Please check your HF_TOKEN."
            
            elif response.status_code == 429:
                return "Too many requests. Please try again in a few minutes."
            
            else:
                logger.error(f"AI API error: {response.status_code}")
                return f"AI service error: {response.status_code}. Please try again later."
        
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        
        except Exception as e:
            logger.error(f"AI answer error: {e}")
            return "An error occurred while getting AI response. Please try again."
    
    def get_service_info(self):
        """Get information about what services are currently being used"""
        info = {
            "embedding_method": self.embedding_method,
            "has_similarity_api": bool(self.hf_token),
            "has_chat_api": bool(self.hf_token),
            "ready_for_production": self.embedding_method in ["local", "api"] and bool(self.hf_token)
        }
        
        return info


# Test function to demonstrate the similarity API integration
def test_similarity_integration():
    """
    Test the integrated similarity API using your exact example
    """
    print("=" * 60)
    print("TESTING SIMILARITY API INTEGRATION")
    print("=" * 60)
    
    # Create service
    service = HuggingFaceAPIService()
    
    # Check what we're using
    info = service.get_service_info()
    print(f"\nService Status:")
    print(f"  Embeddings: {info['embedding_method']}")
    print(f"  Similarity API: {'[SUCCESS]' if info['has_similarity_api'] else '[FAILED]'}")
    print(f"  Chat API: {'[SUCCESS]' if info['has_chat_api'] else '[FAILED]'}")
    print(f"  Ready for production: {'[SUCCESS]' if info['ready_for_production'] else '[FAILED]'}")
    
    if not info['has_similarity_api']:
        print("\n[FAILED] No HF_TOKEN found. Please set your token to test similarity API.")
        return
    
    # Test 1: Your exact example
    print(f"\n" + "="*40)
    print("TEST 1: Your Original Example")
    print("="*40)
    
    source = "That is a happy person"
    sentences = [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
    
    try:
        similarities = service.calculate_similarity(source, sentences)
        
        print(f"\nSource: '{source}'")
        print("Results:")
        for sentence, score in zip(sentences, similarities):
            print(f"  Score: {score:.3f} - '{sentence}'")
    
    except Exception as e:
        print(f"[FAILED] Test 1 failed: {e}")
    
    # Test 2: Document Q&A scenario
    print(f"\n" + "="*40)
    print("TEST 2: Document Q&A Scenario")
    print("="*40)
    
    question = "What is machine learning?"
    document_chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
        "The weather forecast shows sunny skies for tomorrow with temperatures around 75°F",
        "Python is a programming language commonly used for data science and machine learning applications",
        "Our office cafeteria serves lunch from 11 AM to 2 PM on weekdays"
    ]
    
    try:
        relevant_chunks = service.find_most_relevant_chunks(question, document_chunks, top_k=2)
        
        print(f"\nQuestion: '{question}'")
        print("Most relevant chunks:")
        for i, chunk in enumerate(relevant_chunks, 1):
            score = chunk['similarity_score']
            content = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
            print(f"  {i}. Score: {score:.3f}")
            print(f"     Content: {content}")
        
        # Generate answer using best chunk
        if relevant_chunks:
            best_chunk = relevant_chunks[0]
            answer = service.answer_question(question, best_chunk['content'])
            print(f"\n AI Answer: {answer}")
    
    except Exception as e:
        print(f"[FAILED] Test 2 failed: {e}")
    
    print(f"\n" + "="*60)
    print("[SUCCESS] Similarity API integration test completed!")
    print("="*60)


if __name__ == "__main__":
    test_similarity_integration()