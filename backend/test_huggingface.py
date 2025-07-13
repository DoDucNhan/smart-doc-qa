import os
import django
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'docqa_backend.settings')
django.setup()

from documents.huggingface_service import LocalHuggingFaceService

def test_huggingface():
    print("Testing Hugging Face integration...")
    
    try:
        # Initialize service
        hf_service = LocalHuggingFaceService()
        
        # Test embeddings
        texts = ["This is a test document about AI.", "Machine learning is fascinating."]
        embeddings = hf_service.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        
        # Test question answering
        context = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        question = "What is machine learning?"
        answer = hf_service.generate_answer(question, context)
        print(f"Q: {question}")
        print(f"A: {answer}")
        
        print("✅ Hugging Face integration test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_huggingface()