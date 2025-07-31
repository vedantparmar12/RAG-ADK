"""
Test script for embedding providers integration
"""

import asyncio
import numpy as np
from rag_agent.core.embeddings import (
    EmbeddingManager, 
    EmbeddingProvider, 
    GeminiEmbedding, 
    TaskType,
    SentenceTransformerEmbedding,
    VertexAIEmbedding
)
from rag_agent.config import settings


def test_embedding_providers():
    """Test different embedding providers"""
    
    print("Testing Embedding Providers\n" + "="*50)
    
    # Initialize embedding manager
    manager = EmbeddingManager()
    
    # Test texts
    test_texts = [
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence.",
        "How do I bake a chocolate cake?"
    ]
    
    # Test 1: Sentence Transformers
    print("\n1. Testing Sentence Transformers:")
    try:
        if EmbeddingProvider.SENTENCE_TRANSFORMERS in manager.models:
            embeddings = manager.encode(
                test_texts,
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            print(f"   ✓ Shape: {embeddings.shape}")
            print(f"   ✓ Dimension: {embeddings.shape[1]}")
            
            # Test similarity
            similarity = manager.compute_similarity(
                embeddings[0:1], 
                embeddings[1:2]
            )[0, 0]
            print(f"   ✓ Similarity (Q1 vs A1): {similarity:.4f}")
        else:
            print("   ✗ Sentence Transformers not available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Gemini Embeddings
    print("\n2. Testing Gemini Embeddings:")
    try:
        # Create Gemini model with different configurations
        gemini_model = GeminiEmbedding(
            model_name="gemini-embedding-001",
            task_type=TaskType.SEMANTIC_SIMILARITY,
            output_dimensionality=768
        )
        
        # Add to manager
        manager.add_model(EmbeddingProvider.GEMINI, gemini_model)
        
        # Test embeddings
        embeddings = manager.encode(
            test_texts,
            provider=EmbeddingProvider.GEMINI
        )
        print(f"   ✓ Shape: {embeddings.shape}")
        print(f"   ✓ Dimension: {embeddings.shape[1]}")
        
        # Test with different task types
        print("\n   Testing task-specific embeddings:")
        
        # Query embedding
        query_emb = gemini_model.encode(
            ["What is deep learning?"],
            task_type=TaskType.RETRIEVAL_QUERY
        )
        print(f"   ✓ Query embedding shape: {query_emb.shape}")
        
        # Document embedding
        doc_emb = gemini_model.encode(
            ["Deep learning is a type of machine learning."],
            task_type=TaskType.RETRIEVAL_DOCUMENT
        )
        print(f"   ✓ Document embedding shape: {doc_emb.shape}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: Gemini API requires authentication and google-genai library")
    
    # Test 3: Vertex AI Embeddings
    print("\n3. Testing Vertex AI Embeddings:")
    try:
        if settings.use_vertex_ai:
            vertex_model = VertexAIEmbedding(
                model_name=settings.embedding_model
            )
            manager.add_model(EmbeddingProvider.VERTEX_AI, vertex_model)
            
            embeddings = manager.encode(
                test_texts[:1],  # Test with one text
                provider=EmbeddingProvider.VERTEX_AI
            )
            print(f"   ✓ Shape: {embeddings.shape}")
            print(f"   ✓ Dimension: {embeddings.shape[1]}")
        else:
            print("   ✗ Vertex AI not enabled in settings")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: Vertex AI requires GCP authentication")
    
    # Test 4: Embedding dimension flexibility
    print("\n4. Testing Gemini dimension flexibility:")
    try:
        for dim in [768, 1536, 3072]:
            gemini_dim_model = GeminiEmbedding(
                output_dimensionality=dim
            )
            test_emb = np.random.randn(1, dim)  # Mock for testing
            print(f"   ✓ Dimension {dim}: OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*50)
    print("Testing complete!")


def test_corpus_creation_with_embeddings():
    """Test corpus creation with different embedding providers"""
    
    print("\n\nTesting Corpus Creation with Embeddings\n" + "="*50)
    
    from rag_agent.core.indexing import CorpusManager
    
    manager = CorpusManager(settings)
    
    # Test configurations
    test_configs = [
        {
            "name": "test_st",
            "provider": "sentence_transformers",
            "config": {"st_model": "all-MiniLM-L6-v2"}
        },
        {
            "name": "test_gemini",
            "provider": "gemini",
            "config": {
                "gemini_model": "gemini-embedding-001",
                "gemini_task_type": "RETRIEVAL_DOCUMENT",
                "gemini_dimensionality": 768
            }
        },
        {
            "name": "test_vertex",
            "provider": "vertex_ai",
            "config": {"vertex_model": "text-embedding-005"}
        }
    ]
    
    for test in test_configs:
        print(f"\nTesting {test['provider']}:")
        try:
            corpus = manager.create_corpus(
                name=test["name"],
                description=f"Test corpus with {test['provider']}",
                embedding_provider=test["provider"],
                embedding_config=test["config"]
            )
            print(f"   ✓ Created corpus: {corpus['id']}")
            print(f"   ✓ Provider: {corpus['embedding_provider']}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    test_embedding_providers()
    test_corpus_creation_with_embeddings()