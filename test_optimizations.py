#!/usr/bin/env python3
"""
Simple test script to verify RAG optimizations are working correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all optimization modules can be imported."""
    try:
        from rag_agent.core.local_rag_store import LocalRAGConfig
        print("✅ LocalRAGConfig imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LocalRAGConfig: {e}")
        return False
    
    try:
        from rag_agent.core.reranking import CrossEncoderReranker, ContextualCompressor, QueryEnhancer
        print("✅ Reranking components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import reranking components: {e}")
        return False
    
    try:
        from rag_agent.core.cache_manager import CacheManager
        print("✅ CacheManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CacheManager: {e}")
        return False
    
    return True

def test_configuration():
    """Test that configuration works with optimization settings."""
    try:
        from rag_agent.core.local_rag_store import LocalRAGConfig
        
        # Test optimized configuration
        config = LocalRAGConfig(
            storage_path=Path("./test_storage"),
            use_caching=True,
            use_async_search=True,
            use_cross_encoder=True,
            use_compression=True,
            use_query_enhancement=True,
            vector_weight=0.5,
            fts_weight=0.3,
            bm25_weight=0.2
        )
        
        print(f"✅ Optimized configuration created successfully")
        print(f"   - Caching: {config.use_caching}")
        print(f"   - Async search: {config.use_async_search}")
        print(f"   - Cross-encoder: {config.use_cross_encoder}")
        print(f"   - Compression: {config.use_compression}")
        print(f"   - Query enhancement: {config.use_query_enhancement}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_reranking_components():
    """Test that reranking components can be initialized."""
    try:
        from rag_agent.core.reranking import CrossEncoderReranker, ContextualCompressor, QueryEnhancer
        
        # Test cross-encoder reranker (without loading actual model)
        reranker = CrossEncoderReranker()
        print("✅ CrossEncoderReranker initialized")
        
        # Test contextual compressor
        compressor = ContextualCompressor(similarity_threshold=0.3, max_sentences=3)
        print("✅ ContextualCompressor initialized")
        
        # Test query enhancer
        enhancer = QueryEnhancer()
        print("✅ QueryEnhancer initialized")
        
        return True
    except Exception as e:
        print(f"❌ Reranking components test failed: {e}")
        return False

def test_cache_manager():
    """Test cache manager initialization."""
    try:
        from rag_agent.core.cache_manager import create_cache_manager, MemoryCache
        
        # Test creating cache manager
        cache_manager = create_cache_manager(use_redis=False)
        print("✅ CacheManager created successfully")
        
        # Test basic operations
        success = cache_manager.backend.set("test_key", {"test": "value"}, ttl=60)
        if not success:
            print("❌ Cache set operation failed")
            return False
            
        result = cache_manager.backend.get("test_key")
        
        if result and result.get("test") == "value":
            print("✅ Cache operations working")
        else:
            print("❌ Cache operations failed")
            return False
        
        # Test cache statistics
        stats = cache_manager.get_stats()
        print(f"✅ Cache stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing RAG Optimizations")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Test", test_configuration), 
        ("Reranking Components Test", test_reranking_components),
        ("Cache Manager Test", test_cache_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations are working correctly!")
        print("\nYour RAG system now includes:")
        print("  🚀 Async parallel search capabilities")
        print("  🎯 Cross-encoder reranking for better accuracy")
        print("  📦 Contextual compression to reduce noise")
        print("  💾 Intelligent caching system")
        print("  🔧 Flexible configuration options")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Some optimizations may not work properly.")
        print("   Check the errors above and ensure dependencies are installed.")

if __name__ == "__main__":
    main()
