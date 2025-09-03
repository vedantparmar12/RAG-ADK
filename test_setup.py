import sys
import importlib

def test_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import rag_agent.config
        print("✓ Config module loaded")
    except Exception as e:
        print(f"✗ Config module failed: {e}")
        return False
        
    try:
        import rag_agent.agent
        print("✓ Original agent module loaded")
    except Exception as e:
        print(f"✗ Original agent module failed: {e}")
        
    return True

def test_adk_compatibility():
    """Test ADK compatibility"""
    print("\nTesting ADK compatibility...")
    
    try:
        from google.adk import Agent
        from google.adk.agents import LlmAgent
        print("✓ ADK imports successful")
        
        # Test creating a simple agent
        test_agent = Agent(
            name="TestAgent",
            model="gemini-2.0-flash",
            description="Test agent"
        )
        print("✓ Agent creation successful")
        
    except Exception as e:
        print(f"✗ ADK test failed: {e}")
        return False
        
    return True

def test_fastapi():
    """Test FastAPI setup"""
    print("\nTesting FastAPI setup...")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        print("✓ FastAPI imports successful")
        
        # Test creating app
        app = FastAPI(title="Test")
        print("✓ FastAPI app creation successful")
        
    except Exception as e:
        print(f"✗ FastAPI test failed: {e}")
        return False
        
    return True

def test_original_tools():
    """Test original RAG tools"""
    print("\nTesting original RAG tools...")
    
    try:
        from rag_agent.tools.list_corpora import list_corpora
        from rag_agent.tools.rag_query import rag_query
        print("✓ Original tools import successful")
        
    except Exception as e:
        print(f"✗ Original tools test failed: {e}")
        return False
        
    return True

def main():
    print("=" * 50)
    print("Enhanced Vertex AI RAG Agent - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("ADK Compatibility", test_adk_compatibility),
        ("FastAPI Setup", test_fastapi),
        ("Original Tools", test_original_tools)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! The system is ready.")
        print("\nTo start the enhanced RAG agent:")
        print("1. With ADK: adk web")
        print("2. With FastAPI: python3 main.py")
    else:
        print("\n Some tests failed. Please check the errors above.")
        print("\nNote: Some advanced features may require additional dependencies:")
        print("- sentence-transformers (for embeddings)")
        print("- faiss-cpu (for vector search)")
        print("- torch (for neural models)")
        print("- redis (for caching)")

if __name__ == "__main__":
    main()
