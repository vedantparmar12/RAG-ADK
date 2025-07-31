# Context Caching Guide

This guide explains how to use Gemini API's context caching feature to optimize costs and performance in the Enhanced RAG Agent.

## Overview

Context caching allows you to cache large amounts of content (documents, system prompts, embeddings) with the Gemini API and reuse them across multiple requests. This significantly reduces costs and improves response times.

## Key Benefits

1. **Cost Savings**: Cached tokens cost ~75% less than regular input tokens
2. **Performance**: Faster response times by avoiding repeated token processing
3. **Scalability**: Handle larger document sets efficiently
4. **Rate Limit Optimization**: Cached content doesn't count against rate limits

## How It Works

### Automatic Caching

The system automatically creates context caches in these scenarios:

1. **Document Indexing**: When adding documents to a corpus
2. **Query Processing**: For frequently accessed retrieved chunks
3. **System Prompts**: For repeated instructions and contexts

### Cache Lifecycle

```
Document Upload → Chunk Processing → Cache Creation → Multiple Queries → Cache Expiration
     (1x cost)        (processing)     (1x storage)    (0.25x cost)      (auto cleanup)
```

## Configuration

### Environment Variables

```bash
# Enable/disable context caching
ENABLE_CONTEXT_CACHING=true

# Default cache duration (seconds)
CONTEXT_CACHE_TTL=3600  # 1 hour

# Minimum tokens required to create cache
MIN_TOKENS_FOR_CACHE=1024

# Auto-cache corpus documents
CACHE_CORPUS_DOCUMENTS=true

# Cache query contexts
CACHE_QUERY_CONTEXT=true
```

### UI Configuration

When creating a corpus:

1. Enable "Context Caching" checkbox
2. Set cache duration (1-24 hours)
3. Documents will be cached automatically during indexing

## Usage Patterns

### 1. Chatbot with Large Knowledge Base

```python
# Documents are cached when added to corpus
POST /corpus
{
    "name": "support-docs",
    "enable_context_cache": true,
    "cache_ttl_seconds": 14400  # 4 hours
}

# Subsequent queries use cached context
POST /query
{
    "query": "How do I reset my password?",
    "corpus_id": "support-docs"
}
# First query: Full cost
# Subsequent queries: 75% discount on cached tokens
```

### 2. Document Analysis Pipeline

```python
# Cache large documents for repeated analysis
POST /corpus/analysis-corpus/documents
{
    "uris": ["gs://bucket/large-report.pdf"],
    "create_context_cache": true
}

# Multiple analysis queries on same document
# Each query benefits from cached context
```

### 3. System Prompt Caching

```python
# Cache frequently used system instructions
{
    "system_instruction": "You are an expert analyst...",
    "ttl_seconds": 86400  # 24 hours
}
```

## Cost Analysis

### Example Calculation

For a 100,000 token document accessed 10 times per hour:

**Without Caching:**
- Cost per request: 100,000 × $0.00001 = $1.00
- Total cost (10 requests): $10.00

**With Caching (1-hour TTL):**
- Initial cache: 100,000 × $0.00001 = $1.00
- Storage (1 hour): 100,000 × $0.000001 = $0.10
- Subsequent requests (9): 100,000 × $0.0000025 × 9 = $2.25
- **Total cost: $3.35 (66.5% savings)**

### Break-Even Analysis

Caching becomes cost-effective when:
- Content is accessed ≥ 2 times within TTL period
- Content size is ≥ 1,024 tokens
- Access pattern is predictable

## Monitoring

### UI Dashboard

Navigate to "🗄️ Context Cache" to view:
- Active caches and token counts
- Expiring caches with extension options
- Cost savings estimates
- Cache utilization metrics

### API Endpoints

```bash
# Get cache statistics
GET /context-cache/stats

# Update cache TTL
POST /context-cache/{cache_id}/update-ttl
{
    "ttl_seconds": 7200
}

# Delete cache
DELETE /context-cache/{cache_id}
```

## Best Practices

### 1. Optimal TTL Selection

| Use Case | Recommended TTL | Rationale |
|----------|-----------------|-----------|
| Active chat sessions | 5-15 minutes | Short-lived, high frequency |
| Document analysis | 1-4 hours | Medium duration tasks |
| Reference documents | 4-24 hours | Long-term access |
| System prompts | 24 hours | Stable, frequently used |

### 2. Token Threshold Guidelines

- **Minimum**: 1,024 tokens (API requirement)
- **Optimal**: 5,000+ tokens for best cost savings
- **Maximum**: Model's context window limit

### 3. Cache Management

1. **Monitor Expiration**: Set alerts for critical caches
2. **Extend Strategically**: Only extend actively used caches
3. **Clean Up**: Delete unused caches to save storage costs
4. **Batch Similar Content**: Group related documents in single cache

### 4. Performance Optimization

- Pre-cache during off-peak hours
- Use batch processing for document sets
- Implement cache warming strategies
- Monitor hit rates and adjust TTLs

## Troubleshooting

### Common Issues

**Cache not created:**
- Check token count (min 1,024)
- Verify Gemini API credentials
- Ensure caching is enabled

**High storage costs:**
- Review TTL settings
- Delete expired caches
- Monitor cache utilization

**Cache misses:**
- Verify cache hasn't expired
- Check content hasn't changed
- Ensure proper cache ID usage

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("rag_agent.core.context_cache").setLevel(logging.DEBUG)
```

## Advanced Usage

### Custom Cache Strategies

```python
from rag_agent.core.context_cache import context_cache_manager

# Create custom cache for specific use case
cache_entry = context_cache_manager.create_corpus_cache(
    corpus_id="technical-docs",
    documents=large_documents,
    system_instruction="Specialized instructions...",
    ttl_seconds=7200  # 2 hours
)

# Use cache for generation
response = context_cache_manager.generate_with_cache(
    prompt="Analyze the technical specifications",
    cache_entry=cache_entry
)
```

### Cache Warming

Pre-load caches during initialization:

```python
# Warm cache on startup
async def warm_caches():
    critical_corpora = ["faq", "policies", "procedures"]
    for corpus_id in critical_corpora:
        await create_context_cache(corpus_id)
```

## Limitations

1. **Minimum Size**: 1,024 tokens required
2. **Immutable**: Cached content cannot be modified
3. **Model Specific**: Caches are tied to specific models
4. **Regional**: Caches don't transfer across regions

## Future Enhancements

- Automatic cache warming based on usage patterns
- Intelligent TTL adjustment using ML
- Cross-region cache replication
- Cache sharing between projects