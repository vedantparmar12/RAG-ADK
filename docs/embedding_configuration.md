# Embedding Configuration Guide

This guide explains how to configure different embedding providers in the Enhanced RAG Agent, including rate limit management for optimal performance.

## Available Embedding Providers

### 1. Sentence Transformers (Default)
Open-source embedding models that run locally.

**Models available:**
- `all-MiniLM-L6-v2` (384 dimensions) - Fast and efficient
- `all-mpnet-base-v2` (768 dimensions) - Better quality
- `all-distilroberta-v1` (768 dimensions) - Good for longer texts

**Configuration:**
```python
{
    "embedding_provider": "sentence_transformers",
    "st_model": "all-MiniLM-L6-v2"
}
```

### 2. Gemini Embeddings
Google's advanced embedding API with task-specific optimization.

**Features:**
- Model: `gemini-embedding-001`
- Configurable dimensions: 768, 1536, or 3072
- Task-specific embeddings for better performance

**Task Types:**
- `RETRIEVAL_QUERY` - For search queries
- `RETRIEVAL_DOCUMENT` - For documents to be retrieved
- `SEMANTIC_SIMILARITY` - For comparing text similarity
- `CODE_RETRIEVAL_QUERY` - For code search
- `QUESTION_ANSWERING` - For Q&A systems
- `FACT_VERIFICATION` - For fact-checking

**Configuration:**
```python
{
    "embedding_provider": "gemini",
    "gemini_model": "gemini-embedding-001",
    "gemini_task_type": "RETRIEVAL_DOCUMENT",
    "gemini_dimensionality": 768
}
```

### 3. Vertex AI Embeddings
Google Cloud's managed embedding service.

**Models available:**
- `text-embedding-005` - Latest model
- `text-multilingual-embedding-002` - For multilingual content

**Configuration:**
```python
{
    "embedding_provider": "vertex_ai",
    "vertex_model": "text-embedding-005"
}
```

### 4. Multimodal Embeddings
For processing images, text, and video together.

**Features:**
- Fixed 1408 dimensions
- Supports images, text, and video
- Same semantic space for all modalities

**Configuration:**
```python
{
    "embedding_provider": "multimodal"
}
```

## UI Configuration

When creating a corpus in the Streamlit UI:

1. Select your preferred **Embedding Provider** from the dropdown
2. Configure provider-specific settings:
   - For Gemini: Choose task type and output dimensionality
   - For Vertex AI: Select the model version
   - For Sentence Transformers: Pick the model variant
3. Select the **Generation Model** for answer generation

## API Configuration

When using the API, include embedding configuration in the corpus creation request:

```json
POST /corpus
{
    "name": "my-corpus",
    "description": "My document collection",
    "embedding_provider": "gemini",
    "gemini_model": "gemini-embedding-001",
    "gemini_task_type": "RETRIEVAL_DOCUMENT",
    "gemini_dimensionality": 768,
    "generation_model": "gemini-2.0-flash"
}
```

## Best Practices

1. **Choose the right provider:**
   - Use Sentence Transformers for offline/local deployments
   - Use Gemini for best quality and task-specific optimization
   - Use Vertex AI for enterprise GCP deployments
   - Use Multimodal for mixed content types

2. **Optimize dimensions:**
   - 768: Good balance of quality and storage
   - 1536: Better quality, more storage
   - 3072: Best quality, highest storage requirements

3. **Match task types:**
   - Use `RETRIEVAL_QUERY` for user queries
   - Use `RETRIEVAL_DOCUMENT` for indexed documents
   - This improves retrieval accuracy

## Environment Variables

Configure defaults via environment variables:

```bash
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_OUTPUT_DIM=768
GEMINI_TASK_TYPE=RETRIEVAL_DOCUMENT
GENERATION_MODEL=gemini-2.0-flash
```

## Authentication

- **Gemini**: Requires Google AI API key or ADC
- **Vertex AI**: Requires GCP project and authentication
- **Sentence Transformers**: No authentication needed
- **Multimodal**: Requires Vertex AI setup

## Rate Limit Management

### Gemini API Rate Limits

The system automatically handles rate limiting for Gemini embeddings based on your tier:

| Tier | Requirements | Embedding Limits |
|------|--------------|------------------|
| Free | Default | 100 RPM, 30K TPM, 1K RPD |
| Tier 1 | Billing enabled | 1K RPM, 1M TPM, 50K RPD |
| Tier 2 | $250+ spend | 2K RPM, 2M TPM, 100K RPD |
| Tier 3 | $1000+ spend | 4K RPM, 4M TPM, 200K RPD |

*RPM: Requests Per Minute, TPM: Tokens Per Minute, RPD: Requests Per Day*

### Configuration

Set your rate limit tier in environment variables:

```bash
RATE_LIMIT_TIER=tier_1  # Options: free, tier_1, tier_2, tier_3
EMBEDDING_BATCH_SIZE=10  # Number of texts per batch (1-100)
RATE_LIMIT_RETRY_ATTEMPTS=3  # Retry attempts on rate limit errors
RATE_LIMIT_BASE_DELAY=1.0  # Base delay for exponential backoff
```

### Features

1. **Automatic Batching**: Groups multiple embedding requests to optimize API usage
2. **Smart Rate Limiting**: Tracks usage and waits when approaching limits
3. **Exponential Backoff**: Automatically retries with increasing delays
4. **Daily Reset**: Counters reset at midnight Pacific Time
5. **Real-time Monitoring**: View current usage in the UI under "⚡ Rate Limits"

### Best Practices

1. **Batch Processing**
   - Process documents in batches during indexing
   - Use larger batch sizes for better efficiency
   - Example: 10 texts per batch = 10x fewer API calls

2. **Caching Strategy**
   - Enable Redis caching to avoid redundant embeddings
   - Cache frequently used query embeddings
   - Invalidate cache when switching embedding models

3. **Token Optimization**
   - Choose appropriate embedding dimensions:
     - 768: Best balance (default)
     - 1536: Higher quality, more tokens
     - 3072: Highest quality, most tokens
   - Preprocess texts to remove unnecessary content

4. **Error Handling**
   - The system automatically handles rate limit errors
   - Monitor metrics to predict when limits will be reached
   - Consider upgrading tier for production workloads

### Monitoring

Access rate limit metrics via:

1. **UI Dashboard**: Navigate to "⚡ Rate Limits" in the web interface
2. **API Endpoint**: `GET /rate-limits` returns current usage and limits
3. **Logs**: Rate limit events are logged with details

Example API response:
```json
{
  "gemini_embeddings": {
    "current_rpm": 45,
    "current_tpm": 15000,
    "daily_requests": 523,
    "limits": {
      "rpm": 100,
      "tpm": 30000,
      "rpd": 1000
    },
    "utilization": {
      "rpm_percent": 45.0,
      "tpm_percent": 50.0,
      "rpd_percent": 52.3
    }
  },
  "tier": "free",
  "batch_size": 10
}
```

### Troubleshooting

**Issue**: Getting rate limit errors frequently
- **Solution**: Increase batch size, enable caching, or upgrade tier

**Issue**: Slow embedding generation
- **Solution**: Check if rate limiting is causing delays, monitor usage

**Issue**: Daily limit reached
- **Solution**: Process documents during off-peak hours, upgrade tier

**Issue**: Inconsistent performance
- **Solution**: Use batching consistently, monitor token usage