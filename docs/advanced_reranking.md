# Advanced Reranking Guide

This guide explains the advanced reranking capabilities in the Enhanced RAG Agent, including coherence scoring, clustering, and document transformers.

## Overview

The system provides multiple reranking strategies to improve the quality and relevance of retrieved documents:

1. **Coherence Reranking** - Improves document flow and readability
2. **Document Clustering** - Groups semantically similar documents
3. **Diversity Reranking (MMR)** - Reduces redundancy using Maximal Marginal Relevance
4. **Document Reordering** - Sorts by various criteria (relevance, recency, length, source)
5. **Cross-Encoder Reranking** - Uses transformer models for precise relevance scoring

## Reranking Strategies

### 1. Coherence Reranking

Ensures documents flow naturally and maintain topical coherence.

**How it works:**
- Uses cross-encoder models to score query-document relevance
- Calculates coherence between adjacent documents
- Balances relevance with document flow

**Configuration:**
```python
{
    "enable_coherence_reranking": true,
    "coherence_weight": 0.3,  # Weight for coherence vs relevance
}
```

**Best for:**
- Long-form content generation
- Sequential document reading
- Educational materials

### 2. Document Clustering

Groups similar documents together for better topic organization.

**How it works:**
- Encodes documents into embeddings
- Applies K-means or DBSCAN clustering
- Reorders by cluster relevance to query

**Clustering Methods:**
- **K-means**: Automatic cluster count based on document set size
- **DBSCAN**: Density-based clustering for variable cluster sizes

**Best for:**
- Multi-topic queries
- Exploratory search
- Topic summarization

### 3. Diversity Reranking (MMR)

Reduces redundancy while maintaining relevance.

**Algorithm:**
```
MMR = λ × relevance - (1-λ) × max_similarity_to_selected
```

**Configuration:**
```python
{
    "enable_diversity_reranking": true,
    "diversity_lambda": 0.5,  # Balance relevance vs diversity
    "diversity_weight": 0.2   # Weight in final scoring
}
```

**Best for:**
- Avoiding duplicate information
- Comprehensive coverage
- Multi-perspective analysis

### 4. Document Reordering

Final ordering based on specific criteria:

- **Relevance**: Default scoring (highest relevance first)
- **Recency**: Newest documents first (requires timestamp metadata)
- **Length**: Longer, more comprehensive documents first
- **Source**: Groups by document source

### 5. Cross-Encoder Reranking

Uses transformer models for precise relevance scoring.

**Models Used:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Fast and accurate
- Custom models can be configured

**Benefits:**
- More accurate than bi-encoder scoring
- Captures nuanced query-document relationships
- Handles complex queries better

## Configuration

### Environment Variables

```bash
# Enable/disable strategies
ENABLE_COHERENCE_RERANKING=true
ENABLE_DIVERSITY_RERANKING=true
ENABLE_CLUSTERING=false

# Weight parameters
COHERENCE_WEIGHT=0.3      # 0.0-1.0
DIVERSITY_WEIGHT=0.2      # 0.0-1.0
DIVERSITY_LAMBDA=0.5      # 0.0-1.0
```

### API Configuration

When making queries, specify reranking preferences:

```json
POST /query
{
    "query": "What are the benefits of machine learning?",
    "corpus_id": "ml-docs",
    "enable_reranking": true,
    "reranking_config": {
        "strategies": ["coherence", "diversity"],
        "coherence_weight": 0.3,
        "diversity_lambda": 0.6,
        "reorder_strategy": "relevance"
    }
}
```

## Usage Examples

### Example 1: Research Paper Analysis

For academic research requiring comprehensive coverage:

```python
{
    "strategies": ["coherence", "clustering", "diversity"],
    "coherence_weight": 0.2,
    "diversity_lambda": 0.7,  # High diversity
    "reorder_strategy": "source"  # Group by journal/conference
}
```

### Example 2: Customer Support

For support documentation requiring clear, non-redundant answers:

```python
{
    "strategies": ["coherence", "diversity"],
    "coherence_weight": 0.4,  # High coherence for readability
    "diversity_lambda": 0.3,  # Low diversity for focused answers
    "reorder_strategy": "relevance"
}
```

### Example 3: News Aggregation

For news articles requiring temporal ordering and topic grouping:

```python
{
    "strategies": ["clustering", "diversity"],
    "diversity_lambda": 0.6,
    "reorder_strategy": "recency"  # Latest news first
}
```

## Evaluation Metrics

The system provides metrics to evaluate ranking quality:

### 1. Average Coherence Score
- Measures flow between consecutive documents
- Range: 0-1 (higher is better)
- Target: > 0.6 for good readability

### 2. Diversity Score
- Measures non-redundancy in results
- Range: 0-1 (higher = more diverse)
- Target: > 0.5 for comprehensive coverage

### 3. Topic Coverage
- Measures breadth of topics covered
- Calculated from cluster count
- Target: Depends on query complexity

### API Endpoint

Evaluate ranking quality:

```bash
POST /evaluate-ranking
{
    "query": "machine learning applications",
    "documents": [...retrieved documents...]
}

Response:
{
    "metrics": {
        "avg_coherence": 0.72,
        "diversity": 0.65,
        "topic_coverage": 0.8
    },
    "interpretation": {
        "overall_quality": 0.71,
        "quality_label": "Good",
        "recommendations": []
    }
}
```

## Performance Optimization

### 1. Caching
- Cross-encoder scores are cached
- Embeddings are stored with documents
- Cluster assignments are preserved

### 2. Batch Processing
- Documents processed in batches
- Parallel embedding generation
- Efficient similarity computations

### 3. Model Selection
- Use lighter models for real-time queries
- Heavier models for batch processing
- GPU acceleration when available

## Best Practices

### 1. Strategy Selection

| Use Case | Recommended Strategies |
|----------|----------------------|
| Q&A Systems | Coherence + Diversity |
| Research | Clustering + Diversity |
| Documentation | Coherence + Reorder by source |
| News/Updates | Diversity + Reorder by recency |

### 2. Parameter Tuning

- Start with default values
- Increase coherence_weight for better flow
- Increase diversity_lambda for less redundancy
- Use clustering for multi-topic queries

### 3. Monitoring

- Track evaluation metrics
- Monitor processing time
- Adjust strategies based on user feedback
- A/B test different configurations

## Troubleshooting

### Issue: Poor document flow
**Solution**: Increase coherence_weight, enable coherence reranking

### Issue: Redundant results
**Solution**: Enable diversity reranking, increase diversity_lambda

### Issue: Mixed topics
**Solution**: Enable clustering, adjust cluster count

### Issue: Slow performance
**Solution**: Reduce strategies, use lighter models, enable caching

## Advanced Customization

### Custom Cross-Encoder Models

```python
from rag_agent.core.reranking import HybridReranker

reranker = HybridReranker()
reranker.cross_encoder_model = "your-custom-model"
```

### Custom Clustering Parameters

```python
clusterer = DocumentClusterer(clustering_method="dbscan")
clusterer.dbscan_eps = 0.5  # Adjust clustering sensitivity
```

### Custom Scoring Functions

Extend the base transformer classes to implement custom scoring logic.

## Future Enhancements

- Neural reranking with learnable weights
- Query-specific strategy selection
- Multi-stage reranking pipelines
- Real-time learning from user feedback