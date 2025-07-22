# Vertex AI RAG Agent with ADK - Enhanced Edition

This repository contains an advanced Google Agent Development Kit (ADK) implementation of a Retrieval Augmented Generation (RAG) agent using Google Cloud Vertex AI, featuring state-of-the-art retrieval techniques and customizable indexing options.

## Overview

The Enhanced Vertex AI RAG Agent empowers you to build sophisticated document retrieval systems with:

- **Flexible Indexing Strategies** - Choose between dense, sparse, or hybrid embeddings based on your use case
- **Smart Chunking Options** - Configure chunk sizes dynamically and leverage late chunking for better context preservation
- **Advanced Retrieval** - Implement ColBERT-style dense retrieval for superior accuracy
- **Intelligent Caching** - Speed up repeated queries with context-aware caching
- **Context Optimization** - Automatically prune and prioritize the most relevant information
- **Customizable Pipeline** - Fine-tune every aspect of your RAG system

## Core Features

### 1. Query Document Corpora
Ask natural language questions and receive accurate, context-aware answers from your documents. The system now supports multiple retrieval strategies and can be configured to return a specific number of chunks based on your needs.

### 2. Advanced Corpus Management
- **List Corpora** - View all available document collections with detailed metadata
- **Create Custom Corpora** - Build corpora with specific indexing strategies and chunking configurations
- **Add Documents Intelligently** - Import from Google Drive or GCS with automatic optimization
- **Analyze Corpus Performance** - Get insights into retrieval quality and usage patterns
- **Delete with Confidence** - Remove corpora safely with built-in confirmation steps

### 3. Flexible Indexing Options
Choose the indexing strategy that best fits your content:
- **Dense Embeddings** - Best for semantic similarity and conceptual matching
- **Sparse Embeddings** - Ideal for keyword-based retrieval and exact matches
- **Hybrid Approach** - Combines both methods for comprehensive coverage
- **Multilingual Support** - Handle documents in multiple languages seamlessly

### 4. Intelligent Chunking System
Control how your documents are split and processed:
- **Configurable Chunk Sizes** - Set custom sizes from 128 to 2048 tokens
- **Overlap Control** - Define how much content overlaps between chunks
- **Late Chunking** - Maintain document context by chunking after embedding
- **Semantic Segmentation** - Split documents based on meaning, not just size

### 5. Advanced Retrieval Techniques
Leverage cutting-edge retrieval methods:
- **ColBERT-Style Retrieval** - Use late interaction for more accurate matching
- **Multi-Stage Ranking** - First retrieve broadly, then rerank for precision
- **Context-Aware Scoring** - Consider surrounding context when ranking chunks
- **Dynamic Top-K Selection** - Retrieve variable numbers of chunks based on quality scores

### 6. Performance Optimization
Keep your system running efficiently:
- **Smart Context Caching** - Cache frequently asked questions and their results
- **LRU Eviction** - Automatically manage cache size with least-recently-used policies
- **Context Pruning** - Remove redundant information to fit within token limits
- **Batch Processing** - Handle multiple queries efficiently

### 7. Enhanced User Control
Fine-tune your RAG experience:
- **Per-Query Configuration** - Adjust retrieval settings for each question
- **A/B Testing Support** - Compare different configurations side by side
- **Quality Metrics** - Monitor retrieval accuracy and response times
- **Fallback Strategies** - Ensure reliable responses even when advanced features fail

## Prerequisites

- A Google Cloud account with billing enabled
- A Google Cloud project with the Vertex AI API enabled
- Appropriate access to create and manage Vertex AI resources
- Python 3.9+ environment
- Basic understanding of RAG concepts (helpful but not required)

## Setting Up Google Cloud Authentication

Before running the agent, you need to set up authentication with Google Cloud:

1. **Install Google Cloud CLI**:
   Visit the Google Cloud SDK documentation for installation instructions specific to your operating system.

2. **Initialize the Google Cloud CLI**:
   Run the initialization command which will guide you through logging in and selecting your project.

3. **Set up Application Default Credentials**:
   Configure your local environment to authenticate with Google Cloud services automatically.

4. **Verify Authentication**:
   Confirm your setup by listing your authenticated accounts and current configuration.

5. **Enable Required APIs**:
   Ensure the Vertex AI API is enabled in your project, along with any additional services you plan to use.

## Installation

1. **Set up a virtual environment**:
   Create an isolated Python environment to avoid dependency conflicts.

2. **Install Dependencies**:
   Install all required packages including the enhanced modules for advanced features.

## Using the Enhanced Agent

### Quick Start
Begin with the default configuration to get familiar with the system, then gradually enable advanced features as needed.

### Advanced Configuration
When creating a new corpus, you can now specify:
- Indexing strategy (dense, sparse, or hybrid)
- Chunk size and overlap settings
- Whether to enable late chunking
- ColBERT retrieval activation
- Caching preferences
- Context pruning thresholds

### Query Options
When querying your documents, you can control:
- Number of chunks to retrieve
- Whether to use cached results
- Context pruning aggressiveness
- Reranking strategies
- Response format preferences

## Best Practices

1. **Start Simple**: Begin with default settings and enable advanced features gradually
2. **Monitor Performance**: Use built-in metrics to track system performance
3. **Experiment with Settings**: Different document types benefit from different configurations
4. **Cache Wisely**: Enable caching for frequently asked questions
5. **Optimize for Your Use Case**: Tune chunk sizes and retrieval methods based on your specific needs

## Troubleshooting

### Common Issues and Solutions

- **Slow Initial Queries**: First-time queries may be slower due to embedding generation. Enable caching to speed up repeated questions.

- **Irrelevant Results**: Try adjusting chunk size, enabling reranking, or switching to a different indexing strategy.

- **Token Limit Errors**: Enable context pruning to automatically fit responses within model limits.

- **Authentication Problems**: Ensure your credentials are properly configured and have necessary permissions.

- **API Quota Issues**: Monitor your usage and request quota increases if needed. Consider enabling caching to reduce API calls.

## Architecture Overview

The enhanced RAG agent follows a modular architecture:

1. **Indexing Layer**: Handles document processing and embedding generation
2. **Retrieval Layer**: Manages chunk selection and ranking
3. **Optimization Layer**: Provides caching and context pruning
4. **Response Layer**: Generates final answers from retrieved context
5. **Configuration Layer**: Allows fine-tuning of all components

## Future Roadmap

Planned enhancements include:
- Graph-based document relationships
- Multi-modal support (images, tables, charts)
- Streaming responses for real-time applications
- Federation across multiple corpora
- Advanced analytics and monitoring dashboards

## Additional Resources

- Vertex AI RAG Documentation
- Google Agent Development Kit (ADK) Documentation
- Google Cloud Authentication Guide
- Best Practices for RAG Systems
- Community Forums and Support

## Contributing

We welcome contributions! Whether it's bug fixes, new features, or documentation improvements, please feel free to submit pull requests or open issues.

## License

This project is licensed under the Apache 2.0 License. See LICENSE file for details.
