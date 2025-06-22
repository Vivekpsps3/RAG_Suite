# RobustRAG 2

An enhanced RAG (Retrieval-Augmented Generation) system with improved context retrieval and reranking capabilities.

## Features

- Semantic document retrieval using vector embeddings
- Additional context retrieval via query reformulation
- Advanced reranking and deduplication of retrieved context
- Context refinement to provide only the most relevant information
- Support for CSV data processing

## New Reranking Feature

The system now includes an advanced reranking feature that improves the quality of retrieved context:

1. **Initial Retrieval**: Performs standard vector store retrieval for the user query
2. **Additional Context**: Generates a reformulated query to find complementary information
3. **Reranking**: Uses a cross-encoder model to rerank all retrieved contexts based on relevance
4. **Deduplication**: Removes redundant or duplicate information
5. **Context Selection**: Selects only the top 10 most relevant pieces of context

## Usage

Run the main application with:

```bash
python main.py
```

### Command-line options:

- `--new`: Create a new vector store collection (clears existing data)
- `--reload`: Force reload all CSV files into vector store
- `--check`: Check vector store status
- `--no-reranking`: Disable the reranking step (fall back to basic retrieval)

## Testing the Reranker

You can test the reranking functionality specifically using:

```bash
python test_reranker.py
```

This will run a sample query and show detailed information about the reranking process.

## Configuration

Configuration is stored in the `src/__init__.py` file, including paths to models and default parameters.
