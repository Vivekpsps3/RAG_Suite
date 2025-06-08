# 🔍 RobustRAG: Intelligent CSV-based RAG System

A simplified Retrieval-Augmented Generation system that automatically processes CSV files in the documents/csv folder and provides intelligent schema-aware retrieval.

## 🛠 Architecture

RobustRAG is built around five core components that work together to create an efficient and intelligent RAG system:

### 1. HeaderRepository
- Extracts and stores CSV column headers as JSON metadata
- Provides schema information to guide source selection

### 2. CSVProcessor
- Converts CSV rows into textified documents: `"header1: value1 | header2: value2"`
- Uses RecursiveCharacterTextSplitter for handling large rows
- Leverages Sentence-BERT for embedding generation

### 3. VectorStoreManager
- Manages ChromaDB for vector storage and retrieval
- Uses HNSW indexing with cosine similarity

### 4. QueryAgent
- Uses LLM to identify relevant CSV sources based on headers
- Reformulates queries with schema keywords to improve retrieval

### 5. ApplicationEngine
- Automatically processes all CSV files in the documents/csv folder
- Handles the full query pipeline from source selection to answer generation

## 🚀 Execution Flow

### Automatic Ingestion Process
1. System scans documents/csv folder for CSV files
2. Headers are extracted and stored as JSON
3. CSV rows are textified and chunked
4. Embeddings are generated and stored in ChromaDB

### Query Process
1. LLM analyzes headers to select relevant sources
2. Query is reformulated with schema keywords
3. Vector search retrieves relevant chunks
4. LLM generates the final answer

## 💻 Usage

### Installation
```bash
git clone https://github.com/your-username/robust-rag.git
cd robust-rag
pip install -r requirements.txt
```

### Adding CSV Files
Simply copy CSV files to the documents/csv folder. They will be processed automatically when running the application.

### Running Interactive Mode
```bash
python main.py
```

## 📊 Performance

RobustRAG is designed to handle:
- 10+ CSV sources with different schemas
- 50,000+ rows for embedding ingestion
- 80%+ accuracy for factual retrieval

## 📋 MVP Success Criteria

| Area | Requirement | Status |
|------|-------------|--------|
| Header Matching | LLM identifies correct file based on header text | ✅ |
| Embedding Ingestion | ChromaDB stores at least 50,000 rows | ✅ |
| Query Coverage | LLM synthesizes from vector search results | ✅ |
