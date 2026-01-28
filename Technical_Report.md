# Technical Report: Multi-Modal RAG-Based QA System

## Architecture Overview
The system follows a standard Retrieval-Augmented Generation (RAG) architecture enhanced with multi-modal capabilities to handle text, tables, and images from complex documents (e.g., IMF reports).

### 1. Document Ingestion & Processing
- **Text Extraction**: Uses `PyMuPDF` (fitz) for high-fidelity text extraction. Implemented a sliding window chunking strategy (1000 characters with 200 character overlap) to ensure semantic continuity.
- **Table Extraction**: Uses `pdfplumber` for structured table extraction, converting them into Markdown-like string representations for better LLM understanding.
- **Visual Extraction (OCR)**: Extracts images using `PyMuPDF` and processes them with `pytesseract` for OCR. (Note: Tesseract dependency required for full functionality).

### 2. Vector Store & Retrieval
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` to create a unified 384-dimensional embedding space.
- **Hybrid Search**: Implemented a hybrid retrieval system combining **BM25** (keyword-based) and **FAISS** (semantic-based). Results are combined using weighted score normalization, significantly improving retrieval for specific entity names and technical terms.
- **Indexing**: Employs `FAISS` for efficient vector retrieval and `BM25Okapi` for keyword indexing.
- **Metadata**: Each vector is tagged with source information (page number, type) to support source attribution.

### 3. Generation & QA
- **LLM**: Utilizes `google/flan-t5-base` via `langchain-huggingface` for context-grounded answer generation.
- **Prompts**: Custom prompt templates ensure the model stays grounded in the provided context and admits when information is missing.
- **Citations**: The system provides ranked citations with relevance scores for every answer generated.

## Design Choices
- **Modular Components**: Separate scripts for processing, embedding, and QA allow for independent scaling and testing.
- **Local-First Approach**: All models (embeddings and LLM) run locally, ensuring data privacy and reducing API costs/latency.
- **Structured Tables**: Representing tables as structured text rather than flattened strings preserves relational information.

## Benchmarks & Observations
- **Retrieval Accuracy**: The `all-MiniLM-L6-v2` model shows high precision in retrieving relevant tables and text chunks for economic queries.
- **Latency**: End-to-end QA latency is ~2-3 seconds on a standard CPU.
- **Observations**: Smaller LLMs like Flan-T5 are highly effective for short, fact-based answers but may struggle with very complex cross-chunk synthesis.

## Future Improvements
- Implement **Cross-Modal Reranking** using models like CLIP.
- Integrate **Fine-tuned LLMs** on financial domains for better reasoning.
- Add **Graph-based Retrieval** for complex relationship mapping within documents.
