# Multi-Modal RAG-Based QA System

This project implements a Retrieval-Augmented Generation (RAG) system capable of processing text, tables, and images from PDF documents.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   This extracts data and creates the vector index.
   ```bash
   python run_pipeline.py
   ```

3. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

## Features

- **Text Extraction**: Sliding window chunking for better context.
- **Table Extraction**: Uses `pdfplumber` for structured data.
- **OCR Integration**: (Requires Tesseract) for image text extraction.
- **QA Bot**: Powered by LangChain and HuggingFace models.
- **Source Attribution**: Page-level citations for all answers.
- **Summarization**: One-click document briefing.
- **Hybrid Search**: Combines BM25 keyword matching with semantic vector search for superior retrieval accuracy.
- **System Evaluation**: Automated benchmarking of latency and relevance.

## Components

- `document_processor.py`: PDF parsing logic.
- `vector_store.py`: FAISS index management.
- `llm_qa.py`: Answer generation logic.
- `app.py`: Streamlit user interface.
- `Technical_Report.md`: Detailed architecture documentation.
