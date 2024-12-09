# StreamlitAutoRAG: Local AutoRAG Document QA System

A Streamlit-based chat application that implements Automatic Retrieval-Augmented Generation (AutoRAG) using local LLMs. This application automatically optimizes document chunking and uses local models for both embeddings and text generation, making it suitable for CPU-only environments.

## Features

- 🚀 Automatic parameter optimization for document processing
- 💻 Local LLM support using llama.cpp
- 📊 Efficient document chunking with coherence analysis
- 🔍 Smart context retrieval using FAISS
- 📱 User-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/huzaifa525/StreamlitAutoRAG.git
cd StreamlitAutoRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will automatically download the required models on first run

3. Upload your document (PDF or TXT)

4. Start asking questions!

## Models Used

- LLM: Llama 2 7B (4-bit quantized)
- Embeddings: all-MiniLM-L6-v2

## System Requirements

- Python 3.8 or higher
- At least 8GB RAM recommended
- CPU with AVX2 support recommended