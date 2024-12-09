import streamlit as st
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import os
import requests
from tqdm import tqdm
import tempfile
from pathlib import Path
import pdf2image
import pytesseract
from PIL import Image
import io
import cv2
import magic
import re
from huggingface_hub import login

# Hugging Face authentication
HF_TOKEN = "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"
login(token=HF_TOKEN)

# Rest of your imports...

class DocumentProcessor:
    # ... (rest of the DocumentProcessor class remains the same)

class ModelManager:
    MODELS = {
        "tiny": {
            "name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        }
    }
    
    @staticmethod
    def download_model(url: str, save_path: str):
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress.update(size)

    # ... (rest of the ModelManager class remains the same)

class LocalAutoRAGOptimizer:
    def __init__(self):
        self.chunk_sizes = [512, 768, 1024]
        self.overlap_sizes = [50, 100, 150]
        
        # Use Intel's dynamic-tinybert with auth
        self.embedding_model = "Intel/dynamic-tinybert"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={
                'device': 'cpu',
                'token': HF_TOKEN
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize sentence transformer with auth
        self.encoder = SentenceTransformer(
            self.embedding_model,
            token=HF_TOKEN
        )
        
    # ... (rest of the LocalAutoRAGOptimizer class remains the same)

class LocalAutoRAGSystem:
    def __init__(self, model_path: str):
        self.optimizer = LocalAutoRAGOptimizer()
        self.vectorstore = None
        
        # Initialize local LLM with auth
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            max_new_tokens=512,
            context_length=2048,
            gpu_layers=0,
            token=HF_TOKEN
        )
    
    # ... (rest of the LocalAutoRAGSystem class remains the same)

def main():
    st.title("📚 Local AutoRAG Document QA System")
    
    # Initialize HF auth status
    st.sidebar.write("Hugging Face Authentication Status:")
    try:
        login(token=HF_TOKEN)
        st.sidebar.success("✅ Authenticated with Hugging Face")
    except Exception as e:
        st.sidebar.error(f"❌ Authentication failed: {str(e)}")
        st.error("Please check your Hugging Face token.")
        return
    
    # Rest of the main function remains the same...

if __name__ == "__main__":
    main()
