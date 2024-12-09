import streamlit as st
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import os
import requests
from tqdm import tqdm

class ModelManager:
    """Handles downloading and loading of models"""
    
    MODELS = {
        "tiny": {
            "name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        }
    }
    
    @staticmethod
    def download_model(url: str, save_path: str):
        """Download model with progress bar"""
        response = requests.get(url, stream=True)
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

    @staticmethod
    def ensure_model_exists(model_name: str = "tiny") -> str:
        """Check if model exists, download if not"""
        model_info = ModelManager.MODELS[model_name]
        model_path = f"models/{model_name}.gguf"
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            st.info(f"Downloading {model_name} model...")
            ModelManager.download_model(model_info["url"], model_path)
            
        return model_path

class LocalAutoRAGOptimizer:
    def __init__(self):
        # Use lightweight embedding model
        self.embedding_model = "all-MiniLM-L6-v2"
        self.chunk_sizes = [256, 512, 1024]
        self.overlap_sizes = [0, 50, 100]
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(self.embedding_model)
        
    def evaluate_chunks(self, text: str, chunk_size: int, overlap: int) -> float:
        """Evaluate chunk coherence and information retention."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_text(text)
        
        if len(chunks) < 2:
            return 0.0
            
        # Calculate coherence between adjacent chunks
        coherence_scores = []
        embeddings = self.encoder.encode(chunks)
        
        for i in range(len(chunks)-1):
            similarity = cosine_similarity(
                [embeddings[i]],
                [embeddings[i+1]]
            )[0][0]
            coherence_scores.append(similarity)
            
        return np.mean(coherence_scores)

    def optimize_parameters(self, text: str) -> Tuple[int, int]:
        """Find optimal chunking parameters."""
        best_score = -1
        optimal_params = None
        
        for chunk_size in self.chunk_sizes:
            for overlap in self.overlap_sizes:
                score = self.evaluate_chunks(text, chunk_size, overlap)
                if score > best_score:
                    best_score = score
                    optimal_params = (chunk_size, overlap)
        
        return optimal_params[0], optimal_params[1]

class LocalAutoRAGSystem:
    def __init__(self, model_path: str):
        self.optimizer = LocalAutoRAGOptimizer()
        self.vectorstore = None
        
        # Initialize local LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            max_new_tokens=512,
            context_length=2048,
            gpu_layers=0  # CPU only
        )
        
    def process_document(self, file_path: str):
        """Process document with optimized parameters."""
        # Load document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        document = loader.load()
        
        # Get full text
        full_text = " ".join([doc.page_content for doc in document])
        
        # Optimize parameters
        chunk_size, overlap = self.optimizer.optimize_parameters(full_text)
        
        # Create text splitter with optimized parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        splits = splitter.split_documents(document)
        
        # Create embeddings and vectorstore
        self.vectorstore = FAISS.from_documents(
            splits,
            self.optimizer.encoder
        )
        
        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": self.optimizer.embedding_model,
            "num_chunks": len(splits)
        }
    
    def query(self, question: str, num_chunks: int = 3) -> str:
        """Query the document with local LLM."""
        if not self.vectorstore:
            return "Please process a document first."
        
        # Retrieve relevant chunks
        relevant_docs = self.vectorstore.similarity_search(question, k=num_chunks)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer: """
        
        # Generate response using local LLM
        response = self.llm(prompt)
        
        return response.strip()

# Streamlit Interface
def main():
    st.title("📚 Local AutoRAG Document QA System")
    
    # Model setup
    try:
        model_path = ModelManager.ensure_model_exists()
        st.success("Local LLM loaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return
        
    # Initialize AutoRAG system
    try:
        autorag = LocalAutoRAGSystem(model_path)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=["pdf", "txt"])
    
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp_doc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                stats = autorag.process_document("temp_doc")
                st.success("Document processed successfully!")
                
                # Display optimization stats
                st.subheader("📊 Optimization Statistics")
                st.write(f"Optimal chunk size: {stats['chunk_size']}")
                st.write(f"Optimal overlap: {stats['overlap']}")
                st.write(f"Embedding model: {stats['embedding_model']}")
                st.write(f"Number of chunks: {stats['num_chunks']}")
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return
            finally:
                # Clean up temporary file
                if os.path.exists("temp_doc"):
                    os.remove("temp_doc")
    
        # Question input
        question = st.text_input("Ask a question about your document:")
        
        if question:
            with st.spinner("Finding answer..."):
                try:
                    answer = autorag.query(question)
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
