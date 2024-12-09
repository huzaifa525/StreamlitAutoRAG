import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import tempfile
import os
import PyPDF2
import pytesseract
from PIL import Image
import cv2
import io
import re
from huggingface_hub import login

# Hugging Face authentication
HF_TOKEN = "your_token_here"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class DocumentProcessor:
    @staticmethod
    def extract_text_from_image(image) -> str:
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()

    @staticmethod
    def process_pdf(file_content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            text = ""
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    @staticmethod
    def process_file(uploaded_file) -> str:
        if uploaded_file.type == "application/pdf":
            return DocumentProcessor.process_pdf(uploaded_file.read())
        elif uploaded_file.type.startswith('image/'):
            image = Image.open(io.BytesIO(uploaded_file.read()))
            return DocumentProcessor.extract_text_from_image(image)
        elif uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")

class SimpleRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize language model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Store for document chunks and their embeddings
        self.chunks = []
        self.embeddings = None

    def process_document(self, text: str, chunk_size: int = 500):
        # Split text into chunks with overlap
        words = text.split()
        overlap = 50  # words
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        self.chunks = chunks
        # Calculate embeddings for all chunks
        self.embeddings = self.embedding_model.encode(chunks)
        
        return {
            "num_chunks": len(chunks),
            "avg_chunk_length": sum(len(chunk.split()) for chunk in chunks) / len(chunks)
        }

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k chunks
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def query(self, question: str) -> str:
        # Get relevant context
        relevant_chunks = self.get_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt
        prompt = f"""Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer = response[len(prompt):].strip()
        return answer

def main():
    st.title("📚 Document QA with SmolLM2")
    
    # Sidebar for Hugging Face authentication
    st.sidebar.title("Authentication")
    token = st.sidebar.text_input("Enter Hugging Face Token:", type="password")
    if not token:
        st.warning("Please enter your Hugging Face token in the sidebar.")
        return
    
    try:
        login(token=token)
        st.sidebar.success("✅ Authenticated with Hugging Face")
    except Exception as e:
        st.sidebar.error(f"❌ Authentication failed: {str(e)}")
        return
    
    # Initialize RAG system
    @st.cache_resource
    def get_rag_system():
        with st.spinner("Initializing QA system... This may take a few minutes."):
            return SimpleRAG()
    
    try:
        rag = get_rag_system()
        st.success("System initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, Images, or Text files)", 
        type=["pdf", "txt", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                document_text = DocumentProcessor.process_file(uploaded_file)
                if not document_text.strip():
                    st.error("No text could be extracted from the document.")
                    return
                
                stats = rag.process_document(document_text)
                st.success("Document processed successfully!")
                
                st.subheader("📊 Document Statistics")
                st.write(f"Number of chunks: {stats['num_chunks']}")
                st.write(f"Average chunk length: {stats['avg_chunk_length']:.0f} words")
                
                with st.expander("View Extracted Text"):
                    st.text(document_text)
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return
    
        # Question input
        question = st.text_input("Ask a question about your document:")
        if question:
            with st.spinner("Finding answer..."):
                try:
                    answer = rag.query(question)
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
