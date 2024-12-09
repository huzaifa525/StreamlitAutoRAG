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

class DocumentProcessor:
    """Handles document processing including OCR"""
    
    @staticmethod
    def extract_text_from_image(image) -> str:
        """Extract text from an image using OCR"""
        # Convert PIL Image to cv2 format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(binary)
            return text.strip()
        except Exception as e:
            st.warning(f"OCR failed for an image: {str(e)}")
            return ""

    @staticmethod
    def process_pdf(file_content: bytes) -> str:
        """Process PDF file and extract text including OCR if needed"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # First try to extract text directly
            text = ""
            import PyPDF2
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text.strip():
                        text += extracted_text + "\n"
            
            # If no text was extracted, try OCR
            if not text.strip():
                st.info("No text found in PDF, attempting OCR...")
                images = pdf2image.convert_from_path(tmp_file_path)
                text = ""
                for image in images:
                    text += DocumentProcessor.extract_text_from_image(image) + "\n"
            
            return text.strip()
        
        finally:
            # Cleanup
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    @staticmethod
    def process_image(file_content: bytes) -> str:
        """Process image file using OCR"""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_content))
        return DocumentProcessor.extract_text_from_image(image)

    @staticmethod
    def process_file(uploaded_file) -> str:
        """Process uploaded file based on its type"""
        file_content = uploaded_file.read()
        
        # Detect file type
        file_type = magic.from_buffer(file_content, mime=True)
        
        if file_type == 'application/pdf':
            return DocumentProcessor.process_pdf(file_content)
        elif file_type.startswith('image/'):
            return DocumentProcessor.process_image(file_content)
        elif file_type.startswith('text/'):
            return file_content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

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
        self.embedding_model = "all-MiniLM-L6-v2"
        self.chunk_sizes = [256, 512, 1024]
        self.overlap_sizes = [0, 50, 100]
        
        # Initialize Hugging Face embeddings for LangChain
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        # Initialize sentence transformer for coherence scoring
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
        
    def process_document(self, text: str):
        """Process document with optimized parameters."""
        # Optimize parameters
        chunk_size, overlap = self.optimizer.optimize_parameters(text)
        
        # Create text splitter with optimized parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        
        # Create documents
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
        
        # Create vectorstore using HuggingFace embeddings
        self.vectorstore = FAISS.from_documents(docs, self.optimizer.embeddings)
        
        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": self.optimizer.embedding_model,
            "num_chunks": len(docs)
        }
    
    def query(self, question: str, num_chunks: int = 5) -> str:
        """Query the document with local LLM."""
        if not self.vectorstore:
            return "Please process a document first."
        
        # Retrieve relevant chunks
        relevant_docs = self.vectorstore.similarity_search(question, k=num_chunks)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Using the provided context, answer the user's question comprehensively and accurately. If the information cannot be found in the context, say "I cannot find the answer in the provided context."

Some guidelines:
- Synthesize information from multiple chunks if needed
- Provide complete, well-structured answers
- Stay focused on the question asked
- Use your own words to explain clearly
- If asked for a summary, provide key points in a coherent manner
- Cite specific details from the context when relevant

Context:
{context}

Question: {question}

Please provide a clear and complete answer:
"""
        
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
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, Images, or Text files)", 
        type=["pdf", "txt", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                # Process the uploaded file
                document_text = DocumentProcessor.process_file(uploaded_file)
                
                if not document_text.strip():
                    st.error("No text could be extracted from the document.")
                    return
                
                # Process the extracted text
                stats = autorag.process_document(document_text)
                st.success("Document processed successfully!")
                
                # Display optimization stats
                st.subheader("📊 Optimization Statistics")
                st.write(f"Optimal chunk size: {stats['chunk_size']}")
                st.write(f"Optimal overlap: {stats['overlap']}")
                st.write(f"Embedding model: {stats['embedding_model']}")
                st.write(f"Number of chunks: {stats['num_chunks']}")
                
                # Display extracted text
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
                    answer = autorag.query(question)
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
