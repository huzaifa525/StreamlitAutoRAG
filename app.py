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

class DocumentProcessor:
    @staticmethod
    def extract_text_from_image(image) -> str:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            text = pytesseract.image_to_string(binary)
            return text.strip()
        except Exception as e:
            st.warning(f"OCR failed for an image: {str(e)}")
            return ""

    @staticmethod
    def process_pdf(file_content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            text = ""
            import PyPDF2
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text.strip():
                        text += extracted_text + "\n"
            
            if not text.strip():
                st.info("No text found in PDF, attempting OCR...")
                images = pdf2image.convert_from_path(tmp_file_path)
                for image in images:
                    text += DocumentProcessor.extract_text_from_image(image) + "\n"
            
            return text.strip()
        
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    @staticmethod
    def process_image(file_content: bytes) -> str:
        image = Image.open(io.BytesIO(file_content))
        return DocumentProcessor.extract_text_from_image(image)

    @staticmethod
    def process_file(uploaded_file) -> str:
        file_content = uploaded_file.read()
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
    MODELS = {
        "smollm": {
            "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf"
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

    @staticmethod
    def ensure_model_exists(model_name: str = "smollm") -> str:
        model_info = ModelManager.MODELS[model_name]
        model_path = f"models/{model_name}.gguf"
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            st.info(f"Downloading {model_name} model...")
            ModelManager.download_model(model_info["url"], model_path)
            
        return model_path

class LocalAutoRAGOptimizer:
    def __init__(self):
        self.chunk_sizes = [512, 768, 1024]
        self.overlap_sizes = [50, 100, 150]
        
        self.embedding_model = "Intel/dynamic_tinybert"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},  # Removed token parameter
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.encoder = SentenceTransformer(
            self.embedding_model,
            token=HF_TOKEN  # Keep token here as SentenceTransformer accepts it
        )
        
    def evaluate_chunks(self, text: str, chunk_size: int, overlap: int) -> float:
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
        try:
            self.optimizer = LocalAutoRAGOptimizer()
            self.vectorstore = None
            
            # Removed token parameter from model initialization
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="deepseek",
                max_new_tokens=1024,  # Increased for DeepSeek
                context_length=4096,  # Increased context window
                gpu_layers=0,
                top_k=10,
                top_p=0.95,
                temperature=0.7
            )
        except Exception as e:
            raise Exception(f"Failed to initialize AutoRAG system: {str(e)}")
        
    def process_document(self, text: str):
        chunk_size, overlap = self.optimizer.optimize_parameters(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ";", ",", " ", ""]
        )
        
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
        self.vectorstore = FAISS.from_documents(docs, self.optimizer.embeddings)
        
        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": self.optimizer.embedding_model,
            "num_chunks": len(docs)
        }
    
    def clean_response(self, response: str) -> str:
        response = re.sub(r'can you translate.*?\?', '', response, flags=re.IGNORECASE)
        if 'Main St' in response and not re.search(r'\d+.*Main.*St', response, re.IGNORECASE):
            return "I cannot find the specific address in the provided context."
        return response.strip()
    
    def query(self, question: str, num_chunks: int = 5) -> str:
        if not self.vectorstore:
            return "Please process a document first."

        question_type = "unknown"
        if any(word in question.lower() for word in ["address", "location", "place"]):
            question_type = "address"
        elif any(word in question.lower() for word in ["summary", "summarize"]):
            question_type = "summary"
            
        if question_type == "address":
            prompt_template = """Based on the provided context, find the complete and accurate address information. Only return the address if it's explicitly mentioned in the context. If no specific address is found, say "I cannot find the specific address in the provided context."

Context:
{context}

Question: {question}

Let me find the specific address:"""
        else:
            prompt_template = """You are a helpful AI assistant. Answer the question using only the information provided in the context. Be specific and accurate. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Some guidelines:
- Only use information explicitly stated in the context
- Be precise and factual
- Do not make assumptions or add information not present in the context
- If multiple pieces of information are relevant, combine them coherently
- For addresses or specific details, only include them if they are explicitly mentioned

Context:
{context}

Question: {question}

Answer:"""

        relevant_docs = self.vectorstore.similarity_search(question, k=num_chunks)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = prompt_template.format(context=context, question=question)
        response = self.llm(prompt)
        
        return self.clean_response(response)

def main():
    st.title("📚 Local AutoRAG Document QA System")
    
    st.sidebar.write("Hugging Face Authentication Status:")
    try:
        login(token=HF_TOKEN)
        st.sidebar.success("✅ Authenticated with Hugging Face")
    except Exception as e:
        st.sidebar.error(f"❌ Authentication failed: {str(e)}")
        st.error("Please check your Hugging Face token.")
        return
    
    try:
        model_path = ModelManager.ensure_model_exists()
        st.success("Local LLM loaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return
        
    try:
        autorag = LocalAutoRAGSystem(model_path)
        st.success("AutoRAG system initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing AutoRAG system: {str(e)}")
        return
    
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, Images, or Text files)", 
        type=["pdf", "txt", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                document_text = DocumentProcessor.process_file(uploaded_file)
                
                if not document_text.strip():
                    st.error("No text could be extracted from the document.")
                    return
                
                stats = autorag.process_document(document_text)
                st.success("Document processed successfully!")
                
                st.subheader("📊 Optimization Statistics")
                st.write(f"Optimal chunk size: {stats['chunk_size']}")
                st.write(f"Optimal overlap: {stats['overlap']}")
                st.write(f"Embedding model: {stats['embedding_model']}")
                st.write(f"Number of chunks: {stats['num_chunks']}")
                
                with st.expander("View Extracted Text"):
                    st.text(document_text)
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return
    
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
