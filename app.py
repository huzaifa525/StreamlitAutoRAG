import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import io
import gc
import os
import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging
from pathlib import Path
from huggingface_hub import login
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
@dataclass
class Config:
    # Model Configuration
    MODEL_NAME: str = "t5-base"  # Suitable for detailed text generation
    
    # Processing Parameters
    CHUNK_SIZE: int = 512
    OVERLAP: int = 128
    MAX_INPUT_LENGTH: int = 512  # Adjusted to T5's max input length
    MAX_CHUNKS: int = 2000
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    
    # System Parameters
    CACHE_SIZE: int = 1000
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    
    # Paths
    CACHE_DIR: str = "./cache"
    LOG_DIR: str = "./logs"
    
    def __post_init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

config = Config()

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{config.LOG_DIR}/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Document Processing
# ------------------------------
class DocumentProcessor:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.NUM_WORKERS
        )
        self._setup_ocr()

    def _setup_ocr(self):
        """Configure OCR settings for optimal performance"""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not properly configured: {e}")
            st.error("OCR system not properly configured. Please install Tesseract.")
            st.stop()

    async def process_document(self, file) -> str:
        """Process different document types asynchronously"""
        try:
            if file.type == "application/pdf":
                return await self._process_pdf(file.read())
            elif file.type.startswith('image/'):
                return await self._process_image(file)
            elif file.type == "text/plain":
                return file.getvalue().decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file.type}")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            st.error(f"Error processing document: {e}")
            st.stop()

    async def _process_pdf(self, content: bytes) -> str:
        """Process PDF documents with parallel page processing"""
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                tasks = []
                for page in pdf.pages:
                    task = self.executor.submit(self._extract_page_text, page)
                    tasks.append(task)
                
                results = await asyncio.gather(*[
                    asyncio.wrap_future(task) for task in tasks
                ])
                return "\n".join(filter(None, results))
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise

    def _extract_page_text(self, page) -> str:
        """Extract text from a single PDF page"""
        try:
            return page.extract_text() or ""
        except Exception as e:
            logger.error(f"Page extraction error: {e}")
            return ""

    async def _process_image(self, file) -> str:
        """Process images with enhanced OCR"""
        try:
            image = Image.open(io.BytesIO(file.read()))
            return await self._perform_ocr(image)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise

    async def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR with preprocessing"""
        try:
            # Convert to numpy array
            img_np = np.array(image)
            
            # Image preprocessing
            processed = await self._preprocess_image(img_np)
            
            # Perform OCR
            future = self.executor.submit(
                pytesseract.image_to_string,
                processed,
                config='--psm 1 --oem 3'
            )
            text = await asyncio.wrap_future(future)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            raise

    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Thresholding
            _, binary = cv2.threshold(
                denoised, 
                0, 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            return binary
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise

# ------------------------------
# FastRAG System Simplified
# ------------------------------
class FastRAG:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.device = "cpu"
        self._init_models()
        self.temp_file = "temp.txt"  # Initialize temp_file

    def _init_models(self):
        """Initialize and optimize models"""
        try:
            # Load models
            self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
            self.model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
            
            # Optimize model for CPU
            self.model = self._optimize_model(self.model)
            self.model.eval()
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            st.error(f"Model initialization error: {e}")
            st.stop()

    def _optimize_model(self, model):
        """Apply optimizations for CPU inference"""
        try:
            # Dynamic quantization
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            return model
        except Exception as e:
            logger.error(f"Model optimization error: {e}")
            raise

    async def process_and_save_document(self, file):
        """Process the uploaded document and save extracted text to temp.txt"""
        try:
            text = await self.doc_processor.process_document(file)
            if not text.strip():
                st.error("No text could be extracted from the document.")
                st.stop()
            # Save to temp.txt
            with open(self.temp_file, 'w', encoding='utf-8') as f:
                f.write(text)
            st.success("Document processed and text extracted successfully!")
            st.write(f"📊 **Number of Characters Extracted:** {len(text)}")
            with st.expander("📄 Show Extracted Text"):
                st.text_area("Extracted Text", text, height=300)
        except Exception as e:
            logger.error(f"Error during document processing: {e}")
            st.error(f"Error during document processing: {e}")
            st.stop()

    def generate_answer(self, question: str) -> str:
        """Generate answer based on the context from temp.txt"""
        try:
            # Read context from temp.txt
            with open(self.temp_file, 'r', encoding='utf-8') as f:
                context = f.read()
            
            # Construct prompt
            prompt = (
                "System: You are an AI assistant that provides detailed and structured answers based on the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"User: {question}\n"
                "AI:"
            )
            
            # Tokenize input
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=config.MAX_INPUT_LENGTH,
                truncation=True
            )
            
            # Generate response
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=1024,  # Increased max length for detailed answers
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Decode response
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer part
            if "AI:" in answer:
                answer = answer.split("AI:")[-1].strip()
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"An error occurred while generating the answer: {e}"

# ------------------------------
# Streamlit Application
# ------------------------------
def main():
    st.set_page_config(
        page_title="FastRAG Document QA",
        page_icon="💡",
        layout="wide"
    )
    
    st.title("💡 FastRAG Document QA System")
    
    # Sidebar configuration
    st.sidebar.title("Setup")
    api_token = st.sidebar.text_input(
        "Hugging Face Token:",
        type="password"
    )
    
    if not api_token:
        st.sidebar.warning("Please enter your Hugging Face token.")
        st.stop()
    
    try:
        login(token=api_token)
        st.sidebar.success("✅ Authentication successful")
    except Exception as e:
        st.sidebar.error(f"❌ Authentication failed: {e}")
        st.stop()
    
    # Initialize FastRAG system
    @st.cache_resource
    def initialize_system():
        return FastRAG()
    
    try:
        with st.spinner("Initializing system..."):
            rag_system = initialize_system()
        st.success("System initialized successfully!")
    except Exception as e:
        st.error(f"System initialization error: {e}")
        st.stop()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📄 Upload Document (PDF, Image, or Text)",
        type=["pdf", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    rag_system.process_and_save_document(uploaded_file)
                )
        except Exception as e:
            st.error(f"Error processing document: {e}")
            st.stop()

    # Q&A Interface
    if os.path.exists(rag_system.temp_file):
        qa_tab, summary_tab = st.tabs(["🤖 Ask a Question", "📑 View Summary"])
        
        with qa_tab:
            st.subheader("Ask Questions About Your Document")
            question = st.text_input("💭 Enter your question here:")
            
            if st.button("Get Answer"):
                if question.strip() == "":
                    st.warning("Please enter a valid question.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = rag_system.generate_answer(question)
                        if answer:
                            st.success("✅ Answer:")
                            st.write(answer)
                        else:
                            st.warning("⚠️ Could not generate an answer. Please try rephrasing your question.")
    
        with summary_tab:
            st.subheader("📑 Document Summary")
            try:
                with open(rag_system.temp_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Generate summary
                prompt = (
                    "Summarize the following document in a detailed and structured manner:\n\n"
                    f"{text}\n\n"
                    "Summary:"
                )
                input_ids = rag_system.tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                    max_length=config.MAX_INPUT_LENGTH,
                    truncation=True
                )
                outputs = rag_system.model.generate(
                    input_ids=input_ids,
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                summary = rag_system.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Summary:" in summary:
                    summary = summary.split("Summary:")[-1].strip()
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                st.stop()

    # Display system information
    with st.sidebar.expander("ℹ️ System Information"):
        st.write(f"**Model:** {config.MODEL_NAME}")
        st.write(f"**Max Input Length:** {config.MAX_INPUT_LENGTH} tokens")
        st.write(f"**Chunk Size:** {config.CHUNK_SIZE} words")
        st.write(f"**Chunk Overlap:** {config.OVERLAP} words")

    # Add helpful tips
    with st.sidebar.expander("💡 Tips"):
        st.write("""
        - Upload clear and well-formatted documents for optimal text extraction.
        - Ensure that scanned images are legible for accurate OCR.
        - Use specific and clear questions to get the best answers.
        - View the summary for a quick overview of the document.
        """)

if __name__ == "__main__":
    main()
