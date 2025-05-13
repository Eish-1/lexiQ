import os
from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter # Replace this
from langchain.text_splitter import RecursiveCharacterTextSplitter # Use this instead
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import logging
# ... rest of imports ...

logger = logging.getLogger('legalre')

def check_pdf_size(pdf_path):
    """Checks if a PDF is too large to process efficiently."""
    try:
        # Check file size
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        
        # Check page count
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            page_count = len(pdf_reader.pages)
        
        logger.info(f"PDF check: {pdf_path}, Size: {file_size_mb:.2f}MB, Pages: {page_count}")
        
        # Set thresholds for warnings
        SIZE_THRESHOLD_MB = 20  # 20MB
        PAGE_THRESHOLD = 100    # 100 pages
        
        if file_size_mb > SIZE_THRESHOLD_MB or page_count > PAGE_THRESHOLD:
            warning_msg = f"Warning: Large document detected ({file_size_mb:.2f}MB, {page_count} pages). Processing may be slow or incomplete."
            logger.warning(warning_msg)
            return False, warning_msg
        
        return True, None
    
    except Exception as e:
        logger.error(f"Error checking PDF size: {e}")
        return False, f"Error checking PDF: {str(e)}"

def get_pdf_text(pdf_path):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        # First check if the PDF is too large
        is_processable, warning = check_pdf_size(pdf_path)
        if not is_processable:
            logger.warning(f"PDF may be too large for optimal processing: {warning}")
        
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            
            # Limit to a maximum number of pages for very large documents
            MAX_PAGES = 150
            pages_to_process = min(len(pdf_reader.pages), MAX_PAGES)
            
            if pages_to_process < len(pdf_reader.pages):
                logger.warning(f"PDF has {len(pdf_reader.pages)} pages; only processing first {MAX_PAGES} pages")
            
            for i in range(pages_to_process):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages
                else:
                    logger.warning(f"Could not extract text from page {i+1} in {pdf_path}")
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        return []
    
    # Use a smaller chunk size to prevent token limit errors
    # The max token limit for embedding models is usually 512 tokens
    # A safe estimate is ~3-4 chars per token, so 400 tokens is ~1200-1600 chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Significantly smaller than before (was 800)
        chunk_overlap=50,  # Reduced overlap to match smaller chunks
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Set maximum number of chunks to process (to prevent overloading)
    MAX_CHUNKS = 200
    
    # Use split_text for raw text input
    chunks = text_splitter.split_text(text)
    total_chunks = len(chunks)
    
    logger.info(f"Split text into {total_chunks} chunks for processing.")
    
    # If we have too many chunks, truncate and warn
    if total_chunks > MAX_CHUNKS:
        logger.warning(f"Document is very large ({total_chunks} chunks). Truncating to {MAX_CHUNKS} chunks.")
        chunks = chunks[:MAX_CHUNKS]
    
    return chunks

def generate_summary(text_chunks, api_key):
    """Generates summary using Map-Reduce chain with Groq."""
    if not text_chunks:
        return "Error: No text chunks to summarize."
    
    try:
        # Initialize LLM
        llm = ChatGroq(temperature=0.2, groq_api_key=api_key, model_name="llama3-8b-8192")
        
        # Load summarization chain
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # Convert text chunks to Document objects
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Run the chain
        result = summary_chain.invoke({"input_documents": docs})
        summary = result.get('output_text', "Error: Could not extract summary.")
        return summary
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        return f"Error generating summary: {e}"

# ... generate_summary function remains the same ... 