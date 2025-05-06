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

def get_pdf_text(pdf_path):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages
                else:
                    logger.warning(f"Could not extract text from a page in {pdf_path}")
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        return []
    # Switch to RecursiveCharacterTextSplitter for summarization chunks too
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Slightly smaller than RAG chunk size
        chunk_overlap=100,
        length_function=len,
        add_start_index=True, # Good practice
        separators=["\n\n", "\n", ".", " ", ""] # Standard separators
    )
    # Use split_text for raw text input
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks for summarization.")
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