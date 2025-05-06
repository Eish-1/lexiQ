import os
import sys
from PyPDF2 import PdfReader
# from langchain_community.embeddings import HuggingFaceEmbeddings # Deprecated
from langchain_huggingface import HuggingFaceEmbeddings # New import
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import shutil

# Add the parent directory to the sys.path to find relative imports if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Define paths and settings
PDF_ROOT_DIR = "../pdf_data"  # Directory containing your PDFs relative to this script
CHROMA_DB_DIR = "../chroma_db_legal_bot_part1" # Persistence directory relative to this script
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Efficient and popular model
CHUNK_SIZE = 1000 # Text chunk size for embedding
CHUNK_OVERLAP = 100 # Overlap between chunks

# 2. Extract text from a single PDF
def extract_text_from_pdf(pdf_path):

    """Extracts text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add page number to metadata (optional but useful)
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                else:
                    print(f"Warning: Could not extract text from page {page_num + 1} in {pdf_path}")
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# 3. Process all PDFs in the directory structure
def process_pdfs_to_langchain_docs(root_dir):

    """Processes all PDFs in root_dir, extracts text, and returns LangChain Documents with filename metadata."""
    langchain_documents = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")
                text = extract_text_from_pdf(pdf_path)
                if text:
                    # Create a LangChain Document object
                    # Add filename to metadata for filtering
                    metadata = {"source": pdf_path, "filename": file}
                    langchain_documents.append(Document(page_content=text, metadata=metadata))
                else:
                    print(f"Warning: No text extracted from {pdf_path}. Skipping.")
    return langchain_documents

# 4. Split documents into chunks
def split_documents(documents):

    """Splits LangChain Documents into smaller chunks using RecursiveCharacterTextSplitter."""
    # Switch back to RecursiveCharacterTextSplitter for potentially better RAG chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # RAG often benefits from slightly larger chunks
        chunk_overlap=150,
        length_function=len,
        add_start_index=True, 
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    # Important: The metadata from the original document is preserved in the chunks
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# 5. Initialize Embeddings
def get_embedding_model():
    """Initializes the HuggingFace embedding model."""
    print(f"\n Initializing embedding model: {EMBEDDING_MODEL_NAME} \n")
    # Specify 'cpu' or 'cuda' if needed. Defaults usually work well.
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': False} # Keep False for ChromaDB best practices
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# 6. Create or Update ChromaDB
def create_chroma_db(chunks, embeddings):
    
    print("\n ... Creating or updating ChromaDB... \n")

    """Creates or updates the ChromaDB vector store by adding documents in batches."""
    # Clear existing ChromaDB directory if it exists
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Removing existing ChromaDB directory: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
        
    print(f"Initializing new ChromaDB in: {CHROMA_DB_DIR}")
    # Initialize ChromaDB client first, pointing to the persistent directory
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # Add documents in batches
    batch_size = 1000 # Adjust batch size as needed, keep well below the limit (e.g., 5461)
    total_chunks = len(chunks)
    print(f"Adding {total_chunks} chunks to ChromaDB in batches of {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  Adding batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        vector_store.add_documents(documents=batch)

    print(f"Successfully created ChromaDB with {vector_store._collection.count()} documents.")
    return vector_store

# 7. Main execution block
if __name__ == "__main__":
    print("--- Starting PDF Embedding Process ---")

    # Check if PDF directory exists
    if not os.path.isdir(PDF_ROOT_DIR):
        print(f"Error: PDF directory not found at '{PDF_ROOT_DIR}'. Please create it and add PDFs.")
        sys.exit(1)
        
    # Step 1: Process PDFs into LangChain Documents
    raw_documents = process_pdfs_to_langchain_docs(PDF_ROOT_DIR)

    if not raw_documents:
        print(f"No PDF documents found or processed in '{PDF_ROOT_DIR}'. Exiting.")
        sys.exit(0)

    # Step 2: Split documents into chunks
    document_chunks = split_documents(raw_documents)

    if not document_chunks:
        print("No chunks generated from documents. Exiting.")
        sys.exit(0)

    # Step 3: Initialize Embedding Model
    embedding_model = get_embedding_model()

    # Step 4: Create ChromaDB
    try:
        vector_store = create_chroma_db(document_chunks, embedding_model)
        print("--- PDF Embedding Process Completed Successfully ---")
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        print("--- PDF Embedding Process Failed ---")
        sys.exit(1)