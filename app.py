import os
import streamlit as st
import random
import time
import base64
# from legalre_main import LegalRe # Old import
from src.legalre_main import LegalRe # Corrected import path
from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings # Deprecated
from langchain_huggingface import HuggingFaceEmbeddings # New import
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.schema import HumanMessage
import uuid
import logging
from src.utils import get_pdf_text, get_text_chunks, generate_summary

# Set up logger
logger = logging.getLogger('legalre')

#This page implements the streamlit UI
# Set page configuration
st.set_page_config(page_title="LexiQ", page_icon="logo/logo.png", layout="wide")

# Custom CSS for better UI
def add_custom_css():
    """Function for a beautiful streamlit UI"""
    custom_css = """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .st-chat-input {
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            font-size: 16px;
            border-radius: 20px;
            padding: 10px 20px;
            margin-top: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0052a3;
        }
        .st-chat-message-assistant {
            background-color: #f7f7f7;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .st-chat-message-user {
            background-color: #d9f0ff;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f0f0f0;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex-grow: 1;
        }
        .st-title {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: #333;
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .logo {
            width: 40px;
            height: 30px;
        }
        .st-sidebar {
            background-color: #f9f9f9;
            padding: 20px;
        }
        .st-sidebar header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .st-sidebar p {
            font-size: 14px;
            color: #666;
        }
        /* Document mode styles */
        .document-summary {
            background-color: #000000;
            border-left: 4px solid #0066cc;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            color: #FFFFFF; /* Ensure text is white for contrast */
        }
        
        /* Mode selector styles */
        .stRadio > div {
            background-color: #8B0000;
            padding: 10px;
            border-radius: 10px;
        }
        
        /* Document info box */
        .stAlert > div {
            background-color: #8B0000;
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 10px;
            margin-bottom: 15px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()
##Below Code implementation is tha main functioanlity in building the streamlit application
# Title with Logo
logo_path = "logo/logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
    <div class="st-title">
        <img src="data:image/png;base64,{encoded_image}" alt="LexiQ Logo" class="logo">
        <span>LexiQ - An AI Legal Assistant </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="st-title">
        <span>LexiQ - Your Legal Assistant 📖</span>
    </div>
    """, unsafe_allow_html=True)

# Sidebar improvements
st.sidebar.header("About LexiQ")
st.sidebar.markdown("""
**LexiQ** is a free, open-source AI legal assistant that helps answer legal questions based on provided documents.

_Disclaimer_: This tool is in its pilot phase, and responses may not be 100% accurate.
""")

# Function to configure logging verbosity
def configure_logging(verbose=False):
    # Set the base level
    base_level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=base_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set specific modules to warning or higher to reduce noise
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Keep LegalRe's own logs at the selected verbosity
    logging.getLogger('legalre').setLevel(base_level)

# Add to the sidebar (after the about section):
with st.sidebar:
    st.markdown("---")
    verbose_logging = st.sidebar.checkbox("Verbose logging", value=False, help="Show detailed processing logs in the terminal")
    # Configure logging based on user preference
    configure_logging(verbose_logging)

load_dotenv()

#random thread id for session
# Ensure session state for thread_id if needed across reruns, 
# but for now, generating per run might be fine for basic history.
if 'thread_id' not in st.session_state:
     st.session_state.thread_id = str(uuid.uuid4())
thread_id = st.session_state.thread_id

# Load API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Check if the Groq API key is available
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop execution if key is missing

# Caching functions
@st.cache_resource(show_spinner=False)
def get_llm(api_key):
    with st.spinner("Initializing language model..."):
        logging.info("Initializing Groq LLM...")
        return ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            groq_api_key=api_key
        )

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    with st.spinner("Loading embedding model..."):
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

@st.cache_resource(show_spinner=False)
def get_vector_store(_embeddings):
    with st.spinner("Connecting to document database..."):
        CHROMA_DB_DIR = "chroma_db_legal_bot_part1"
        logging.info(f"Loading vector store from: {CHROMA_DB_DIR}")
        if not os.path.isdir(CHROMA_DB_DIR):
            st.error(f"ChromaDB directory not found at '{CHROMA_DB_DIR}'. Please run the embedding script first.")
            st.stop()
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=_embeddings
        )
        logging.info(f"Loaded vector store with {vector_store._collection.count()} documents.")
        return vector_store

@st.cache_resource # Cache the main RAG class instance
def get_legalre_instance(_llm, _embeddings, _vector_store):
    print("Initializing LegalRe instance...")
    return LegalRe(_llm, _embeddings, _vector_store)

# Use cached functions for initialization
llm = get_llm(groq_api_key)
embeddings = get_embedding_model()
vector_store = get_vector_store(embeddings)
law = get_legalre_instance(llm, embeddings, vector_store)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    
    # Check if this is a summary message (needs special styling)
    is_summary = message.get("is_summary", False)
    
    with st.chat_message(role):
        if is_summary:
            # Apply special styling for summary messages
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 10px; background-color: #000000; border-left: 4px solid #0066cc; color: #FFFFFF;">
                {message["content"]}
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # Normal message display
            st.markdown(message["content"])

# Chat input prompt fixed at the bottom
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
# User Input
prompt = st.chat_input("Have a legal question? Let's work through it.")

st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state variables for document handling
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "general"  # Default mode: "general" or "document"
if "uploaded_document" not in st.session_state:
    st.session_state.uploaded_document = None  # Will store document text
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None  # Will store the summary
if "document_name" not in st.session_state:
    st.session_state.document_name = None  # Will store the document name

# Sidebar with mode selection and file upload
with st.sidebar:
    st.header("Document Interaction")
    
    # File uploader for PDF documents
    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
    
    # Process uploaded document
    if uploaded_file is not None and (st.session_state.document_name != uploaded_file.name):
        with st.spinner("Processing document..."):
            # Create temp directory if it doesn't exist
            temp_dir = "temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Save the uploaded file temporarily
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Extract text from PDF
                doc_text = get_pdf_text(temp_path)
                
                if doc_text:
                    # Check if the document might be too large to handle properly
                    if len(doc_text) > 500000:  # ~500K characters is quite large
                        st.warning(f"This document is very large ({len(doc_text)/1000:.0f}K characters). It may be processed partially and results might be incomplete.")
                    
                    # Split into chunks for processing
                    text_chunks = get_text_chunks(doc_text)
                    
                    if text_chunks:
                        # Show progress to user for large documents
                        with st.status("Generating document summary...") as status:
                            # Generate summary
                            summary = generate_summary(text_chunks, groq_api_key)
                            status.update(label="Summary complete! Processing document for Q&A...", state="running")
                            
                            # Clear previous document conversation history when a new document is uploaded
                            if "messages" in st.session_state:
                                # Keep only non-document related messages
                                st.session_state.messages = [msg for msg in st.session_state.messages 
                                                           if not msg.get("is_summary", False) and 
                                                           not (msg.get("document_related", False))]
                            
                            # Clean up old document embeddings
                            if "document_embeddings" in st.session_state:
                                try:
                                    # Try to delete the collection if it exists
                                    if hasattr(st.session_state, "doc_collection_name"):
                                        logger.info(f"Cleaning up old document collection: {st.session_state.doc_collection_name}")
                                        from langchain_chroma import Chroma
                                        client = st.session_state.document_embeddings._client
                                        if client.get_collection(st.session_state.doc_collection_name) is not None:
                                            client.delete_collection(st.session_state.doc_collection_name)
                                except Exception as e:
                                    logger.warning(f"Error cleaning up old document collection: {e}")
                                
                                # Remove the reference
                                st.session_state.document_embeddings = None
                            
                            # Store in session state
                            st.session_state.uploaded_document = doc_text
                            st.session_state.document_summary = summary
                            st.session_state.document_name = uploaded_file.name
                            
                            # Switch to document mode automatically
                            st.session_state.active_mode = "document"
                            
                            # Add summary as a special message in the chat
                            if "messages" in st.session_state:
                                # Add a system message with the summary
                                summary_message = {
                                    "role": "assistant",
                                    "content": f"📄 **Document Summary: {uploaded_file.name}**\n\n{summary}",
                                    "is_summary": True,  # Custom flag to style differently
                                    "document_related": True
                                }
                                st.session_state.messages.append(summary_message)
                            
                            status.update(label="Document ready for Q&A!", state="complete")
                    else:
                        st.error("Could not process document - no text chunks generated.")
                else:
                    st.error("Could not extract text from the PDF.")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error processing document: {e}")
                logger.error(f"Document processing error: {e}")
    
    # Mode selector (only show if a document is loaded)
    if st.session_state.uploaded_document is not None:
        st.divider()
        
        # Store previous mode before updating
        previous_mode = st.session_state.active_mode
        
        mode = st.radio(
            "Conversation Mode:",
            options=["Document Summary", "General Legal Q&A"],
            index=0 if st.session_state.active_mode == "document" else 1,
            key="mode_selector"
        )
        
        # Update active mode based on selection
        new_mode = "document" if mode == "Document Summary" else "general"
        
        # If mode has changed, clear conversation history for that mode
        if previous_mode != new_mode:
            # Clear the relevant part of conversation history
            if "messages" in st.session_state:
                if new_mode == "document":
                    # Keep only general messages when switching to document mode
                    # And add the document summary again
                    st.session_state.messages = [msg for msg in st.session_state.messages 
                                              if not msg.get("document_related", False)]
                    
                    # Re-add the document summary message
                    if st.session_state.document_summary:
                        summary_message = {
                            "role": "assistant",
                            "content": f"📄 **Document Summary: {st.session_state.document_name}**\n\n{st.session_state.document_summary}",
                            "is_summary": True,
                            "document_related": True
                        }
                        st.session_state.messages.append(summary_message)
                else:
                    # Keep only non-document messages when switching to general mode
                    st.session_state.messages = [msg for msg in st.session_state.messages 
                                              if not msg.get("document_related", False)]
            
            # Clear the appropriate session history in LegalRe
            law.clear_history(f"{st.session_state.thread_id}_{new_mode}")
            
            # Update the mode
            st.session_state.active_mode = new_mode
            
            # Show a message about switching modes
            st.info(f"Switched to {mode} mode. Previous conversation context has been cleared.")
            
            # Force a rerun to refresh the page with the updated state
            st.rerun()
        else:
            # Just update the mode if it hasn't changed
            st.session_state.active_mode = new_mode
        
        # Show current mode
        if st.session_state.active_mode == "document":
            st.info(f"📄 Answering questions about: {st.session_state.document_name}")
        else:
            st.info("💬 Using general legal knowledge")

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "document_related": st.session_state.active_mode == "document"
    })

    # Different processing based on mode
    try:
        with st.spinner("Thinking..."):
            if st.session_state.active_mode == "document" and st.session_state.uploaded_document:
                # Document mode: Create on-the-fly embeddings for the document
                # and query against it
                
                # Check if we need to generate document embeddings
                if "document_embeddings" not in st.session_state or st.session_state.document_name != getattr(st.session_state, "last_embedded_doc", None):
                    logger.info(f"Generating embeddings for document: {st.session_state.document_name}")
                    
                    try:
                        # Split the document text into chunks
                        doc_chunks = get_text_chunks(st.session_state.uploaded_document)
                        
                        if not doc_chunks:
                            st.error("Unable to extract meaningful chunks from the document.")
                            final_response = "I couldn't process this document properly. The document might be too complex or in a format I can't handle well."
                            raise ValueError("No document chunks generated")
                        
                        # Create document objects with metadata
                        from langchain.schema import Document
                        doc_objects = [
                            Document(
                                page_content=chunk, 
                                metadata={"filename": st.session_state.document_name, "source": "uploaded_document"}
                            ) 
                            for chunk in doc_chunks
                        ]
                        
                        # Generate a unique collection ID for this document
                        import hashlib
                        import time
                        doc_hash = hashlib.md5(st.session_state.document_name.encode()).hexdigest()[:10]
                        collection_name = f"user_doc_{doc_hash}_{int(time.time())}"
                        
                        # Create a persistent path for document collections
                        doc_persist_dir = os.path.join("temp_uploads", "doc_collections")
                        if not os.path.exists(doc_persist_dir):
                            os.makedirs(doc_persist_dir)
                        
                        # Create persistent vector store with document embeddings
                        from langchain_chroma import Chroma
                        st.session_state.document_embeddings = Chroma.from_documents(
                            documents=doc_objects,
                            embedding=embeddings,
                            collection_name=collection_name,
                            persist_directory=doc_persist_dir
                        )
                        
                        # Mark which document was embedded
                        st.session_state.last_embedded_doc = st.session_state.document_name
                        st.session_state.doc_collection_name = collection_name
                        logger.info(f"Successfully created document embeddings with {len(doc_chunks)} chunks in collection {collection_name}")
                        
                    except Exception as e:
                        logger.error(f"Error creating document embeddings: {e}")
                        st.error(f"Error creating document embeddings: {e}")
                        final_response = f"I had trouble processing your document. Error: {str(e)}"
                        # Default to general mode if document processing fails
                        st.session_state.active_mode = "general"
                        raise e
                
                # Use our own custom RAG for document-specific queries
                from langchain.chains import create_retrieval_chain
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain_core.prompts import ChatPromptTemplate

                # Create a document-specific prompt
                doc_prompt_template = ChatPromptTemplate.from_template("""
                You are a legal assistant analyzing a specific document uploaded by the user.
                
                Document Summary:
                {summary}
                
                <document_content>
                {context}
                </document_content>
                
                Answer the following question based ONLY on the information in the document above.
                If the answer is not in the document, say "I don't see information about that in the document."
                
                Always reference specific parts of the document in your answer to show where you found the information.
                
                Always conclude your answer with:
                
                References:
                - User-uploaded document: {document_name}
                
                Question: {input}
                """)
                
                # Create a chain to retrieve from the document embeddings
                document_chain = create_stuff_documents_chain(llm, doc_prompt_template)
                retriever = st.session_state.document_embeddings.as_retriever(
                    search_kwargs={"k": 5}
                )
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Run the chain on the query
                response = retrieval_chain.invoke({
                    "input": prompt,
                    "summary": st.session_state.document_summary,
                    "document_name": st.session_state.document_name
                })
                
                # Extract the answer and clean it up
                raw_response = response.get("answer", "I couldn't process that document query.")
                
                # Remove any <think> tags and their contents (some models include reasoning this way)
                import re
                final_response = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                # GENERAL LEGAL Q&A MODE
                # Use the LegalRe class with a fresh session ID to avoid context contamination
                # We'll create a session ID that's unique to the general mode to avoid mixing with document mode
                general_session_id = f"{thread_id}_general"
                
                # Make sure we're not filtering by any document
                result = law.conversational(
                    query=prompt,
                    session_id=general_session_id,  # Use a separate session for general queries
                    filename_filter=None  # No filter - search all documents in RAG
                )
                
                # Ensure the response doesn't mention the uploaded document
                if "Context document provided by the user" in result:
                    # Replace incorrect reference with proper citation
                    result = result.replace("Context document provided by the user", "Legal documents in knowledge base")
                
                final_response = result
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error during conversation: {e}")
        final_response = f"Sorry, I encountered an error: {e}"

    # Display the response
    with st.chat_message("assistant"):
        st.markdown(final_response)
    
    # Add to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_response, 
        "document_related": st.session_state.active_mode == "document"
    })
