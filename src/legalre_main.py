from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from sentence_transformers import CrossEncoder # Import CrossEncoder
import os # Import os for basename
import re # Import the regex module
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set default level
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Create a logger for this module
logger = logging.getLogger('legalre')

#This Class deals with working of Chatbot

class LegalRe:
    """This is the class which deals mainly with a conversational RAG
      It takes llm, embeddings and vector store as input to initialise.

      create an instance of it using law = LegalRe(llm,embeddings,vectorstore)
      In order to run the instance

      law.conversational(query)

      Example:
      law = LegalRe(llm,embeddings,vectorstore)
      query1 = "What is rule of Law?"
      law.conversational(query1)
      query2 = "Is it applicable in India?"
      law.conversational(query2)
    """
    store = {}
    # Make CrossEncoder lazy-loaded (only initialize when first needed)
    _cross_encoder = None
    
    @classmethod
    def get_cross_encoder(cls):
        if cls._cross_encoder is None:
            cls._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._cross_encoder

    def __init__(self,llm,embeddings,vector_store):
      self.llm = llm
      self.embeddings = embeddings
      self.vector_store = vector_store

    def __retriever(self, filename_filter=None):
      """The function to define the properties of retriever, optionally filtering by filename."""
      
      search_kwargs={ # Base search settings
              "k": 20, 
              "score_threshold": 0.2
          }
      
      # Add metadata filter if filename_filter is provided
      if filename_filter:
          search_kwargs['filter'] = {'filename': filename_filter}
          logger.info(f"Retriever filter: '{filename_filter}'")
      else:
           logger.info("General search (no filter)")
           
      retriever = self.vector_store.as_retriever(
          search_type="similarity_score_threshold", 
          search_kwargs=search_kwargs
          )
      return retriever

    @staticmethod 
    def rerank_documents(inputs):
        query = inputs['input']
        docs = inputs['context']
        
        if not docs:
            logger.info("No documents retrieved for re-ranking")
            return []
        
        logger.info(f"Re-ranking {len(docs)} documents")
        pairs = [[query, doc.page_content] for doc in docs]
        scores = LegalRe.get_cross_encoder().predict(pairs)
        docs_with_scores = list(zip(docs, scores))
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        top_n = 5
        reranked_docs = [doc for doc, score in sorted_docs[:top_n]]
        
        # Only log the essential information for top docs
        logger.info(f"Top {top_n} documents:")
        for i, doc in enumerate(reranked_docs):
            filename = doc.metadata.get('filename', 'Unknown')
            score = sorted_docs[i][1]
            logger.info(f"  Doc {i+1}: {filename} (Score: {score:.4f})")
        
        return reranked_docs

    def llm_answer_generator(self, query, filename_filter=None):
      llm = self.llm
      # Pass filter to retriever
      retriever = self.__retriever(filename_filter=filename_filter) 

      contextualize_q_system_prompt = (
          "Given a chat history and the latest user question "
          "which might reference context in the chat history, "
          "formulate a standalone question which can be understood "
          "without the chat history. Do NOT answer the question, "
          "just reformulate it if needed and otherwise return it as is."
      )

      contextualize_q_prompt = ChatPromptTemplate.from_messages(
          [
              ("system", contextualize_q_system_prompt),
              MessagesPlaceholder("chat_history"),
              ("human", "{input}"),
          ]
      )
      history_aware_retriever = create_history_aware_retriever(
          llm, retriever, contextualize_q_prompt
      )
      
      reranker_step = RunnableLambda(LegalRe.rerank_documents)

      # --- Enhanced System Prompt (Slightly adapted for clarity) ---
      # This prompt is used by BOTH branches. The presence/absence of {context}
      # in the final formatted prompt tells the LLM which path to follow.
      system_prompt_template = """You are LegalRe, a highly specialized AI assistant for analyzing Indian legal texts. Your PRIMARY RESPONSIBILITY is to answer questions using ONLY the specific documents that are retrieved for each query.
      
Follow this workflow STRICTLY:

1. **Document Analysis:** When given a query and context documents, your FIRST priority is to thoroughly analyze the provided documents.

2. **Response Generation Rules:**
   - **MANDATORY:** Base your answers EXCLUSIVELY on the information found in retrieved documents. This is your PRIMARY DIRECTIVE.
   - Do NOT use your general knowledge or training data unless EXPLICITLY stated that no relevant documents were found.
   - If the documents contain partial information, state what was found and acknowledge what's missing - DO NOT fill gaps with general knowledge.
   - Cite evidence from documents frequently, using format "According to [filename]..."
   - When multiple documents contain relevant information, synthesize a response that incorporates all sources.
   - **Always** include a References section listing the exact filenames used.
   
3. **If No Documents Found:**
   - Only in this specific case, clearly state: "I don't have specific documents about this topic."
   - Then provide a general response with a disclaimer that it's based on general knowledge.
   - Mark references as "General knowledge" only in this case.

4. **Formatting:**
   - Use standard Markdown 
   - Structure answers with paragraphs for readability
   - Include bullet points where appropriate
   - Always end with References section

Remember: Your value comes primarily from accurately reporting document content, not generating answers from general knowledge. Think of yourself as a legal document analyst first, AI assistant second.

User Query: {input}

Context: {context}

Answer:"""
      # --- End Enhanced System Prompt ---

      # --- Define QA chains --- 
      
      # 1. Chain for when documents ARE found (uses create_stuff_documents_chain)
      # This prompt expects 'input', 'chat_history', and 'context' (as List[Document])
      qa_prompt_with_context = ChatPromptTemplate.from_messages(
              [
                  ("system", system_prompt_template),
                  MessagesPlaceholder(variable_name="chat_history"),
                  ("human", "{input}")
              ]
          )
      question_answer_chain_with_context = create_stuff_documents_chain(llm, qa_prompt_with_context)

      # 2. Chain for when documents are NOT found
      # This prompt template only needs 'input' and 'chat_history'.
      # The system message instructs the LLM on fallback behavior.
      qa_prompt_without_context = ChatPromptTemplate.from_messages(
          [
              ("system", """You are LegalRe, a highly specialized AI assistant for analyzing Indian legal texts.
              
I don't have specific documents about this topic in my database. I'll provide a general response based on my training.

Please note that this answer is NOT based on specific legal documents but on general knowledge."""),
              MessagesPlaceholder(variable_name="chat_history"),
              ("human", "{input}")
          ]
      )
      # This chain just formats the prompt and sends it to the LLM.
      question_answer_chain_without_context = qa_prompt_without_context | llm 

      # --- Define the Branching Logic ---
      
      # Condition: Check if the 'context' key (after re-ranking) is empty list
      def check_if_docs_exist(inputs):
            context = inputs.get('context', [])
            logger.info(f"Checking docs for branching. Found: {len(context)} documents.")
            return bool(context) # True if list is not empty

      # Define the branch runnable
      branch = RunnableBranch(
            # If check_if_docs_exist is True, run the context-aware chain
            (check_if_docs_exist, question_answer_chain_with_context),
            # Otherwise (no docs found), run the fallback chain
            question_answer_chain_without_context 
      )

      # --- Final RAG Chain Construction ---
      rag_chain_pipeline = (
            # Initial retrieval step - get documents based on history
            RunnablePassthrough.assign(
                context=history_aware_retriever # Output: {'input':..., 'chat_history':..., 'context': List[Docs] or []}
            )
            # Re-ranking step - refine the context list
            | RunnablePassthrough.assign(
                context=reranker_step # Output: {'input':..., 'chat_history':..., 'context': List[Docs] or []}
            )
            # Branching step - choose the correct QA chain based on context
            | branch # Output: String (from LLM or chain)
            # Wrap the final output (string from LLM or AIMessage) into the required dict
            | RunnableLambda(lambda final_output: {"answer": final_output.content if hasattr(final_output, 'content') else str(final_output)}) 
      )
      # --- End Final RAG Chain Construction ---
      
      return rag_chain_pipeline

    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
      # Updated reference to class name for store
      if session_id not in LegalRe.store:
          LegalRe.store[session_id] = ChatMessageHistory()
      return LegalRe.store[session_id]
    
    def conversational(self, query, session_id, filename_filter=None):
      # Pass filter to generator
      rag_chain_with_history_handling = self.llm_answer_generator(query, filename_filter=filename_filter) 
      
      conversational_rag_chain = RunnableWithMessageHistory(
          rag_chain_with_history_handling, 
          self.get_session_history,
          input_messages_key="input",
          history_messages_key="chat_history",
          output_messages_key="answer"
      )
      response = conversational_rag_chain.invoke(
          {"input": query}, 
          config={
              "configurable": {"session_id": session_id}
          },
      )
      
      # --- Add Debugging ---
      # print(f"--- DEBUG: Full response from wrapped chain: {response}") 
      # print(f"--- DEBUG: Type of response: {type(response)}")

      # Extract the answer string
      final_answer = "Error: Could not extract answer."
      if isinstance(response, dict):
          final_answer = response.get('answer', "Error: 'answer' key not found in response dict.")
      else:
          # print(f"--- WARNING: Response was not a dict, received type: {type(response)}")
          final_answer = str(response) if response is not None else "Error: Received None response."
      
      # Handle potential nested dicts (less likely now but safe)
      if isinstance(final_answer, dict): 
          final_answer = final_answer.get('answer', "Error: Unexpected nested response structure.")

      final_answer_str = str(final_answer) # Ensure it's a string

      # --- Add Output Parsing --- 
      # Use regex to remove <think>...</think> blocks (case-insensitive, multi-line)
      cleaned_answer = re.sub(r"<think>.*?</think>\s*", "", final_answer_str, flags=re.IGNORECASE | re.DOTALL).strip()
      # --- End Output Parsing --- 
      
      # --- Add Debugging ---
      # print(f"--- DEBUG: Original final_answer_str: {final_answer_str}")
      # print(f"--- DEBUG: Cleaned answer: {cleaned_answer}")
      return cleaned_answer 