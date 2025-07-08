# ⚖️ **LexiQ: AI-Powered Legal Assistant**

### _Bridging the Gap Between People and Legal Access_ 🌍

---

## 📚 **Legal Coverage**

**LexiQ** works with the PDF documents you provide in the `pdf_data` directory. The accuracy and scope of its responses depend entirely on the content of those documents.

Then upon running this file in **/src/pdf_emb.py** , vector embeddings of these pdf's get created, which is what is stored in **chroma_db_legal_bot_part1** folder

Then all that is required is to use **uv** installer , first you will have to set it up and then follow the steps below for successful setup.

A concise guide for developers on how to set this project up and run it is provided later below. 

## About LexiQ

 LexiQ integrates a dual-mode pipeline: a summarization module that 
generates concise, plain-language synopses of user-uploaded PDF documents, and a 
retrieval-augmented Q&A system built on a preloaded corpus of statutes, case laws, and 
regulations. The system is implemented in Python 3.11+ using Streamlit for the user 
interface, LangChain (with Groq LLM and Hugging Face embeddings), and ChromaDB for 
vector storage. Documents are parsed and chunked with PyMuPDF, embedded with all
MiniLM-L6-v2, and re-ranked using a cross-encoder (ms-marco-MiniLM-L-6-v2). LexiQ 
first retrieves contextually relevant excerpts for each query, synthesizes polished, Markdown
formatted responses with source citations, and falls back to its internal knowledge only when 
no documents match. 

## How the application looks like

![image](https://github.com/user-attachments/assets/940e7c33-3638-482f-a187-42d7991f993a)

*Query:*
*Hi, I recently moved out of a rented apartment, but my landlord is refusing to return my security deposit. He claims there was damage, but didn't provide any proof or receipts. It's been over 45 days. What are my legal rights and what steps can I take to get my money back?*

*Response: *

![image](https://github.com/user-attachments/assets/894dd687-0471-4b84-9440-2e717994ed9f)

![image](https://github.com/user-attachments/assets/d81f0055-a36d-4135-988d-5b679d903184)

*Say an user uploads a legal document, the bot will also summarize the document and would explain what the document is about concisely:*

![image](https://github.com/user-attachments/assets/28c8d2bf-24fe-4031-a701-e3b32340aac4)

*After taking around 10 ~ 40 seconds (Based on the size of the document) a response will be delivered*

*Here as an example I had uploaded a legal text document in pdf format. The user has also the ability to switch between two modes as well, either the initial QnA mode or the document summary mode as you can see below*

![image](https://github.com/user-attachments/assets/8281ae2f-6a39-4b9f-9943-26c220d2b25b)

*So this is the way the project is right now, and in the future I plan to deploy it and anyone would be able to access it through web!*


---

# 💻 **Developer Quick Start Guide**

Ready to get started? Follow these simple steps to set up **LegalRe** on your machine:

1. **Clone the Repository** 🌀

2. **Install uv** 📂

Install uv and set up your Python project and environment

MacOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

3. **Install Dependencies** 📦

   ```bash
   uv sync
   ```

4. **Set Your Groq API Key** 🔑

   Create a `.env` file and add your Groq API key as such:

   ```bash
   GROQ_API_KEY=your-api-key-here
   ```

5. **Add Your Documents** 📄

   Create a folder named `pdf_data` in the root of the project directory.
   Place all your legal PDF documents inside this `pdf_data` folder.

   If and only if you want to add new documents to existing files.
   If you have no new data to add **SKIP this and the NEXT step** - the embeddings already exist.

   Documents upon which the current chroma db database was made on :
   https://drive.google.com/drive/folders/1wOyEXU9lTpYsakiXuc0Lo3GgD_zC-wnm?usp=sharing

6. **Generate Embeddings** ✨
   Run the embedding script to process your PDFs and create the vector database:

   ```bash
   cd src
   python pdf_emb.py
   cd ..
   ```

7. **Run the Application** 🚀

   ```bash
   uv run streamlit run app.py
   ```

8. **Access the App** 🌐  
   Open your browser and visit:
   ```bash
   http://127.0.0.1:8501
   ```

---

## 🔧 **Tools & Technologies**

| 💡 **Technology**         | 🔍 **Description**                           |
| ------------------------- | -------------------------------------------- |
| **LangChain**             | Framework for building language applications |
| **ChromaDB**              | Vector database for RAG implementation       |
| **Groq API**              | Powering fast large language model inference |
| **Sentence Transformers** | Generating text embeddings                   |
| **Streamlit**             | Python web framework for the UI              |
| **PyPDF2**                | Extracting text from PDF files               |

---

---

## 🤝 **Contribute**

We are always looking for contributors! Whether you want to help with development, report issues, or request features, we welcome you to fork the repo and submit a pull request. Every contribution helps to make **LexiQ** better for everyone! 🚀

---

**LexiQ** is more than just an AI tool—it's a framework for building custom RAG applications for specific document sets. Together, let's make information more accessible! ✨
