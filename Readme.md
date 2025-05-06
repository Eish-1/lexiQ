# ⚖️ **LexiQ: AI-Powered Legal Assistant**

### _Bridging the Gap Between People and Legal Access_ 🌍

---

## 📚 **Legal Coverage**

**LegalRe** works with the PDF documents you provide in the `pdf_data` directory. The accuracy and scope of its responses depend entirely on the content of those documents.

Then upon running this file in **/src/pdf_emb.py** , vector embeddings of these pdf's get created, which is what is stored in **chroma_db_legal_bot_part1** folder

Then all that is required is to use **uv** installer , first you will have to set it up and then follow the steps below for successful setup.

---

## 💻 **Developer Quick Start Guide**

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

   Create a  `.env` file and add your Groq API key as such:

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
   

7. **Generate Embeddings** ✨
   Run the embedding script to process your PDFs and create the vector database:

   ```bash
   cd src
   python pdf_emb.py
   cd ..
   ```

8. **Run the Application** 🚀

   ```bash
   uv run streamlit run app.py
   ```

9. **Access the App** 🌐  
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

We are always looking for contributors! Whether you want to help with development, report issues, or request features, we welcome you to fork the repo and submit a pull request. Every contribution helps to make **LegalRe** better for everyone! 🚀

---

**LegalRe** is more than just an AI tool—it's a framework for building custom RAG applications for specific document sets. Together, let's make information more accessible! ✨
