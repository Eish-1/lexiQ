# ⚖️ **LegalRe: AI-Powered Legal Assistant**

### _Bridging the Gap Between People and Legal Access_ 🌍

---

## 📚 **Legal Coverage**

**LegalRe** works with the PDF documents you provide in the `pdf_data` directory. The accuracy and scope of its responses depend entirely on the content of those documents.

---

## 💻 **Developer Quick Start Guide**

Ready to get started? Follow these simple steps to set up **LegalRe** on your machine:

1. **Clone the Repository** 🌀

2. **Install uv** 📂

   First, let's install uv and set up our Python project and environment

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

   Open `.env` and add your Groq API key:

   ```bash
   GROQ_API_KEY=your-api-key-here
   ```

5. **Add Your Documents** 📄
   Create a folder named `pdf_data` in the root of the project directory.
   Place all your legal PDF documents inside this `pdf_data` folder.

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

We are always looking for contributors! Whether you want to help with development, report issues, or request features, we welcome you to fork the repo and submit a pull request. Every contribution helps to make **LegalRe** better for everyone! 🚀

---

**LegalRe** is more than just an AI tool—it's a framework for building custom RAG applications for specific document sets. Together, let's make information more accessible! ✨
