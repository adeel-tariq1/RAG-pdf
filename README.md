# 📖 Chat with PDF (RAG-powered)

A **Streamlit web app** that allows you to **chat with your PDF documents** using a **Retrieval-Augmented Generation (RAG) pipeline**. Upload one or more PDFs, and ask questions about their content — the app retrieves relevant information and provides AI-generated answers in **concise, detailed, or bullet-point formats**.

---

## **Features**

* **Multi-PDF support:** Upload multiple PDFs and combine their content for querying.
* **RAG-powered:** Uses embeddings and a vector store (Chroma) to retrieve relevant text chunks.
* **Multiple LLM providers:** Groq, OpenAI, and Gemini supported.
* **Answer style options:**

  * **Concise:** 1–2 clear sentences.
  * **Detailed:** Well-structured short paragraphs.
  * **Bullet Points:** Key points listed line by line.
* **Simulated streaming:** Answers appear **sentence by sentence**, giving a live typing effect.
* **Export chat history:** Save your chat as a `.txt` file for reference.
* **Interactive Streamlit UI:** Clean, simple, and user-friendly interface.
* **Cached embeddings:** Repeated queries are faster thanks to caching.

---

## **How It Works**

1. **Upload PDFs:** Drag and drop one or more PDF files.
2. **Text Extraction:** The app reads and combines text from all pages.
3. **Text Splitting:** The text is split into manageable chunks.
4. **Embeddings:** Each chunk is converted into a vector using `SentenceTransformer`.
5. **Vector Store:** ChromaDB stores the embeddings and enables semantic search.
6. **Querying:** When you ask a question, the app finds relevant chunks and passes them to the selected LLM.
7. **Streaming Answer:** The response is displayed **sentence by sentence**, simulating live typing.

---

## **Installation**

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/rag-pdf-chat.git
cd rag-pdf-chat
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Select your **LLM provider** (Groq, OpenAI, or Gemini) and enter your **API key**.

3. Upload one or more **PDF files**.

4. Choose your **answer style**: Concise, Detailed, or Bullet Points.

5. Ask questions in the chat input.

6. Optionally, **export chat history** as a `.txt` file.

---

## **Dependencies**

* `streamlit` – Web app interface
* `pypdf` – PDF reading
* `sentence-transformers` – Embeddings
* `chromadb` – Vector store
* `langchain` – RAG orchestration
* LLM connectors:

  * `langchain-groq`
  * `langchain-openai`
  * `langchain-google-genai`
* `torch` – Required by `sentence-transformers`
* `numpy` – Required for embeddings

---

## **Folder Structure**

```
rag-pdf-chat/
│
├─ app.py             # Main Streamlit app
├─ requirements.txt   # Python dependencies
└─ README.md          # This documentation
```

> Since the app is single-file, no additional modules are required.

---

## **Tips**

* Large PDFs may take longer to process — caching helps for repeated queries.
* Always use **hosted LLMs**; running very large models locally is not feasible.
* Keep your API keys secure; for deployment on **Streamlit Cloud**, use **Streamlit secrets**.

---

## **Future Enhancements**

* Multi-language PDF support
* Persistent storage for uploaded PDFs and embeddings
* Improved streaming with true token-level generation
* Integration with cloud storage (e.g., S3) for PDF uploads
