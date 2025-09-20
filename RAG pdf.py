# MUST BE THE FIRST LINES IN YOUR SCRIPT
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import time


def validate_api(provider, api_key):
    """Initialize and return the LLM based on the provider and API key."""
    if not api_key:
        st.warning(f"Please enter your {provider} API key to continue.")
        st.stop()
    
    try:
        if provider == "Groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.5, streaming=True)
        elif provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.5, streaming=True)
        elif provider == "Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key, temperature=0.5, streaming=True)
        st.success(f"✅ {provider} API key validated successfully!")
        return llm
    except Exception as e:
        st.error(f"❌ Invalid API Key for {provider}: {e}")
        st.stop()


def read_pdfs(uploaded_files):
    """Read and combine text from uploaded PDFs."""
    combined_text = ""
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                combined_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")
    if not combined_text.strip():
        st.error("⚠️ No text could be extracted from the uploaded PDFs.")
        st.stop()
    return combined_text


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings(text_chunks, model):
    """Create embeddings for the text chunks."""
    return model.encode(text_chunks).tolist()


def setup_chroma(collection_name, chunks, embeddings):
    """Initialize Chroma client and collection, add documents."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection


def build_prompt():
    """Return a ChatPromptTemplate for the LLM."""
    return ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the information from the provided context to answer the user’s question clearly and meaningfully. 

- Always rely only on the context. Do not use outside knowledge.  
- If the context has partial information, give the best possible answer and note that some details may be missing.  
- If the answer truly cannot be found in the context, say: "I don't know."  

Context:
{context}

Question:
{question}

Answer Style: {answer_style}

Instructions for formatting:
- If style = "Concise": reply in 1–2 clear sentences.  
- If style = "Detailed": write only in paragraph style and it should be short, well-structured paragraph.  
- If style = "Bullet Points": list the key points in proper bullet form. Print it line by line, not like paragraph style
""")


def stream_response(chain, context, query, answer_style):
    """Get LLM response and simulate streaming by sentence chunks."""
    placeholder = st.empty()
    full_response = chain.invoke({
        "context": context,
        "question": query,
        "answer_style": answer_style
    })
    text = full_response.content
    displayed = ""
    for s in text.split("."):
        if s.strip():
            displayed += s.strip() + ". "
            placeholder.markdown(displayed)
            time.sleep(0.3)
    return text


# Sidebar -------------------
st.markdown("""
<style>
    section[data-testid="stSidebar"] .stMarkdown { margin-top: -30px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📖 Chat with PDF (RAG-powered)")
    provider = st.selectbox(" Choose LLM Provider", ["Groq", "OpenAI", "Gemini"])
    api_key = st.text_input(f"🔑 Enter your {provider} API Key", type="password")

# Validate API and initialize LLM
llm = validate_api(provider, api_key)

with st.sidebar:
    uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    answer_style = st.radio("✍️ Answer Style", ["Concise", "Detailed", "Bullet Points"])

if not uploaded_files:
    st.warning("Please upload at least one PDF.")
    st.stop()

# ------------------- Main Processing -------------------
text = read_pdfs(uploaded_files)

rec_split = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = rec_split.split_text(text)

model = load_embedding_model()
embeddings = create_embeddings(chunks, model)

collection = setup_chroma("story", chunks, embeddings)

prompt = build_prompt()
chain = prompt | llm

# ------------------- Streamlit Chat UI -------------------
st.write("✅ Now you can chat with your uploaded PDF!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about the PDF..."):
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    retrieved_documents = results["documents"][0]
    context = " ".join(retrieved_documents)

    # Streamed assistant response
    response_text = stream_response(chain, context, query, answer_style)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

#  Export Chat -------------------
if st.button("💾 Export Chat as TXT"):
    chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
    st.download_button(" Download Chat", chat_text, file_name="chat_history.txt")


