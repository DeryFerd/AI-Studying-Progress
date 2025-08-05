import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI (MESIN RAG KITA)
# =====================================================================================

@st.cache_resource
def load_llm():
    """Memuat LLM dari Google Gemini API."""
    # Kita hanya perlu menginisialisasi model dengan nama dan API key
    # API key akan dibaca secara otomatis dari Streamlit Secrets
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return llm

# Versi baru yang menggunakan ChromaDB dan tugasnya lebih spesifik
def create_vector_store(text_chunks, embedding_model):
    """Membuat vector store ChromaDB dari potongan teks."""
    vector_store = Chroma.from_documents(documents=text_chunks, embedding=embedding_model)
    return vector_store

# GANTI SELURUH FUNGSI create_qa_chain DENGAN VERSI FINAL INI

# Ganti fungsi create_qa_chain lama
def create_qa_chain(vector_store, llm):
    """Membuat QA Chain, kembali ke 'stuff' karena Gemini punya konteks besar."""
    
    # Kita kembali ke retriever yang lebih sederhana
    retriever = vector_store.as_retriever()

    # Kita kembali ke satu prompt simpel
    prompt_template = """
    Berdasarkan konteks yang diberikan, jawablah pertanyaan berikut dengan jelas dan ringkas dalam Bahasa Indonesia.
    Konteks:
    {context}
    Pertanyaan:
    {question}
    Jawaban:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # KEMBALI KE "STUFF" yang lebih cepat!
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain
# =====================================================================================
# BAGIAN 2: TAMPILAN APLIKASI STREAMLIT
# =====================================================================================

st.set_page_config(page_title="Asisten Dokumen AI", page_icon="🤖")
st.title("🤖 AI Asisten Dokumen Pribadi")
st.write("Upload dokumen PDF Anda dan ajukan pertanyaan tentang isinya.")

# Inisialisasi session state untuk menyimpan QA chain
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar untuk upload file
with st.sidebar:
    st.header("Upload Dokumen Anda")
    uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

    # Ganti blok 'if uploaded_file' Anda dengan ini
if uploaded_file is not None:
    # Simpan file yang diupload ke disk sementara
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tombol untuk memproses dokumen
    if st.button("Proses Dokumen"):
        # PASTIKAN BLOK 'with' INI MENJOROK KE DALAM (ada 4 spasi)
        with st.spinner("Memproses dokumen... Ini bisa memakan waktu beberapa menit."):
            # 1. Load dan potong dokumen
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = text_splitter.split_documents(documents)

            # 2. Siapkan model embedding
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            device = 'cpu' # Pastikan menggunakan CPU
            embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})

            # 3. Buat Vector Store ChromaDB
            vector_store = create_vector_store(text_chunks, embedding_model)

            # 4. Load LLM
            llm = load_llm()

            # 5. Buat QA Chain dan simpan di session state
            st.session_state.qa_chain = create_qa_chain(vector_store, llm)

            st.success("Dokumen berhasil diproses! Silakan ajukan pertanyaan.")

# Kotak chat utama
st.header("Ajukan Pertanyaan")
user_question = st.text_input("Ketik pertanyaan Anda di sini...")

if user_question:
    if st.session_state.qa_chain is not None:
        with st.spinner("AI sedang berpikir..."):
           # LANGKAH TAMBAHAN: Cek dulu apakah ada dokumen yang relevan
            retriever = st.session_state.qa_chain.retriever
            relevant_docs = retriever.get_relevant_documents(user_question)
            
            if not relevant_docs:
                # Jika tidak ada dokumen relevan, beri pesan ke pengguna
                st.warning("Maaf, saya tidak dapat menemukan bagian yang relevan di dalam dokumen untuk menjawab pertanyaan tersebut. Coba ajukan pertanyaan yang lebih spesifik mengenai isi dokumen.")
            else:
                # Jika ada, baru jalankan proses tanya jawab lengkap
                result = st.session_state.qa_chain.invoke({"query": user_question})
                st.write("### Jawaban AI:")
                st.write(result['result'])
    else:
        st.warning("Mohon upload dan proses dokumen terlebih dahulu.")
