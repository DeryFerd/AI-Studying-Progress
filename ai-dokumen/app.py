
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
from langchain_community.vectorstores import FAISS

# =====================================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI (MESIN RAG KITA)
# =====================================================================================

@st.cache_resource # Trik agar model tidak di-load ulang setiap kali ada interaksi
def load_llm():
    """Memuat LLM yang sudah dikuantisasi."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def create_vector_store(pdf_path):
    """Membuat vector store dari file PDF."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
    
    vector_store = FAISS.from_documents(text_chunks, embedding_model)
    return vector_store

def create_qa_chain(vector_store, llm):
    """Membuat QA Chain dengan prompt kustom."""
    retriever = vector_store.as_retriever()
    
    prompt_template = """
    Gunakan potongan-potongan konteks berikut untuk menjawab pertanyaan di akhir.
    Jawablah dengan ringkas dan jelas dalam Bahasa Indonesia.
    Jika Anda tidak tahu jawabannya berdasarkan konteks, katakan saja "Saya tidak menemukan jawaban di dalam dokumen."

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Jawaban Membantu:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False, # Kita set False agar jawaban lebih ringkas
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
            with st.spinner("Memproses dokumen... Ini bisa memakan waktu beberapa menit."):
                # 1. Buat Vector Store
                vector_store = create_vector_store(file_path)
                
                # 2. Load LLM (ini akan menggunakan cache)
                llm = load_llm()
                
                # 3. Buat QA Chain dan simpan di session state
                st.session_state.qa_chain = create_qa_chain(vector_store, llm)
                
                st.success("Dokumen berhasil diproses! Silakan ajukan pertanyaan.")

# Kotak chat utama
st.header("Ajukan Pertanyaan")
user_question = st.text_input("Ketik pertanyaan Anda di sini...")

if user_question:
    if st.session_state.qa_chain is not None:
        with st.spinner("AI sedang berpikir..."):
            result = st.session_state.qa_chain.invoke({"query": user_question})
            st.write("### Jawaban AI:")
            st.write(result['result'])
    else:
        st.warning("Mohon upload dan proses dokumen terlebih dahulu.")
