import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

def process_pdf(uploaded_file):
    file_hash = get_file_hash(uploaded_file)
    cache_dir = Path(f"./cache/{file_hash}")
    faiss_index_path = cache_dir / "index.faiss"

    if faiss_index_path.exists():
        logger.info(f"Loading from cache: {cache_dir}")
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading cached index: {str(e)}. Reprocessing PDF.")
    else:
        logger.info("Cache not found. Processing new PDF.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        chunk_size = 1000
        chunk_overlap = 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        texts = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.from_documents(texts, embeddings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))

        logger.info(f"Vectorstore saved to {cache_dir}")
        
        os.unlink(temp_file_path)
        return vectorstore

    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return None

def get_query_engine(vectorstore):
    llm = OpenAI(temperature=0.7)  # Increased temperature for more varied responses
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    
    return qa_chain

st.title("Chat with your PDF")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF... This may take a while for large files."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
    
    if st.session_state.vectorstore:
        st.success("PDF processed successfully!")
    else:
        st.error("Failed to process the PDF. Please try again.")

if st.session_state.vectorstore:
    query_engine = get_query_engine(st.session_state.vectorstore)
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        with st.spinner("Generating response..."):
            try:
                response = query_engine({"query": user_question})
                st.write(response["result"])
            except Exception as e:
                logger.error(f"An error occurred while generating the response: {str(e)}")
                st.error("An error occurred while generating the response. Please try again.")
else:
    st.info("Please upload a PDF file to get started.")