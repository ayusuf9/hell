# Import necessary libraries
import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import hashlib
import logging
from tqdm import tqdm
from langchain.vectorstores import Chroma
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()
    return file_hash

def process_pdf(file_path):
    file_hash = get_file_hash(file_path)
    cache_dir = Path(f"./cache/{file_hash}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_persist_dir = cache_dir / "chroma"
    texts_cache_path = cache_dir / "split_texts.pkl"

    # Check if processed data already exists
    if chroma_persist_dir.exists() and texts_cache_path.exists():
        logger.info(f"Loading from cache: {cache_dir}")
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(persist_directory=str(chroma_persist_dir), embedding_function=embeddings)
            with open(texts_cache_path, "rb") as f:
                texts = pickle.load(f)
            return vectorstore, texts
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}. Reprocessing PDF.")

    logger.info("Cache not found or incomplete. Processing new PDF.")

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        chunk_size = 2000
        chunk_overlap = 300

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Process pages in batches
        batch_size = 50
        all_texts = []
        for i in tqdm(range(0, len(pages), batch_size), desc="Processing PDF"):
            batch = pages[i:i+batch_size]
            texts = text_splitter.split_documents(batch)
            all_texts.extend(texts)

        # Save split texts
        with open(texts_cache_path, "wb") as f:
            pickle.dump(all_texts, f)

        embeddings = OpenAIEmbeddings()
        
        # Create and persist Chroma vectorstore
        vectorstore = Chroma.from_documents(
            documents=all_texts,
            embedding=embeddings,
            persist_directory=str(chroma_persist_dir)
        )
        vectorstore.persist()

        logger.info(f"Vectorstore saved to {chroma_persist_dir}")
        
        return vectorstore, all_texts

    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        return None, None

def get_query_engine(vectorstore):
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4-1106-preview")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    return qa_chain

def ask_question(question, vectorstore):
    qa_chain = get_query_engine(vectorstore)
    response = qa_chain.invoke({"query": question})
    return response["result"]

# Main execution
pdf_path = '/Users/ayusuf/Desktop/Finance /Cohere/annualreport-2023.pdf'
vectorstore, texts = process_pdf(pdf_path)

def get_answer(question):
    if vectorstore:
        answer = ask_question(question, vectorstore)
        print(f"Q: {question}\nA: {answer}\n")
    else:
        print("PDF not processed successfully. Please check the file path and try again.")

# Example usage
question = "From the Selected income statement data, what is the Total net revenue in 2023"
get_answer(question)