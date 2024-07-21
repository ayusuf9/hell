# Import necessary libraries
import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain_community.chat_models import AzureChatOpenAI
import hashlib
import logging
from tqdm import tqdm
from langchain.vectorstores import Chroma
import pickle
from langchain.embeddings.base import Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Azure OpenAI Embeddings class
class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, azure_openai_client, model="text-embedding-ada-002"):
        self.client = azure_openai_client
        self.model = model

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.client.embeddings.create(input=text, model=self.model).data[0].embedding

def get_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()
    return file_hash

def process_pdf(file_path, llm):
    file_hash = get_file_hash(file_path)
    cache_dir = Path(f"./cache/{file_hash}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_persist_dir = cache_dir / "chroma"
    texts_cache_path = cache_dir / "split_texts.pkl"

    # Check if processed data already exists
    if chroma_persist_dir.exists() and texts_cache_path.exists():
        logger.info(f"Loading from cache: {cache_dir}")
        try:
            openai_client = llm.get_azure_openai_client()
            embeddings = AzureOpenAIEmbeddings(openai_client)
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

        openai_client = llm.get_azure_openai_client()
        embeddings = AzureOpenAIEmbeddings(openai_client)
        
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

def get_query_engine(vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    return qa_chain

def ask_question(question, vectorstore, llm):
    qa_chain = get_query_engine(vectorstore, llm)
    response = qa_chain.invoke({"query": question})
    return response["result"]

# Main execution
pdf_path = '/Users/ayusuf/Desktop/Finance /Cohere/annualreport-2023.pdf'

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="your-deployment-name",
    temperature=0
)

vectorstore, texts = process_pdf(pdf_path, llm)

def get_answer(question):
    if vectorstore:
        answer = ask_question(question, vectorstore, llm)
        print(f"Q: {question}\nA: {answer}\n")
    else:
        print("PDF not processed successfully. Please check the file path and try again.")

# Example usage
question = "From the Selected income statement data, what is the Total net revenue in 2023"
get_answer(question)