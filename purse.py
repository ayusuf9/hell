# Cell 1: Imports and Setup
import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import hashlib
import logging
from quanthub.util import llm
import pickle
import faiss
import numpy as np
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the OpenAI client using your custom method
openai = llm.get_llm_client(llm.GPT_4_MODEL)

# Cell 2: Helper Functions and Classes
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def create_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        deployment_id="text-embedding-ada-002",
        input=[text]
    )
    return response["data"][0]["embedding"]

class EmbeddingFunction:
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [create_embedding(text) for text in texts]

def process_pdf(file_path, cache_dir="./pdf_cache"):
    file_hash = get_file_hash(file_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"{file_hash}_index.faiss"
    pkl_path = cache_dir / f"{file_hash}_index.pkl"

    embedding_func = EmbeddingFunction()

    if index_path.exists() and pkl_path.exists():
        logger.info(f"Loading existing index for {file_hash}")
        try:
            with open(pkl_path, "rb") as f:
                stored_data = pickle.load(f)
            
            if isinstance(stored_data, dict) and "docstore" in stored_data and "index_to_docstore_id" in stored_data:
                docstore = stored_data["docstore"]
                index_to_docstore_id = stored_data["index_to_docstore_id"]
                index = faiss.read_index(str(index_path))
                vectorstore = FAISS(embedding_func, index, docstore, index_to_docstore_id)
                logger.info("Successfully loaded existing index")
                return vectorstore
            else:
                logger.warning("Stored data format is incorrect. Creating new index.")
                return create_new_index(file_path, cache_dir, file_hash, embedding_func)
        except Exception as e:
            logger.error(f"Error loading existing index: {str(e)}. Creating new index.")
            return create_new_index(file_path, cache_dir, file_hash, embedding_func)
    else:
        logger.info("No existing index found. Creating new index.")
        return create_new_index(file_path, cache_dir, file_hash, embedding_func)

def create_new_index(file_path, cache_dir, file_hash, embedding_func):
    logger.info(f"Creating new index for {file_hash}")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_documents(pages)
    
    vectorstore = FAISS.from_documents(texts, embedding_func)
    
    # Save the FAISS index
    vectorstore.save_local(str(cache_dir), index_name=f"{file_hash}_index")
    
    logger.info("New index created and saved successfully")
    return vectorstore

# Cell 3: Query Engine Setup
def get_query_engine(vectorstore):
    def query_func(query: str) -> str:
        try:
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(query, k=5)
            
            # Prepare context from retrieved documents
            context = "\n".join([doc.page_content for doc in docs])
            
            # Prepare the prompt
            prompt = f"""Given the following context, please answer the question.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model=llm.GPT_35_16K_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            answer = response.choices[0].message['content'].strip()
            usage = response['usage']['total_tokens']
            cost = float(usage) / 1000 * 0.03
            logger.info(f"Tokens: {usage}, Cost of Call: ${cost}")
            return answer
        except Exception as e:
            logger.error(f"Error in query_func: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."
    
    return query_func

# Cell 4: Process PDF and Create Query Engine
pdf_path = '/users/CFII_DataScience/USERs/SPTADM/test_llm/doc.pdf'
vectorstore = process_pdf(pdf_path)

if vectorstore:
    query_engine = get_query_engine(vectorstore)
    print("PDF processed successfully and query engine is ready!")
else:
    print("Failed to process the PDF. Please check the file path and try again.")

# Cell 5: Query the PDF
def ask_question(question):
    if vectorstore:
        try:
            response = query_engine(question)
            return response
        except Exception as e:
            logger.error(f"An error occurred while generating the response: {str(e)}")
            return "An error occurred while generating the response. Please try again."
    else:
        return "Please process a PDF file first."

# Example usage
question = "What is the main topic of the PDF?"
answer = ask_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")