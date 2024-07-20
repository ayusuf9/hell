# Cell 1: Imports and Setup
import os
import tempfile
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
import hashlib
import logging
from quanthub.util import llm
import pickle
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the OpenAI client using your custom method
openai = llm.get_llm_client(llm.GPT_4_MODEL)

# Cell 2: Custom Embeddings and Helper Functions
class CustomOpenAIEmbeddings(Embeddings):
    def __init__(self, openai_client):
        self.openai = openai_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.openai.Embedding.create(
                model="text-embedding-ada-002",
                deployment_id="text-embedding-ada-002",
                input=[text]
            )
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.openai.Embedding.create(
            model="text-embedding-ada-002",
            deployment_id="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

# Cell 3: PersistentFAISS and PDF Processing
class PersistentFAISS:
    def __init__(self, cache_dir, openai_client):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = CustomOpenAIEmbeddings(openai_client)

    def load_or_create(self, texts, file_hash):
        index_path = self.cache_dir / f"{file_hash}_index.faiss"
        pkl_path = self.cache_dir / f"{file_hash}_index.pkl"

        if index_path.exists() and pkl_path.exists():
            with open(pkl_path, "rb") as f:
                stored_data = pickle.load(f)
            vectorstore = FAISS.load_local(str(self.cache_dir), self.embeddings, index_name=f"{file_hash}_index")
            logger.info(f"Loaded existing index for {file_hash}")
            return vectorstore
        else:
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(str(self.cache_dir), index_name=f"{file_hash}_index")
            with open(pkl_path, "wb") as f:
                pickle.dump({"hash": file_hash}, f)
            logger.info(f"Created and saved new index for {file_hash}")
            return vectorstore

def process_pdf(file_path, persistent_faiss):
    file_hash = get_file_hash(file_path)
    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_documents(pages)
        vectorstore = persistent_faiss.load_or_create(texts, file_hash)
        return vectorstore

    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        return None

# Cell 4: CustomGPT and Query Engine
class CustomGPT:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt, **kwargs):
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
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
            logger.error(f"Error in CustomGPT: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

def get_query_engine(vectorstore):
    custom_gpt = CustomGPT(openai, llm.GPT_35_16K_MODEL)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=custom_gpt,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    
    return qa_chain

# Cell 5: Process PDF and Create Query Engine
# Initialize PersistentFAISS
persistent_faiss = PersistentFAISS(cache_dir="./pdf_cache", openai_client=openai)

# Replace 'path/to/your/pdf/file.pdf' with the actual path to your PDF file
pdf_path = 'path/to/your/pdf/file.pdf'
vectorstore = process_pdf(pdf_path, persistent_faiss)

if vectorstore:
    query_engine = get_query_engine(vectorstore)
    print("PDF processed successfully and query engine is ready!")
else:
    print("Failed to process the PDF. Please check the file path and try again.")

# Cell 6: Query the PDF
def ask_question(question):
    if vectorstore:
        try:
            response = query_engine({"query": question})
            return response["result"]
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