# Cell 1: Imports and Setup
import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import hashlib
import logging
from quanthub.util import llm
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the OpenAI client using your custom method
openai = llm.get_llm_client(llm.GPT_4_MODEL)

# Cell 2: Helper Functions
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def create_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        deployment_id="text-embedding-ada-002",
        input=[text]
    )
    return response["data"][0]["embedding"]

def process_pdf(file_path, cache_dir="./pdf_cache"):
    file_hash = get_file_hash(file_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"{file_hash}_index.faiss"
    pkl_path = cache_dir / f"{file_hash}_index.pkl"

    if index_path.exists() and pkl_path.exists():
        logger.info(f"Loading existing index for {file_hash}")
        vectorstore = FAISS.load_local(
            str(cache_dir),
            embeddings=create_embeddings,
            index_name=f"{file_hash}_index",
            allow_dangerous_deserialization=True
        )
    else:
        logger.info(f"Creating new index for {file_hash}")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(texts, create_embeddings)
        vectorstore.save_local(str(cache_dir), index_name=f"{file_hash}_index")
        with open(pkl_path, "wb") as f:
            pickle.dump({"hash": file_hash}, f)

    return vectorstore

# Cell 3: Query Engine Setup
def get_query_engine(vectorstore):
    def custom_gpt(prompt):
        try:
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
            logger.error(f"Error in custom_gpt: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

    qa_chain = RetrievalQA.from_chain_type(
        llm=custom_gpt,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    
    return qa_chain

# Cell 4: Process PDF and Create Query Engine
# Replace 'path/to/your/pdf/file.pdf' with the actual path to your PDF file
pdf_path = 'path/to/your/pdf/file.pdf'
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