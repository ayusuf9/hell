# Cell 1: Imports and Setup
import os
import tempfile
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import hashlib
import logging
from quanthub.util import llm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the OpenAI client using your custom method
openai = llm.get_llm_client(llm.GPT_4_MODEL)
openai_api_key = openai.api_key

# Cell 2: Helper Functions
def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

def process_pdf(file_path):
    file_hash = get_file_hash(open(file_path, 'rb'))
    cache_dir = Path(f"./cache/{file_hash}")
    faiss_index_path = cache_dir / "index.faiss"

    if faiss_index_path.exists():
        logger.info(f"Loading from cache: {cache_dir}")
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading cached index: {str(e)}. Reprocessing PDF.")
    
    logger.info("Cache not found. Processing new PDF.")
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        chunk_size = 1000
        chunk_overlap = 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        texts = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        vectorstore = FAISS.from_documents(texts, embeddings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))

        logger.info(f"Vectorstore saved to {cache_dir}")
        
        return vectorstore

    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        return None

# Cell 3: Custom GPT Class
class CustomGPT:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt, **kwargs):
        try:
            response = self.client.ChatCompletion.create(
                deployment_id=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            answer = response.choices[0].message['content'].strip()
            usage = response["usage"]['total_tokens']
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