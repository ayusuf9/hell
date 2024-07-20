from langchain_openai import OpenAIEmbeddings

class PersistentFAISS:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai.api_key,
                model="text-embedding-ada-002"
            )
        except TypeError:
            # If the above fails, try without specifying the model
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai.api_key
            )

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
        
class CustomGPT:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            usage = response.usage.total_tokens
            cost = float(usage) / 1000 * 0.03
            logger.info(f"Tokens: {usage}, Cost of Call: ${cost}")
            return answer
        except AttributeError:
            # Fall back to older API style if the above fails
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
        except Exception as e:
            logger.error(f"Error in CustomGPT: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."