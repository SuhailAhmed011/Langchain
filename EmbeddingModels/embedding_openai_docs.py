from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32);

documents = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Madrid is the capital of Spain",
    "Rome is the capital of Italy"
]

result = embedding.embed_documents(documents)

print(str(result))