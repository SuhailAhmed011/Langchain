from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

documents = [
    "RAG combines retrieval with generation.",
    "Embeddings convert text into numerical vectors.",
    "Chroma is a vector database.",
    "Python is a programming language.",
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_texts(
    documents = documents,
    embedding = embeddings,
    collection_name = "demo"
)

retriever = vector_store.as_retriever(
    search_type = "Similarity",
    search_krwags = {"k" : 3}
)

results = retriever.invoke(
    "What are embeddings?"
)

for doc in results:
    print(doc.page_content)