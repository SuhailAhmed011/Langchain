from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Create documents
documents = [
    Document(
        page_content="Black cotton T-shirt for men",
        metadata={"category": "clothing"}
    ),
    Document(
        page_content="Blue denim jeans for men",
        metadata={"category": "clothing"}
    ),
    Document(
        page_content="Running shoes with comfortable cushioning",
        metadata={"category": "shoes"}
    ),
    Document(
        page_content="Wireless Bluetooth headphones",
        metadata={"category": "electronics"}
    )
]


# Create embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="products"
)


# Similarity search
results = vector_store.similarity_search(
    "dark colored shirt",
    k=2
)


# Display results
for result in results:
    print("Content:", result.page_content)
    print("Metadata:", result.metadata)
    print("-" * 50)