from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


documents = [
    Document(
        page_content="RAG combines retrieval with generation."
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors."
    ),
    Document(
        page_content="Chroma is a vector database used for similarity search."
    ),
    Document(
        page_content="Vector databases store embeddings and allow semantic search."
    ),
    Document(
        page_content="Retrievers find relevant documents for a user query."
    ),
]


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="multi_query_demo"
)


base_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)


results = retriever.invoke(
    "How does RAG find information?"
)


for i, doc in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(doc.page_content)