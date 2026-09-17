from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor



documents = [
    Document(
        page_content="""
        Artificial Intelligence is a field of computer science.
        It is used in healthcare, finance, education, and many other industries.
        AI systems can analyze large amounts of data.
        """
    ),

    Document(
        page_content="""
        RAG stands for Retrieval-Augmented Generation.
        RAG retrieves relevant information from external documents
        and provides that information to an LLM as context.
        RAG is commonly used for question answering.
        """
    ),

    Document(
        page_content="""
        Chroma is a vector database.
        It stores embeddings and allows semantic similarity search.
        Chroma can be used as the vector store in a RAG application.
        """
    ),
]


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)



vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="compression_demo"
)



base_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


compressor = LLMChainExtractor.from_llm(
    llm
)


retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)



results = retriever.invoke(
    "What is RAG?"
)


for i, doc in enumerate(results):

    print(f"\nResult {i + 1}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)