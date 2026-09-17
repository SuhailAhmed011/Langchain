from langchain_community.retrievers import WikipediaRetriever

# used llm here

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

result = retriever.invoke("Artificial Intelligence")

for doc in result:
    print(doc.page_content)