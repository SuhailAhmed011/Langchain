from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag.pdf")

doc = loader.load()

print(doc[0].metadata)