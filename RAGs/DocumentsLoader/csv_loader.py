from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="docs/dummy.csv", encoding="utf-8")

docs = loader.load()

print(docs[2].page_content)
print(docs[2].metadata)