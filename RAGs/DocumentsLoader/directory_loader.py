from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    path="RAGs/DocumentsLoader/docs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

doc = loader.load() # if we have many documents than we can use lazy_loading to load the documents one by one instead of loading all at once

print(doc[0].page_content)