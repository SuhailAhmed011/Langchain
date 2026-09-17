#  Text Splitters in LangChain

# Text splitting is the process of breaking a large document into smaller
# pieces called **chunks**.

# Why do we split documents?

# - Large documents may exceed the model's context window.
# - Smaller chunks improve retrieval accuracy in RAG.
# - Relevant information can be retrieved more precisely.
# - Smaller chunks are easier to process and embed.

# Basic RAG flow:

# Document
#    ↓
# Text Splitter
#    ↓
# Chunks
#    ↓
# Embeddings
#    ↓
# Vector Database
#    ↓
# Retriever
#    ↓
# LLM
#    ↓
# Answer


#  1. Length-Based Text Splitter

#  Concept

# Length-based splitters divide text according to a fixed size such as
# characters or tokens.

# They do not understand the meaning of the text 



from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../DocumentsLoader/docs/rag.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
    separator = " "
)

result = splitter.split_documents(docs)

print(result)