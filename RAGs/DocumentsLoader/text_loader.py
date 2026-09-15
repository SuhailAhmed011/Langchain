from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'write a summary of the following article - \n {article}',
    input_variables = ["article"]
)

parser = StrOutputParser()

loader = TextLoader("ai.txt", encoding="utf8")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"article": docs[0].page_content})

print(result)
