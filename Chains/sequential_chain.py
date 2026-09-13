from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

model = ChatHuggingFace(llm=llm)

prompt1 = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

chain1 = prompt1 | model | StrOutputParser()

prompt2 = ChatPromptTemplate.from_template(
    "Summarize the following explanation in 3 points:\n\n{text}"
)

chain2 = prompt2 | model | StrOutputParser()

result1 = chain1.invoke({"topic": "AI in healthcare"})

result2 = chain2.invoke({"text": result1})

print("AI Explanation: ", result1)
print("AI Summary: ", result2)
