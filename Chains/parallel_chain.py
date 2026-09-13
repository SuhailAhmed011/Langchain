from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

explaination_prompt = ChatPromptTemplate.from_template(
    """
    Explain the following topic in simple words:

    Topic: {topic}
    """
)

explaination_chain = explaination_prompt | model | parser


summary_prompt = ChatPromptTemplate.from_template(
    """
    Give a short summary of the following topic
    in 3 simple points:

    Topic: {topic}
    """
)

summary_chain = summary_prompt | model | parser


keywords_prompt = ChatPromptTemplate.from_template(
    """
    Give a short summary of the following topic
    in 3 simple points:

    Topic: {topic}
    """
)

keywords_chain = keywords_prompt | model | parser

parallel_chain = RunnableParallel({
    "explaination": explaination_chain,
    "summary": summary_chain,
    "keywords": keywords_chain
}

)

result = parallel_chain.invoke({"topic": "AI in finance"})

print("\n========== EXPLANATION ==========")
print(result["explaination"])

print("\n========== SUMMARY ==========")
print(result["summary"])

print("\n========== KEYWORDS ==========")
print(result["keywords"])