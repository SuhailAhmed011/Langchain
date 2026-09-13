from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=200,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

technical_prompt = ChatPromptTemplate.from_template(
    """
    You are a technical support assistant.

    Answer the following technical question
    clearly and provide practical guidance.

    Question:
    {question}
    """
)

general_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful general assistant.

    Answer the following question
    in simple and easy-to-understand language.

    Question:
    {question}
    """
)

technical_chain = technical_prompt | model | parser


general_chain = general_prompt | model | parser


def is_technical_question(data):
    question = data["question"].lower()

    technical_words = [
        "python",
        "code",
        "programming",
        "error",
        "bug",
        "api",
        "database",
        "server",
        "javascript",
        "docker",
        "git",
        "github",
        "langchain"
    ]

    return any(word in question for word in technical_words)


conditional_chain = RunnableBranch(
    (
        is_technical_question,
        technical_chain
    ),
    general_chain
)


result = conditional_chain.invoke(
    {
        "question": "I am getting an error in my Python code."
    }
)

print("\n========== TECHNICAL QUESTION ==========")
print(result)



result = conditional_chain.invoke(
    {
        "question": "What is the capital of India?"
    }
)

print("\n========== GENERAL QUESTION ==========")
print(result)