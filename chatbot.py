from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

while True:
    user_input = input("you: ")
    if user_input == "exit":
        break
    model = ChatHuggingFace(llm=llm)
    result = model.invoke(user_input)
    print("AI: " + result.content)