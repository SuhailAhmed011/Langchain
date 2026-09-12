from langchain_openai import chatmodel_openai
from dotenv import load_dotenv

load_dotenv()

model = chatmodel_openai.chatmodel_openai(model="gpt-4", temperature=0.9)

result = model.invoke("What is the capital of France?")

print(result)
