from langchain_google_genai import chatmodel_google
from dotenv import load_dotenv

load_dotenv()

model = chatmodel_google.chatmodel_google(model="gemini-1.5", temperature=0.9)

result = model.invoke("What is the capital of France?")

print(result)

