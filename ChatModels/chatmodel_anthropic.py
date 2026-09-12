from langchain_anthropic import chatmodel_anthropic
from dotenv import load_dotenv

load_dotenv()

model = chatmodel_anthropic.chatmodel_anthropic(model="claude-3.5-sonnect", temperature=0.9)

result = model.invoke("What is the capital of France?")

print(result)

