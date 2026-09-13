from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

class review(TypedDict):
    review: str
    sentiments: str

chat_model = ChatHuggingFace(llm=model)

structured_output = chat_model.with_structured_output(review)

result = structured_output.invoke("The movie was fantastic! I loved the acting and the plot but the ending was a bit disappointing. Overall, I would recommend it to others.")


print(result)