from langchain_community.document_loaders import WebBaseLoader
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
    template = 'answer the following \n {question} from the following \n {text}',
    input_variables = ["question", "text"]
)

parser  = StrOutputParser()

loader = WebBaseLoader("https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR1JWX1G/ref=sr_1_1_sspa?adgrpid=60047617118&dib=eyJ2IjoiMSJ9.Sw_7V4289LkeROzaDw-w8nymkE7uSvI6Lam3yOxsSPLmIsuMVyWO4GzQHaI2Otdk0rgLef0hF4ToM6dH7p7l7P0JUoWt-wCQPVJcu87A1CSSrjM5Kc3-d3n3rr9RJ9BcDCMOcju1P6wHgE2yVRfFE8st7Osdc7uKamXt0KL_JWQ.Mpczo0rUQ8lipUHA-xgyfO6TGSWitIei3qEs7UR25aU&dib_tag=se&gad_source=1&hvadid=398050919194&hvdev=c&hvexpln=0&hvlocphy=9145306&hvnetw=g&hvocijid=3940254712639517826--&hvqmt=b&hvrand=3940254712639517826&hvtargid=kwd-2475419516319&hydadcr=26944_2178277&keywords=macbook%2Bm5%2Bair%2B16gb&mcid=4f9b90f80dee37618d4bb3bbeda5c2f7&qid=1789457776&sr=8-1-spons&aref=UE9SZy51xs&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "what is the price of this product?", "text": docs[0].page_content})

print(result)