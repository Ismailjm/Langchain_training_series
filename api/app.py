from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv 

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # this solved the problem of the project not appearing in langsmith dashboard
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)


llm = ChatOllama(model="llama3.1")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words for a 5 years old child.")

prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} in the style of shakespear.")

add_routes(
    app,
    prompt1 | llm,
    path="/essay_llama"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem_llama"
)

if __name__=="__main__":
    uvicorn.run("app:app",host="localhost",port=8000)
