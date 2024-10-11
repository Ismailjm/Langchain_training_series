from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langsmith import traceable
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # this solved the problem of the project not appearing in langsmith dashboard
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## ollama = llama 3.1
llm = ChatOllama(model="llama3.1")
output_parser = StrOutputParser()

@traceable
def run_chain(user_input):
    chain = prompt | llm | output_parser
    return chain.invoke({'question': user_input})

## streamlit interface
st.title('Langchain Demo With LLAMA3.1')
input_text = st.text_input("Start a convo :)")

if input_text:
    st.write(run_chain(input_text))