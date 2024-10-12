# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
print(os.getenv("GEMINI_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit interface

st.title('Langchain Demo With OPENAI API key')
input_text = st.text_input("Start a convo :)")

## OpenAI LLM 
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))