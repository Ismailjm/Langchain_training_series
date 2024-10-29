import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # this solved the problem of the project not appearing in langsmith dashboard
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader=WebBaseLoader("https://en.wikipedia.org/wiki/World_War_II")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.1-70b-versatile")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response baed on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

input_txt = st.text_input("Write your prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":input_txt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        ## Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------")