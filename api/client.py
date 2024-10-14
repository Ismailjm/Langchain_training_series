import streamlit as st
import requests

def get_essay_response(user_txt):
    response = requests.post("http://localhost:8000/essay_llama/invoke",
                             json={'input':{'topic':user_txt}})
    
    return response.json()['output']['content']

def get_poem_response(user_txt):
    response = requests.post("http://localhost:8000/poem_llama/invoke",
                             json={'input':{'topic':user_txt}})
    
    return response.json()['output']['content']

# interface
st.title('Langchain Demo With LLAMA3.1 API')
input_text1 = st.text_input("Write an essay :)")
input_text2 = st.text_input("Write a poem :)")

if input_text1:
    st.write(get_essay_response(input_text1))
    
if input_text2:
    st.write(get_poem_response(input_text2))

