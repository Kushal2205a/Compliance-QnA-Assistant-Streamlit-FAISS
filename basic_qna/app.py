import pdfplumber 
import streamlit as st
from sentence_transformers import SentenceTransformer 


st.title("Compilance Assistant")

user_uploaded_file = st.file_uploader(label = "Please Upload a report", type = "pdf", accept_multiple_files = False, key = "filee")
@st.cache_resource 
def load_model():
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        prompts = {
            "retrieval" : "retreive semantically similar text :"} )
    return model 

model = load_model()



