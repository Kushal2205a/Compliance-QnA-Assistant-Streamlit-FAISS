import pdfplumber 
import streamlit as st
from sentence_transformers import SentenceTransformer 
import spacy 



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

if user_uploaded_file is not None:
    user_uploaded_file.seek(0)

    with pdfplumber.open(user_uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    def spacy_chunker(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        chunked_sentence = [sentences.text for sentences in doc.sents]
        return chunked_sentence

    actual_chunked_sentence = spacy_chunker(text)

    embeddings = model.encode(actual_chunked_sentence)











