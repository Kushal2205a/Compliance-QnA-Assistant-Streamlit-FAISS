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
            
    @st.cache_resource
    def spacy_chunker(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        chunked_sentence = [sentences.text for sentences in doc.sents]
        return chunked_sentence

    actual_chunked_sentence = spacy_chunker(text)

    embeddings_pdf = model.encode(actual_chunked_sentence)


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

chat_history = st.session_state["chat_history"]

if st.button("Clear Chat"):
    st.session_state["chat_history"].clear()
user_prompt = st.chat_input("Say Something")

#messages = st.container(height=800)


for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_prompt:
    st.chat_message("user").write(user_prompt)
    chat_history.append({"role" : "user", "content" : user_prompt})
    embeddings_query = model.encode(  "Retreive semantically similar text :" + user_prompt )
















