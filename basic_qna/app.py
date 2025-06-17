import pdfplumber 
import streamlit as st
from sentence_transformers import SentenceTransformer 
import spacy 
import faiss 
import numpy as np 

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

    embeddings_pdf = model.encode(actual_chunked_sentence)
    document_embeddings = np.array(embeddings_pdf).astype("float32")
    faiss.normalize_L2(document_embeddings)
    index = faiss.IndexFlatL2(384)
    index.add(document_embeddings)


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

top_k = 5 

def rerank(chunk:str, query:str ) ->int :
    score = 0
    words = chunk.split()

    if len(words) > 12 :
        score += 1 
    
    if len(words) > 20 :
        score += 1 
    
    if chunk.strip().endswith('.'):
        score += 1 
    
    for word in query.lower().split():
        if word in chunk.lower():
            score += 1 
    
    if len(words) < 10 :
        score -= 2

    if len(words) < 5 :
        score -= 3 
    
    return score 
    

if user_prompt:
    st.chat_message("user").write(user_prompt)
    chat_history.append({"role" : "user", "content" : user_prompt})
    
    embeddings_query = model.encode(user_prompt )
    embeddings_query = np.array(embeddings_query).astype("float32")
    if embeddings_query.ndim == 1:
        embeddings_query = embeddings_query.reshape(1,-1)

    D, I = index.search(embeddings_query,top_k)

    matching_chunks = [actual_chunked_sentence[i] for i in I[0]]
   

    scored_chunks = []
    for chunk in matching_chunks:
        score = rerank(chunk, user_prompt)
        scored_chunks.append((score,chunk))
    scored_chunks.sort(reverse=True)

    best_match = scored_chunks[0][1]
    st.chat_message("assistant").write(f"Best match is : \n{best_match}\n\n")
    for score,chunk in scored_chunks[1:3]:
        st.chat_message("assistant").write(f"🔷{chunk}")
    chat_history.append({"role" : "assistant", "content" : best_match})
    
















