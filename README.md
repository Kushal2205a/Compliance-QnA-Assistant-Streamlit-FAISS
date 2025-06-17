# Compliance Q&A Assistant

A lightweight, local document assistant that extracts text from compliance PDFs, indexes it using vector embeddings (FAISS), and answers user questions via semantic search.

---

## Features

- Upload a compliance-related PDF document  
- Ask natural language questions about its contents  
- Returns the most relevant chunks from the document  
- Uses FAISS for fast similarity search  
- Implements custom re-ranking to improve answer quality  
- Streamlit-based interactive UI with chat history  

---

## Tech Stack

- Python  
- Streamlit – front-end interface  
- SentenceTransformers – for semantic embeddings (`all-MiniLM-L6-v2`)  
- spaCy – for sentence chunking  
- FAISS – vector similarity search  
- pdfplumber – PDF text extraction  

---

## Installation

1. Clone the repository:  
   `git clone https://github.com/Kushal2205a/Compliance-QnA-Assistant-Streamlit-FAISS.git`  
   `cd Compliance-QnA-Assistant-Streamlit-FAISS`

2. (Optional) Create and activate a virtual environment:  
   `python -m venv venv`  
   `venv\Scripts\activate` (on Windows)

3. Install the dependencies:  
   `pip install -r requirements.txt`  
   `python -m spacy download en_core_web_sm`

---

## Running the App

Run the Streamlit application:  
`streamlit run basic_qna/app.py`

Once the app opens in your browser:
- Upload a PDF report
- Ask a question in the chat input
- The assistant will retrieve the most relevant sections

---

## Notes

- Works only with text-based PDFs (not scanned images)
- Runs completely locally – no API calls required
- Includes a basic scoring system to improve result relevance

---

