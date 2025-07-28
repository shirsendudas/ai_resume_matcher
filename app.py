import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("🧠 AI Resume Matcher for Recruitment Teams")

# Step 1: Paste Job Description
st.header("Step 1: Paste Job Description")
job_desc = st.text_area("Paste the Job Description here")

# Step 2: Upload Resume Files
st.header("Step 2: Upload Resume Files (PDF or DOCX)")
uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx"], accept_multiple_files=True)

# Extractors
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Main logic
if uploaded_files and job_desc:
    resume_texts = []
    file_names = []

    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(file)
        else:
            text = ""
        resume_texts.append(text)
        file_names.append(file.name)

    # TF-IDF Matching
    corpus = [job_desc] + resume_texts
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(corpus)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Result display
    result_df = pd.DataFrame({
        "File Name": file_names,
        "Similarity Score (%)": [round(score * 100, 2) for score in similarities],
        "Resume Preview": [text[:300] + "..." for text in resume_texts]
    }).sort_values(by="Similarity Score (%)", ascending=False)

    st.success("Top matching resume:")
    st.dataframe(result_df)
