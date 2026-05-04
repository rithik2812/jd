# ------------------------------
# IMPORTS
# ------------------------------
import streamlit as st
import PyPDF2
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="GenAI Resume Matcher (RAG)", layout="wide")
st.title("GenAI Resume–Job Matcher (RAG Enabled ATS)")


# ------------------------------
# LOAD OPENAI CLIENT
# ------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------------------
# LOAD EMBEDDING MODEL (cached)
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()


# ------------------------------
# PDF TEXT EXTRACTION
# ------------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text


def read_file(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    else:
        return file.read().decode("utf-8", errors="ignore")


# ------------------------------
# CLEAN TEXT
# ------------------------------
def clean_text(text):
    return " ".join(text.lower().split())


# ------------------------------
# CHUNKING (RAG STEP 1)
# ------------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


# ------------------------------
# BUILD FAISS INDEX (RAG STEP 2)
# ------------------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# ------------------------------
# RETRIEVE TOP CHUNKS (RAG STEP 3)
# ------------------------------
def retrieve_chunks(jd_text, chunks, index, top_k=3):
    jd_embedding = embedder.encode([jd_text])
    jd_embedding = np.array(jd_embedding).astype("float32")

    distances, indices = index.search(jd_embedding, top_k)

    return [chunks[i] for i in indices[0]]


# ------------------------------
# SIMILARITY SCORE (FULL RESUME)
# ------------------------------
def compute_similarity(jd_text, resume_text):
    jd_emb = embedder.encode([jd_text])
    resume_emb = embedder.encode([resume_text])

    return cosine_similarity(jd_emb, resume_emb)[0][0]


# ------------------------------
# VERDICT LOGIC
# ------------------------------
def verdict_from_score(score):
    if score >= 70:
        return "Shortlist"
    elif score >= 50:
        return "Maybe"
    return "Reject"


# ------------------------------
# LLM WITH RAG
# ------------------------------
def generate_explanation_rag(jd_text, retrieved_chunks, score, verdict):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an expert recruiter.

    Job Description:
    {jd_text}

    Relevant Resume Sections:
    {context}

    Match Score: {score}%

    Instructions:
    - Only use given resume sections
    - Do NOT assume missing info

    Output format:
    Strengths:
    - ...

    Missing Skills:
    - ...

    Final Verdict:
    - {verdict}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# ------------------------------
# UI INPUT
# ------------------------------
st.header("📂 Upload Files")

jd_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"])
resume_files = st.file_uploader("Upload Resumes", type=["pdf", "txt"], accept_multiple_files=True)


# ------------------------------
# MAIN PIPELINE
# ------------------------------
if st.button("🚀 Run Matching"):

    if not jd_file or not resume_files:
        st.error("Upload JD and resumes")
        st.stop()

    jd_text = clean_text(read_file(jd_file))

    results = []

    for file in resume_files:
        resume_text = clean_text(read_file(file))

        # -------- RAG PIPELINE --------
        chunks = chunk_text(resume_text)
        index, _ = build_faiss_index(chunks)
        retrieved_chunks = retrieve_chunks(jd_text, chunks, index)

        # -------- SCORING --------
        sim_score = compute_similarity(jd_text, resume_text)
        score_percent = round(sim_score * 100, 2)
        verdict = verdict_from_score(score_percent)

        # -------- LLM --------
        explanation = generate_explanation_rag(
            jd_text, retrieved_chunks, score_percent, verdict
        )

        results.append((file.name, score_percent, verdict, explanation))

    results.sort(key=lambda x: x[1], reverse=True)

    st.success("Done")

    for i, (name, score, verdict, exp) in enumerate(results, 1):
        with st.expander(f"Rank {i}: {name} | {score}% | {verdict}"):
            st.write(exp)
