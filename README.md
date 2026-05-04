# GenAI Resume Matcher — RAG Enabled ATS

A Streamlit app that matches resumes against a job description using a full RAG (Retrieval-Augmented Generation) pipeline. Resumes are chunked, indexed with FAISS, and the most relevant sections are retrieved before GPT-4o-mini generates a structured candidate evaluation.

---

## Overview

Upload a job description and multiple resumes. The app chunks each resume, builds a FAISS vector index, retrieves the top matching sections against the job description, scores the candidate using cosine similarity, and generates a recruiter-style explanation grounded only in the retrieved resume content.

---

## Features

- Full RAG pipeline: chunk, embed, index, retrieve, generate
- FAISS vector index per resume for fast similarity search
- Retrieves top-k most relevant resume sections before passing to LLM
- LLM explanation strictly grounded in retrieved context — no hallucinated assumptions
- Cosine similarity score across full resume for overall ranking
- Verdict assignment: Shortlist, Maybe, or Reject
- Supports PDF and TXT for both job description and resumes
- Embedding model cached with `st.cache_resource` for performance
- Candidates ranked by match score in descending order

---

## How It Works

### RAG Pipeline

```
Resume Text
    |
    v
Chunk into 200-word segments
    |
    v
Embed chunks with all-MiniLM-L6-v2
    |
    v
Build FAISS IndexFlatL2
    |
    v
Embed Job Description
    |
    v
Retrieve top-k most similar chunks
    |
    v
Pass retrieved chunks + JD to GPT-4o-mini
    |
    v
Structured explanation: Strengths, Missing Skills, Verdict
```

### Scoring

Cosine similarity is computed between the full job description embedding and the full resume embedding, independent of the RAG retrieval step. This gives a stable percentage score for ranking.

### Verdict Thresholds

| Score Range   | Verdict   |
|---------------|-----------|
| 70% and above | Shortlist |
| 50% to 69%    | Maybe     |
| Below 50%     | Reject    |

---

## Tech Stack

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| Frontend        | Streamlit                         |
| Embeddings      | sentence-transformers (MiniLM-L6) |
| Vector Index    | FAISS (IndexFlatL2)               |
| Similarity      | scikit-learn cosine similarity    |
| LLM             | OpenAI GPT-4o-mini                |
| PDF Parsing     | PyPDF2                            |
| Language        | Python 3.9+                       |
| Deployment      | Streamlit Community Cloud         |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key

### Installation

```bash
git clone https://github.com/your-username/genai-resume-matcher-rag.git
cd genai-resume-matcher-rag

pip install -r requirements.txt
```

### Requirements

```
streamlit
openai
PyPDF2
sentence-transformers
scikit-learn
faiss-cpu
numpy
```

### Running Locally

```bash
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
```

---

## Deployment on Streamlit Community Cloud

1. Push your code to a public GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Add your secret in the app settings:

```toml
# .streamlit/secrets.toml (do NOT commit this file)
OPENAI_API_KEY = "your_key_here"
```

---

## Project Structure

```
genai-resume-matcher-rag/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml        # API key (local only, not committed)
```

---

## Difference from Basic Version

| Feature                        | Basic Matcher     | RAG Matcher              |
|--------------------------------|-------------------|--------------------------|
| Resume passed to LLM           | Full text         | Top-k retrieved chunks only |
| Vector index                   | None              | FAISS per resume         |
| LLM context size               | Large             | Focused and minimal      |
| Hallucination risk             | Higher            | Lower (grounded context) |
| Chunking                       | No                | 200-word sliding chunks  |

---

## Limitations

- FAISS index is built in-memory per run; not persisted between sessions
- Chunk size is fixed at 200 words; no overlap between chunks
- PDF extraction depends on text-based PDFs; scanned documents are not supported
- Verdict thresholds are hardcoded with no UI control to adjust them
- API costs apply per GPT-4o-mini call, one call per resume

---

## Roadmap

- [ ] Overlapping chunks for better boundary coverage
- [ ] Adjustable top-k and chunk size via UI sliders
- [ ] Adjustable verdict threshold controls
- [ ] CSV export of ranked results
- [ ] DOCX resume support
- [ ] Persistent FAISS index across sessions

---

## License

This project is licensed under the [MIT License](./LICENSE).
