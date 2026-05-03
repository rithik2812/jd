# GenAI Resume Matcher — Mini ATS

A Streamlit app that matches resumes against a job description using sentence embeddings and GPT-4o-mini. Ranks candidates by similarity score and generates an AI explanation for each match.

---

## Overview

Upload a job description and multiple resumes (PDF or TXT). The app computes semantic similarity using a SentenceTransformer model, scores each resume, assigns a verdict, and generates a plain-English explanation of why each candidate was ranked the way they were.

---

## Features

- Accepts PDF and TXT files for both job description and resumes
- Semantic similarity scoring via `all-MiniLM-L6-v2` (SentenceTransformers)
- Automatic verdict assignment: Shortlist, Maybe, or Reject
- GPT-4o-mini generated explanation per candidate covering strengths, gaps, and recommendation
- Candidates ranked by match score in descending order
- Expandable result cards per candidate
- Embedding model cached for performance

---

## Tech Stack

| Layer            | Technology                        |
|------------------|-----------------------------------|
| Frontend         | Streamlit                         |
| Embeddings       | sentence-transformers (MiniLM-L6) |
| Similarity       | scikit-learn cosine similarity    |
| LLM Explanation  | OpenAI GPT-4o-mini                |
| PDF Parsing      | PyPDF2                            |
| Language         | Python 3.9+                       |
| Deployment       | Streamlit Community Cloud         |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key

### Installation

```bash
git clone https://github.com/your-username/genai-resume-matcher.git
cd genai-resume-matcher

pip install -r requirements.txt
```

### Requirements

```
streamlit
openai
PyPDF2
sentence-transformers
scikit-learn
```

### Running Locally

```bash
# Create the secrets file
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Run the app
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

The app reads the key via `st.secrets["OPENAI_API_KEY"]` and stops gracefully if it is missing.

---

## How It Works

1. User uploads a job description file (PDF or TXT)
2. User uploads one or more resume files (PDF or TXT)
3. Both job description and resumes are extracted, lowercased, and whitespace-normalised
4. The `all-MiniLM-L6-v2` model encodes all texts into embeddings
5. Cosine similarity is computed between the job description and each resume
6. Each resume receives a percentage score and a verdict based on thresholds
7. GPT-4o-mini generates a structured explanation per candidate
8. Results are sorted by score and displayed in expandable cards

### Verdict Thresholds

| Score Range | Verdict   |
|-------------|-----------|
| 70% and above | Shortlist |
| 50% to 69%    | Maybe     |
| Below 50%     | Reject    |

---

## Project Structure

```
genai-resume-matcher/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml        # API key (local only, not committed)
```

---

## Limitations

- Similarity is computed on full resume text; section-level matching (skills, experience) is not supported
- PDF extraction quality depends on the PDF structure; scanned or image-based PDFs may yield poor results
- Verdict thresholds are fixed; no UI control to adjust them
- No persistent storage — results are lost on page refresh
- API costs apply for each GPT-4o-mini explanation call

---

## Roadmap

- [ ] Adjustable verdict threshold sliders
- [ ] Section-level matching (skills, education, experience separately)
- [ ] CSV export of ranked results
- [ ] Support for DOCX resume format
- [ ] Highlight matched and missing keywords inline

---

## License

This project is licensed under the [MIT License](./LICENSE).
