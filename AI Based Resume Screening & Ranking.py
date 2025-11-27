import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# 🧠 Load BERT Model (Cached)
@st.cache_resource
def load_bert_model():
    # 'all-MiniLM-L6-v2' is an efficient model for semantic similarity tasks.
    return SentenceTransformer('all-MiniLM-L6-v2')

# 🧾 Extract text from PDF resumes
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# 📊 Rank resumes using BERT (Semantic Similarity)
def calculate_bert_scores(job_description, resumes):
    model = load_bert_model()
    # Encode text into vector embeddings
    job_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resumes)
    
    # Calculate Cosine Similarity between JD and all resumes
    scores = cosine_similarity(job_embedding, resume_embeddings).flatten()
    return scores

# 🔑 Extract Top Keywords using TF-IDF (for interpretability)
def extract_top_keywords_tfidf(job_description, resumes, index, top_n=5):
    documents = [job_description] + resumes
    # TF-IDF helps identify the most statistically important terms
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the vector for the specific resume (index + 1 because the JD is at index 0)
    resume_vector = tfidf_matrix[index + 1].toarray().flatten()
    
    # Sort and get top keywords based on TF-IDF weight
    indices = np.argsort(resume_vector)[::-1][:top_n]
    keywords = [feature_names[i] for i in indices]
    return keywords

# --- MAIN APP LOGIC ---
def main():
    st.title("📊 AI Resume Screening (BERT & TF-IDF)")
    st.markdown("Using **BERT** for semantic scoring and **TF-IDF** for keyword extraction.*")

    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description")

    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    score_threshold = st.sidebar.slider("Minimum BERT Score Threshold", 0.0, 1.0, 0.0, 0.01)

    if uploaded_files and job_description:
        st.header("🏆 Resume Ranking Results")
        
        with st.spinner("Analyzing resumes with BERT..."):
            resumes_text = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes_text.append(text)

            # 1. Get Semantic Scores using BERT
            bert_scores = calculate_bert_scores(job_description, resumes_text)

            results = []
            for i, score in enumerate(bert_scores):
                if score >= score_threshold:
                    # 2. Get Keywords using TF-IDF (for display only)
                    keywords = extract_top_keywords_tfidf(job_description, resumes_text, i)
                    
                    results.append({
                        "Resume": uploaded_files[i].name,
                        "BERT Score": score,
                        "Top Keywords": ", ".join(keywords),
                        "Text": resumes_text[i]
                    })

        if results:
            df = pd.DataFrame(results).sort_values(by="BERT Score", ascending=False)
            st.dataframe(df[["Resume", "BERT Score", "Top Keywords"]])
            st.subheader("📈 Similarity Scores (BERT)")
            st.bar_chart(df.set_index("Resume")["BERT Score"])

            st.subheader("📄 Resume Preview")
            for item in results:
                with st.expander(f"{item['Resume']} - Score: {item['BERT Score']:.2f}"):
                    st.write(item["Text"])
                    st.markdown(f"**Top Keywords (TF-IDF):** {item['Top Keywords']}")
        else:
            st.warning("🚫 No resumes met the score threshold. Try lowering the threshold or upload new files.")

if __name__ == "__main__":
    main()
