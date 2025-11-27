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
    # It ensures the model is loaded only once across sessions.
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

# 📊 Rank resumes using BERT (Semantic Similarity via Cosine Similarity)
def calculate_bert_scores(job_description, resumes):
    model = load_bert_model()
    # Encode text into vector embeddings
    job_embedding = model.encode([job_description])
    resume_embeddings = model.encode(resumes)
    
    # Cosine Similarity is used to compare the semantic closeness of the high-dimensional embeddings
    scores = cosine_similarity(job_embedding, resume_embeddings).flatten()
    return scores

# 🔑 Analyze resumes using TF-IDF (Keyword Similarity and Extraction via Cosine Similarity)
def analyze_tfidf(job_description, resumes, top_n=5):
    documents = [job_description] + resumes
    # TF-IDF for vectorization and feature importance
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Cosine Similarity is used here to compare the keyword overlap of the sparse TF-IDF vectors
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    tfidf_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    all_keywords = []
    # Loop through resumes to extract top keywords from each
    for i in range(len(resumes)):
        # Get the vector for the specific resume (index i + 1 because the JD is at index 0)
        resume_vector = tfidf_matrix[i + 1].toarray().flatten()

        # Sort and get top keywords based on TF-IDF weight
        indices = np.argsort(resume_vector)[::-1][:top_n]
        keywords = [feature_names[j] for j in indices]
        all_keywords.append(", ".join(keywords))

    return tfidf_scores, all_keywords

# 🌟 Calculate Combined Score
def calculate_combined_score(bert_score, tfidf_score, weight_bert):
    """Calculates a weighted average of the two scores for final ranking."""
    weight_tfidf = 1.0 - weight_bert
    combined_score = (weight_bert * bert_score) + (weight_tfidf * tfidf_score)
    return combined_score


# --- MAIN APP LOGIC ---
def main():
    st.set_page_config(layout="wide")
    st.title("📊 AI Resume Screening: Combined Ranking")
    st.markdown("We use **Cosine Similarity** to calculate both the **Semantic Score (BERT)** and the **Keyword Match Score (TF-IDF)**. The final ranking uses a customizable weighted average of these two scores.")

    # --- Sidebar Configuration ---
    st.sidebar.title("Ranking Configuration")
    
    # Custom weighting slider
    weight_bert = st.sidebar.slider(
        "Weight for Semantic Score (BERT)", 0.0, 1.0, 0.7, 0.05, 
        help="Controls the influence of conceptual similarity on the final rank. (0.7 means 70% BERT, 30% TF-IDF)"
    )
    weight_tfidf = 1.0 - weight_bert
    st.sidebar.caption(f"Keyword Match (TF-IDF) Weight: **{weight_tfidf:.2f}**")
    
    score_threshold = st.sidebar.slider("Minimum BERT Score Threshold", 0.0, 1.0, 0.0, 0.01)
    
    # --- Main Input Area ---
    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description here.", height=200)

    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files for screening", type=["pdf"], accept_multiple_files=True)


    if uploaded_files and job_description:
        st.header("🏆 Resume Ranking Results")
        
        with st.spinner("Analyzing resumes and calculating scores..."):
            resumes_text = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes_text.append(text)

            # 1. Get Semantic Scores (BERT + Cosine Similarity)
            bert_scores = calculate_bert_scores(job_description, resumes_text)

            # 2. Get Keyword Scores and Keywords (TF-IDF + Cosine Similarity)
            tfidf_scores, all_keywords = analyze_tfidf(job_description, resumes_text)

            results = []
            for i, bert_score in enumerate(bert_scores):
                if bert_score >= score_threshold:
                    tfidf_score = tfidf_scores[i]
                    keywords = all_keywords[i]
                    
                    # 3. Calculate Combined Score
                    combined_score = calculate_combined_score(bert_score, tfidf_score, weight_bert)
                    
                    results.append({
                        "Resume": uploaded_files[i].name,
                        "Combined Score": f"{combined_score:.4f}",
                        "BERT Score": f"{bert_score:.4f}",
                        "TF-IDF Score": f"{tfidf_score:.4f}",
                        "Top Keywords": keywords,
                        "Text": resumes_text[i]
                    })

        if results:
            df = pd.DataFrame(results)
            # Convert scores back to numeric for sorting and charting
            df["Combined Score (Numeric)"] = df["Combined Score"].astype(float)
            df["BERT Score (Numeric)"] = df["BERT Score"].astype(float)
            df["TF-IDF Score (Numeric)"] = df["TF-IDF Score"].astype(float)
            
            # Rank by the new Combined Score
            df_sorted = df.sort_values(by="Combined Score (Numeric)", ascending=False).reset_index(drop=True)
            
            # Display main ranking table
            st.dataframe(
                df_sorted[["Resume", "Combined Score", "BERT Score", "TF-IDF Score", "Top Keywords"]],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📈 Ranking based on Combined Score")
            
            # Prepare data for combined chart
            chart_df = df_sorted.set_index("Resume")[["Combined Score (Numeric)", "BERT Score (Numeric)", "TF-IDF Score (Numeric)"]]
            chart_df.columns = ["Combined Score", "Semantic (BERT)", "Keyword Match (TF-IDF)"]

            st.bar_chart(chart_df)

            st.subheader("📄 Resume Preview and Details")
            for _, item in df_sorted.iterrows():
                with st.expander(f"**RANKED** | {item['Resume']} | Combined Score: {item['Combined Score']}"):
                    st.markdown(f"**Combined Score:** `{item['Combined Score']}` (Primary Ranking Metric)")
                    st.markdown(f"**Semantic Score (BERT):** `{item['BERT Score']}` - *Conceptual alignment via Cosine Similarity on BERT embeddings.*")
                    st.markdown(f"**Keyword Score (TF-IDF):** `{item['TF-IDF Score']}` - *Keyword overlap via Cosine Similarity on TF-IDF vectors.*")
                    st.markdown(f"**Top Keywords:** {item['Top Keywords']}")
                    st.text_area(f"Full Text of {item['Resume']}", item["Text"], height=300)
        else:
            st.warning("🚫 No resumes met the configured score threshold. Please try lowering the 'Minimum BERT Score Threshold' in the sidebar or upload different files.")

if __name__ == "__main__":
    main()
