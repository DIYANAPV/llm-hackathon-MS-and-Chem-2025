# Install necessary libraries if not already installed:
# pip install pymupdf nltk sklearn sentence-transformers textstat

import fitz  # PyMuPDF
import pandas as pd
import nltk
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat  # alternative for readability

nltk.download('punkt')
nltk.download('punkt_tab')

def read_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def compute_readability(text):
    try:
        flesch = textstat.flesch_reading_ease(text)
    except:
        flesch = 0
    return flesch

def compute_semantic_similarity(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences1 = nltk.sent_tokenize(text1)
    sentences2 = nltk.sent_tokenize(text2)

    # Average embeddings for each paper
    emb1 = model.encode(sentences1)
    emb2 = model.encode(sentences2)

    avg_emb1 = emb1.mean(axis=0, keepdims=True)
    avg_emb2 = emb2.mean(axis=0, keepdims=True)

    similarity = cosine_similarity(avg_emb1, avg_emb2)[0][0]
    return similarity

def compute_length_similarity(text1, text2):
    # Compare total word count
    len1 = len(text1.split())
    len2 = len(text2.split())
    return 1 - abs(len1 - len2) / max(len1, len2)

def evaluate_human_likeness(human_text, ai_text):
    # Readability score comparison (normalized 0-1)
    human_read = compute_readability(human_text)
    ai_read = compute_readability(ai_text)
    readability_score = 1 - abs(human_read - ai_read) / max(human_read, ai_read, 1)

    # Semantic similarity
    semantic_score = compute_semantic_similarity(human_text, ai_text)

    # Length similarity
    length_score = compute_length_similarity(human_text, ai_text)

    # Weighted Human-Likeness Score
    human_likeness = 0.4 * semantic_score + 0.3 * readability_score + 0.3 * length_score

    return {
        'Semantic Similarity': round(semantic_score, 3),
        'Readability Similarity': round(readability_score, 3),
        'Length Similarity': round(length_score, 3),
        'Overall Human-Likeness Score': round(human_likeness, 3)
    }

if __name__ == "__main__":


    base_dir = "llm_hackathon/"
    original_dir = os.path.join(base_dir, "original_papers")
    model_folders = ["generated_papers/gpt-5", "generated_papers/o4-mini", "generated_papers/o3"]

    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Collect results in a list of dicts
    all_results = []

    for model in model_folders:
        model_path = os.path.join(base_dir, model)
        print(model_path)
        for paper_file in os.listdir(model_path):
            if not paper_file.endswith(".pdf"):
                continue
            print(f"Processing {paper_file} for model {model}") 
            ai_pdf = os.path.join(model_path, paper_file)
            human_pdf = os.path.join(original_dir, paper_file)
            print(human_pdf)

            if not os.path.exists(human_pdf):
                print(f"⚠️ Skipping {paper_file} (no matching original found)")
                all_results.append({
                    "model": model,
                    "paper": paper_file,
                    **{}  # no values
                })
                continue

            # Read both PDFs
            human_text = read_pdf(human_pdf)
            ai_text = read_pdf(ai_pdf)

            # Run evaluation
            result = evaluate_human_likeness(human_text, ai_text)

            # Round float values to 2 decimals
        
            all_results.append({
                "model": model,
                "paper": paper_file,
                **result
            })

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Save as CSV and Excel
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    excel_path = os.path.join(output_dir, "evaluation_results.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print(f"✅ Saved tabular results -> {csv_path} and {excel_path}")
