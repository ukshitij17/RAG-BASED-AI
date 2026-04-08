import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding

# -----------------------------
# LLM Inference Function
# -----------------------------
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]


# -----------------------------
# Faithfulness Function (Improved)
# -----------------------------
def check_faithfulness(context, answer):
    prompt = f"""
You are an evaluator.

Context:
{context}

Answer:
{answer}

Question:
Is the answer grounded in the context?

Rules:
- Minor wording differences are OK
- Paraphrasing is OK
- Time format differences are OK
- If MOST of the answer is supported → say YES

Answer only YES or NO.
"""

    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    })

    result = r.json()["response"].strip().lower()

    print("Faithfulness raw response:", result)  # DEBUG

    return 1 if "yes" in result else 0


# -----------------------------
# Load Embeddings
# -----------------------------
df = joblib.load("embeddings.joblib")


# -----------------------------
# Test Dataset (Ground Truth)
# -----------------------------
test_data = [
    {"query": "where is seo taught?", "answer": "6"},
    {"query": "where are inline elements explained?", "answer": "8"},
    {"query": "where is audio tag taught?", "answer": "10"},
]


# -----------------------------
# Evaluation Variables
# -----------------------------
recall_correct = 0
faithfulness_scores = []

total = len(test_data)


# -----------------------------
# MAIN LOOP (IMPORTANT)
# -----------------------------
for item in test_data:
    query = item["query"]
    true_video = item["answer"]

    print("\n==============================")
    print("Query:", query)

    # -------- RETRIEVAL --------
    query_embedding = create_embedding([query])[0]

    similarity = cosine_similarity(
        [query_embedding],
        np.vstack(df['embedding'].values)
    ).flatten()

    top_k = 5
    max_idx = similarity.argsort()[::-1][:top_k]

    new_df = df.iloc[max_idx]

    retrieved_videos = new_df['number'].astype(str).values

    print("Retrieved videos:", retrieved_videos)

    if true_video in retrieved_videos:
        recall_correct += 1

    # -------- GENERATION --------
    prompt = f'''
You are a helpful teaching assistant.

Context:
{new_df[['title','number','start','end','text']].to_json(orient="records")}

Question:
{query}

Answer clearly:
- Mention video number
- Mention video title
- Mention timestamp (MM:SS format)
'''

    response = inference(prompt)

    print("Answer:", response)

    # -------- FAITHFULNESS --------
    context = new_df[['text']].to_string()

    print("\n--- DEBUG ---")
    print("Context preview:", context[:200])
    print("--------------")

    faith = check_faithfulness(context, response)

    faithfulness_scores.append(faith)


# -----------------------------
# FINAL METRICS
# -----------------------------
recall_at_5 = recall_correct / total
final_faithfulness = sum(faithfulness_scores) / total

print("\n==============================")
print("FINAL RESULTS")
print("==============================")
print("Recall@5:", recall_at_5)
print("Faithfulness:", final_faithfulness)