import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding

# def create_embedding(text_list):
#     try:
#         r = requests.post("http://localhost:11434/api/embed", json={
#             "model": "bge-m3",
#             "input": text_list
#         }, timeout=30)
#         r.raise_for_status()
#         return r.json()["embeddings"]
#     except requests.exceptions.Timeout:
#         print("Error: The request to Ollama timed out. The server might be stuck or loading the model.")
#         return None
#     except Exception as e:
#         print(f"Error creating embedding: {e}")
#         return None

# try:
#     df = joblib.load("embeddings.joblib")
# except FileNotFoundError:
#     print("embeddings.joblib not found. Please run read_chunks.py first.")
#     exit()

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")

query_embedding = create_embedding([incoming_query])[0]

# print(df['embedding'].values)

# Find question similarity with other embeddings

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)

similarity = cosine_similarity([query_embedding], np.vstack(df['embedding'].values)).flatten()
top_result = 30
max_idx = similarity.argsort()[: : -1][0: top_result]

new_df = df.iloc[max_idx]

# print(new_df[['title','number','text']])

for index,item in new_df.iterrows():
    print(item['title'], item['number'], item['text'])  