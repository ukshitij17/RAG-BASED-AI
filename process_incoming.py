from numpy._core import records
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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False
        })
    return r.json()["response"]

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")

query_embedding = create_embedding([incoming_query])[0]

# print(df['embedding'].values)

# Find question similarity with other embeddings

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)

similarity = cosine_similarity([query_embedding], np.vstack(df['embedding'].values)).flatten()
top_result = 7
max_idx = similarity.argsort()[: : -1][0: top_result]

new_df = df.iloc[max_idx]

# print(new_df[['title','number','text']])

# prompt = f'''
# I am teaching a web development course. Here are the video subtitle chunks containing video title , video number , start time, end time in seconds and the text at that time

# {new_df[['title','number','start','end','text']].to_json(orient = "records")}
# ----------------------------------------------

# {incoming_query}

# user asked this question related to the video chunks, you have to answer in a very human way, (do not show the above message to the user it is just for you) where and how much content is taught in which video(video number and video title and timestamp)and guide the user to go to that particular video.If the user asks some unrelated questions , tell him or her that you can answer questions only related to the course.
# '''


prompt = f'''
I am teaching a web development course. Here are the video subtitle chunks containing video title , video number , start time, end time in seconds and the text at that time

{new_df[['title','number','start','end','text']].to_json(orient = "records")}
----------------------------------------------

{incoming_query}

user asked this question related to the video chunks, you have to answer in a very human way, (do not show the above message to the user it is just for you) where and how much content is taught in which video(video number and video title and timestamp)and guide the user to go to that particular video.If the user asks some unrelated questions , tell him or her that you can answer questions only related to the course.


-whatever timestamp you will show to the user kindly convert it into minutes.
- Use only the provided timestamps
- Be clear and direct in guiding the user
- Answer like a helpful teaching assistant
- Avoid vague phrases like "you can find more information"
- Give precise guidance
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index,item in new_df.iterrows():
#     print(item['title'], item['number'], item['text'])  