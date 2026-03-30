import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
# def create_embedding(text):
#     r = requests.post("http://localhost:11434/api/embeddings", json={
#         "model":"bge-m3",
#         "prompt": text
#         })
#     embedding = r.json()["embedding"]
#     return embedding

def create_embedding(text_list):
    try:
        # Added a 30-second timeout
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=30)
        
        # Raise an exception for bad status codes
        r.raise_for_status()
        
        # The /api/embed endpoint returns 'embeddings' (plural)
        return r.json()["embeddings"]
    except requests.exceptions.Timeout:
        print("Error: The request to Ollama timed out. The server might be stuck or loading the model.")
        return None
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None


def main():
    jsons = os.listdir("jsons")
    my_dicts = []
    chunk_id = 0
    for json_file in jsons:
        if not json_file.endswith(".json"):
            continue
        # Reading the json file
        with open(f"jsons/{json_file}", "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f) #converts the json file into python dictionary(chunk: value, text: value)
        print(f"Creating embeddings for : {json_file}")
        embeddings = create_embedding([chunk['text'] for chunk in data["chunks"]])
        for i,chunk in enumerate(data["chunks"]):
            chunk["chunk_id"] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)
            



    df = pd.DataFrame.from_records(my_dicts)
    # print(df)
    joblib.dump(df, "embeddings.joblib")

if __name__ == "__main__":
    main()

