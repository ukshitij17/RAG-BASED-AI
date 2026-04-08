An intelligent **Retrieval-Augmented Generation (RAG)** system that helps students navigate video-based courses efficiently.  
It allows users to ask questions and get **precise video references with timestamps**.

---

## 🚀 Features

- 🎧 Converts lecture audio → text using Whisper  
- ✂️ Splits content into meaningful chunks  
- 🧠 Generates semantic embeddings using `bge-m3`  
- 🔍 Retrieves most relevant chunks using cosine similarity  
- 🤖 Uses LLM (LLaMA 3.2 via Ollama) to generate answers  
- ⏱️ Provides **video number + timestamp (MM:SS)** for navigation  

---

## 🧠 How It Works

```text
Audio → Transcription → Chunking → Embeddings → Retrieval → LLM → Answer
Transcription
Audio files are converted into text using Whisper.
Chunking
Transcripts are split into smaller segments with metadata:
Video number
Title
Start & end timestamps
Embedding Generation
Each chunk is converted into vector embeddings using bge-m3 via Ollama.
Query Processing
User query → embedding
Cosine similarity → top relevant chunks
Response Generation
LLM generates a human-like answer with:
Video reference
Timestamp guidance

📂 Project Structure
RAG-BASED-AI/
├── audios/                 
├── jsons/                  
├── videos/                 
├── embeddings.joblib       
├── mp3_to_json.py          
├── read_chunks.py          
├── process_incoming.py     
├── prompt.txt              
├── response.txt

       
⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/ukshitij17/RAG-BASED-AI.git
cd RAG-BASED-AI
2. Install dependencies
pip install -r requirements.txt
3. Install & run Ollama

Download from: https://ollama.com

Then run:

ollama pull llama3.2
ollama pull bge-m3
▶️ Usage

Step 1: Convert audio to JSON
python mp3_to_json.py

Step 2: Generate embeddings
python read_chunks.py

Step 3: Ask questions
python process_incoming.py

💡 Example
Ask a Question: where is seo taught?
Video #6 - SEO and Core Web Vitals in HTML  
Timestamp: 01:00 - 01:03
💡 Key Highlights
📌 Built a RAG-based system for educational video search
⏱️ Provides precise timestamps for quick navigation
🧠 Uses semantic search for better understanding of queries
🤖 Generates helpful, human-like responses using LLM
🔮 Future Improvements
Add UI for better interaction
Support multiple courses
Improve response accuracy further
Integrate vector databases (FAISS)

🧑‍💻 Author
Kshitij Upadhyay


⭐ If you like this project

Give it a star ⭐ and share it!
