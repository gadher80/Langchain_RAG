# YouTube Transcript Chat RAG App

A Streamlit-based RAG application that lets you paste a YouTube link and chat with the video transcript.  
The app fetches the transcript, builds embeddings using OpenAI, stores them in Chroma, and answers user questions using a retrieval-augmented generation pipeline.

---

## Features

- Paste any YouTube video link with available transcripts
- Automatically fetch and chunk the transcript
- Build and cache vector embeddings using Chroma
- Ask natural language questions about the video content
- Fast repeated queries using Streamlit caching
- Windows-safe implementation without file locking errors

---

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain
- OpenAI Embeddings and Chat Models
- Chroma Vector Store
- YouTube Transcript API

---

## Project Structure

```text
.
├── app.py                # Streamlit application
├── chroma_db/            # Local Chroma vector store (auto-created)
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
