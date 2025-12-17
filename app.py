from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import shutil
import re
import streamlit as st
import os

load_dotenv()

# ------------------ Utils ------------------

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def load_transcript(video_id: str, lang="en") -> str:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=[lang])
    return " ".join(item.text for item in transcript)


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )
    return splitter.split_text(text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------ Vector Store ------------------

def build_vectorstore(chunks):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="youtube_transcripts"
    )

# ------------------ Prompt ------------------

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
)

# ------------------ Chain ------------------

def build_rag_chain(retriever):
    return (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
        )
        | RAG_PROMPT
        | ChatOpenAI(model="gpt-4o", temperature=0.2)
        | StrOutputParser()
    )

# ------------------ Main ------------------

if __name__ == "__main__":
    st.set_page_config(page_title="YouTube Transcript Chat", page_icon="🎬", layout="centered")
    st.title("YouTube Transcript Chat")
    st.write("Paste a YouTube link and ask questions about the video transcript.")

    youtube_url = st.text_input("YouTube URL")
    user_question = st.text_input("Your Question")

    # ------------------ Caching ------------------
    @st.cache_resource
    def get_vectorstore(chunks):
        return build_vectorstore(chunks)

    @st.cache_data
    def get_chunks_from_url(url):
        video_id = extract_video_id(url)
        transcript = load_transcript(video_id)
        chunks = chunk_text(transcript)
        return chunks

    if st.button("Get Answer"):
        if not youtube_url or not user_question:
            st.warning("Please provide both YouTube URL and a question.")
        else:
            try:
                with st.spinner("Processing..."):
                    chunks = get_chunks_from_url(youtube_url)
                    vector_db = get_vectorstore(chunks)
                    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
                    rag_chain = build_rag_chain(retriever)
                    answer = rag_chain.invoke(user_question)
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
