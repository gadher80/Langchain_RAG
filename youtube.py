from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, faiss
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser



from dotenv import load_dotenv
import shutil

import re
load_dotenv()

# Extract video id from youtube url
def extract_video_id(youtube_url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_youtube_transcript_simple(video_id, language='en'):
    """Fetch transcript using the list and fetch method correctly"""
    
    # Validate video ID format
    if not video_id or len(video_id) != 11:
        print(f"Error: Invalid video ID format. Expected 11 characters, got: {video_id}")
        return None
    
    try:
        print(f"Fetching transcript for video: {video_id}")
        
        api = YouTubeTranscriptApi()
        
        # First, try to list available transcripts
        try:
            print(f"  Checking available transcripts...")
            transcripts = api.list(video_id)
            #print(f"  Available: {transcripts}")
        except Exception as e:
            print(f"  Could not list transcripts: {e}")
        
        # Now fetch the transcript directly (pass as single video_id string, not list)
        transcript_obj = api.fetch(video_id, languages=[language])
        
        # Convert FetchedTranscriptSnippet objects to dictionaries
        transcript = []
        for snippet in transcript_obj:
            transcript.append({
                'text': snippet.text,
                'start': snippet.start,
                'end': snippet.start + snippet.duration,
                'duration': snippet.duration
            })
        
        # If successful, transcript should be a list of dicts
        return transcript
    
    except VideoUnavailable:
        print(f"Error: Video is unavailable or has been removed (ID: {video_id})")
        return None
    
    except TranscriptsDisabled:
        print(f"Error: Transcripts are disabled for this video (ID: {video_id})")
        return None
    
    except NoTranscriptFound:
        print(f"Error: No transcript found for video (ID: {video_id}) in language '{language}'")
        print("Try using a different language or a different video with captions enabled")
        return None
    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error ({error_type}): {error_msg}")
        
        # If YouTube request failed, it's likely the video doesn't have transcripts
        if "YouTubeRequestFailed" in error_type or "400" in error_msg:
            print("\nTip: This video may not have transcripts enabled or is unavailable.")   
        return None

def chunk_transcript(transcript, chunk_size=100, chunk_overlap=20):
    if transcript is None:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Convert transcript to a single string
    transcript_text = " ".join([segment['text'] for segment in transcript])

    # Split into chunks
    chunks = splitter.split_text(transcript_text)

    return chunks

def embed_and_store_chunks(chunks, model_name="text-embedding-3-small"):
    """Create embeddings and store them in a vector database using OpenAI and Chroma"""
    
    if not chunks:
        print("Error: No chunks provided for embedding")
        return None
    
    try:
        print(f"Creating embeddings using {model_name}...")
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model=model_name)
   
        # Create vector store and embed chunks
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="youtube_transcripts"
        )

        # faiss_db = Faiss.from_documents(
        #     documents=chunks,
        #     embedding=embeddings
        # )
        
        #print(f"✓ Successfully embedded and stored {len(chunks)} chunks in vector database")
        return vector_db
    
    except Exception as e:
        error_type = type(e).__name__
        print(f"Error ({error_type}): Failed to create embeddings - {str(e)}")
        return None
    
def create_retriever(vector_db, search_type="similarity", k=1):
    if vector_db is None:
        print("Error: Vector database is None")
        return None

    try:
        retriever = vector_db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        #print(f"✓ Created {search_type} retriever with top {k} results")
        return retriever

    except Exception as e:
        print(f"Failed to create retriever: {e}")
        return None
    


if __name__ == "__main__":
    url_id = "https://www.youtube.com/watch?v=vJOGC8QJZJQ"
    language = "en"
    
    # Usage
    transcript = get_youtube_transcript_simple(extract_video_id(url_id), language)
    
    # Get chunks
    chunks = chunk_transcript(transcript, chunk_size=150, chunk_overlap=30)
    
    # Embed and store
    #delete data in chroma_db if exists using delete method of chroma
    if shutil.os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    vector_db = embed_and_store_chunks(chunks, model_name="text-embedding-3-small")
    retriever_data = create_retriever(vector_db, search_type="similarity", k=10)

    question = "What is Langraph?"
    similarity_list = retriever_data.invoke(question)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        """,
    )

    final_prompt = prompt.invoke({
        "context": "\n\n".join(item.page_content for item in similarity_list),
        "question": question
    })

    # Use this prompt template to invoke the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    response = llm.invoke(final_prompt)
    print(f"\nFinal Response from LLM:\n{response.content}\n")

    

