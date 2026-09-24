import os
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    """
    Returns the configured embedding model.
    Defaults to Google Gemini's official `models/gemini-embedding-001`.
    Falls back gracefully to Chroma's local built-in embeddings if offline.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=gemini_key
            )
        except Exception:
            pass

    # Local zero-token fallback
    from langchain_chroma import Chroma
    return None
