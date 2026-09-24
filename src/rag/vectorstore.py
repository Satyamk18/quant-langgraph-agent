import os
import glob
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag.embeddings import get_embeddings

COLLECTION_NAME = "sec_filings"
DEFAULT_DB_DIR = os.path.join("data", "chroma_db")
DEFAULT_FILINGS_DIR = os.path.join("data", "filings")

def get_chroma_store(persist_directory: str = DEFAULT_DB_DIR) -> Chroma:
    """Initializes or loads persistent Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

def extract_ticker_from_path(file_path: str) -> str:
    """Infers stock ticker from filename (e.g. boeing_10k -> BA, apple_10k -> AAPL)."""
    name = os.path.basename(file_path).lower()
    if "boeing" in name or "ba_" in name:
        return "BA"
    elif "apple" in name or "aapl" in name:
        return "AAPL"
    elif "tesla" in name or "tsla" in name:
        return "TSLA"
    elif "nvidia" in name or "nvda" in name:
        return "NVDA"
    return "UNKNOWN"

def ingest_filings(filings_dir: str = DEFAULT_FILINGS_DIR, persist_directory: str = DEFAULT_DB_DIR) -> int:
    """
    Ingests all markdown and text filing reports from filings_dir into ChromaDB.
    Chunks text with semantic overlap to preserve financial tables and footnotes.
    """
    os.makedirs(filings_dir, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)
    
    files = glob.glob(os.path.join(filings_dir, "*.md")) + glob.glob(os.path.join(filings_dir, "*.txt"))
    if not files:
        return 0
        
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        ticker = extract_ticker_from_path(file_path)
        base_name = os.path.basename(file_path)
        
        raw_doc = Document(
            page_content=text,
            metadata={"source": base_name, "ticker": ticker}
        )
        splits = text_splitter.split_documents([raw_doc])
        documents.extend(splits)
        
    if documents:
        vectorstore = get_chroma_store(persist_directory)
        vectorstore.add_documents(documents)
        
    return len(documents)

def query_filings(query: str, ticker: Optional[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Performs similarity search in ChromaDB.
    Filters by ticker if provided, extracts source citations.
    """
    vectorstore = get_chroma_store()
    
    # Check if collection is empty; if so, auto-ingest
    count = vectorstore._collection.count()
    if count == 0:
        ingest_filings()
        
    filter_dict = None
    if ticker and ticker not in ["SPY", "UNKNOWN"]:
        filter_dict = {"ticker": ticker.upper()}
        
    try:
        if filter_dict:
            results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_dict)
        else:
            results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception:
        # Fallback if filter or score isn't supported by backend
        docs = vectorstore.similarity_search(query, k=top_k)
        results = [(doc, 0.85) for doc in docs]
        
    sources = []
    for doc, score in results:
        sources.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "10-K Filing"),
            "ticker": doc.metadata.get("ticker", "UNKNOWN"),
            "score": round(float(score), 3) if score else 0.8
        })
        
    return sources
