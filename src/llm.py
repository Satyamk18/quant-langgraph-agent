import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Initializes and returns the configured Chat LLM.
    Supports Google Gemini, OpenAI, or falls back to prompt configuration.
    Includes automated exponential backoff retries for rate limits (HTTP 429).
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # 1. Google Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "gemini" and gemini_key and gemini_key != "your_gemini_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=temperature,
            max_retries=6
        )
        
    # 2. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if (provider == "openai" or openai_key) and openai_key != "your_openai_api_key_here" and openai_key:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=openai_key,
            temperature=temperature,
            max_retries=6
        )
        
    raise ValueError(
        "No valid LLM API key detected!\n"
        "Please set GEMINI_API_KEY (recommended, free at https://aistudio.google.com/app/apikey) "
        "or OPENAI_API_KEY in your .env file or environment variables."
    )
