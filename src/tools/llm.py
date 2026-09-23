"""LLM client factory for Google Gemini and Groq with fallback support."""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.2, model_name: Optional[str] = None):
    """Factory to retrieve the configured ChatModel.

    Defaults to Google Gemini (gemini-2.0-flash / gemini-1.5-flash),
    or falls back to Groq (llama-3.3-70b-versatile) based on configuration.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not gemini_key and not groq_key:
        raise ValueError(
            "No LLM API key detected!\n"
            "Please set GEMINI_API_KEY (get free key at https://aistudio.google.com/) "
            "or GROQ_API_KEY (from https://console.groq.com/) in your .env file."
        )

    # If provider is explicitly specified as groq, or if groq key is present and gemini is not
    if provider == "groq" or (groq_key and not gemini_key):
        from langchain_groq import ChatGroq
        selected_model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"Initializing Groq LLM with model: {selected_model}")
        return ChatGroq(
            model=selected_model,
            temperature=temperature,
            api_key=groq_key
        )

    # Otherwise default to Google Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if selected_model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
        selected_model = "gemini-3.6-flash"

    logger.info(f"Initializing Google Gemini LLM with model: {selected_model}")
    return ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=temperature,
        google_api_key=gemini_key
    )


def extract_text_content(content) -> str:
    """Normalize LLM response content into a clean string across different SDK formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)
