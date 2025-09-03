"""
Vertex AI RAG Agent

A package for interacting with Google Gemini API for RAG capabilities.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if we're using Gemini API or Vertex AI
USE_VERTEX_AI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if USE_VERTEX_AI:
    # Initialize Vertex AI
    try:
        import vertexai
        PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
        LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
        
        if PROJECT_ID and LOCATION:
            print(f"Initializing Vertex AI with project={PROJECT_ID}, location={LOCATION}")
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            print("Vertex AI initialization successful")
        else:
            print(
                f"Missing Vertex AI configuration. PROJECT_ID={PROJECT_ID}, LOCATION={LOCATION}. "
                f"Switching to Gemini API mode."
            )
            USE_VERTEX_AI = False
    except Exception as e:
        print(f"Failed to initialize Vertex AI: {str(e)}")
        print("Switching to Gemini API mode.")
        USE_VERTEX_AI = False
else:
    # Initialize Gemini API
    if GOOGLE_API_KEY:
        print("Using Gemini API with API key")
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API configured successfully")
    else:
        print("Warning: No GOOGLE_API_KEY found. Please set it in your .env file")

# Import agent after initialization is complete
from . import agent
