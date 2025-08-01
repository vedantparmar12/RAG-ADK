import os

import vertexai
from dotenv import load_dotenv


load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")


try:
    if PROJECT_ID and LOCATION:
        print(f"Initializing Vertex AI with project={PROJECT_ID}, location={LOCATION}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print("Vertex AI initialization successful")
    else:
        print(
            f"Missing Vertex AI configuration. PROJECT_ID={PROJECT_ID}, LOCATION={LOCATION}. "
            f"Tools requiring Vertex AI may not work properly."
        )
except Exception as e:
    print(f"Failed to initialize Vertex AI: {str(e)}")
    print("Please check your Google Cloud credentials and project settings.")


from . import agent
