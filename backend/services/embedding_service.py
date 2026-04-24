from openai import OpenAI
import os
from dotenv import load_dotenv

'''FILE OBJECTIVE: How raw text data gets converted (embedded)'''

# Load environment variables from .env
load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str):
    """
    Converts text into an embedding vector (list of numbers).
    For the purpose
    - Allowing us to compare meaning between texts
    - Used for semantic search later
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    # Return the embedding vector
    return response.data[0].embedding