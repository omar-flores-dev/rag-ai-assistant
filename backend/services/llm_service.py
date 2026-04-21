from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Will need to investigate in optimizing and providing context to the LLM to prevent
# Ambiguous Abbreviation Hallucination, based on intial tests with the system. 
# Specifying what the LLM is, i.e expert AI assistant, I reduce AAH in the LLM's response when asked
# to explain what RAG is. Before: Talked about Red/Amber/Green, After: Talked about Retrieval Aug. Gen.

def get_llm_response(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI assistant. When explaining technical terms, always prefer machine learning and AI meanings unless context clearly indicates otherwise."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"