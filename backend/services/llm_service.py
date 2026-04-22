from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Will need to investigate in optimizing and providing context to the LLM to prevent
# Ambiguous Abbreviation Hallucination, based on intial tests with the system. 
# Specifying what the LLM is, i.e expert AI assistant, I reduce AAH in the LLM's response when asked
# to explain what RAG is. Before: Talked about Red/Amber/Green, After: Talked about Retrieval Aug. Gen.

def get_llm_response(prompt: str, context: list[str]):
    """
    Generates a response using retrieved context.
    context = list of relevant document chunks
    This will ultimately:
    - Force the model to use our data
    - Reduce potential hallucination in responses
    - Make answers much more grounded
    """
    # Combine retrieved chunks into a single string
    context_text = "\n\n".join(context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that answers questions using provided context. If the answer is not in the context, say you don't know."
            },
            {
                "role": "user",
                "content": f"""
                Use the following context to answer the question.

                Context:
                {context_text}

                Question:
                {prompt}
                """
            }
        ]
    )
    return response.choices[0].message.content
