import os

DATA_PATH = "data/"

'''FILE OBJECTIVE: How data is prepared'''

# Read files from /data and load the text into memory, the goal is to essentially provide useful context/data
# to the llm, thus provide better quality responses.

# read all files within /data, load them into memory, and return a list of docs
def load_documents():
    """
    Reads all files from the data directory and returns their content as a list of strings.
    Where each file = one document.
    """
    documents = []
    # Loop through every file in /data
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)

         # Make sure to focus on file types
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Add the full document text to our list
                documents.append(content)

    return documents

# Need to split documents into data chunks, this makes it easier for an LLM to digest info, as well
# as get relevant info based on certain chunks rather than getting all of them.

def chunk_text(documents, chunk_size=100, overlap=20):
    """
    Splits a large piece of text into smaller overlapping chunks.
    So that:
    - LLMs and embeddings work better with smaller text segments
    - Later, each chunk will become an embedding

    chunk_size = number of words per chunk
    Overlaps will allow to improve quality of the retrievals but it will also preserve continuity in context
    """
    chunks = []

    for doc in documents:
        words = doc.split() # Split text into individual words
        # Loop through words in steps of chunk_size

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end]) # add the words together as a chunk
            chunks.append(chunk) # add chunk to list
            start += chunk_size - overlap  # overlap step

    return chunks


def load_and_chunk_documents():
    """
    Full pipeline:
    1. Load documents
    2. Break each document into chunks
    3. Return ALL chunks in one list
    """
    docs = load_documents()
    all_chunks = []

    # Process each document individually
    for doc in docs:
        chunks = chunk_text(doc)

        # Add all chunks from this document into the master list (all_chunks)
        all_chunks.extend(chunks)

    return all_chunks