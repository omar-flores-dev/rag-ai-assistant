import os
import json

DATA_PATH = "backend/data/raw"

"""
FILE OBJECTIVE:
Structured document ingestion layer for the RAG pipeline.

This module is responsible for:
- Loading documents from multiple source types (research papers, websites, documentation, etc.)
- Parsing structured metadata (metadata.json)
- Extracting section-level content from each document
- Preserving source-level metadata for future citation and traceability
- Preparing data for downstream hybrid chunking and embedding

PIPELINE STAGE:
This is a structured preprocessing stage in the RAG pipeline.

Each document is first broken into semantic sections, then further
processed into hybrid retrieval chunks downstream.

NOTE:
This is NOT final chunking.
Final hybrid chunking (section-aware + overlapping retrieval chunks)
happens after this stage before embedding generation.
"""


def chunk_section_text(text, chunk_size=250, overlap=50):
    """
    Splits a section into smaller overlapping word-based chunks. I.e chunk with 250 words, but overlap with 50 words (helps preserve context in some cases)

    Why:
    Large sections reduce embedding precision.
    Smaller chunks improve retrieval relevance.

    Potential Future upgrade:
    Replace with token-based chunking or sentence-aware chunking.
    """

    #split words
    words = text.split()
        
    chunks = []

    #start at the begining of a section (the first word)
    start = 0

    # loop through all words of a given section,
    while start < len(words):

        # set the end of the 250 word collection based on the current position of the word
        end = start + chunk_size

        # collect all the words from the starting pos to the end pos
        chunk_words = words[start:end]

        # form it back into corresponding sentences
        chunks.append(
            " ".join(chunk_words)
        )

        # set the new starting point but account for overlap to preserve context (want to avoid finishing a chunk in the middle of a sentence)
        start += (chunk_size - overlap)

    return chunks

def load_documents_structured():
    """
    Loads structured document folders like:

    backend/data/raw/
      research_papers/
        doc_001/
          metadata.json
          sections/
            abstract.txt
            introduction.txt
    """

    all_chunks = []

    # Iterate over source categories (extensible design)
    # Examples:
    # - research_papers
    # - websites
    # - documentation
    for source_type in os.listdir(DATA_PATH):

        source_path = os.path.join(DATA_PATH, source_type)

        if not os.path.isdir(source_path):
            continue

        # Iterate over individual documents within a source type
        for doc_folder in os.listdir(source_path):

            doc_path = os.path.join(source_path, doc_folder)

            if not os.path.isdir(doc_path):
                continue

            metadata_path = os.path.join(doc_path, "metadata.json")

            if not os.path.exists(metadata_path):
                continue

            # Load document metadata (used for structure + future citation support)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            sections_path = os.path.join(doc_path, "sections")

            # Load each section defined in metadata
            for section in metadata.get("sections", []):
                safe_section_name = section.lower().replace(" ", "_")
                section_file = os.path.join(sections_path, f"{safe_section_name}.txt")

                if not os.path.exists(section_file):
                    print(f"Missing section file: {section_file}")
                    continue

                with open(section_file, "r", encoding="utf-8") as f:
                    text = f.read()
                # Hybrid chunking:
                # Section = semantic boundary
                # Chunk = retrieval boundary
                section_chunks = chunk_section_text(text)

                for chunk_index, chunk in enumerate(section_chunks):

                    all_chunks.append({
                        "text": chunk,
                        "source": metadata["document_id"],
                        "source_type": source_type,
                        "section": section,
                        "chunk_id": chunk_index,
                        "title": metadata.get("title", ""),
                        "url": metadata.get("source", {}).get("url", ""),
                        "pdf": metadata.get("source", {}).get("pdf", ""),
                        "authors": metadata.get("authors", []),
                        "year": metadata.get("year", "")
                    })

    return all_chunks