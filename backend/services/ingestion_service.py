import os
import json

DATA_PATH = "backend/data/raw"

"""
FILE OBJECTIVE:
Structured document ingestion layer for the RAG pipeline.

This module is responsible for:
- Loading documents from multiple source types (research papers, websites, docs, etc.)
- Parsing structured metadata (metadata.json)
- Extracting section-based content from each document
- Returning normalized document objects for downstream embedding + chunking

NOTE:
This is NOT final chunking yet. It is a structured pre-processing stage.
Final chunking (semantic or hybrid) happens downstream in the pipeline.
"""


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

                # Structured document unit (NOT final embedding chunk yet)
                all_chunks.append({
                    "text": text,
                    "source": metadata["document_id"],
                    "source_type": source_type,
                    "section": section,
                    "title": metadata.get("title", ""),
                    "url": metadata.get("source", {}).get("url", ""),
                    "pdf": metadata.get("source", {}).get("pdf", ""),
                    "authors": metadata.get("authors", []),
                    "year": metadata.get("year", "")
                })

    print(f"Loaded {len(all_chunks)} structured document sections.")
    return all_chunks