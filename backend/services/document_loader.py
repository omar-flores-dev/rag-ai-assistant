import os
import json

DATA_PATH = "backend/data/raw"


def load_documents_structured():
    """
    Loads all documents in structured format:
    - reads metadata.json
    - reads section files
    - returns structured chunks with metadata
    """

    all_chunks = []

    for doc_folder in os.listdir(DATA_PATH):
        doc_path = os.path.join(DATA_PATH, doc_folder)

        if not os.path.isdir(doc_path):
            continue

        metadata_path = os.path.join(doc_path, "metadata.json")

        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        sections = metadata.get("sections", [])

        for section in sections:
            section_file = os.path.join(doc_path, f"{section}.txt")

            if not os.path.exists(section_file):
                continue

            with open(section_file, "r", encoding="utf-8") as f:
                text = f.read()

            all_chunks.append({
                "text": text,
                "source": metadata["document_id"],
                "section": section,
                "title": metadata.get("title", "")
            })

    return all_chunks