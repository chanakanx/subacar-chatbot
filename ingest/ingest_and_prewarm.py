"""
Ingest + Preload Embedding Script
รันก่อน Streamlit start
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import time
import json

print("🚀 Starting ingest + preload embedding...")

# =====================
# 1. Load documents
# =====================
def load_documents():
    docs = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    jsonl_files = [
        os.path.join(PROJECT_ROOT, "subacar_chunksV2.jsonl"),
        os.path.join(PROJECT_ROOT, "subacar_allFAQ_chunks.jsonl"),
    ]

    for path in jsonl_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                docs.append(
                    Document(
                        page_content=item["text"],
                        metadata=item.get("metadata", {})
                    )
                )

    return docs


def main():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = splitter.split_documents(documents)

    print(f"📄 Documents split into {len(splits)} chunks")

    start = time.time()

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64}
    )

    print(f"🧠 Embedding model loaded in {time.time() - start:.2f}s")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="chroma_db",
        collection_name="subacar_all"
    )

    print("✅ Vector DB persisted to chroma_db")

    print("🔥 Pre-warming query embedding...")

    _ = embedding.embed_query("ทดลองระบบ")

    print("🎉 Ingest + preload finished successfully")

if __name__ == "__main__":
    main()