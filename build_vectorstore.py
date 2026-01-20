# build_vectorstore.py
"""
สร้าง Chroma vector store จาก .jsonl สองไฟล์
เวอร์ชันแก้ไข 20 ม.ค. 2026 - เพิ่ม import json + อ่านไฟล์ด้วยมือ
"""

import logging
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ปิด warning urllib3 บน macOS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_jsonl_to_docs(file_path: str) -> list[Document]:
    """
    อ่าน .jsonl ทีละบรรทัด สร้าง LangChain Document เอง
    """
    docs = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                metadata = data.get("metadata", {})
                if text:
                    docs.append(Document(page_content=text, metadata=metadata))
                else:
                    logger.warning(f"บรรทัด {i}: ไม่มี 'text' → ข้าม")
            except json.JSONDecodeError as e:
                logger.warning(f"บรรทัด {i} ไม่ใช่ JSON ที่ถูกต้อง → ข้าม: {e}")
            except Exception as e:
                logger.warning(f"บรรทัด {i} มีปัญหา: {e}")
    return docs


def create_chroma_from_jsonl(
    faq_path: str = "subacar_allFAQ_chunks.jsonl",
    car_path: str = "subacar_chunksV2.jsonl",
    persist_dir: str = "./chroma_db_subacar",
    collection_name: str = "subacar_all_v2026",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_model: str = "BAAI/bge-m3",
):
    logger.info(f"เริ่มสร้าง vector store | model: {embedding_model}")

    # ── 1. โหลด embedding ──
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},  # เปลี่ยนเป็น "cuda" ถ้ามี GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("โหลด embedding model สำเร็จ")

    # ── 2. โหลด FAQ ──
    logger.info(f"กำลังโหลด FAQ จาก: {faq_path}")
    faq_docs = load_jsonl_to_docs(faq_path)
    logger.info(f"โหลด FAQ ได้ {len(faq_docs)} เอกสาร")

    # ── 3. โหลดข้อมูลรถ ──
    logger.info(f"กำลังโหลดข้อมูลรถจาก: {car_path}")
    car_docs = load_jsonl_to_docs(car_path)
    logger.info(f"โหลดข้อมูลรถได้ {len(car_docs)} เอกสาร")

    all_docs = faq_docs + car_docs
    logger.info(f"รวมเอกสารทั้งหมด {len(all_docs)} ชิ้น")

    # ── 4. Split ข้อความ ──
    if chunk_size > 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        split_docs = text_splitter.split_documents(all_docs)
        logger.info(f"หลัง split เหลือ {len(split_docs)} chunks")
    else:
        split_docs = all_docs
        logger.info("ไม่ split (chunk_size=0)")

    # ── 5. สร้าง Chroma ──
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    count = vectorstore._collection.count()
    logger.info(f"สร้าง Chroma เสร็จสิ้น → {persist_dir}")
    logger.info(f"จำนวนเอกสารใน collection: {count}")

    # ── ทดสอบ retrieval สั้น ๆ ──
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        test_query = "รถ Toyota Yaris ราคาเท่าไหร่"
        results = retriever.invoke(test_query)
        if results:
            logger.info("ตัวอย่างผล retrieval สำเร็จ:")
            logger.info(results[0].page_content[:180] + "...")
            logger.info(f"Metadata: {results[0].metadata}")
    except Exception as e:
        logger.warning(f"ทดสอบ retrieval ล้มเหลว: {e}")

    return vectorstore


if __name__ == "__main__":
    print("=" * 70)
    print("สร้าง Chroma vector store สำหรับ SubaCar RAG")
    print("เวอร์ชันแก้ไขล่าสุด - อ่าน JSONL ด้วยมือ + import json")
    print("=" * 70)
    
    create_chroma_from_jsonl()
    
    print("\n" + "=" * 70)
    print("เสร็จสิ้น! โฟลเดอร์ ./chroma_db_subacar พร้อมนำไปใช้ใน app.py แล้ว")
    print("=" * 70)