import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faulthandler
faulthandler.enable()

import json
import torch
import faiss

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent

# 1. PDF okuma

reader = PdfReader(
    BASE_DIR / "data" / "teb_bankacilik_mevzuati.pdf"
)

text = "\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

print("PDF okundu.", flush=True)

# 2. Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

if not chunks:
    raise ValueError("PDF metni okunamadı.")

print("Chunk sayısı:", len(chunks), flush=True)

# 3. Embedding model

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    model_kwargs={
        "torch_dtype": torch.float32
    }
)

print("Embedding modeli yüklendi.", flush=True)

# 4. Embedding oluşturma

print("Embedding işlemi başlıyor.", flush=True)

embeddings = model.encode(
    chunks,
    batch_size=4,
    prompt="",
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

print("Embedding shape:", embeddings.shape, flush=True)

# 5. FAISS

print("FAISS oluşturuluyor.", flush=True)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)

print("FAISS oluşturuldu.", flush=True)

# 6. Kaydetme

directory = BASE_DIR / "vector_db" / "character"

directory.mkdir(
    parents=True,
    exist_ok=True
)

faiss.write_index(
    index,
    str(directory / "index.faiss")
)

with open(
    directory / "chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Vector DB başarıyla kaydedildi.")
print("Toplam vektör:", index.ntotal)