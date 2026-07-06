from sentence_transformers import SentenceTransformer, CrossEncoder
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import subprocess
import os

HF_CACHE = "/root/.cache/huggingface"
PIPER_DIR = "/root/.cache/piper"

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(PIPER_DIR, exist_ok=True)

print("=" * 60)
print("Downloading Hugging Face models...")
print("=" * 60)

# ------------------------------------------------------------------
# Embedding Model
# ------------------------------------------------------------------
print("\n[1/6] Embedding model...")
SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    cache_folder=HF_CACHE
)

# ------------------------------------------------------------------
# Cross Encoder (Reranker)
# ------------------------------------------------------------------
print("\n[2/6] Cross Encoder...")
CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cache_folder=HF_CACHE
)

# ------------------------------------------------------------------
# Faster Whisper
# ------------------------------------------------------------------
print("\n[3/6] Faster Whisper...")
WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    download_root=HF_CACHE
)

# ------------------------------------------------------------------
# Docling Layout Model
# ------------------------------------------------------------------
print("\n[4/6] Docling Layout Model...")
snapshot_download(
    repo_id="docling-project/docling-layout-heron",
    cache_dir=HF_CACHE
)

# ------------------------------------------------------------------
# Docling Models
# ------------------------------------------------------------------
print("\n[5/6] Docling Models...")
snapshot_download(
    repo_id="docling-project/docling-models",
    cache_dir=HF_CACHE
)

# ------------------------------------------------------------------
# Piper Voice
# ------------------------------------------------------------------
print("\n[6/6] Piper Voice...")

voice_url = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/main/en/en_US/amy/medium/"
    "en_US-amy-medium.onnx"
)

config_url = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/main/en/en_US/amy/medium/"
    "en_US-amy-medium.onnx.json"
)

subprocess.run(
    [
        "wget",
        "-nc",
        "-P",
        PIPER_DIR,
        voice_url
    ],
    check=False
)

subprocess.run(
    [
        "wget",
        "-nc",
        "-P",
        PIPER_DIR,
        config_url
    ],
    check=False
)

print("\n✓ All Hugging Face models downloaded successfully.")