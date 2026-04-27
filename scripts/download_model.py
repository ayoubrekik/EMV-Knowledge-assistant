from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(f"Downloading model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"  # optional
)

print("Download completed!")