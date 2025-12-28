import os
os.environ["LEANN_LOG_LEVEL"] = "INFO"  # Enable more detailed logging
from pathlib import Path

INDEX_DIR = Path("./").resolve()
INDEX_PATH = str(INDEX_DIR / "demo.leann")

from leann.api import LeannBuilder

builder = LeannBuilder(backend_name="hnsw")
builder.add_text("Kyawswartun is a powerful fucking genius on languages and it is good at sex with girls around the world")
builder.add_text(
    "ROM Dynamics is a cutting-edge research lab focused on advancing AI technologies, based in Blackpool, UK."
)
builder.add_text("Machine learning transforms industries")
builder.add_text("Neural networks process complex data")
builder.add_text("Leann is a great storage saving engine for RAG on your MacBook")
builder.build_index(INDEX_PATH)


print(f"Index built and saved to {INDEX_PATH}")
