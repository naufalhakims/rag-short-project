# Pocket Librarian — container image.
#
# On a server, use docker-compose.yml (adds HTTPS and keeps data); see deploy/ORACLE.md.
# On its own:
#   docker build -t pocket-librarian .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... -e APP_PASSWORD=... pocket-librarian
#
# Builds on x86-64 and Arm64 (for example Oracle Cloud's Ampere servers). Runs as an
# unprivileged user (uid 1000) on port 7860; this layout also suits Hugging Face Spaces.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/data \
    MODEL_DIR=/home/user/models \
    SEED_SAMPLE_DOCS=true
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Bake the retrieval models into the image so the app starts without downloading them.
# Keep these in step with EMBEDDING_MODEL and RERANKER_MODEL if you change them.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
RUN python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='${MODEL_DIR}'); \
TextCrossEncoder('${RERANKER_MODEL}', cache_dir='${MODEL_DIR}')"

COPY --chown=user . .
RUN mkdir -p "$DATA_DIR"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
