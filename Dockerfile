FROM python:3.10-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install base packages
RUN pip install --no-cache-dir langchain langchain-ollama langchain-community langchain-huggingface langchain-chroma

# Install vector store and database packages
RUN pip install --no-cache-dir chromadb llama-index faiss-cpu

# Install web framework and utilities
RUN pip install --no-cache-dir streamlit tqdm

# Install ML and embedding packages
RUN pip install --no-cache-dir sentence-transformers torch

# Fix permissions for model cache directory
RUN mkdir -p model_cache && chmod -R 755 model_cache

COPY . .
RUN chmod +x start.sh

CMD ["./start.sh"]