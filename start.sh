#!/bin/bash

# Wait for Ollama server to be ready
until curl -s http://ollama:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama server..."
    sleep 1
done

# Pull required models
echo "Pulling required models..."
ollama pull mistral
#ollama pull mxbai-embed-large
ollama pull nomic-embed-text

# Start the application
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 