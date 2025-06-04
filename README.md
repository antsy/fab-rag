# Flesh and Blood RAG Assistant

### 🍖🩸🤖

Felt cute, might delete later. 🤠

## Setup and Installation

### Building the Container
```bash
docker compose build
```

### Running the Application
```bash
docker compose up -d
```

### Data Ingestion
To ingest data into the system, run:
```bash
docker exec -it fab-assistant python ingest.py
```
Note: the data ingestion will take some time, be patient!

### Ask a question

```bash
docker exec -it fab-assistant python ask.py
```
