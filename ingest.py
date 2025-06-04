import requests
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
import time

print("📥 Downloading and loading files...")
# Load .txt manual
if not os.path.exists("en-fab-cr.txt"):
    response = requests.get("https://rules.fabtcg.com/txt/latest/en-fab-cr.txt")
    with open("en-fab-cr.txt", "w") as f:
        f.write(response.text)
with open("en-fab-cr.txt", "r") as f:
    raw_text = f.read()

# Load card database
card_response = requests.get("https://the-fab-cube.github.io/flesh-and-blood-cards/json/english/card.json")
cards = card_response.json()

print(f"🃏 Processing {len(cards)} cards...")
# Process cards into text format
card_texts = []
for card in tqdm(cards, desc="Processing cards"):
    # Create a structured card description
    card_info = [
        f"=== CARD: {card['name']} ===",
        f"Type: {card.get('type_text', '')}",
        f"Subtypes: {', '.join(card['types'])}",
        f"Cost: {card['cost']}",
        f"Pitch Value: {card['pitch']}",
    ]

    # Add power and defense if present
    if card.get('power'):
        card_info.append(f"Power: {card['power']}")
    if card.get('defense'):
        card_info.append(f"Defense: {card['defense']}")

    # Add all keywords in a structured way
    if card.get('card_keywords'):
        card_info.append(f"Card Keywords: {', '.join(card['card_keywords'])}")
    if card.get('ability_and_effect_keywords'):
        card_info.append(f"Ability Keywords: {', '.join(card['ability_and_effect_keywords'])}")
    if card.get('granted_keywords'):
        card_info.append(f"Granted Keywords: {', '.join(card['granted_keywords'])}")
    if card.get('removed_keywords'):
        card_info.append(f"Removed Keywords: {', '.join(card['removed_keywords'])}")
    if card.get('interacts_with_keywords'):
        card_info.append(f"Interacts With Keywords: {', '.join(card['interacts_with_keywords'])}")

    # Add card text with clear formatting
    if card.get('functional_text'):
        card_info.append(f"Card Text: {card['functional_text']}")

    # Add legality information in a structured way
    # legality_info = []
    # if card.get('blitz_banned'):
    #     legality_info.append("Banned in Blitz")
    # if card.get('cc_banned'):
    #     legality_info.append("Banned in Classic Constructed")
    # if card.get('commoner_banned'):
    #     legality_info.append("Banned in Commoner")
    # if card.get('ll_banned'):
    #     legality_info.append("Banned in Living Legend")

    # if card.get('blitz_suspended'):
    #     legality_info.append("Suspended in Blitz")
    # if card.get('cc_suspended'):
    #     legality_info.append("Suspended in Classic Constructed")
    # if card.get('commoner_suspended'):
    #     legality_info.append("Suspended in Commoner")

    # if card.get('ll_restricted'):
    #     legality_info.append("Restricted in Living Legend")

    # if legality_info:
    #     card_info.append(f"Legality Status: {', '.join(legality_info)}")

    card_texts.append("\n".join(card_info))

# Combine rules and card texts
all_texts = [raw_text] + card_texts

print("📚 Creating text chunks...")
# Split text into documents with smaller chunks and more overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
chunks = splitter.split_text("\n\n".join(all_texts))
documents = [Document(page_content=chunk) for chunk in chunks]

print("🔢 Creating embeddings and database...")
# Create embeddings and database
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # This model has 768 dimensions
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder="model_cache"  # Add local caching for the model
)

# Process documents in batches
BATCH_SIZE = 20
db = None

# Clear existing database if it exists
if os.path.exists("db"):
    import shutil
    shutil.rmtree("db")
    print("Cleared existing database")

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Creating embeddings"):
    batch = documents[i:i + BATCH_SIZE]
    try:
        if db is None:
            db = Chroma.from_documents(batch, embedding=embedding, persist_directory="db")
        else:
            db.add_documents(batch)
    except Exception as e:
        print(f"\nWarning: Error processing document {i + 1}: {str(e)}")
        continue

if db is not None:
    db.persist()
    print("✅ Manual and card database indexed!")
else:
    print("❌ Failed to create database!")