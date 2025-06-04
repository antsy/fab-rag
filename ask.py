import os
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def main():
    # Initialize the model and vector store
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)
    db = Chroma(persist_directory="db", embedding_function=embedding)
    retriever = db.as_retriever()
    llm = OllamaLLM(model="mistral", base_url=ollama_host)

    # Create the QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("Flesh and Blood - Rule Assistant")
    print("Type 'exit' to quit")
    print("-" * 50)

    while True:
        query = input("\nAsk about the rules (in English): ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            continue

        try:
            answer = qa.invoke({"query": query})
            print("\nAnswer:", answer["result"])
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 