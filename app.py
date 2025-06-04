import streamlit as st
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

st.set_page_config(page_title="Flesh and Blood - RAG", layout="wide")
st.title("🧠 Flesh and Blood - Sääntöassistentti")

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="db", embedding_function=embedding)
retriever = db.as_retriever()
llm = Ollama(model="mistral")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = st.text_input("Kysy säännöistä (englanniksi):")

if query:
    with st.spinner("Haetaan sääntötulkintaa..."):
        answer = qa.run(query)
    st.success(answer)