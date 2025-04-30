import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MergerRetriever

def get_combined_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    user_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    site_index = FAISS.load_local("site_faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever1 = user_index.as_retriever()
    retriever2 = site_index.as_retriever()

    merged = MergerRetriever(retrievers=[retriever1, retriever2])
    return merged
