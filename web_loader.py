import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# import os

# def scrape_single_page_site(url):
#     print(f"Scraping site: {url}")
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.content, "html.parser")

#         # Remove script and style tags
#         for script in soup(["script", "style", "noscript"]):
#             script.extract()

#         # Get all visible text
#         text = soup.get_text(separator=" ", strip=True)
#         return text

#     except Exception as e:
#         print(f"Error scraping site: {e}")
#         return ""


# def build_faiss_from_web(site_url, save_path="site_faiss_index"):
#     content = scrape_single_page_site(site_url)
    
#     if not content:
#         print("No content scraped.")
#         return
    
#     # Split text
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     docs = splitter.create_documents([content])

#     # Load embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Build FAISS vector store
#     vector_store = FAISS.from_documents(docs, embeddings)

#     # Save
#     vector_store.save_local(save_path)
#     print(f"Saved FAISS index to {save_path}")

# if __name__ == "__main__":
#     build_faiss_from_web("https://vanshajraghuvanshi.me/")

from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load JS-rendered content from your React site
loader = PlaywrightURLLoader(["https://vanshajraghuvanshi.me/"], remove_selectors=["header", "footer"])
docs = loader.load()

# Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 3: Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Save to disk
vectorstore.save_local("site_faiss_index")
print("✅ JS-rendered site indexed and saved as site_faiss_index/")
