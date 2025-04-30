import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

def scrape_single_page_site(url):
    print(f"Scraping site: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Define the known section headers on your site
        section_titles = ["Overview", "Expertise", "Projects", "Contact"]
        extracted_sections = []

        for title in section_titles:
            section = soup.find(lambda tag: tag.name in ["h1", "h2", "h3"] and title.lower() in tag.text.lower())
            if section:
                content = []
                for sibling in section.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3"]:
                        break
                    content.append(sibling.get_text(separator=" ", strip=True))
                extracted_sections.append(f"{title}:\n" + "\n".join(content))
        
        return "\n\n".join(extracted_sections)
    
    except Exception as e:
        print(f"Error scraping site: {e}")
        return ""

def build_faiss_from_web(site_url, save_path="site_faiss_index"):
    content = scrape_single_page_site(site_url)
    
    if not content:
        print("No content scraped.")
        return
    
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([content])

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save
    vector_store.save_local(save_path)
    print(f"Saved FAISS index to {save_path}")

if __name__ == "__main__":
    build_faiss_from_web("https://vanshajraghuvanshi.me/")
