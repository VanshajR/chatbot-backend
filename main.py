import json
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from retriever import get_combined_retriever
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Load user name from profile
with open("user_profile.json") as f:
    user_data = json.load(f)
user_name = user_data["personal_info"]["name"]

app = FastAPI()

# CORS for frontend on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class ChatRequest(BaseModel):
    message: str
    model: str = "Llama3-70b-8192"

# Set up memory (in-memory session; can switch to Redis if needed)
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# Create prompt
prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant created to answer questions about {name}. You are **not** {name}, but you use the provided context to give accurate responses.

Context about {name}:
{context}

Conversation History:
{history}

**Rules:**
1. Be respectful and professional.
2. Answer only using the given context.
3. If unsure, say "I don't have that information."
4. Keep responses professional and concise.

**User's Question:** {input}
""")

@app.get("/ping")
def health_check():
    return JSONResponse({"status": "ok"})

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        retriever = get_combined_retriever()
        llm = ChatGroq(model_name=req.model, api_key=os.getenv("GROQ_API_KEY"))
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, prompt_template),
        )

        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages[-5:]])
        context_input = {
            "input": req.message,
            "history": history_str,
            "name": user_name,
            "context": ""
        }

        response = chain.invoke(context_input)
        answer = response.get("answer", "I don't have that information.")

        memory.chat_memory.add_user_message(req.message)
        memory.chat_memory.add_ai_message(answer)

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
