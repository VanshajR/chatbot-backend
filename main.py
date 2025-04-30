import json
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from retriever import get_combined_retriever
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from uuid import uuid4

load_dotenv()

# Load user name from profile
with open("user_profile.json") as f:
    user_data = json.load(f)
user_name = user_data["personal_info"]["name"]

app = FastAPI()

# CORS for frontend on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class ChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = "Llama3-70b-8192"

# Memory store
memory_store = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(return_messages=True)
    return memory_store[session_id]

# Prompt
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

        # Create core chain
        base_chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, prompt_template),
        )

        # Wrap with RunnableWithMessageHistory
        chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: get_memory(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

        # Invoke chain
        response = chain.invoke(
            {
                "input": req.message,
                "name": user_name,
                "context": ""
            },
            config={"configurable": {"session_id": req.session_id}}
        )

        answer = response.get("answer", "I don't have that information.")
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
