import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import get_combined_retriever
from dotenv import load_dotenv

load_dotenv()

# Load API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load user profile once
with open("user_profile.json") as f:
    user_data = json.load(f)
user_name = user_data["personal_info"]["name"]

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history store
chat_histories = {}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
        user_prompt = body.get("prompt")
        session_id = body.get("session_id", "default")

        if not user_prompt:
            return {"error": "No prompt provided."}
        if not GROQ_API_KEY:
            return {"error": "GROQ_API_KEY not found in environment variables."}

        # Load combined retriever (user + site index)
        # retriever = get_combined_retriever()
        try:
            retriever = get_combined_retriever()
        except Exception as e:
            print(f"[ERROR] Failed to load retriever: {e}")


        # Load LLM
        ChatGroq.model_rebuild()
        llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", api_key=GROQ_API_KEY)

        # Chat prompt template
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

        # Chain setup
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, prompt_template)
        )

        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(user_prompt)

        # Get last 5 messages from session history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        history_msgs = chat_histories[session_id][-5:]
        history = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])

        # Invoke chain
        result = chain.invoke({
            "input": user_prompt,
            "name": user_name,
            "context": "\n\n".join([doc.page_content for doc in retrieved_docs]),
            "history": history
        })

        answer = result.get("answer", "I don't have that information.")

        # Update history
        chat_histories[session_id].append({"role": "user", "content": user_prompt})
        chat_histories[session_id].append({"role": "assistant", "content": answer})

        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


# import os
# import json
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from retriever import get_combined_retriever
# from dotenv import load_dotenv

# load_dotenv()

# # Load API keys
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Load user profile once
# with open("user_profile.json") as f:
#     user_data = json.load(f)
# user_name = user_data["personal_info"]["name"]

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory history
# chat_histories = {}

# # ===== GLOBAL INIT BLOCK =====

# # Try to load retriever ONCE
# try:
#     print("[INIT] Loading retriever...")
#     retriever = get_combined_retriever()
#     print("[INIT] Retriever loaded.")
# except Exception as e:
#     print(f"[ERROR] Failed to load retriever: {e}")
#     retriever = None  # fallback to safe value

# # Initialize Groq LLM and chain
# print("[INIT] Initializing ChatGroq...")
# ChatGroq.model_rebuild()
# llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", api_key=GROQ_API_KEY)

# prompt_template = ChatPromptTemplate.from_template("""
# You are an AI assistant created to answer questions about {name}. You are **not** {name}, but you use the provided context to give accurate responses.

# Context about {name}:
# {context}

# Conversation History:
# {history}

# **Rules:**
# 1. Be respectful and professional.
# 2. Answer only using the given context.
# 3. If unsure, say "I don't have that information."
# 4. Keep responses professional and concise.

# **User's Question:** {input}
# """)

# chain = create_retrieval_chain(
#     retriever,
#     create_stuff_documents_chain(llm, prompt_template)
# )

# # ===== API ROUTES =====

# # @app.post("/chat")
# # async def chat_endpoint(request: Request):
# #     try:
# #         body = await request.json()
# #         user_prompt = body.get("prompt")
# #         session_id = body.get("session_id", "default")

# #         if not user_prompt:
# #             return {"error": "No prompt provided."}
# #         if not GROQ_API_KEY:
# #             return {"error": "GROQ_API_KEY not found in environment variables."}
# #         if not retriever:
# #             return {"error": "Retriever could not be initialized."}

# #         # Get relevant docs
# #         try:
# #             print(f"[RETRIEVAL] Getting relevant documents for: {user_prompt}")
# #             retrieved_docs = retriever.get_relevant_documents(user_prompt)
# #             print(f"[RETRIEVAL] Retrieved {len(retrieved_docs)} docs.")
# #         except Exception as e:
# #             print(f"[ERROR] Document retrieval failed: {e}")
# #             retrieved_docs = []

# #         # Get history
# #         if session_id not in chat_histories:
# #             chat_histories[session_id] = []
# #         history_msgs = chat_histories[session_id][-5:]
# #         history = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])

# #         # Run chain
# #         print("[CHAIN] Invoking chain...")
# #         result = chain.invoke({
# #             "input": user_prompt,
# #             "name": user_name,
# #             "context": "\n\n".join([doc.page_content for doc in retrieved_docs]),
# #             "history": history
# #         })

# #         answer = result.get("answer", "I don't have that information.")

# #         # Save history
# #         chat_histories[session_id].append({"role": "user", "content": user_prompt})
# #         chat_histories[session_id].append({"role": "assistant", "content": answer})

# #         return {"response": answer}

# #     except Exception as e:
# #         print(f"[ERROR] /chat route crashed: {e}")
# #         return {"error": str(e)}

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     try:
#         body = await request.json()
#         user_prompt = body.get("prompt")

#         dummy_context = "Vanshaj Raghuvanshi is a computer science student with experience in AI and full-stack development."

#         result = chain.invoke({
#             "input": user_prompt,
#             "name": user_name,
#             "context": dummy_context,
#             "history": ""
#         })

#         answer = result.get("answer", "No answer found.")
#         return {"response": answer}

#     except Exception as e:
#         print(f"[ERROR] /chat route crashed: {e}")
#         return {"error": str(e)}


# @app.get("/ping")
# def ping():
#     return {"status": "ok"}
