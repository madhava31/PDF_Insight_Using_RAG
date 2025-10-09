# app.py
import os
import threading
import time
from dotenv import load_dotenv

import gradio as gr
from fastapi import FastAPI
from starlette.responses import RedirectResponse, PlainTextResponse

# LangChain / HF imports (kept here but heavy init deferred)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ========== Globals & placeholders ==========
load_dotenv()  # local dev: read .env
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_API")  # set this in Render secrets too

# state
model_ready = False
init_error = None

# placeholders for objects we will create in background init
splitter = None
embeddings = None
load_pdf_runnable = None
split_runnable = None
vectorstore_runnable = None
prompt = None
llm = None
model = None
parser = None

# conversation / retrieval state
conversation_history = []
retriever = None
main_chain = None

# ========== Helper functions (UI-safe, check model_ready) ==========
def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def safe_load_pdf(file):
    """Gradio wrapper for load_pdf with safety checks."""
    if not model_ready:
        return "⚠️ Model is still initializing. Please wait a bit and try again."
    # delegate to actual loader
    return load_pdf(file)

def safe_ask_question_and_update(chat_history, question):
    if not question.strip():
        return chat_history, ""
    if not model_ready:
        return chat_history + [{"role":"assistant","content":"⚠️ Model still initializing. Please wait..."}], ""
    answer = ask_question(question)
    chat_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""

# ========== Actual RAG functions (will use objects created after init) ==========
def load_pdf(file):
    """Load the uploaded PDF and build FAISS index (uses runnables created after init)."""
    global retriever, main_chain, conversation_history
    conversation_history = []
    if file is None:
        return "⚠️ Please upload a PDF first."
    pdf_path = file.name  # Gradio gives local temp path
    # Build FAISS index using Runnables
    retriever = (load_pdf_runnable | split_runnable | vectorstore_runnable).invoke(pdf_path).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
        "history": RunnableLambda(lambda _: format_history(conversation_history)),
        "question": RunnablePassthrough()
    })
    # set main chain
    # `prompt`, `model`, `parser` set in init
    main_chain = parallel_chain | prompt | model | parser
    return f"✅ PDF loaded successfully: {os.path.basename(file.name)}. You can now ask questions!"

def ask_question(question):
    """Ask the RAG chain."""
    global conversation_history, main_chain
    if main_chain is None:
        return "⚠️ Please upload a PDF first."
    response = main_chain.invoke(question).strip()
    conversation_history.append({"question": question, "answer": response})
    return response

# ========== Build Gradio UI (uses safe wrappers) ==========
with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 📚 PDF Insight RAG Model (mounted at /gradio)")
    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        load_btn = gr.Button("Load PDF")
    status = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)
    chatbot = gr.Chatbot(label="Chat with PDF", height=400, type="messages")
    query = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    submit_btn = gr.Button("Ask")

    # connect safe functions
    load_btn.click(safe_load_pdf, inputs=pdf_file, outputs=status)
    submit_btn.click(safe_ask_question_and_update, inputs=[chatbot, query], outputs=[chatbot, query])

# optional: enable queue if you expect concurrent requests
demo.queue()

# ========== FastAPI app & mount Gradio ==========
app = FastAPI()

# root -> redirect to mounted Gradio UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/gradio")

@app.get("/health", include_in_schema=False)
def health():
    if model_ready:
        return PlainTextResponse("ok")
    if init_error:
        return PlainTextResponse(f"error: {init_error}", status_code=500)
    return PlainTextResponse("initializing", status_code=202)

# mount gradio ASGI app at /gradio
app.mount("/gradio", demo.app)

# ========== Background initialization ==========
def initialize_heavy_components():
    """Initialize HuggingFace endpoint, embeddings, runnables, prompt, etc.
       This runs in a background thread so the server starts fast.
    """
    global splitter, embeddings, load_pdf_runnable, split_runnable, vectorstore_runnable
    global prompt, llm, model, parser, model_ready, init_error

    try:
        # Re-read token (in case Render provides it)
        load_dotenv()
        token = os.getenv("HUGGINGFACE_TOKEN_API")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN_API not found in environment.")

        # Basic components (may be heavy)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Runnables
        load_pdf_runnable = RunnableLambda(lambda pdf_path: PyPDFLoader(pdf_path).load())
        split_runnable = RunnableLambda(lambda docs: splitter.split_documents(docs))
        vectorstore_runnable = RunnableLambda(lambda docs: FAISS.from_documents(docs, embeddings))

        # Prompt template
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context and conversation history.
If the context is insufficient, say "I don't know."

Conversation History:
{history}

Transcript Context:
{context}

Question: {question}
""",
            input_variables=["history", "context", "question"]
        )

        # HuggingFace LLM endpoint - pass token correctly
        llm = HuggingFaceEndpoint(
            repo_id="google/gemma-2-2b-it",
            task="conversational",
            huggingfacehub_api_token=token
        )
        model = ChatHuggingFace(llm=llm)
        parser = StrOutputParser()

        # mark ready
        model_ready = True
        print("[init] Model and runnables initialized successfully.")
    except Exception as e:
        init_error = str(e)
        print("[init] ERROR during initialization:", init_error)

# start background init on FastAPI startup
@app.on_event("startup")
def on_startup():
    # do not block startup; run heavy init in daemon thread
    t = threading.Thread(target=initialize_heavy_components, daemon=True)
    t.start()

# ========== Run app with Uvicorn (Render start) ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render will set PORT env var
    # run the ASGI app (FastAPI) which has Gradio mounted at /gradio
    uvicorn.run(app, host="0.0.0.0", port=port)
