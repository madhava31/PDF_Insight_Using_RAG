import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI
import gradio as gr

# LangChain and PDF-related imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ================================
# Load Environment Variables
# ================================
load_dotenv()
api_key = os.getenv("HUGGINGFACE_TOKEN_API")
if not api_key:
    raise ValueError("❌ HUGGINGFACE_TOKEN_API not found. Set it in Render Secrets.")

# ================================
# Embeddings, Splitter, LLM Setup
# ================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# ================================
# Helper Functions
# ================================
def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ================================
# Load LLM
# ================================
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    max_new_tokens=256,
    temperature=0.3
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# ================================
# Conversation + Chain Logic
# ================================
conversation_history = []
retriever = None
main_chain = None

def load_pdf(file_path):
    """Load PDF and initialize FAISS retriever & chain."""
    global retriever, main_chain, conversation_history
    conversation_history = []

    if file_path is None:
        return "⚠️ Please upload a PDF first."

    # file_path is already a string path (Gradio type="file")
    docs = PyPDFLoader(file_path).load()
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
        "history": RunnableLambda(lambda _: format_history(conversation_history)),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | parser
    return f"✅ PDF loaded successfully. You can now ask questions!"

def ask_question(question):
    global conversation_history, main_chain
    if main_chain is None:
        return "⚠️ Please upload a PDF first."

    response = main_chain.invoke(question).strip()
    conversation_history.append({"question": question, "answer": response})
    return response

def ask_question_and_update(chat_history, question):
    if not question.strip():
        return chat_history, ""

    # 1️⃣ First, show the user's question immediately
    chat_history.append({"role": "user", "content": question})
    yield chat_history, ""  # This updates the chatbox immediately

    # 2️⃣ Then process the answer
    answer = ask_question(question)
    chat_history.append({"role": "assistant", "content": answer})

    # 3️⃣ Update chat with the assistant's response
    yield chat_history, ""



# ================================
# Gradio UI
# ================================
with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 📚 PDF Insight RAG — Chat with your PDF")
    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        load_btn = gr.Button("Load PDF")

    status = gr.Textbox(label="Status", interactive=False, lines=2)
    chatbot = gr.Chatbot(label="Chat", height=400, type="messages")
    query = gr.Textbox(label="Ask a question", placeholder="Type your question here...")

    load_btn.click(load_pdf, inputs=pdf_file, outputs=status)
    query.submit(ask_question_and_update, inputs=[chatbot, query], outputs=[chatbot, query])

# ================================
# FastAPI App Wrapper for Render
# ================================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 PDF Insight RAG is running successfully!"}

# Mount Gradio at /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

# ================================
# Launch Server (local / Render)
# ================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Render automatically assigns PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
