import os
from fastapi import FastAPI
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# =========================================================
# 1️⃣ Load environment variables
# =========================================================
load_dotenv()
api_key = os.getenv("HUGGINGFACE_TOKEN_API")
if not api_key:
    raise ValueError("❌ HUGGINGFACE_TOKEN_API not found. Please set it in Render Secrets.")

# =========================================================
# 2️⃣ Initialize text splitter & embeddings
# =========================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =========================================================
# 3️⃣ Define prompt
# =========================================================
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

Instructions:
- If the new question is about previous questions or answers, use ONLY the conversation history.
- If the question is about the PDF, use the PDF context.
- If you cannot find the answer in either, respond with "I don't know."
""",
    input_variables=["history", "context", "question"]
)

# =========================================================
# 4️⃣ Helper functions
# =========================================================
def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# =========================================================
# 5️⃣ Load LLM
# =========================================================
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational",
    model_kwargs={"api_key": api_key}
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# =========================================================
# 6️⃣ Global state
# =========================================================
conversation_history = []
retriever = None
main_chain = None

# =========================================================
# 7️⃣ PDF Loader
# =========================================================
def load_pdf(file):
    global retriever, main_chain, conversation_history
    conversation_history = []
    if file is None:
        return "⚠️ Please upload a PDF first."

    pdf_path = file.name
    docs = PyPDFLoader(pdf_path).load()
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
        "history": RunnableLambda(lambda _: format_history(conversation_history)),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | parser
    return f"✅ PDF loaded successfully: {file.name}. You can now ask questions!"

# =========================================================
# 8️⃣ Ask question logic
# =========================================================
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
    answer = ask_question(question)
    chat_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""

# =========================================================
# 9️⃣ Build Gradio UI
# =========================================================
with gr.Blocks(theme="default") as gradio_app:
    gr.Markdown("## 📘 PDF Insight RAG Assistant")
    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        load_btn = gr.Button("Load PDF")
    status = gr.Textbox(label="Status", interactive=False, lines=2)
    chatbot = gr.Chatbot(label="Chat with your PDF", height=400, type="messages")
    query = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    submit_btn = gr.Button("Ask")

    load_btn.click(load_pdf, inputs=pdf_file, outputs=status)
    submit_btn.click(ask_question_and_update, inputs=[chatbot, query], outputs=[chatbot, query])

# =========================================================
# 🔟 FastAPI wrapper to expose Gradio on Render
# =========================================================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 PDF Insight RAG is running on Render!"}

# Mount Gradio interface to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# =========================================================
# 🚀 Launch Uvicorn server
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
