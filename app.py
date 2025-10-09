import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_TOKEN_API")
if not api_key:
    raise ValueError("❌ HUGGINGFACE_TOKEN_API not found. Set it in Render Secrets.")

# Step 1: Splitter & Embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Runnables for PDF processing
load_pdf_runnable = RunnableLambda(lambda pdf_path: PyPDFLoader(pdf_path).load())
split_runnable = RunnableLambda(lambda docs: splitter.split_documents(docs))
vectorstore_runnable = RunnableLambda(lambda docs: FAISS.from_documents(docs, embeddings))

# Step 3: Prompt Template
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

# Helper functions
def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Step 4: LLM via HuggingFace Inference API (tiny model)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",  # lightweight
    task="text2text-generation",
    model_kwargs={"api_key": api_key, "max_new_tokens": 200}
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Globals
conversation_history = []
retriever = None
main_chain = None

# PDF loader
def load_pdf(file):
    global retriever, main_chain, conversation_history
    conversation_history = []
    if file is None:
        return "⚠️ Please upload a PDF first."
    pdf_path = file.name
    retriever = (load_pdf_runnable | split_runnable | vectorstore_runnable).invoke(pdf_path).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
        "history": RunnableLambda(lambda _: format_history(conversation_history)),
        "question": RunnablePassthrough()
    })
    main_chain = parallel_chain | prompt | model | parser
    return f"✅ PDF loaded successfully: {file.name}. You can now ask questions!"

# Chat logic
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

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📚 PDF Insight RAG Model")
    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        load_btn = gr.Button("Load PDF")
    status = gr.Textbox(label="Status", interactive=False, lines=2)
    chatbot = gr.Chatbot(label="Chat with PDF", height=400, type="messages")
    query = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    submit_btn = gr.Button("Ask")
    load_btn.click(load_pdf, inputs=pdf_file, outputs=status)
    submit_btn.click(ask_question_and_update, inputs=[chatbot, query], outputs=[chatbot, query])

# Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
