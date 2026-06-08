import os
from fastapi import FastAPI
import gradio as gr
from openai import OpenAI

# Config imports
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from config.templates import TemplateLoader
from config.pdf_processor import EnhancedPDFProcessor


# ================================
# PDF Processor, Template & Client Setup
# ================================
pdf_processor = EnhancedPDFProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    enable_preprocessing=True,
    enable_semantic_split=True,
)
template_loader = TemplateLoader()
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
# Conversation + Chain Logic
# ================================
conversation_history = []
pdf_chunks = []

def load_pdf(file_path):
    """Load PDF with enhanced processing and metadata preservation."""
    global conversation_history, pdf_chunks
    conversation_history = []
    pdf_chunks = []

    if file_path is None:
        return "⚠️ Please upload a PDF first."

    try:
        # Use enhanced processor for better chunking and metadata
        pdf_chunks = pdf_processor.process(file_path)
        summary = pdf_processor.get_chunk_summary(pdf_chunks)
        
        return (
            f"✅ PDF loaded successfully!\n"
            f"📊 Chunks: {summary['total_chunks']} | "
            f"Pages: {summary['pages_extracted']} | "
            f"Avg chunk: {int(summary['avg_chunk_size'])} chars"
        )
    except ValueError as e:
        return f"⚠️ Error loading PDF: {e}"

def ask_question(question):
    global conversation_history, pdf_chunks

    if client is None:
        return "⚠️ Set OPENAI_API_KEY or NVIDIA_API_KEY in your .env file first."

    if not pdf_chunks:
        return "⚠️ Please upload a PDF first."

    if not question.strip():
        return "Please ask a question about the PDF."

    question_terms = [term.lower() for term in question.split() if len(term) > 2]

    scored_chunks = []
    for doc in pdf_chunks:
        text = doc.page_content.strip()
        lowered = text.lower()
        score = sum(1 for term in question_terms if term in lowered)
        scored_chunks.append((score, text))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [text for score, text in scored_chunks[:4] if score > 0]

    if not top_chunks:
        return "I don't know."

    context = "\n\n".join(top_chunks)
    history_text = format_history(conversation_history)
    
    system_prompt = template_loader.render("system_prompt", {})
    user_prompt = template_loader.render(
        "user_prompt",
        {"history_text": history_text, "context": context, "question": question},
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=True,
        )
        answer_parts = []
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                answer_parts.append(content)
        answer = "".join(answer_parts).strip()
    except Exception as e:
        fallback = top_chunks[0].strip()
        answer = f"I don't know. ({e})" if not fallback else fallback

    conversation_history.append({"question": question, "answer": answer})
    return answer

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
with gr.Blocks() as demo:
    gr.Markdown("## 📚 PDF Insight RAG — Chat with your PDF")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False, lines=8)
    
    chatbot = gr.Chatbot(label="Chat", height=400)
    query = gr.Textbox(label="Ask a question", placeholder="Type your question here...")

    # Auto-load PDF when file is uploaded
    pdf_file.change(load_pdf, inputs=pdf_file, outputs=status)
    query.submit(ask_question_and_update, inputs=[chatbot, query], outputs=[chatbot, query])

# ================================
# FastAPI App Wrapper for Render
# ================================
app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)