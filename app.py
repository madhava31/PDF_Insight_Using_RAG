"""PDF Insight RAG - Main Application

This module provides a web interface for uploading PDFs and asking questions about their contents.
It uses a LangChain-style runnable chain (RunnableParallel | prompt | model) to compose the RAG pipeline.

Components:
- Gradio UI: interactive chat interface for users
- FastAPI: serves the Gradio app and handles HTTP requests
- RAGChain: orchestrates PDF processing, retrieval, prompting, and LLM calls
- EnhancedPDFProcessor: loads and chunks PDFs with metadata preservation
- TemplateLoader: renders Jinja2 prompts for system and user messages
"""
import os
from fastapi import FastAPI
import gradio as gr
from openai import OpenAI

# Config imports
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from config.templates import TemplateLoader
from config.pdf_processor import EnhancedPDFProcessor
from config.chain import RAGChain


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

# Instantiate the RAG chain (runnable-based: RunnableParallel | prompt | model)
rag_chain = RAGChain(pdf_processor=pdf_processor, template_loader=template_loader, client=client, top_k=4)

# ================================
# Helper Functions
# ================================
def format_history(history):
    """Format conversation history into a readable string.
    
    Args:
        history: List of dicts with 'question' and 'answer' keys
        
    Returns:
        Formatted history string for inclusion in prompts
    """
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    """Join retrieved documents into a single context string (not currently used).
    
    Args:
        retrieved_docs: List of Document objects with page_content
        history: Conversation history (unused in this function)
        
    Returns:
        Concatenated document texts separated by newlines
    """
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ================================
# Conversation + Chain Logic
# ================================

def load_pdf(file_path):
    """Load a PDF file, process it into chunks, and update the RAG chain.
    
    This function:
    1. Accepts a PDF file path from Gradio file upload
    2. Passes it to rag_chain.load_pdf() which calls EnhancedPDFProcessor
    3. Clears conversation history (starts fresh with new PDF)
    4. Returns a status message with chunk/page statistics
    
    Args:
        file_path: Path to uploaded PDF file
        
    Returns:
        Status message with success info or error description
    """
    if file_path is None:
        return "Please upload a PDF first."

    try:
        summary = rag_chain.load_pdf(file_path)
        return (
            f"PDF loaded successfully!\n"
            f"Chunks: {summary['total_chunks']} | "
            f"Pages: {summary['pages_extracted']} | "
            f"Avg chunk: {int(summary['avg_chunk_size'])} chars"
        )
    except ValueError as e:
        return f"Error loading PDF: {e}"

def ask_question(question):
    """Process a user question through the RAG chain and return an answer.
    
    This function:
    1. Validates that a PDF is loaded and API key is configured
    2. Invokes rag_chain.answer(question) which runs: RunnableParallel | prompt | model
    3. Returns the LLM-generated answer grounded in the PDF context
    
    Args:
        question: User's question about the PDF
        
    Returns:
        LLM-generated answer or error message
    """
    if client is None:
        return "Set OPENAI_API_KEY or NVIDIA_API_KEY in your .env file first."

    if not rag_chain.chunks:
        return "Please upload a PDF first."

    if not question.strip():
        return "Please ask a question about the PDF."

    try:
        answer = rag_chain.answer(question)
    except ValueError as e:
        return str(e)
    except RuntimeError as e:
        return str(e)

    return answer

def ask_question_and_update(chat_history, question):
    """Handle chat UI updates: show user question immediately, then stream the answer.
    
    This generator function:
    1. Appends the user's question to chat history
    2. Yields to update UI immediately (responsive UX)
    3. Gets the answer via ask_question()
    4. Appends the answer and yields final update
    
    Args:
        chat_history: List of chat messages from Gradio Chatbot
        question: User's input question
        
    Yields:
        Tuples of (updated_chat_history, cleared_query_field)
    """
    if not question.strip():
        return chat_history, ""

    # First, show the user's question immediately.
    chat_history.append({"role": "user", "content": question})
    yield chat_history, ""  # This updates the chatbox immediately

    # Then process the answer via the chain.
    answer = ask_question(question)
    chat_history.append({"role": "assistant", "content": answer})

    # Update chat with the assistant's response.
    yield chat_history, ""



# ================================
# Gradio UI
# ================================
with gr.Blocks() as demo:
    gr.Markdown("## PDF Insight RAG - Chat with your PDF")
    
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
