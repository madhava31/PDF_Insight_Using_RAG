import threading
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
import sys

# Load environment variables (assuming you have a .env file locally, or secrets in Render)
load_dotenv()
# -----------------------------------------
# Load environment variables
api_key = os.getenv("HUGGINGFACE_TOKEN_API")
if not api_key:
    print("❌ HUGGINGFACE_TOKEN_API not found. The application will not be able to query the LLM.")


# -----------------------------------------
# STEP 1: Reusable objects 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load Embeddings Robustly
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    # Do not exit the process, but log the error.
    print(f"FATAL EMBEDDINGS ERROR: Could not load HuggingFaceEmbeddings. Error: {e}", file=sys.stderr)


# -----------------------------------------
# STEP 2: Runnables for preprocessing
load_pdf_runnable = RunnableLambda(lambda pdf_path: PyPDFLoader(pdf_path).load())
split_runnable = RunnableLambda(lambda docs: splitter.split_documents(docs))
# vectorstore_runnable will rely on the global 'embeddings' variable
vectorstore_runnable = RunnableLambda(lambda docs: FAISS.from_documents(docs, embeddings))

# -----------------------------------------
# Prompt
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

def format_history(history):
    """Formats the conversation history for the LLM prompt."""
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

def format_docs_with_history(retrieved_docs, history):
    """Formats retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# -----------------------------------------
# Load LLM Robustly
# -----------------------------------------
model = None
parser = None
llm_ready = False
try:
    if api_key:
        llm = HuggingFaceEndpoint(
            repo_id="google/gemma-2-2b-it",
            task="conversational", 
            model_kwargs={"api_key": api_key} 
        )
        model = ChatHuggingFace(llm=llm)
        parser = StrOutputParser()
        llm_ready = True
    else:
        print("LLM initialization skipped due to missing API key.", file=sys.stderr)
except Exception as e:
    # Do not exit the process, but log the error. This allows Gradio to open the port.
    print(f"FATAL LLM INITIALIZATION ERROR: The HuggingFaceEndpoint could not be initialized. Error: {e}", file=sys.stderr)


# -----------------------------------------
# Globals
conversation_history = []     # for retrieval memory
retriever = None
main_chain = None

# -----------------------------------------
# Function to load and preprocess PDF
def load_pdf(file):
    global retriever, main_chain, conversation_history
    
    # 1. Reset state
    conversation_history = []  # reset on new PDF
    
    if file is None:
        return "⚠️ Please upload a PDF first.", []
    
    # Critical check for embeddings before processing
    if embeddings is None:
        return "❌ The document processing component failed to load at startup. Check console logs for details.", []

    pdf_path = file.name
    
    # 2. Build FAISS index
    try:
        docs = load_pdf_runnable.invoke(pdf_path)
        split_docs = split_runnable.invoke(docs)
        # Pass embeddings implicitly via the global vectorstore_runnable
        vectorstore = vectorstore_runnable.invoke(split_docs)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 3. Define the RAG chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
            "history": RunnableLambda(lambda _: format_history(conversation_history)),
            "question": RunnablePassthrough()
        })

        # Check if LLM loaded before building the full chain
        if model is None:
             return f"⚠️ PDF loaded successfully, but the LLM is not available due to a startup error. Cannot answer questions.", []
             
        main_chain = parallel_chain | prompt | model | parser
        return f"✅ PDF loaded successfully: {file.name}. You can now ask questions!", []
        
    except Exception as e:
        return f"❌ Error processing PDF: {e}", []


# -----------------------------------------
# Function to handle user query for the RAG model
def ask_question(question):
    global conversation_history, main_chain
    
    # Critical check for LLM
    if model is None:
        return "❌ RAG Model failed to initialize at startup. Please check the HUGGINGFACE_TOKEN_API environment variable in your deployment environment."

    if main_chain is None:
        return "⚠️ Please upload and load a PDF first."

    # Invoke the chain
    response = main_chain.invoke(question).strip()
    
    # Store history for conversational context
    conversation_history.append({"question": question, "answer": response})
    return response

# -----------------------------------------
# NEW: Function to handle chatbot update in messages format
def ask_question_and_update(chat_history, question):
    """Handles the query, gets the answer, and updates the Gradio Chatbot history."""
    if not question.strip():
        return chat_history, ""
    
    # Call the main RAG logic
    answer = ask_question(question)
    
    # Append messages in the new format for Gradio Chatbot
    chat_history.append([question, answer])

    return chat_history, "" # return updated chat + clear textbox

# -----------------------------------------
# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray
)) as demo:
    gr.Markdown("# 📚 PDF Insight RAG Model")
    gr.Markdown("Upload a PDF document to start a conversation with its content using the Gemma 2B model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="1. Upload PDF", file_types=[".pdf"])
            load_btn = gr.Button("Load PDF and Start Chat")

        with gr.Column(scale=3):
            # The Chatbot will display the conversation history
            chatbot = gr.Chatbot(label="Chat with PDF", height=400, layout="panel", show_copy_button=True)
            
            # Status box to show loading result
            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                max_lines=5,
                value="Upload a PDF and click 'Load PDF and Start Chat'."
            )
            
            # Input query box
            with gr.Row():
                query = gr.Textbox(label="2. Ask a question", placeholder="e.g., What are the key findings mentioned in the document?", scale=4)
                submit_btn = gr.Button("Ask", scale=1, variant="primary")

    # Connect components
    load_btn.click(
        load_pdf, 
        inputs=pdf_file, 
        outputs=[status, chatbot]
    )

    submit_event = submit_btn.click(
        fn=ask_question_and_update,
        inputs=[chatbot, query],
        outputs=[chatbot, query]
    )
    
    # Allow sending with Enter key
    query.submit(
        fn=ask_question_and_update,
        inputs=[chatbot, query],
        outputs=[chatbot, query]
    )

# -----------------------------------------
# LAUNCH GRADIO APP IN THE MAIN THREAD
# This is the critical change for Render deployment
if __name__ == "__main__":
    # Get the port from the environment variable (Render sets this)
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Gradio server on 0.0.0.0:{port}")
    # Launch Gradio in the main thread to allow Render to detect the open port
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
