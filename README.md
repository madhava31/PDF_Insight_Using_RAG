# PDF_Insight_Using_RAG
Project Overview

PDF Insight RAG is an AI-powered application that allows users to upload a PDF and ask questions about its content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant text from the PDF and generate accurate answers using a Large Language Model.

This ensures answers are grounded in the document instead of hallucinating information.

#🚀 Features

Upload any PDF document

Ask questions about the document

AI retrieves relevant context from the PDF

Answers are generated using an LLM

Maintains conversation history

Clean chat interface

Deployable using FastAPI + Gradio

#🧠 Technologies Used
Backend

FastAPI – API framework

LangChain – RAG pipeline orchestration

AI / LLM

Google Gemma 2B Instruct

Accessed via HuggingFace Endpoint

Embeddings

sentence-transformers/all-MiniLM-L6-v2

Vector Database

FAISS – used to store and retrieve document embeddings

Document Processing

PyPDFLoader – loads PDF files

RecursiveCharacterTextSplitter – splits large documents into smaller chunks

Frontend

Gradio – interactive chat UI

Deployment

Render

#🏗️ Project Architecture
User Uploads PDF
        │
        ▼
PDF Loader (PyPDFLoader)
        │
        ▼
Text Splitter
(RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings
(HuggingFace Sentence Transformers)
        │
        ▼
Vector Database
(FAISS)
        │
        ▼
Retriever
        │
        ▼
Prompt Template
        │
        ▼
LLM (Gemma 2B via HuggingFace)
        │
        ▼
Generated Answer
⚙️ How the System Works
Step 1 — Upload PDF

User uploads a PDF using the Gradio interface.

Step 2 — Document Processing

The system:

Loads the PDF

Splits it into smaller chunks

chunk_size = 1000
chunk_overlap = 200
Step 3 — Create Embeddings

Each chunk is converted into vector embeddings using:

sentence-transformers/all-MiniLM-L6-v2
Step 4 — Store in FAISS

All embeddings are stored in FAISS vector database.

This allows fast semantic search.

Step 5 — Retrieval

When the user asks a question:

The question is converted into an embedding

FAISS retrieves top 4 relevant chunks

k = 4
Step 6 — Prompt Construction

The model receives:

Retrieved context

Conversation history

User question

Prompt Example:

You are a helpful assistant.
Answer ONLY from the provided transcript context and conversation history.

Conversation History:
{history}

Transcript Context:
{context}

Question:
{question}
Step 7 — Answer Generation

The retrieved context is passed to Gemma 2B Instruct model, which generates the final answer.

This allows the model to understand previous interactions.

📂 Project Structure
PDF_Insight_RAG
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
└── modules
      ├── rag.py
      ├── embeddings.py
      └── utils.py
🔑 Environment Variables

Create a .env file:

HUGGINGFACE_TOKEN_API=your_huggingface_token
🖥️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/pdf-insight-rag.git
cd pdf-insight-rag
2️⃣ Create Virtual Environment
python -m venv venv

Activate:

venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python app.py

or

uvicorn app:app --reload

#📊 Example Use Cases

Research paper analysis

Legal document search

Study material Q&A

Business reports analysis

Policy document understanding

⚠️ Limitations

Works best with text-based PDFs

Very large PDFs may increase processing time

Answers depend on retrieved context quality


https://github.com/madhava31
