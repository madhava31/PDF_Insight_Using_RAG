# PDF Insight Using RAG

PDF Insight is a web application for uploading a PDF and asking questions about
its contents. It extracts and cleans the document text, splits it into chunks,
selects relevant chunks for each question, and sends the grounded context to an
OpenAI-compatible language model.

## Features

- **PDF Processing**: Upload text-based PDFs and automatically extract, clean, and chunk content
- **Smart Retrieval**: Term-frequency scoring to find relevant PDF sections for each question
- **Conversation History**: Multi-turn interactions with conversation context preserved in prompts
- **Metadata Preservation**: Page numbers, source filenames, and chunk IDs tracked for each context segment
- **LangChain Integration**: Professional runnable-based RAG pattern with composable pipeline stages
- **Template-Driven Prompting**: Jinja2 templates for flexible, maintainable system and user prompts
- **OpenAI-Compatible API**: Works with any OpenAI-compatible provider (NVIDIA API, HuggingFace, local models, etc.)
- **Gradio Chat UI**: Interactive web interface for PDF upload and Q&A without coding
- **FastAPI Backend**: Production-ready HTTP server with proper logging and error handling
- **Easy Deployment**: Render platform configuration included for one-click deployment

## Project Structure

```text
.
|-- app.py                          # Main FastAPI app with Gradio UI
|-- config/
|   |-- __init__.py
|   |-- chain.py                    # RAGChain: runnable-based RAG pipeline
|   |-- pdf_processor.py            # EnhancedPDFProcessor: PDF loading & chunking
|   |-- settings.py                 # Configuration: environment variables
|   `-- templates.py                # TemplateLoader: Jinja2 template rendering
|-- templates/
|   |-- system_prompt.jinja         # System instruction constraining LLM to PDF context
|   `-- user_prompt.jinja           # User message template with history & context
|-- requirements.txt                # Python dependencies
|-- render.yaml                     # Render platform deployment config
`-- README.md                       # This file
```

## Requirements

- Python 3.10 or newer
- An API key for an OpenAI-compatible model provider

## Local Setup

1. Clone the repository and enter the project directory:

   ```powershell
   git clone https://github.com/madhava31/PDF_Insight_Using_RAG-2.git
   cd PDF_Insight_Using_RAG-2
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Create a `.env` file:

   ```env
   OPENAI_API_KEY=your_api_key
   OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
   LLM_MODEL=openai/gpt-oss-120b
   ```

   `NVIDIA_API_KEY` and `API_KEY` are also accepted as API key variable names.

5. Start the application:

   ```powershell
   python app.py
   ```

6. Open `http://localhost:7860/gradio`.

## How It Works

### RAG Pipeline Flow

1. **PDF Upload & Loading** (`EnhancedPDFProcessor.process()`)
   - `PyPDFLoader` extracts text and metadata from the uploaded PDF
   - Text preprocessing removes headers, footers, and normalizes whitespace
   - `RecursiveCharacterTextSplitter` chunks text into overlapping segments with semantic awareness (splits on paragraph/sentence boundaries when possible)
   - Metadata enrichment adds source filename, page numbers, and chunk IDs

2. **Question Retrieval** (`RAGChain._score_and_get_docs()`)
   - User question is scored against all PDF chunks using term-frequency matching
   - Non-trivial terms (length > 2) from the question are extracted
   - Each chunk receives a score based on term occurrence count
   - Top-k (default: 4) chunks with positive scores are retrieved

3. **Context Assembly & Prompting** (`RAGChain.answer()`)
   - Retrieved chunks are joined into a context string
   - Conversation history is formatted from previous Q/A pairs
   - Jinja2 templates render system and user prompts with the context, history, and question
   - System prompt constrains the LLM to answer only from the PDF

4. **LLM Generation & Response** (`call_model()`)
   - OpenAI-compatible API is called with the structured messages
   - LLM generates an answer grounded in the PDF context
   - Q/A pair is stored in conversation history for multi-turn interaction
   - Answer is returned to the user via Gradio UI

## Architecture: LangChain Runnable Pattern

### Core Components

**RAGChain** (`config/chain.py`)
- Main RAG orchestrator implementing the LangChain runnable pattern
- Manages PDF chunks, conversation history, and the retrieval-prompting-generation pipeline
- `load_pdf(file_path)`: Processes PDF and stores chunks with metadata
- `answer(question)`: Executes the full RAG pipeline and returns LLM response

**EnhancedPDFProcessor** (`config/pdf_processor.py`)
- Loads PDF files using PyPDFLoader
- Preprocesses text: removes headers/footers, normalizes whitespace
- Splits documents with semantic awareness (paragraph/sentence boundaries)
- Preserves page numbers, source filename, and chunk IDs in metadata

**TemplateLoader** (`config/templates.py`)
- Renders Jinja2 templates from the `templates/` directory
- System prompt: Constrains LLM to answer only from PDF context
- User prompt: Structures context, history, and question for the LLM

**FastAPI + Gradio** (`app.py`)
- FastAPI serves the HTTP server on port 7860
- Gradio chat interface mounted at `/gradio/` endpoint
- Handles file uploads, chat history display, and streaming responses

### Runnable Composition Pattern

The `RAGChain` class demonstrates professional LangChain composition using runnables:

```python
# The RAG pipeline uses runnable composition:
parallel = RunnableParallel({
    "context": RunnableLambda(lambda q: retrieve_docs(q)),
    "history": RunnableLambda(lambda _: format_history()),
    "question": RunnablePassthrough(),
})

prompt_runnable = RunnableLambda(render_jinja2_templates)
model_runnable = RunnableLambda(call_openai_api)

# Compose into a chain
main_chain = parallel | prompt_runnable | model_runnable

# Invoke with a question
answer = main_chain.invoke(question)
```

**Key Components:**
- **RunnableParallel**: Prepares context, history, and question simultaneously
- **RunnableLambda**: Wraps custom functions (template rendering, API calls) as runnables
- **RunnablePassthrough**: Passes the question through unchanged
- **Pipe operator `|`**: Chains components in a declarative, composable way

**Benefits:**
- Declarative pipeline definition matching LangChain conventions
- Easy to modify, test, and extend individual pipeline stages
- Built-in support for streaming and async execution
- Clear separation of concerns (retrieval → prompting → generation)

## Deployment

The included `render.yaml` installs dependencies from `requirements.txt` and
starts the application with `python app.py`. Add the API key and optional model
settings as environment variables in the Render dashboard.

## Usage Examples

### Via Gradio Web Interface

1. Open `http://localhost:7860/gradio` in your browser
2. Click "Upload a file" and select a PDF
3. Wait for the status message showing chunk count and processing details
4. Type your question in the text input
5. View the LLM response and ask follow-up questions

The chat history is displayed in the chatbot widget and automatically included in subsequent queries for context.

### Environment Configuration

The application reads from a `.env` file in the project root:

```env
# Required: API key for your LLM provider
OPENAI_API_KEY=your_api_key_here

# Optional: Custom API endpoint (default: NVIDIA endpoint)
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1

# Optional: Model name (default: NVIDIA Gemma 2)
LLM_MODEL=openai/gpt-oss-120b
```

Supported API key variable names: `OPENAI_API_KEY`, `NVIDIA_API_KEY`, `API_KEY`

## Troubleshooting

**PDF upload fails**
- Verify the PDF is text-based (not scanned/image-based). Scanned PDFs require OCR preprocessing.
- Check file size and try a smaller PDF first.

**No API key error**
- Ensure `.env` file exists in the project root with `OPENAI_API_KEY` set
- Restart the application after updating `.env`

**Connection refused on localhost:7860**
- Verify no other application is using port 7860
- Try `netstat -ano | findstr :7860` (Windows) or `lsof -i :7860` (macOS/Linux)

**Empty responses from LLM**
- Check your API key is valid and has quota remaining
- Verify API endpoint is accessible
- Ensure the PDF was successfully uploaded (check status message)

## Notes

- Scanned PDFs require OCR before this application can extract their text.
- `.env`, virtual environments, caches, and editor settings are ignored by Git.
- Do not commit API keys or other credentials.
