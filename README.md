# PDF Insight Using RAG

PDF Insight is a web application for uploading a PDF and asking questions about
its contents. It extracts and cleans the document text, splits it into chunks,
selects relevant chunks for each question, and sends the grounded context to an
OpenAI-compatible language model.

## Features

- Upload and process text-based PDF files
- Preserve page and source metadata while chunking
- Ask follow-up questions through a Gradio chat interface
- Configure any OpenAI-compatible API endpoint and model
- Run locally with FastAPI/Uvicorn or deploy on Render

## Project Structure

```text
.
|-- app.py
|-- config/
|   |-- pdf_processor.py
|   |-- settings.py
|   `-- templates.py
|-- templates/
|   |-- system_prompt.jinja
|   `-- user_prompt.jinja
|-- requirements.txt
`-- render.yaml
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

1. `PyPDFLoader` extracts text from the uploaded PDF.
2. `EnhancedPDFProcessor` cleans the text and splits it into overlapping chunks.
3. The application scores chunks using terms from the user's question.
4. The four highest-scoring chunks are added to the prompt as context.
5. The configured language model generates an answer grounded in that context.

## Deployment

The included `render.yaml` installs dependencies from `requirements.txt` and
starts the application with `python app.py`. Add the API key and optional model
settings as environment variables in the Render dashboard.

## Notes

- Scanned PDFs require OCR before this application can extract their text.
- `.env`, virtual environments, caches, and editor settings are ignored by Git.
- Do not commit API keys or other credentials.
