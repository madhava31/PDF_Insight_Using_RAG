"""Runnable-based RAG Chain Implementation

This module provides RAGChain, which implements a LangChain-style runnable pattern:
    
    chain = RunnableParallel({"context": ..., "history": ..., "question": ...}) 
         | PromptRunnable (renders Jinja templates)
         | ModelRunnable (calls OpenAI API)

The RAGChain class:
- Wraps the retrieval pipeline (scoring PDF chunks)
- Composes runnables for prompt templating and LLM calls
- Manages conversation history and state
- Provides simple load_pdf() and answer() APIs for Gradio integration
"""
from typing import List
from langchain_core.documents import Document

from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough


class RAGChain:
    """Runnable-based RAGChain that composes a RunnableParallel -> prompt -> model flow.

    This class preserves the `load_pdf(file_path)` and `answer(question)` API while
    internally using LangChain-style runnables so you can demonstrate the
    `parallel_chain | prompt | model` pattern.
    """

    def __init__(self, pdf_processor, template_loader, client=None, top_k: int = 4):
        self.pdf_processor = pdf_processor
        self.template_loader = template_loader
        self.client = client
        self.top_k = top_k

        self.chunks: List[Document] = []
        self.conversation_history: List[dict] = []

    def load_pdf(self, file_path: str) -> dict:
        """Load and chunk the PDF using the provided processor.
        
        This method:
        1. Resets conversation history
        2. Calls pdf_processor.process() to extract and chunk text
        3. Returns statistics about the chunks (total, pages, avg size)
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with keys: total_chunks, total_characters, avg_chunk_size, pages_extracted
        """
        self.conversation_history = []
        self.chunks = self.pdf_processor.process(file_path)
        return self.pdf_processor.get_chunk_summary(self.chunks)

    def _format_history(self) -> str:
        """Format conversation history into a string for inclusion in the prompt.
        
        Returns:
            Formatted string with Q/A pairs or "No prior conversation." if empty
        """
        if not self.conversation_history:
            return "No prior conversation."
        return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in self.conversation_history])

    def _score_and_get_docs(self, question: str) -> List[Document]:
        """Retrieve and rank relevant chunks using simple term-frequency scoring.
        
        Algorithm:
        1. Extract non-trivial terms from the question (len > 2)
        2. Score each chunk by counting term occurrences
        3. Sort by score and return top-k chunks
        
        Args:
            question: User's question
            
        Returns:
            List of top-k Document objects ranked by relevance
        """
        question_terms = [term.lower() for term in question.split() if len(term) > 2]
        scored = []
        for doc in self.chunks:
            text = doc.page_content.strip()
            lowered = text.lower()
            score = sum(1 for term in question_terms if term in lowered)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[: self.top_k] if score > 0]

    def answer(self, question: str) -> str:
        """Answer a question by composing a RunnableParallel -> prompt -> model flow.

        Pipeline:
        1. RunnableParallel retrieves context, history, and passes question
        2. PromptRunnable renders system and user Jinja2 templates
        3. ModelRunnable calls OpenAI API 
        4. Result is stored in conversation history
        
        The runnable composition demonstrates the LangChain pattern:
            main_chain = parallel_chain | prompt_runnable | model_runnable
            result = main_chain.invoke(question)
        
        Args:
            question: User's question about the PDF
            
        Returns:
            LLM-generated answer string grounded in PDF context
            
        Raises:
            ValueError: If no PDF is loaded
            RuntimeError: If no LLM client is configured
        """
        if not self.chunks:
            raise ValueError("No PDF loaded. Upload a PDF first.")

        if not question.strip():
            return ""

        # Build the parallel runnable: it produces context and history values
        parallel = RunnableParallel(
            {
                "context": RunnableLambda(lambda q: "\n\n".join([d.page_content for d in self._score_and_get_docs(q)])),
                "history": RunnableLambda(lambda _: self._format_history()),
                "question": RunnablePassthrough(),
            }
        )

        # Prompt runnable: render templates into a single user message
        def render_prompt(inputs: dict) -> dict:
            # inputs is a mapping with 'context','history','question'
            system = self.template_loader.render("system_prompt", {})
            user = self.template_loader.render(
                "user_prompt",
                {
                    "history_text": inputs.get("history", ""),
                    "context": inputs.get("context", ""),
                    "question": inputs.get("question", ""),
                },
            )
            return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

        prompt_runnable = RunnableLambda(render_prompt)

        # Model runnable: call the configured client (if present)
        def call_model(inputs: dict) -> str:
            """
            Call OpenAI API with rendered messages.
            
            Extracts message list from inputs dict and sends to LLM client.
            Returns LLM-generated text or error message.
            """
            messages = inputs.get("messages")
            if self.client is None:
                raise RuntimeError("No LLM client configured. Set OPENAI_API_KEY or similar.")
            try:
                completion = self.client.chat.completions.create(
                    messages=messages, model=self.client.default_model if hasattr(self.client, "default_model") else None
                )
                # Non-streaming fallback: join content if present
                if getattr(completion, "choices", None):
                    return "".join(getattr(c, "message", {}).get("content", "") or getattr(c, "text", "") for c in completion.choices).strip()
                return str(completion)
            except Exception as e:
                return f"I don't know. ({e})"

        model_runnable = RunnableLambda(call_model)

        # Stage 1: Compose the parallel retrieval pipeline
        # This creates inputs dict with 'context', 'history', 'question' keys
        main_chain = parallel | prompt_runnable | model_runnable

        # Stage 2: Invoke the full pipeline with the user's question
        # The question flows through: parallel retrieval → template rendering → LLM call
        result = main_chain.invoke(question)

        # Stage 3: Store the Q/A pair in conversation history for future context
        answer = result if isinstance(result, str) else str(result)
        self.conversation_history.append({"question": question, "answer": answer})
        return answer
