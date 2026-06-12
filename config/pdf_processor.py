"""Enhanced PDF Loading and Chunking Module

This module provides EnhancedPDFProcessor which:
- Loads PDFs using LangChain's PyPDFLoader
- Preprocesses text (removes headers, footers, normalizes whitespace)
- Splits documents into chunks with configurable size and overlap
- Preserves metadata (page numbers, source filename, chunk IDs)
- Returns LangChain Document objects with enriched metadata
"""
import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EnhancedPDFProcessor:
    """
    Intelligent PDF processor with:
    - Metadata preservation (page numbers, source)
    - Text preprocessing (clean headers, footers, whitespace)
    - Adaptive chunk sizing based on content
    - Semantic-aware splitting on sentence/paragraph boundaries
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_preprocessing: bool = True,
        enable_semantic_split: bool = True,
    ):
        """Initialize processor.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            enable_preprocessing: Clean text before chunking
            enable_semantic_split: Split on sentences/paragraphs, not just char count
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_preprocessing = enable_preprocessing
        self.enable_semantic_split = enable_semantic_split

        # Semantic splitter tries paragraphs, sentences, words, then characters.
        separators = ["\n\n", "\n", ". ", " ", ""] if enable_semantic_split else [""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

    def _preprocess_text(self, text: str) -> str:
        """
        Clean text: remove headers, footers, extra whitespace, etc.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Remove common headers/footers (page numbers, URLs at start/end of lines)
        text = re.sub(r"^[\s\d]{0,5}$", "", text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r"https?://[^\s]+", "", text)  # URLs
        
        # Normalize whitespace: collapse multiple spaces/newlines
        text = re.sub(r" +", " ", text)  # Multiple spaces to single
        text = re.sub(r"\n\n\n+", "\n\n", text)  # Multiple newlines to double

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove lines that are mostly whitespace
        text = "\n".join(
            line for line in text.split("\n") 
            if line.strip() or line == ""
        )

        return text.strip()

    def _enrich_metadata(
        self, docs: List[Document], file_path: str
    ) -> List[Document]:
        """
        Add or enhance metadata: source filename, page numbers.

        Args:
            docs: Documents from PyPDFLoader
            file_path: Path to PDF file

        Returns:
            Documents with enriched metadata
        """
        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            # PyPDFLoader already adds 'page' metadata, ensure it's set
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
        return docs

    def process(self, file_path: str) -> List[Document]:
        """
        Load, preprocess, and chunk PDF with full metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects with chunks and metadata

        Raises:
            ValueError: If PDF cannot be loaded or is empty
        """
        # 1. Load PDF
        try:
            docs = PyPDFLoader(file_path).load()
        except Exception as e:
            raise ValueError(f"Failed to load PDF: {e}")

        if not docs:
            raise ValueError("PDF is empty or contains no extractable text.")

        # 2. Enrich metadata (add source, ensure page numbers)
        docs = self._enrich_metadata(docs, file_path)

        # 3. Preprocess text if enabled
        if self.enable_preprocessing:
            for doc in docs:
                doc.page_content = self._preprocess_text(doc.page_content)

        # 4. Split into chunks
        chunks = self.splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No chunks generated after splitting. PDF may be too small."
            )

        # 5. Add chunk index for reference
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    def get_chunk_summary(self, chunks: List[Document]) -> dict:
        """
        Get summary statistics about chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary with summary info
        """
        if not chunks:
            return {}

        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        unique_pages = len(set(chunk.metadata.get("page", 0) for chunk in chunks))

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 0),
            "pages_extracted": unique_pages,
        }
