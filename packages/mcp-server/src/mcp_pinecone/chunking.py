"""
Smart document chunking with token awareness and recursive splitting.
Provides configurable text splitting strategies optimized for LLM context windows.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
import tiktoken
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("smart_chunker")


class ChunkingError(Exception):
    """Base exception for chunking errors"""

    pass


class Chunk(BaseModel):
    """Represents a document chunk with metadata"""

    id: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary format for embed-document"""
        return {"id": self.id, "content": self.content, "metadata": self.metadata}


class ChunkingConfig(BaseModel):
    """Configuration for chunking behavior"""

    target_tokens: int = Field(
        default=512,
        description="Target chunk size in tokens",
        gt=0,  # Must be positive
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum allowed tokens per chunk",
        gt=0,
    )
    overlap_tokens: int = Field(
        default=50,
        description="Number of tokens to overlap",
        ge=0,
    )
    tokenizer_model: str = Field(
        default="cl100k_base", description="Tokenizer model to use"
    )

    # Separators in priority order
    separators: List[str] = Field(
        default=[
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            "? ",  # Questions
            "! ",  # Exclamations
            ", ",  # Clauses
            " ",  # Words
            "",  # Characters
        ],
        description="Separators in order of preference",
    )

    @model_validator(mode="after")
    def validate_tokens(self):
        """Ensure overlap tokens are less than target tokens"""
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError(
                "max_tokens must be greater than or equal to target_tokens"
            )
        return self


class BaseChunker(ABC):
    """
    Abstract base for all chunking strategies.
    We can add more chunking strategies here as we learn more approaches for certain document types.
    """

    @abstractmethod
    def chunk_document(
        self, document_id: str, content: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        pass


class SmartChunker(BaseChunker):
    """
    Intelligent chunking implementation that combines:
    - Token awareness
    - Recursive splitting
    - Smart overlap handling
    - Configurable behavior
    This is inspired by approaches highlighted in https://js.langchain.com/docs/concepts/text_splitters/
    In order to keep dependencies minimal, we're not using LangChain here.
    Just taking inspiration from their approaches.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.tokenizer = tiktoken.get_encoding(self.config.tokenizer_model)

    def count_tokens(self, text: str) -> int:
        """
        Get exact token count for text
        """
        return len(self.tokenizer.encode(text))

    def create_chunk(
        self,
        document_id: str,
        content: str,
        chunk_number: int,
        total_chunks: int,
        base_metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a chunk with complete metadata"""
        token_count = self.count_tokens(content)

        metadata = {
            "document_id": document_id,
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "token_count": token_count,
            "char_count": len(content),
            "chunk_type": "smart",
            **base_metadata,
        }

        return Chunk(
            id=f"{document_id}#chunk{chunk_number}",
            content=content.strip(),
            metadata=metadata,
        )

    def chunk_document(
        self, document_id: str, content: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk document with intelligent boundary detection and token awareness
        This works by recursively splitting the document into chunks with overlap
        and then trying to find the best boundaries using progressively smaller separators
        """
        if not content or not content.strip():
            raise ChunkingError("Cannot chunk empty content")
        if not document_id:
            raise ChunkingError("Document ID is required")
        try:
            # Get initial splits
            chunks = self._split_with_overlap(
                content,
                self.config.separators,
                self.config.target_tokens,
                self.config.overlap_tokens,
            )

            # Convert to chunk objects with metadata
            processed_chunks = []
            for i, text in enumerate(chunks, 1):
                chunk = self.create_chunk(
                    document_id=document_id,
                    content=text,
                    chunk_number=i,
                    total_chunks=len(chunks),
                    base_metadata=metadata,
                )
                processed_chunks.append(chunk)

            # Log stats
            total_tokens = sum(c.metadata["token_count"] for c in processed_chunks)
            avg_tokens = total_tokens / len(processed_chunks)
            logger.info(
                f"Split document {document_id} into {len(processed_chunks)} chunks. "
                f"Average tokens per chunk: {avg_tokens:.0f}"
            )

            return processed_chunks

        except Exception as e:
            raise ChunkingError(f"Error chunking document {document_id}: {e}")

    def _split_with_overlap(
        self, text: str, separators: List[str], target_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """
        Split text recursively while handling overlap

        Args:
            text: The text to split
            separators: List of separators to try, in order of preference
            target_tokens: Target number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks

        Returns:
            List of text chunks with overlap

        Raises:
            ChunkingError: If text cannot be split into chunks
        """

        # Base case - text is small enough
        text_tokens = self.count_tokens(text)
        if text_tokens <= target_tokens:
            return [text]

        # Try each separator in order
        for separator in separators:
            splits = text.split(separator)

            # Skip if separator doesn't help
            if len(splits) == 1:
                continue

            # Process splits with overlap
            chunks = []
            current_chunk = []
            current_tokens = 0

            for split in splits:
                split_tokens = self.count_tokens(split)

                # Check if adding split would exceed target
                if current_tokens + split_tokens > target_tokens and current_chunk:
                    # Add current chunk
                    chunks.append(separator.join(current_chunk))

                    # Start new chunk with overlap
                    overlap_tokens_remaining = overlap_tokens
                    current_chunk = []

                    # Add previous splits until we hit overlap target
                    prev_splits = current_chunk.copy()
                    current_chunk = []
                    for prev_split in reversed(prev_splits):
                        prev_tokens = self.count_tokens(prev_split)
                        if overlap_tokens_remaining - prev_tokens < 0:
                            break
                        current_chunk.insert(0, prev_split)
                        overlap_tokens_remaining -= prev_tokens

                    current_tokens = self.count_tokens(separator.join(current_chunk))

                current_chunk.append(split)
                current_tokens += split_tokens

            # Add final chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))

            # If we found valid splits, return them
            if chunks:
                return chunks

        # If no good splits found, fall back to token boundary
        return self._split_by_tokens(text, target_tokens, overlap_tokens)

    def _split_by_tokens(
        self, text: str, target_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """
        Split on token boundaries as a last resort
        This is a simple approach that splits the document into chunks of the target size
        with an overlap of the overlap size.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), target_tokens - overlap_tokens):
            chunk_tokens = tokens[i : i + target_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks


# Factory for creating chunkers
def create_chunker(
    chunk_type: str = "smart", config: Optional[ChunkingConfig] = None
) -> BaseChunker:
    """Create appropriate chunker based on type"""
    chunkers = {"smart": lambda: SmartChunker(config)}

    if chunk_type not in chunkers:
        raise ValueError(f"Unknown chunker type: {chunk_type}")

    return chunkers[chunk_type]()


__all__ = [
    "Chunk",
    "ChunkingConfig",
    "BaseChunker",
    "SmartChunker",
    "create_chunker",
]
