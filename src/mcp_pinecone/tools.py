import json
import logging
from typing import Dict, Any, TypedDict
from typing import Union, Sequence
import mcp.types as types
from mcp.server import Server
from .pinecone import PineconeClient, PineconeRecord
from .utils import MCPToolError
from .chunking import MarkdownChunker, ChunkingResponse

logger = logging.getLogger("pinecone-mcp")

ServerTools = [
    types.Tool(
        name="semantic-search",
        description="Search pinecone knowledge base",
        category="search",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to search in",
                },
                "category": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"},
                    },
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="read-document",
        description="Read a document from the pinecone knowledge base",
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to read from",
                },
            },
            "required": ["document_id"],
        },
    ),
    types.Tool(
        name="process-document",
        description="Process a document by optionally chunking, embedding, and upserting it into the knowledge base. Returns the document ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "text": {"type": "string"},
                "metadata": {"type": "object"},
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to store the document in",
                },
                "chunk_enabled": {
                    "type": "boolean",
                    "description": "Whether to chunk the document (default: false)",
                    "default": False,
                },
            },
            "required": ["document_id", "text", "metadata"],
        },
    ),
    types.Tool(
        name="list-documents",
        description="List all documents in the knowledge base by namespace",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to list documents in",
                }
            },
            "required": ["namespace"],
        },
    ),
    types.Tool(
        name="pinecone-stats",
        description="Get stats about the Pinecone index specified in this server",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


def register_tools(server: Server, pinecone_client: PineconeClient):
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return ServerTools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> Sequence[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
        try:
            if name == "semantic-search":
                return list_documents(arguments, pinecone_client)
            if name == "pinecone-stats":
                return pinecone_stats(pinecone_client)
            if name == "read-document":
                return read_document(arguments, pinecone_client)
            if name == "process-document":
                return process_document(arguments, pinecone_client)
            if name == "list-documents":
                return list_documents(arguments, pinecone_client)

        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            raise


def list_documents(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    List all documents in the knowledge base by namespace
    """
    namespace = arguments.get("namespace")
    results = pinecone_client.list_records(namespace=namespace)
    return [types.TextContent(type="text", text=json.dumps(results))]


def pinecone_stats(pinecone_client: PineconeClient) -> list[types.TextContent]:
    """
    Get stats about the Pinecone index specified in this server
    """
    stats = pinecone_client.stats()
    return [types.TextContent(type="text", text=json.dumps(stats))]


def read_document(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Read a document from the pinecone knowledge base
    """
    query = arguments.get("query")
    top_k = arguments.get("top_k", 10)
    filters = arguments.get("filters")
    namespace = arguments.get("namespace")

    results = pinecone_client.search_records(
        query=query,
        top_k=top_k,
        filter=filters,
        include_metadata=True,
        namespace=namespace,
    )

    matches = results.get("matches", [])

    # Format results with rich context
    formatted_text = "Retrieved Contexts:\n\n"
    for i, match in enumerate(matches, 1):
        metadata = match.get("metadata", {})
        formatted_text += f"Result {i} | Similarity: {match['score']:.3f} | Document ID: {match['id']}\n"
        formatted_text += f"{metadata.get('text', '').strip()}\n"
        formatted_text += "-" * 10 + "\n\n"

    return [types.TextContent(type="text", text=formatted_text)]


def process_document(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Process a document by optionally chunking, embedding, and upserting it into the knowledge base. Returns the document ID.
    """
    document_id = arguments.get("document_id")
    text = arguments.get("text")
    chunk_enabled = arguments.get("chunk_enabled", False)
    namespace = arguments.get("namespace")
    metadata = arguments.get("metadata", {})

    if chunk_enabled:
        chunks_result = chunk_document(document_id, text, "markdown", metadata)
        chunks_data = chunks_result.to_dict()

        embed_result = embed_document(chunks_data["chunks"], pinecone_client)

        embedded_chunks = embed_result.get("embedded_chunks", None)

        if embedded_chunks is None:
            raise MCPToolError("No embedded chunks found")

        upsert_documents(embedded_chunks, pinecone_client)
    else:
        # Process the document as a single piece
        embedding = pinecone_client.generate_embeddings(text)
        record = PineconeRecord(
            id=document_id,
            embedding=embedding,
            text=text,
            metadata=metadata,
        )

        pinecone_client.upsert_records([record], namespace=namespace)

    return [
        types.TextContent(
            type="text",
            text=f"Successfully processed document {'with' if chunk_enabled else 'without'} chunking. The document ID is {document_id}",
        )
    ]


def chunk_document(
    document_id: str, text: str, chunk_type: str, metadata: dict
) -> ChunkingResponse:
    """
    Chunk a document into smaller chunks.
    Default chunk type is markdown.
    """
    chunker = MarkdownChunker()

    chunks = chunker.chunk_document(
        document_id=document_id, content=text, metadata=metadata
    )

    chunk_type = chunk_type or "markdown"

    response = ChunkingResponse(
        chunks=chunks,
        total_chunks=len(chunks),
        document_id=document_id,
        chunk_type=chunk_type,
    )

    # Return the chunks as a list of text content
    return response


class EmbeddingResult(TypedDict):
    embedded_chunks: list[PineconeRecord]
    total_embedded: int


def embed_document(
    chunks: list[dict[str, Any]], pinecone_client: PineconeClient
) -> EmbeddingResult:
    """
    Embed a list of chunks.
    Uses the Pinecone client to generate embeddings with the inference API.
    """
    embedded_chunks = []
    for chunk in chunks:
        content = chunk.get("content")
        chunk_id = chunk.get("id")
        metadata = chunk.get("metadata", {})

        if not content or not chunk_id:
            logger.warning(f"Skipping invalid chunk: {chunk}")
            continue

        embedding = pinecone_client.generate_embeddings(content)
        record = PineconeRecord(
            id=chunk_id,
            embedding=embedding,
            text=content,
            metadata=metadata,
        )
        embedded_chunks.append(record)
    return EmbeddingResult(
        embedded_chunks=embedded_chunks,
        total_embedded=len(embedded_chunks),
    )


def upsert_documents(
    records: list[PineconeRecord],
    pinecone_client: PineconeClient,
    namespace: str | None = None,
) -> Dict[str, Any]:
    """
    Upsert a list of Pinecone records into the knowledge base.
    """
    result = pinecone_client.upsert_records(records, namespace=namespace)
    return result


__all__ = [
    "register_tools",
]
