import json
import logging
from typing import Dict, Any, TypedDict
from enum import Enum
from typing import Union, Sequence
import mcp.types as types
from mcp.server import Server
from .pinecone import PineconeClient, PineconeRecord
from .utils import MCPToolError
from .chunking import create_chunker, Chunk


logger = logging.getLogger("pinecone-mcp")


class ToolName(str, Enum):
    SEMANTIC_SEARCH = "semantic-search"
    READ_DOCUMENT = "read-document"
    PROCESS_DOCUMENT = "process-document"
    LIST_DOCUMENTS = "list-documents"
    PINECONE_STATS = "pinecone-stats"


ServerTools = [
    types.Tool(
        name=ToolName.SEMANTIC_SEARCH,
        description="Search pinecone for documents",
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
        name=ToolName.READ_DOCUMENT,
        description="Read a document from pinecone",
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
        name=ToolName.PROCESS_DOCUMENT,
        description="Process a document. This will optionally chunk, then embed, and upsert the document into pinecone.",
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
            },
            "required": ["document_id", "text", "metadata"],
        },
    ),
    types.Tool(
        name=ToolName.LIST_DOCUMENTS,
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
        name=ToolName.PINECONE_STATS,
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
            if name == ToolName.SEMANTIC_SEARCH:
                return semantic_search(arguments, pinecone_client)
            if name == ToolName.PINECONE_STATS:
                return pinecone_stats(pinecone_client)
            if name == ToolName.READ_DOCUMENT:
                return read_document(arguments, pinecone_client)
            if name == ToolName.PROCESS_DOCUMENT:
                return process_document(arguments, pinecone_client)
            if name == ToolName.LIST_DOCUMENTS:
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


def semantic_search(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Read a document from the pinecone knowledge base
    """
    query = arguments.get("query")
    top_k = arguments.get("top_k", 10)
    filters = arguments.get("filters", {})
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
    Process a document by chunking, embedding, and upserting it into the knowledge base. Returns the document ID.
    """
    document_id = arguments.get("document_id")
    text = arguments.get("text")
    namespace = arguments.get("namespace")
    metadata = arguments.get("metadata", {})

    chunker = create_chunker(chunk_type="smart")
    chunks = chunker.chunk_document(document_id, text, metadata)

    embed_result = embed_document(chunks, pinecone_client)

    embedded_chunks = embed_result.get("embedded_chunks", None)

    if embedded_chunks is None:
        raise MCPToolError("No embedded chunks found")

    upsert_documents(embedded_chunks, pinecone_client, namespace)

    return [
        types.TextContent(
            type="text",
            text=f"Successfully processed document. The document ID is {document_id}",
        )
    ]


class EmbeddingResult(TypedDict):
    embedded_chunks: list[PineconeRecord]
    total_embedded: int


def embed_document(
    chunks: list[Chunk], pinecone_client: PineconeClient
) -> EmbeddingResult:
    """
    Embed a list of chunks.
    Uses the Pinecone client to generate embeddings with the inference API.
    """
    embedded_chunks = []
    for chunk in chunks:
        content = chunk.content
        chunk_id = chunk.id
        metadata = chunk.metadata

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


def read_document(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Read a single Pinecone document by ID
    """
    document_id = arguments.get("document_id")
    namespace = arguments.get("namespace")
    if not document_id:
        raise ValueError("document_id is required")

    # Fetch the record using your existing fetch_records method
    record = pinecone_client.fetch_records([document_id], namespace=namespace)

    # Get the vector data for this document
    vector = record.vectors.get(document_id)
    if not vector:
        raise ValueError(f"Document {document_id} not found")

    # Get metadata from the vector
    metadata = vector.metadata if hasattr(vector, "metadata") else {}

    # Format the document content
    formatted_content = []
    formatted_content.append(f"Document ID: {document_id}")
    formatted_content.append("")  # Empty line for spacing

    if metadata:
        formatted_content.append("Metadata:")
        for key, value in metadata.items():
            formatted_content.append(f"{key}: {value}")

    return [types.TextContent(type="text", text="\n".join(formatted_content))]


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
