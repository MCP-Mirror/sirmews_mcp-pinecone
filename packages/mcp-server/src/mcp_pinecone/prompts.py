import logging
from enum import Enum
import mcp.types as types
from mcp.server import Server
from .pinecone import PineconeClient
from datetime import datetime
from .tools import semantic_search


logger = logging.getLogger("pinecone-mcp")


class PromptName(str, Enum):
    PINECONE_QUERY = "pinecone-query"
    PINECONE_STORE = "pinecone-store"
    PINECONE_SEMANTIC_SEARCH = "pinecone-semantic-search"
    PINECONE_READ_DOCUMENT = "pinecone-read-document"
    PINECONE_PROCESS_DOCUMENT = "pinecone-process-document"
    PINECONE_LIST_DOCUMENTS = "pinecone-list-documents"
    PINECONE_STATS = "pinecone-stats"


ServerPrompts = [
    types.Prompt(
        name=PromptName.PINECONE_QUERY,
        description="Search Pinecone index and construct an answer based on relevant pinecone documents",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The question to answer, or the context to search for",
                required=True,
            )
        ],
    ),
    types.Prompt(
        name=PromptName.PINECONE_STORE,
        description="Store content as document in Pinecone",
        arguments=[
            types.PromptArgument(
                name="content",
                description="The content to store as a Pinecone document",
                required=True,
            ),
            types.PromptArgument(
                name="namespace",
                description="The namespace to store the document in",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name=PromptName.PINECONE_SEMANTIC_SEARCH,
        description="Search Pinecone index and construct an answer based on relevant pinecone documents",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The query to search for",
                required=True,
            ),
            types.PromptArgument(
                name="top_k",
                description="The number of documents to return",
                required=False,
            ),
            types.PromptArgument(
                name="namespace",
                description="The namespace to search in",
                required=False,
            ),
        ],
    ),
]


def register_prompts(server: Server, pinecone_client: PineconeClient):
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return ServerPrompts

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        try:
            if name == PromptName.PINECONE_QUERY:
                return pinecone_query(arguments, pinecone_client)
            elif name == PromptName.PINECONE_STORE:
                return pinecone_store(arguments, pinecone_client)
            elif name == PromptName.PINECONE_SEMANTIC_SEARCH:
                return pinecone_semantic_search(arguments, pinecone_client)
            else:
                raise ValueError(f"Unknown prompt: {name}")

        except Exception as e:
            logger.error(f"Error calling prompt {name}: {e}")
            raise


def pinecone_semantic_search(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Search Pinecone index and construct an answer based on relevant pinecone documents
    """
    results = semantic_search(arguments, pinecone_client)

    # loop through each result and add it to messages
    messages = []
    for result in results:
        messages.append(
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=result.text),
            )
        )

    return types.GetPromptResult(messages=messages)


def pinecone_store(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Store content as document in Pinecone
    """
    content = arguments.get("content")
    namespace = arguments.get("namespace")

    metadata = {
        "date": datetime.now().isoformat(),
    }

    if not content:
        raise ValueError("Content required")

    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"The namespace is {namespace if namespace else 'not specified'}. \n"
                    "If the namespace is not specified, use pinecone-stats to find an appropriate namespace or use the default namespace.",
                ),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Based on the content, generate metadata that can be relevant to the content and used for filtering. \n"
                    "The metadata should be a dictionary with keys and values that are relevant to the content. \n"
                    f"Append the metdata to {metadata} \n",
                ),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Run the process-document tool with the content: {content} \n"
                    "Do not truncate or use placeholders to describe an intent to add a document. Always add the raw content. \n"
                    "Include generated metadata in the document. \n"
                    f"Store in the {namespace} if specified",
                ),
            ),
        ]
    )


def pinecone_query(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Search Pinecone index and construct an answer based on relevant pinecone documents
    """
    query = arguments.get("query")
    if not query:
        raise ValueError("Query required")

    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="First use pinecone-stats to get a list of namespaces that might contain relevant documents. Ignore if a namespace is specified in the query",
                ),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Do a semantic search for the query: {query} with the chosen namespace",
                ),
            ),
        ]
    )


__all__ = [
    "register_prompts",
]
