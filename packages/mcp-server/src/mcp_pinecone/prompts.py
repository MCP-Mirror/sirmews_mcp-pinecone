import logging
from enum import Enum
import mcp.types as types
from mcp.server import Server
from .pinecone import PineconeClient
from .tools import semantic_search


logger = logging.getLogger("pinecone-mcp")


class PromptName(str, Enum):
    PINECONE_SEMANTIC_SEARCH = "pinecone-search"


ServerPrompts = [
    types.Prompt(
        name=PromptName.PINECONE_SEMANTIC_SEARCH,
        description="Search Pinecone index and construct an answer based on relevant pinecone documents",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The query to search for",
                required=True,
            )
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
            if name == PromptName.PINECONE_SEMANTIC_SEARCH:
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

    logger.info(f"Searching Pinecone index for query: {arguments['query']}")
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


__all__ = [
    "register_prompts",
]
