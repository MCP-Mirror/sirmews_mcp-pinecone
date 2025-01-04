# Index name
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def get_pinecone_config():
    parser = argparse.ArgumentParser(description="Pinecone MCP Configuration")
    parser.add_argument(
        "--index-name",
        default=None,
        help="Name of the Pinecone index to use. Will use environment variable PINECONE_INDEX_NAME if not provided.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Pinecone. Will use environment variable PINECONE_API_KEY if not provided.",
    )
    args = parser.parse_args()

    # Use command line arguments if provided, otherwise fall back to environment variables
    index_name = args.index_name or os.getenv("PINECONE_INDEX_NAME")
    api_key = args.api_key or os.getenv("PINECONE_API_KEY")

    # Set default index name if none provided
    if not index_name:
        index_name = "mcp-pinecone-index"
        print(f"No index name provided, using default: {index_name}")

    # Validate API key
    if not api_key:
        raise ValueError(
            "Pinecone API key is required. Provide it via --api-key argument or PINECONE_API_KEY environment variable"
        )

    return index_name, api_key


# Get configuration values
PINECONE_INDEX_NAME, PINECONE_API_KEY = get_pinecone_config()

# Validate configuration after loading
if not PINECONE_INDEX_NAME or not PINECONE_API_KEY:
    raise ValueError(
        "Missing required configuration. Ensure PINECONE_INDEX_NAME and PINECONE_API_KEY "
        "are set either via environment variables or command line arguments."
    )

# Inference API model name
INFERENCE_MODEL = "multilingual-e5-large"

# Inference API embedding dimension
INFERENCE_DIMENSION = 1024

# Export values for use in other modules
__all__ = [
    "PINECONE_INDEX_NAME",
    "PINECONE_API_KEY",
    "INFERENCE_MODEL",
    "INFERENCE_DIMENSION",
]
