from . import server
import asyncio


def main():
    asyncio.run(server.main())


# Optionally expose other important items at package level
__all__ = ["main", "server"]
