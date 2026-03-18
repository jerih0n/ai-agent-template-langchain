"""
Agent middlewares.

Register hooks by passing them to `create_agent(middleware=[...])`.
Reference: https://docs.langchain.com/oss/python/langchain/middleware/custom
"""

from app.ai.middlewares.thread_summary import summarise_if_new

__all__ = ["summarise_if_new"]
