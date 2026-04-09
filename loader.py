"""
loader.py — Document ingestion and Markdown chunking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode

from config import DOC_CACHE_FILE

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_document(source_path: str, force_reload: bool = False) -> str:
    """
    Convert a PDF to Markdown using Docling.
    Result is cached to DOC_CACHE_FILE to avoid reprocessing.

    Args:
        source_path:  Absolute path to the source PDF.
        force_reload: If True, ignore the cache and re-convert.

    Returns:
        The document content as a Markdown string.
    """
    cache = Path(DOC_CACHE_FILE)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force_reload:
        logger.info(f"Cache hit — loading from {cache}")
        return cache.read_text(encoding="utf-8")

    logger.info(f"Converting PDF: {source_path}")
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    markdown_text = converter.convert(source_path).document.export_to_markdown()

    cache.write_text(markdown_text, encoding="utf-8")
    logger.info(f"Saved to cache: {cache}")
    return markdown_text


def chunk_markdown(markdown_text: str, node_slice: tuple[int, int] | None = None) -> List[BaseNode]:
    """
    Split Markdown text into section-level nodes.

    Args:
        markdown_text: Full document as a Markdown string.
        node_slice:    Optional (start, end) tuple to process a subset of nodes
                       (useful for testing on limited hardware).

    Returns:
        List of LlamaIndex BaseNode objects.
    """
    doc = Document(text=markdown_text)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([doc])

    if node_slice is not None:
        start, end = node_slice
        nodes = nodes[start:end]
        logger.info(f"Sliced nodes [{start}:{end}] → {len(nodes)} nodes")
    else:
        logger.info(f"Total nodes: {len(nodes)}")

    return nodes
