"""
loader.py — Document ingestion and Markdown chunking.
Processes PDFs page by page, falling back to pypdf for pages Docling can't handle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode

from config import DOC_CACHE_FILE

logger = logging.getLogger(__name__)


def _build_converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()
    opts.do_ocr                  = False
    opts.do_table_structure      = False
    opts.images_scale            = 0.5
    opts.generate_page_images    = False
    opts.generate_picture_images = False

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _page_count(source_path: str) -> int:
    import pypdf
    with open(source_path, "rb") as f:
        return len(pypdf.PdfReader(f).pages)


def _pypdf_fallback(source_path: str, page_number: int) -> str:
    """Direct text extraction for pages Docling can't handle. page_number is 0-indexed."""
    import pypdf
    with open(source_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        return (reader.pages[page_number].extract_text() or "").strip()


def load_document(source_path: str, force_reload: bool = False) -> str:
    cache = Path(DOC_CACHE_FILE)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force_reload:
        logger.info(f"Cache hit — loading from {cache}")
        return cache.read_text(encoding="utf-8")

    logger.info(f"Converting PDF: {source_path}")

    total  = _page_count(source_path)
    logger.info(f"Total pages: {total}")

    converter  = _build_converter()
    chunks     = []
    failed     = []

    for page_num in range(total):
        label = page_num + 1
        try:
            result = converter.convert(
                source_path,
                raises_on_error=True,
                page_range=(label, label),
            )
            md = result.document.export_to_markdown()
            if md.strip():
                chunks.append(md)
            logger.info(f"  ✓ Page {label}")

        except Exception as e:
            logger.warning(f"  ⚠ Page {label} Docling failed ({e}) — falling back to pypdf")
            failed.append(label)
            text = _pypdf_fallback(source_path, page_num)
            if text:
                chunks.append(f"## Page {label}\n\n{text}")
                logger.info(f"  ↩ Page {label} recovered via pypdf")
            else:
                logger.warning(f"  ✗ Page {label} empty after fallback — skipped")

    if failed:
        logger.warning(f"Pages that needed pypdf fallback: {failed}")

    markdown_text = "\n\n---\n\n".join(chunks)
    cache.write_text(markdown_text, encoding="utf-8")
    logger.info(f"Saved to cache: {cache} ({len(chunks)}/{total} pages)")
    return markdown_text


def chunk_markdown(markdown_text: str, node_slice: tuple[int, int] | None = None) -> List[BaseNode]:
    doc    = Document(text=markdown_text)
    parser = MarkdownNodeParser()
    nodes  = parser.get_nodes_from_documents([doc])

    if node_slice is not None:
        start, end = node_slice
        nodes = nodes[start:end]
        logger.info(f"Sliced nodes [{start}:{end}] → {len(nodes)} nodes")
    else:
        logger.info(f"Total nodes: {len(nodes)}")

    return nodes
