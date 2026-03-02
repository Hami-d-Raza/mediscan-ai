"""
utils/file_handler.py
---------------------
Text extraction utilities for MediScan AI report analysis.

Supports:
  - Plain text files  (.txt  | MIME: text/plain)
  - PDF documents     (.pdf  | MIME: application/pdf)

Public API:
  extract_text(content: bytes, content_type: str, filename: str) -> str
"""

import io
import logging

logger = logging.getLogger(__name__)

# Supported MIME types
SUPPORTED_REPORT_TYPES = {"text/plain", "application/pdf"}


# ---------------------------------------------------------------------------
# Internal extractors
# ---------------------------------------------------------------------------

def _extract_from_txt(content: bytes) -> str:
    """Decode raw bytes as UTF-8 text, replacing un-decodable bytes."""
    return content.decode("utf-8", errors="replace").strip()


def _extract_from_pdf(content: bytes) -> str:
    """
    Extract all text from a PDF byte stream using PyPDF2.
    Returns concatenated text from every page, separated by newlines.
    Raises ValueError if the PDF cannot be parsed.
    """
    try:
        import PyPDF2
    except ImportError as exc:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Install it with: pip install PyPDF2"
        ) from exc

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
            else:
                logger.debug("Page %d yielded no text (may be image-only).", i + 1)

        extracted = "\n\n".join(pages_text).strip()

        if not extracted:
            raise ValueError("PDF contains no extractable text (possibly a scanned image PDF).")

        return extracted

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(content: bytes, content_type: str, filename: str = "") -> str:
    """
    Dispatch to the correct extractor based on MIME type.

    Args:
        content:      Raw file bytes.
        content_type: MIME type string (e.g. "application/pdf").
        filename:     Original filename — used only for logging.

    Returns:
        Extracted plain text string.

    Raises:
        ValueError:  If the content_type is unsupported or extraction fails.
    """
    logger.info("Extracting text from '%s' (type: %s, size: %d bytes)",
                filename, content_type, len(content))

    if content_type == "text/plain":
        return _extract_from_txt(content)

    if content_type == "application/pdf":
        return _extract_from_pdf(content)

    raise ValueError(
        f"Unsupported file type '{content_type}'. "
        f"Accepted types: {sorted(SUPPORTED_REPORT_TYPES)}"
    )
