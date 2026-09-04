"""
pdf_extractor.py

PDF extraction utilities for the PDF -> AI -> PowerPoint pipeline.

Responsibilities:
    1. Extract page text
    2. Split pages into paragraphs
    3. Extract embedded images
    4. Preserve source/page metadata

Dependencies:
    pip install pypdf pymupdf
"""

from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# ---------------------------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_pages(
    pdf_path: str,
    include_empty_pages: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path:
            Path to the PDF.

        include_empty_pages:
            If True, include pages even if no text could be extracted.

    Returns:
        List of dictionaries:

        [
            {
                "page": 1,
                "text": "...",
                "char_count": 1234
            },
            ...
        ]
    """

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    reader = PdfReader(pdf_path)

    pages: List[Dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"Warning: Could not extract text from page "
                f"{page_number}: {exc}"
            )
            text = ""

        text = clean_page_text(text)

        if text or include_empty_pages:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "char_count": len(text)
                }
            )

    return pages


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_page_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts.

    This intentionally does NOT aggressively rewrite the text because
    preserving source content is important for downstream AI processing.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Fix words broken across lines:
    #
    # "transmis-\n sion"
    #
    # becomes:
    #
    # "transmission"
    #
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PARAGRAPH EXTRACTION
# ---------------------------------------------------------------------------

def split_paragraphs(
    text: str,
    min_chars: int = 40
) -> List[str]:
    """
    Split page text into paragraph-like blocks.

    PDFs frequently don't preserve paragraph boundaries cleanly.
    Therefore this function supports both:

        paragraph\n\nparagraph

    and PDFs where lines are separated only by single newlines.

    Args:
        text:
            Page text.

        min_chars:
            Ignore extremely short fragments.

    Returns:
        List of paragraph strings.
    """

    if not text:
        return []

    text = clean_page_text(text)

    # First attempt: explicit paragraph breaks.
    blocks = re.split(r"\n\s*\n", text)

    paragraphs: List[str] = []

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        # Normalize remaining line breaks inside the paragraph.
        #
        # This helps turn:
        #
        # "The physical layer is responsible
        # for transmitting bits..."
        #
        # into one paragraph.
        block = re.sub(r"\s*\n\s*", " ", block)

        block = re.sub(r"\s+", " ", block).strip()

        if len(block) >= min_chars:
            paragraphs.append(block)

    return paragraphs


def page_to_paragraphs(
    pages: List[Dict[str, Any]],
    min_chars: int = 40
) -> List[Dict[str, Any]]:
    """
    Convert page objects into paragraph objects.

    Each paragraph gets a stable paragraph ID and source page.

    Returns:

        [
            {
                "paragraph_id": 0,
                "page": 1,
                "text": "...",
                "char_count": 500
            },
            ...
        ]
    """

    paragraphs: List[Dict[str, Any]] = []

    paragraph_id = 0

    for page in pages:

        page_number = page["page"]
        text = page.get("text", "")

        page_paragraphs = split_paragraphs(
            text,
            min_chars=min_chars
        )

        for paragraph in page_paragraphs:

            paragraphs.append(
                {
                    "paragraph_id": paragraph_id,
                    "page": page_number,
                    "text": paragraph,
                    "char_count": len(paragraph)
                }
            )

            paragraph_id += 1

    return paragraphs


# ---------------------------------------------------------------------------
# IMAGE EXTRACTION
# ---------------------------------------------------------------------------

def _require_pymupdf():
    """
    Make sure PyMuPDF is installed.
    """

    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for image extraction.\n"
            "Install it with:\n\n"
            "pip install pymupdf"
        )


def _safe_filename(value: str) -> str:
    """
    Convert a string into a filesystem-safe filename.
    """

    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)

    return value.strip("_") or "image"


def _clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving the actual source wording.
    """

    if not text:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    return re.sub(r"\s+", " ", text).strip()


def _looks_like_caption(text: str) -> bool:
    """
    Determine whether a piece of text looks like an image/figure caption.
    """

    if not text:
        return False

    text = _clean_text(text).lower()

    caption_patterns = [
        r"^figure\s+\d+",
        r"^fig\.\s*\d+",
        r"^fig\s+\d+",
        r"^table\s+\d+",
        r"^diagram\s+\d+",
        r"^chart\s+\d+",
        r"^image\s+\d+",
        r"^illustration\s+\d+",
    ]

    return any(
        re.match(pattern, text)
        for pattern in caption_patterns
    )


def _find_image_caption(
    image_bbox,
    page_blocks,
    max_distance: float = 100,
):
    """
    Find the most likely caption associated with an image.

    Captions are searched for immediately above or below the image.

    Args:
        image_bbox:
            Tuple of (x0, y0, x1, y1).

        page_blocks:
            Blocks returned by PyMuPDF page.get_text("blocks").

        max_distance:
            Maximum vertical distance between image and caption.

    Returns:
        Caption text or None.
    """

    if not image_bbox:
        return None

    x0, y0, x1, y1 = image_bbox

    candidates = []

    for block in page_blocks:

        if len(block) < 5:
            continue

        bx0, by0, bx1, by1, text = block[:5]

        text = _clean_text(text)

        if not text:
            continue

        # Caption below the image.
        distance_below = by0 - y1

        if 0 <= distance_below <= max_distance:
            if _looks_like_caption(text):
                candidates.append(
                    (distance_below, text)
                )

        # Caption above the image.
        distance_above = y0 - by1

        if 0 <= distance_above <= max_distance:
            if _looks_like_caption(text):
                candidates.append(
                    (distance_above, text)
                )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def _find_nearby_context(
    image_bbox,
    page_blocks,
    max_distance: float = 180,
    max_blocks: int = 3,
):
    """
    Find text near an image that can help explain its purpose.

    This is used when an explicit caption is not available.

    Returns:
        List of nearby text blocks, ordered by distance from image.
    """

    if not image_bbox:
        return []

    x0, y0, x1, y1 = image_bbox

    candidates = []

    for block in page_blocks:

        if len(block) < 5:
            continue

        bx0, by0, bx1, by1, text = block[:5]

        text = _clean_text(text)

        if not text:
            continue

        # Don't duplicate a caption as nearby context.
        if _looks_like_caption(text):
            continue

        distance_below = by0 - y1
        distance_above = y0 - by1

        if 0 <= distance_below <= max_distance:
            candidates.append(
                (distance_below, text)
            )

        elif 0 <= distance_above <= max_distance:
            candidates.append(
                (distance_above, text)
            )

    candidates.sort(key=lambda item: item[0])

    return [
        text
        for _, text in candidates[:max_blocks]
    ]


def _find_section_heading(
    image_bbox,
    page_blocks,
    max_distance: float = 500,
):
    """
    Attempt to find a section heading associated with an image.

    This is intentionally conservative.

    A text block is considered a possible heading if:

    - it appears above the image
    - it is reasonably close to the image
    - it is relatively short
    - it does not look like normal paragraph text
    """

    if not image_bbox:
        return None

    x0, y0, x1, y1 = image_bbox

    candidates = []

    for block in page_blocks:

        if len(block) < 5:
            continue

        bx0, by0, bx1, by1, text = block[:5]

        text = _clean_text(text)

        if not text:
            continue

        # Only consider text above the image.
        distance = y0 - by1

        if not (0 <= distance <= max_distance):
            continue

        # Don't treat captions as section headings.
        if _looks_like_caption(text):
            continue

        # Headings are usually relatively short.
        if len(text) > 150:
            continue

        # Avoid obviously sentence-like blocks.
        if text.endswith((".", "?", "!")):
            continue

        candidates.append(
            (distance, text)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def _build_image_description(
    image,
    caption=None,
    alt_text=None,
    nearby_context=None,
    section_heading=None,
):
    """
    Build a semantic description for an image.

    Priority:

        1. Alt text
        2. Caption
        3. Caption + nearby context
        4. Nearby context
        5. Section heading
        6. Generic fallback

    The description is generated entirely from information
    available in the PDF. No external AI/vision model is used.
    """

    alt_text = _clean_text(alt_text)
    caption = _clean_text(caption)
    section_heading = _clean_text(section_heading)

    nearby_context = [
        _clean_text(text)
        for text in (nearby_context or [])
        if _clean_text(text)
    ]

    # ---------------------------------------------------------------
    # 1. ALT TEXT
    # ---------------------------------------------------------------

    if alt_text:
        return {
            "description": alt_text,
            "description_source": "alt_text",
        }

    # ---------------------------------------------------------------
    # 2. CAPTION
    # ---------------------------------------------------------------

    if caption:
        return {
            "description": caption,
            "description_source": "caption",
        }

    # ---------------------------------------------------------------
    # 3. CAPTION + CONTEXT
    # ---------------------------------------------------------------

    if section_heading and nearby_context:

        description = (
            f"Image related to '{section_heading}'. "
            f"Nearby text: {' '.join(nearby_context)}"
        )

        return {
            "description": description,
            "description_source": "section_heading_and_context",
        }

    # ---------------------------------------------------------------
    # 4. NEARBY CONTEXT
    # ---------------------------------------------------------------

    if nearby_context:

        description = (
            f"Image related to: "
            f"{' '.join(nearby_context)}"
        )

        return {
            "description": description,
            "description_source": "nearby_context",
        }

    # ---------------------------------------------------------------
    # 5. SECTION HEADING
    # ---------------------------------------------------------------

    if section_heading:

        return {
            "description": (
                f"Image associated with the section "
                f"'{section_heading}'."
            ),
            "description_source": "section_heading",
        }

    # ---------------------------------------------------------------
    # 6. FALLBACK
    # ---------------------------------------------------------------

    page = image.get("page")

    if page is not None:
        description = (
            f"Image extracted from page {page}."
        )
    else:
        description = "Image extracted from the PDF."

    return {
        "description": description,
        "description_source": "fallback",
    }


def extract_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    pages=None,
):
    """
    Extract embedded images from a PDF.

    Each image is saved to disk and receives semantic metadata
    derived from the PDF itself:

        - caption
        - nearby context
        - section heading
        - description
        - description source

    No vision model is required.

    Args:
        pdf_path:
            Path to the PDF.

        output_dir:
            Directory where images should be saved.

        pages:
            Optional iterable of 1-based page numbers to process.

    Returns:
        List of image dictionaries.
    """

    _require_pymupdf()

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    if output_dir is None:
        output_dir = (
            Path(pdf_path).parent /
            f"{Path(pdf_path).stem}_images"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Open PDF with PyMuPDF.
    # ---------------------------------------------------------------

    document = fitz.open(pdf_path)

    images = []

    # Convert requested 1-based page numbers to 0-based indexes.
    if pages is None:
        page_numbers = range(len(document))
    else:
        page_numbers = [
            page - 1
            for page in pages
            if 1 <= page <= len(document)
        ]

    try:

        for page_index in page_numbers:

            page = document[page_index]
            page_number = page_index + 1

            # -------------------------------------------------------
            # Extract text blocks.
            # -------------------------------------------------------

            page_blocks = page.get_text("blocks")

            # -------------------------------------------------------
            # Extract images.
            # -------------------------------------------------------

            page_images = page.get_images(full=True)

            for image_index, image_info in enumerate(page_images):

                xref = image_info[0]

                try:
                    image_data = document.extract_image(xref)

                except Exception as exc:

                    print(
                        f"Warning: Could not extract image "
                        f"on page {page_number}: {exc}"
                    )

                    continue

                extension = image_data.get(
                    "ext",
                    "png",
                )

                image_id = (
                    f"page_{page_number}"
                    f"_image_{image_index + 1}"
                )

                filename = (
                    f"{_safe_filename(image_id)}."
                    f"{extension}"
                )

                image_path = (
                    output_dir /
                    filename
                )

                # ---------------------------------------------------
                # Save image.
                # ---------------------------------------------------

                with open(image_path, "wb") as f:
                    f.write(image_data["image"])

                # ---------------------------------------------------
                # Find image location.
                # ---------------------------------------------------

                image_bbox = None

                try:

                    rects = page.get_image_rects(xref)

                    if rects:

                        rect = rects[0]

                        image_bbox = (
                            rect.x0,
                            rect.y0,
                            rect.x1,
                            rect.y1,
                        )

                except Exception:
                    image_bbox = None

                # ---------------------------------------------------
                # Find caption.
                # ---------------------------------------------------

                caption = _find_image_caption(
                    image_bbox=image_bbox,
                    page_blocks=page_blocks,
                )

                # ---------------------------------------------------
                # Find nearby context.
                # ---------------------------------------------------

                nearby_context = _find_nearby_context(
                    image_bbox=image_bbox,
                    page_blocks=page_blocks,
                )

                # ---------------------------------------------------
                # Find possible section heading.
                # ---------------------------------------------------

                section_heading = _find_section_heading(
                    image_bbox=image_bbox,
                    page_blocks=page_blocks,
                )

                # ---------------------------------------------------
                # Build initial image metadata.
                # ---------------------------------------------------

                image = {
                    "image_id": image_id,
                    "page": page_number,
                    "path": str(image_path),
                    "xref": xref,
                    "width": image_data.get("width"),
                    "height": image_data.get("height"),
                    "extension": extension,
                    "bbox": image_bbox,

                    # Semantic information
                    "alt_text": None,
                    "caption": caption,
                    "section_heading": section_heading,
                    "nearby_context": nearby_context,
                }

                # ---------------------------------------------------
                # Build final description.
                # ---------------------------------------------------

                description_data = _build_image_description(
                    image=image,
                    alt_text=image.get("alt_text"),
                    caption=caption,
                    nearby_context=nearby_context,
                    section_heading=section_heading,
                )

                image.update(description_data)

                images.append(image)

    finally:
        document.close()

    return images

# ---------------------------------------------------------------------------
# PAGE + IMAGE INDEX
# ---------------------------------------------------------------------------

def build_page_image_index(
    images: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group images by PDF page.

    Returns:

        {
            1: [image1, image2],
            2: [image3],
            ...
        }
    """

    index: Dict[int, List[Dict[str, Any]]] = {}

    for image in images:

        page = image["page"]

        if page not in index:
            index[page] = []

        index[page].append(image)

    return index


# ---------------------------------------------------------------------------
# COMPLETE EXTRACTION PIPELINE
# ---------------------------------------------------------------------------

def extract_pdf(
    pdf_path: str,
    image_output_dir: Optional[str] = None,
    min_paragraph_chars: int = 40,
    extract_pdf_images: bool = True
) -> Dict[str, Any]:
    """
    Run the complete PDF extraction pipeline.

    Returns:

        {
            "pdf_path": "...",
            "title": "...",
            "page_count": 15,
            "pages": [...],
            "paragraphs": [...],
            "images": [...],
            "images_by_page": {...}
        }
    """

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    title = Path(pdf_path).stem

    # -------------------------------------------------------
    # Text
    # -------------------------------------------------------

    pages = extract_pages(pdf_path)

    paragraphs = page_to_paragraphs(
        pages,
        min_chars=min_paragraph_chars
    )

    # -------------------------------------------------------
    # Images
    # -------------------------------------------------------

    if extract_pdf_images:

        images = extract_images(
            pdf_path,
            output_dir=image_output_dir
        )

    else:

        images = []

    images_by_page = build_page_image_index(images)

    return {
        "pdf_path": pdf_path,
        "title": title,
        "page_count": len(pages),
        "pages": pages,
        "paragraphs": paragraphs,
        "images": images,
        "images_by_page": images_by_page
    }


# ---------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract text, paragraphs, and images from a PDF."
    )

    parser.add_argument(
        "pdf",
        help="Path to PDF"
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not extract images"
    )

    args = parser.parse_args()

    result = extract_pdf(
        args.pdf,
        extract_pdf_images=not args.no_images
    )

    print("\n==============================")
    print("PDF EXTRACTION")
    print("==============================")

    print(f"Title: {result['title']}")
    print(f"Pages: {result['page_count']}")
    print(f"Paragraphs: {len(result['paragraphs'])}")
    print(f"Images: {len(result['images'])}")

    print("\nFirst 5 paragraphs:")

    for paragraph in result["paragraphs"][:5]:

        print(
            f"\n[{paragraph['paragraph_id']}] "
            f"Page {paragraph['page']}"
        )

        print(paragraph["text"][:500])

    print("\nImages:")

    for image in result["images"]:

        print(
            f"{image['image_id']} | "
            f"Page {image['page']} | "
            f"{image['width']}x{image['height']} | "
            f"{image['path']}"
        )