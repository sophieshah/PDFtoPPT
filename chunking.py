"""
chunking.py

Semantic PDF chunking + E5 embedding preparation.

Pipeline:

    PDF
      |
      v
    pdf_extractor.extract_pdf()
      |
      v
    paragraphs
      |
      v
    LLM semantic merging
      |
      v
    token-limit splitting
      |
      v
    provenance-aware chunks
      |
      v
    E5 embedding text

Dependencies:

    pip install pypdf pymupdf spacy tiktoken openai sentence-transformers

Optional:

    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from pdf_extractor import extract_pdf


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_LLM_MODEL = "gpt-oss-120b"

DEFAULT_MAX_TOKENS = 350

DEFAULT_MIN_TOKENS = 150


# ============================================================================
# INITIALIZE NLP / TOKENIZER / MODELS
# ============================================================================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is not installed.\n"
        "Run:\n\n"
        "python -m spacy download en_core_web_sm"
    )


enc = tiktoken.get_encoding("cl100k_base")


def get_openai_client() -> OpenAI:
    """
    Create the OpenAI-compatible client.

    Your existing UF API configuration is preserved.

    Environment variables:

        OPENAI_API_KEY
        OPENAI_BASE_URL

    If OPENAI_BASE_URL is not provided, the UF endpoint is used.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://api.ai.it.ufl.edu"
    )

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


client = get_openai_client()


# E5 model
e5_model = SentenceTransformer(
    "intfloat/e5-large"
)


# ============================================================================
# TEXT / TOKEN HELPERS
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Count cl100k_base tokens.
    """

    return len(enc.encode(text))


def split_if_needed(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[str]:
    """
    Split text into chunks that are guaranteed to be <= max_tokens.

    This uses token boundaries rather than character boundaries.

    Note:
        The splitting is deliberately conservative. Later we can make
        this sentence-aware if desired.
    """

    if not text:
        return []

    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return [text.strip()]

    chunks: List[str] = []

    start = 0

    while start < len(tokens):

        end = min(
            start + max_tokens,
            len(tokens)
        )

        chunk = enc.decode(
            tokens[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


# ============================================================================
# SPA CY HEADING DETECTION
# ============================================================================

def potential_heading(text: str) -> bool:
    """
    Identify paragraphs that look like headings.

    This is a heuristic only.

    The LLM remains responsible for semantic organization.
    """

    text = text.strip()

    if not text:
        return False

    doc = nlp(text)

    # Short ALL CAPS heading
    if len(text) < 120 and text.isupper():
        return True

    # Short noun-heavy phrase
    if (
        len(doc) < 10
        and sum(
            1 for token in doc
            if token.pos_ == "NOUN"
        ) >= 2
    ):
        return True

    # Numbered heading
    if re.match(
        r"^\s*\d+[\.\)]?\s+",
        text
    ):
        return True

    # Layer-style headings
    if re.match(
        r"^\s*(layer|chapter|section)\s+\d+",
        text,
        re.IGNORECASE
    ):
        return True

    return False


def classify_paragraph(text: str) -> Dict[str, Any]:
    """
    Add lightweight linguistic metadata to a paragraph.
    """

    doc = nlp(text)

    noun_count = sum(
        1
        for token in doc
        if token.pos_ == "NOUN"
    )

    return {
        "is_heading": potential_heading(text),
        "noun_ratio": (
            noun_count / max(len(doc), 1)
        ),
        "sentence_count": len(
            list(doc.sents)
        ),
        "token_count": count_tokens(text)
    }


# ============================================================================
# LLM JSON EXTRACTION
# ============================================================================

def extract_response_text(response) -> str:
    """
    Extract output text from the OpenAI Responses API.
    """

    for item in response.output:

        if item.type != "message":
            continue

        for content in item.content:

            if content.type == "output_text":

                return content.text.strip()

    raise ValueError(
        "No output_text found in model response."
    )


def sanitize_json(text: str) -> str:
    """
    Clean common LLM JSON formatting problems.
    """

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Escape invalid backslashes
    text = re.sub(
        r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        r"\\\\",
        text
    )

    return text.strip()


def parse_llm_json(raw_text: str) -> Any:
    """
    Parse JSON from an LLM response.
    """

    cleaned = sanitize_json(raw_text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as first_error:

        # Try extracting the outer JSON object/list.
        first_bracket = min(
            [
                position
                for position in [
                    cleaned.find("["),
                    cleaned.find("{")
                ]
                if position >= 0
            ],
            default=-1
        )

        if first_bracket >= 0:

            last_bracket = max(
                cleaned.rfind("]"),
                cleaned.rfind("}")
            )

            if last_bracket > first_bracket:

                candidate = cleaned[
                    first_bracket:last_bracket + 1
                ]

                try:
                    return json.loads(candidate)

                except json.JSONDecodeError:
                    pass

        raise ValueError(
            "Could not parse valid JSON from LLM response.\n\n"
            f"Raw response:\n{raw_text}"
        ) from first_error


# ============================================================================
# PARAGRAPH MERGING
# ============================================================================

def merge_paragraphs_llm(
    paragraphs: List[Dict[str, Any]],
    model: str = DEFAULT_LLM_MODEL,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[Dict[str, Any]]:
    """
    Use an LLM to merge related paragraphs into semantic chunks.

    Important:
        The LLM is instructed to preserve paragraph IDs.

    Input:

        [
            {
                "paragraph_id": 0,
                "page": 1,
                "text": "..."
            }
        ]

    Output:

        [
            {
                "chunk_title": "...",
                "chunk_text": "...",
                "paragraph_ids": [0, 1]
            }
        ]
    """

    if not paragraphs:
        return []

    # Keep the prompt focused on source content.
    input_paragraphs = []

    for paragraph in paragraphs:

        input_paragraphs.append(
            {
                "paragraph_id": paragraph["paragraph_id"],
                "page": paragraph["page"],
                "text": paragraph["text"]
            }
        )

    prompt = f"""
You are preparing a source document for semantic retrieval and
eventually for an educational presentation.

Group related paragraphs into coherent semantic chunks.

RULES:

1. Each chunk must cover ONE clear topic or concept.

2. A chunk should be understandable without requiring the reader
   to see the surrounding chunks.

3. Prefer approximately {min_tokens}-{max_tokens} tokens per chunk.

4. Do not merge unrelated topics.

5. Preserve important definitions, explanations, terminology,
   examples, and relationships.

6. Do not introduce information that is not present in the source.

7. Preserve the paragraph IDs that support each chunk.

8. A paragraph may belong to only ONE chunk.

9. Do not omit important paragraphs simply because they are short.

10. If a paragraph is a heading, use it as contextual information
    when determining the topic of the following paragraphs.

11. The output must contain ONLY valid JSON.

OUTPUT FORMAT:

[
    {{
        "chunk_title": "Short descriptive title",
        "chunk_text": "Self-contained chunk text",
        "paragraph_ids": [0, 1, 2]
    }}
]

SOURCE PARAGRAPHS:

{json.dumps(input_paragraphs, indent=2, ensure_ascii=False)}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2
    )

    raw = extract_response_text(response)

    result = parse_llm_json(raw)

    if not isinstance(result, list):
        raise ValueError(
            "Expected LLM output to be a JSON list."
        )

    validated_chunks: List[Dict[str, Any]] = []

    valid_paragraph_ids = {
        p["paragraph_id"]
        for p in paragraphs
    }

    for index, chunk in enumerate(result):

        if not isinstance(chunk, dict):
            continue

        title = str(
            chunk.get(
                "chunk_title",
                f"Chunk {index + 1}"
            )
        ).strip()

        text = str(
            chunk.get(
                "chunk_text",
                ""
            )
        ).strip()

        paragraph_ids = chunk.get(
            "paragraph_ids",
            []
        )

        if not isinstance(paragraph_ids, list):
            paragraph_ids = []

        paragraph_ids = [
            pid
            for pid in paragraph_ids
            if pid in valid_paragraph_ids
        ]

        if not text:
            continue

        validated_chunks.append(
            {
                "chunk_id": index,
                "chunk_title": title,
                "chunk_text": text,
                "paragraph_ids": paragraph_ids
            }
        )

    return validated_chunks


# ============================================================================
# PROVENANCE
# ============================================================================

def get_chunk_source_metadata(
    chunk: Dict[str, Any],
    paragraphs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determine exactly which pages and paragraphs support a chunk.
    """

    paragraph_lookup = {
        p["paragraph_id"]: p
        for p in paragraphs
    }

    source_paragraphs = []

    for paragraph_id in chunk["paragraph_ids"]:

        paragraph = paragraph_lookup.get(
            paragraph_id
        )

        if paragraph is not None:
            source_paragraphs.append(
                paragraph
            )

    source_pages = sorted(
        {
            p["page"]
            for p in source_paragraphs
        }
    )

    if source_pages:

        if len(source_pages) == 1:
            page_range = str(
                source_pages[0]
            )

        else:
            page_range = (
                f"{min(source_pages)}-"
                f"{max(source_pages)}"
            )

    else:
        page_range = None

    return {
        "source_pages": source_pages,
        "page_range": page_range,
        "source_paragraphs": source_paragraphs
    }


# ============================================================================
# E5 FORMAT
# ============================================================================

def format_for_e5(
    chunk_text: str,
    doc_title: Optional[str] = None,
    section_header: Optional[str] = None,
    page_range: Optional[str] = None
) -> str:
    """
    Format text for E5 passage embedding.
    """

    metadata = []

    if doc_title:
        metadata.append(
            f"Document: {doc_title}"
        )

    if section_header:
        metadata.append(
            f"Section: {section_header}"
        )

    if page_range:
        metadata.append(
            f"Pages: {page_range}"
        )

    metadata_block = " | ".join(metadata)

    if metadata_block:

        return (
            f"passage: {metadata_block}\n"
            f"{chunk_text}"
        )

    return f"passage: {chunk_text}"


# ============================================================================
# COMPLETE PDF -> CHUNKS PIPELINE
# ============================================================================

def pdf_to_embedded_chunks(
    pdf_path: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_paragraph_chars: int = 40,
    extract_pdf_images: bool = True,
    image_output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete PDF processing pipeline.

    This function now returns BOTH:

        1. embedding chunks
        2. extracted images

    This is intentional because the future PowerPoint planner
    will need both.

    Returns:

        {
            "document": {...},
            "paragraphs": [...],
            "chunks": [...],
            "images": [...]
        }
    """

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    print("\n==============================")
    print("PDF -> SEMANTIC CHUNKS")
    print("==============================")

    # -------------------------------------------------------
    # Step 1: Extract PDF
    # -------------------------------------------------------

    print("\n[1/5] Extracting PDF...")

    extracted = extract_pdf(
        pdf_path,
        image_output_dir=image_output_dir,
        min_paragraph_chars=min_paragraph_chars,
        extract_pdf_images=extract_pdf_images
    )

    doc_title = extracted["title"]

    pages = extracted["pages"]

    paragraphs = extracted["paragraphs"]

    images = extracted["images"]

    print(
        f"Pages: {len(pages)}"
    )

    print(
        f"Paragraphs: {len(paragraphs)}"
    )

    print(
        f"Images: {len(images)}"
    )

    # -------------------------------------------------------
    # Step 2: Add linguistic metadata
    # -------------------------------------------------------

    print("\n[2/5] Classifying paragraphs...")

    enriched_paragraphs = []

    for paragraph in paragraphs:

        classification = classify_paragraph(
            paragraph["text"]
        )

        enriched_paragraphs.append(
            {
                **paragraph,
                **classification
            }
        )

    # -------------------------------------------------------
    # Step 3: LLM semantic merging
    # -------------------------------------------------------

    print("\n[3/5] Merging semantic paragraphs with LLM...")

    merged_chunks = merge_paragraphs_llm(
        enriched_paragraphs,
        model=llm_model,
        max_tokens=max_tokens
    )

    print(
        f"LLM created {len(merged_chunks)} semantic chunks."
    )

    # -------------------------------------------------------
    # Step 4: Enforce token limits
    # -------------------------------------------------------

    print("\n[4/5] Enforcing token limits...")

    final_chunks: List[Dict[str, Any]] = []

    for chunk in merged_chunks:

        text_splits = split_if_needed(
            chunk["chunk_text"],
            max_tokens=max_tokens
        )

        for split_index, split_text in enumerate(
            text_splits,
            start=1
        ):

            title = chunk["chunk_title"]

            if len(text_splits) > 1:

                title = (
                    f"{title} "
                    f"({split_index}/{len(text_splits)})"
                )

            source_metadata = (
                get_chunk_source_metadata(
                    chunk,
                    enriched_paragraphs
                )
            )

            final_chunks.append(
                {
                    "chunk_id": len(final_chunks),

                    "chunk_title": title,

                    "chunk_text": split_text,

                    "token_count": count_tokens(
                        split_text
                    ),

                    "paragraph_ids": (
                        chunk["paragraph_ids"]
                    ),

                    "source_pages": (
                        source_metadata[
                            "source_pages"
                        ]
                    ),

                    "page_range": (
                        source_metadata[
                            "page_range"
                        ]
                    ),

                    "source_paragraphs": [
                        {
                            "paragraph_id": p[
                                "paragraph_id"
                            ],
                            "page": p["page"],
                            "text": p["text"]
                        }
                        for p in source_metadata[
                            "source_paragraphs"
                        ]
                    ]
                }
            )

    print(
        f"Final chunks: {len(final_chunks)}"
    )

    # -------------------------------------------------------
    # Step 5: Format for E5
    # -------------------------------------------------------

    print("\n[5/5] Preparing E5 embedding text...")

    embedding_chunks = []

    for chunk in final_chunks:

        embedding_text = format_for_e5(
            chunk["chunk_text"],
            doc_title=doc_title,
            section_header=chunk[
                "chunk_title"
            ],
            page_range=chunk[
                "page_range"
            ]
        )

        embedding_chunks.append(
            {
                "text_for_embedding": embedding_text,

                "chunk_text": chunk[
                    "chunk_text"
                ],

                "metadata": {
                    "chunk_id": chunk[
                        "chunk_id"
                    ],

                    "title": doc_title,

                    "section_header": chunk[
                        "chunk_title"
                    ],

                    "pages": chunk[
                        "page_range"
                    ],

                    "source_pages": chunk[
                        "source_pages"
                    ],

                    "paragraph_ids": chunk[
                        "paragraph_ids"
                    ],

                    "token_count": chunk[
                        "token_count"
                    ]
                }
            }
        )

    # -------------------------------------------------------
    # Final result
    # -------------------------------------------------------

    return {
        "document": {
            "title": doc_title,
            "pdf_path": pdf_path,
            "page_count": len(pages)
        },

        "pages": pages,

        "paragraphs": enriched_paragraphs,

        "chunks": final_chunks,

        "embedding_chunks": embedding_chunks,

        "images": images
    }


# ============================================================================
# E5 EMBEDDINGS
# ============================================================================

def embed_with_e5(
    chunks: List[str],
    batch_size: int = 32
):
    """
    Generate normalized E5 embeddings.

    IMPORTANT:
        E5 expects the passage/query prefixes to be used consistently.

    This function expects `chunks` to already contain:

        passage: ...
    """

    return e5_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size
    )


# ============================================================================
# QDRANT PAYLOAD PREPARATION
# ============================================================================

def build_qdrant_records(
    processed_document: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert processed chunks into records convenient for Qdrant.

    This does NOT create the Qdrant collection or upload anything.

    Returns:

        [
            {
                "text": "...",
                "payload": {...}
            }
        ]
    """

    records = []

    for chunk in processed_document[
        "embedding_chunks"
    ]:

        records.append(
            {
                "text": chunk[
                    "text_for_embedding"
                ],

                "payload": {
                    "doc_title": chunk[
                        "metadata"
                    ]["title"],

                    "section_header": chunk[
                        "metadata"
                    ]["section_header"],

                    "text": chunk[
                        "chunk_text"
                    ],

                    "pages": chunk[
                        "metadata"
                    ]["pages"],

                    "source_pages": chunk[
                        "metadata"
                    ]["source_pages"],

                    "paragraph_ids": chunk[
                        "metadata"
                    ]["paragraph_ids"],

                    "chunk_id": chunk[
                        "metadata"
                    ]["chunk_id"],

                    "token_count": chunk[
                        "metadata"
                    ]["token_count"]
                }
            }
        )

    return records


# ============================================================================
# SAVE INTERMEDIATE RESULTS
# ============================================================================

def save_processed_document(
    processed_document: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save extraction/chunking results as JSON.

    Useful for debugging the pipeline before generating PowerPoint.
    """

    output_path = os.path.abspath(
        output_path
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            processed_document,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved processed document to:\n"
        f"{output_path}"
    )


# ============================================================================
# DEBUG / TEST
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF into semantic chunks "
            "and extracted images."
        )
    )

    parser.add_argument(
        "pdf",
        help="Path to PDF"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help="LLM model name"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens per chunk"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path"
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not extract PDF images"
    )

    args = parser.parse_args()

    result = pdf_to_embedded_chunks(
        pdf_path=args.pdf,
        llm_model=args.model,
        max_tokens=args.max_tokens,
        extract_pdf_images=not args.no_images
    )

    print("\n==============================")
    print("RESULT")
    print("==============================")

    print(
        f"Document: "
        f"{result['document']['title']}"
    )

    print(
        f"Pages: "
        f"{result['document']['page_count']}"
    )

    print(
        f"Paragraphs: "
        f"{len(result['paragraphs'])}"
    )

    print(
        f"Chunks: "
        f"{len(result['chunks'])}"
    )

    print(
        f"Images: "
        f"{len(result['images'])}"
    )

    print("\nChunks:")

    for chunk in result["chunks"]:

        print(
            f"\n[{chunk['chunk_id']}] "
            f"{chunk['chunk_title']}"
        )

        print(
            f"Pages: "
            f"{chunk['page_range']}"
        )

        print(
            f"Tokens: "
            f"{chunk['token_count']}"
        )

        print(
            chunk["chunk_text"][:500]
        )

    # Optional JSON output
    if args.output:

        save_processed_document(
            result,
            args.output
        )