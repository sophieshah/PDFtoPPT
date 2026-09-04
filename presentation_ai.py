"""
presentation_ai.py

AI-powered presentation planning layer.

Pipeline:

    Processed PDF
          |
          v
    Rank chunks by importance
          |
          v
    Select important content
          |
          v
    Generate 20-slide presentation plan
          |
          v
    Select appropriate PDF visuals
          |
          v
    Validate presentation
          |
          v
    Final slide JSON

This module DOES NOT create PowerPoint files.

It creates a structured presentation specification that
ppt_generator.py can later turn into a PowerPoint.

Dependencies:

    pip install openai

This module expects the output of chunking.py:

    pdf_to_embedded_chunks()

"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-oss-120b"

DEFAULT_NUM_SLIDES = 20

DEFAULT_MAX_CHUNKS = 40

DEFAULT_MAX_BULLETS = 4

DEFAULT_MAX_BULLET_WORDS = 25


# ============================================================================
# OPENAI CLIENT
# ============================================================================

def get_openai_client() -> OpenAI:
    """
    Create the OpenAI-compatible client.

    Environment variables:

        OPENAI_API_KEY

        OPENAI_BASE_URL

    If OPENAI_BASE_URL is not set, the UF endpoint from your
    existing chunking script is used.
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


# ============================================================================
# JSON HELPERS
# ============================================================================

def extract_response_text(response) -> str:
    """
    Extract text from an OpenAI Responses API response.
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
    Remove common formatting problems from LLM JSON.
    """

    text = text.strip()

    # Remove ```json
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove ```
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
    Parse JSON returned by an LLM.

    Includes a fallback that attempts to locate the outer
    JSON object/list if the model included extra text.
    """

    cleaned = sanitize_json(raw_text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as first_error:

        candidates = []

        first_list = cleaned.find("[")
        last_list = cleaned.rfind("]")

        if (
            first_list >= 0
            and last_list > first_list
        ):
            candidates.append(
                cleaned[
                    first_list:last_list + 1
                ]
            )

        first_object = cleaned.find("{")
        last_object = cleaned.rfind("}")

        if (
            first_object >= 0
            and last_object > first_object
        ):
            candidates.append(
                cleaned[
                    first_object:last_object + 1
                ]
            )

        for candidate in candidates:

            try:
                return json.loads(candidate)

            except json.JSONDecodeError:
                continue

        raise ValueError(
            "Could not parse valid JSON from LLM response.\n\n"
            f"Raw response:\n{raw_text}"
        ) from first_error


def llm_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> Any:
    """
    Send a prompt to the LLM and return parsed JSON.
    """

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature
    )

    raw = extract_response_text(response)

    return parse_llm_json(raw)


# ============================================================================
# CHUNK SUMMARIZATION
# ============================================================================

def prepare_chunk_catalog(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create a compact representation of chunks for the AI.

    We don't need to send every internal field to the planner.
    """

    catalog = []

    for chunk in chunks:

        catalog.append(
            {
                "chunk_id": chunk.get("chunk_id"),

                "title": chunk.get(
                    "chunk_title",
                    ""
                ),

                "text": chunk.get(
                    "chunk_text",
                    ""
                ),

                "source_pages": chunk.get(
                    "source_pages",
                    []
                ),

                "page_range": chunk.get(
                    "page_range"
                ),

                "paragraph_ids": chunk.get(
                    "paragraph_ids",
                    []
                )
            }
        )

    return catalog


# ============================================================================
# STEP 1: RANK CHUNKS
# ============================================================================

def rank_chunks_for_presentation(
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_chunks: int = DEFAULT_MAX_CHUNKS
) -> List[Dict[str, Any]]:
    """
    Rank chunks according to how useful they are for a presentation.

    The model considers:

        - conceptual importance
        - educational value
        - uniqueness
        - relevance to the document's main topic
        - whether the concept deserves visual presentation
        - whether it is foundational for later concepts

    Returns:

        [
            {
                "chunk_id": 12,
                "importance_score": 0.96,
                "presentation_role": "core_concept",
                "reason": "..."
            }
        ]

    """

    if not chunks:
        return []

    catalog = prepare_chunk_catalog(chunks)

    prompt = f"""
You are selecting the most important source material for an
educational PowerPoint presentation.

The presentation will contain approximately 20 slides.

Your job is to evaluate the supplied semantic chunks and rank
them by presentation importance.

IMPORTANT:

- Use ONLY information contained in the supplied chunks.
- Do not add outside facts.
- Do not rewrite the source into new concepts.
- Avoid rewarding repeated information.
- Foundational concepts should generally rank highly.
- Definitions, major mechanisms, processes, comparisons,
  classifications, examples, and conclusions can be important.
- Minor details, repetition, trivial examples, and peripheral
  information should rank lower.
- A chunk does not automatically deserve an entire slide.
- Some chunks will be combined with other chunks later.

For every chunk provide:

importance_score:
    Number from 0.0 to 1.0.

presentation_role:
    One of:
        "core_concept"
        "supporting_concept"
        "example"
        "process"
        "comparison"
        "definition"
        "detail"
        "background"

reason:
    Short explanation of why the chunk has that importance.

Return ONLY valid JSON.

FORMAT:

[
    {{
        "chunk_id": 0,
        "importance_score": 0.95,
        "presentation_role": "core_concept",
        "reason": "..."
    }}
]

CHUNKS:

{json.dumps(catalog, indent=2, ensure_ascii=False)}
"""

    result = llm_json(
        prompt,
        model=model,
        temperature=0.1
    )

    if not isinstance(result, list):
        raise ValueError(
            "Chunk ranking response must be a JSON list."
        )

    valid_ids = {
        chunk.get("chunk_id")
        for chunk in chunks
    }

    ranked = []

    for item in result:

        if not isinstance(item, dict):
            continue

        chunk_id = item.get("chunk_id")

        if chunk_id not in valid_ids:
            continue

        try:
            score = float(
                item.get(
                    "importance_score",
                    0
                )
            )
        except (TypeError, ValueError):
            score = 0.0

        score = max(
            0.0,
            min(1.0, score)
        )

        ranked.append(
            {
                "chunk_id": chunk_id,
                "importance_score": score,
                "presentation_role": item.get(
                    "presentation_role",
                    "supporting_concept"
                ),
                "reason": item.get(
                    "reason",
                    ""
                )
            }
        )

    # Sort highest importance first
    ranked.sort(
        key=lambda x: x["importance_score"],
        reverse=True
    )

    # Keep only the highest-ranked chunks.
    if max_chunks:
        ranked = ranked[:max_chunks]

    return ranked


# ============================================================================
# STEP 2: SELECT IMPORTANT CHUNKS
# ============================================================================

def select_important_chunks(
    chunks: List[Dict[str, Any]],
    rankings: List[Dict[str, Any]],
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    minimum_score: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Combine chunk content with its presentation ranking.

    Returns chunks sorted by importance.
    """

    chunk_lookup = {
        chunk.get("chunk_id"): chunk
        for chunk in chunks
    }

    selected = []

    for ranking in rankings:

        chunk_id = ranking["chunk_id"]

        if ranking["importance_score"] < minimum_score:
            continue

        chunk = chunk_lookup.get(
            chunk_id
        )

        if chunk is None:
            continue

        selected.append(
            {
                **chunk,

                "presentation_importance": (
                    ranking[
                        "importance_score"
                    ]
                ),

                "presentation_role": (
                    ranking[
                        "presentation_role"
                    ]
                ),

                "importance_reason": (
                    ranking[
                        "reason"
                    ]
                )
            }
        )

    selected.sort(
        key=lambda x: x[
            "presentation_importance"
        ],
        reverse=True
    )

    return selected[:max_chunks]


# ============================================================================
# VISUAL CATALOG
# ============================================================================

def build_visual_catalog(
    images: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prepare PDF images for the presentation planner.

    The image description may be populated later by a vision model.

    For now we provide metadata and page information.
    """

    catalog = []

    for image in images:

        # Ignore duplicate records that do not have their own file.
        if image.get("is_duplicate"):
            continue

        catalog.append(
            {
                "image_id": image.get(
                    "image_id"
                ),

                "page": image.get(
                    "page"
                ),

                "path": image.get(
                    "path"
                ),

                "width": image.get(
                    "width"
                ),

                "height": image.get(
                    "height"
                ),

                "extension": image.get(
                    "extension"
                ),

                "description": image.get(
                    "description",
                    ""
                ),

                "visual_type": image.get(
                    "visual_type",
                    "unknown"
                )
            }
        )

    return catalog


# ============================================================================
# VISUAL SELECTION
# ============================================================================

def select_visuals_for_slides(
    slide_plan: Dict[str, Any],
    images: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Select existing PDF images for slides.

    This is deliberately a separate AI step from slide planning.

    Why?

        Slide planning answers:
            "What should this slide communicate?"

        Visual selection answers:
            "Which existing PDF visual best communicates it?"

    Returns an updated slide plan.
    """

    visual_catalog = build_visual_catalog(
        images
    )

    if not visual_catalog:
        return slide_plan

    prompt = f"""
You are selecting visuals from a source PDF for a PowerPoint
presentation.

The presentation slide plan is supplied below.

The available visuals are also supplied below.

Your job is to decide whether an existing PDF image should be
used on each slide.

RULES:

1. Use a visual ONLY when it meaningfully supports the slide.

2. Do not use a visual merely because one exists.

3. Prefer diagrams, charts, figures, screenshots, tables,
   and other instructional visuals over decorative images.

4. If no available visual is appropriate, use:
       "none"

5. Do not invent image IDs.

6. An image may be reused only when doing so is genuinely useful.

7. Do not put visuals on title slides unless the source image
   is genuinely appropriate.

8. If a slide would benefit from a diagram but no suitable PDF
   image exists, use:
       "generated_diagram"

9. If the visual is useful but should occupy only a small area,
   indicate that in the placement.

Return ONLY valid JSON.

FORMAT:

[
    {{
        "slide_number": 1,
        "visual": {{
            "type": "pdf_image",
            "image_id": "page_2_image_1",
            "placement": "right",
            "reason": "..."
        }}
    }}
]

OR:

[
    {{
        "slide_number": 2,
        "visual": {{
            "type": "generated_diagram",
            "description": "...",
            "placement": "center",
            "reason": "..."
        }}
    }}
]

OR:

[
    {{
        "slide_number": 3,
        "visual": {{
            "type": "none",
            "reason": "No visual is necessary."
        }}
    }}
]

SLIDE PLAN:

{json.dumps(slide_plan, indent=2, ensure_ascii=False)}

AVAILABLE PDF VISUALS:

{json.dumps(visual_catalog, indent=2, ensure_ascii=False)}
"""

    result = llm_json(
        prompt,
        model=model,
        temperature=0.2
    )

    if not isinstance(result, list):
        return slide_plan

    image_ids = {
        image["image_id"]
        for image in visual_catalog
    }

    slide_lookup = {
        slide.get("slide_number"): slide
        for slide in slide_plan.get(
            "slides",
            []
        )
    }

    for item in result:

        if not isinstance(item, dict):
            continue

        slide_number = item.get(
            "slide_number"
        )

        visual = item.get(
            "visual"
        )

        if slide_number not in slide_lookup:
            continue

        if not isinstance(visual, dict):
            continue

        visual_type = visual.get(
            "type",
            "none"
        )

        if visual_type == "pdf_image":

            image_id = visual.get(
                "image_id"
            )

            if image_id not in image_ids:
                continue

        elif visual_type == "generated_diagram":

            if not visual.get(
                "description"
            ):
                continue

        elif visual_type == "none":

            pass

        else:

            continue

        slide_lookup[
            slide_number
        ]["visual"] = visual

    return slide_plan


# ============================================================================
# STEP 3: GENERATE SLIDE PLAN
# ============================================================================

def generate_slide_plan(
    document: Dict[str, Any],
    selected_chunks: List[Dict[str, Any]],
    num_slides: int = DEFAULT_NUM_SLIDES,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generate a coherent presentation structure.

    This is the core planning function.

    The model is NOT asked to make PowerPoint slides directly.

    Instead, it produces a structured semantic representation
    that the future ppt_generator.py will render.
    """

    if not selected_chunks:
        raise ValueError(
            "No selected chunks available."
        )

    title = document.get(
        "title",
        "Presentation"
    )

    prompt = f"""
You are an expert instructional presentation designer.

Create a coherent {num_slides}-slide presentation based ONLY
on the supplied source material.

DOCUMENT:

Title:
{title}

Source page count:
{document.get("page_count")}

IMPORTANT CONTENT RULES:

1. The source document is the authority.

2. Do NOT introduce outside facts.

3. Do NOT fabricate examples, definitions, statistics,
   terminology, or conclusions.

4. Every substantive slide must be traceable to one or more
   source chunks.

5. Use the source terminology and framing.

6. Prioritize the most important ideas.

7. Avoid repeating the same concept across multiple slides.

8. Combine closely related chunks when appropriate.

9. Do not give minor details an entire slide unless they are
   necessary for understanding a major concept.

10. The presentation should have a logical teaching progression.

A GOOD GENERAL STRUCTURE MAY INCLUDE:

- title / introduction
- context or motivation
- foundational concepts
- major concepts
- mechanisms or processes
- comparisons/classifications
- examples
- synthesis
- conclusion

However, adapt the structure to the actual source document.
Do not force this structure if the source does not support it.

SLIDE RULES:

- Exactly {num_slides} slides.
- Every slide must have a clear purpose.
- Titles should be concise.
- Use 2-4 key points on most content slides.
- Each key point should normally be one sentence.
- Avoid paragraphs.
- Avoid excessive text.
- A slide should communicate ONE primary idea.
- Slides can reference multiple source chunks.
- Source pages must come from the source chunks.
- Never invent a source page.
- Title slides may have no source chunk.
- A conclusion slide should synthesize the source rather than
  introduce new information.

SLIDE TYPES:

Use one of:

"title"
"overview"
"concept"
"process"
"comparison"
"example"
"diagram"
"summary"
"conclusion"

OUTPUT FORMAT:

Return ONLY valid JSON.

{{
    "presentation_title": "...",
    "presentation_objective": "...",
    "slides": [
        {{
            "slide_number": 1,
            "slide_type": "title",
            "title": "...",
            "subtitle": "...",
            "purpose": "...",
            "key_points": [],
            "source_chunks": [],
            "source_pages": []
        }},

        {{
            "slide_number": 2,
            "slide_type": "overview",
            "title": "...",
            "subtitle": "",
            "purpose": "...",
            "key_points": [
                "...",
                "...",
                "..."
            ],
            "source_chunks": [1, 2],
            "source_pages": [1, 2]
        }}
    ]
}}

SELECTED SOURCE CHUNKS:

{json.dumps(selected_chunks, indent=2, ensure_ascii=False)}
"""

    result = llm_json(
        prompt,
        model=model,
        temperature=0.2
    )

    if not isinstance(result, dict):
        raise ValueError(
            "Slide plan must be a JSON object."
        )

    slides = result.get(
        "slides",
        []
    )

    if not isinstance(slides, list):
        raise ValueError(
            "Slide plan must contain a slides list."
        )

    # -------------------------------------------------------
    # Basic validation
    # -------------------------------------------------------

    valid_chunk_ids = {
        chunk.get("chunk_id")
        for chunk in selected_chunks
    }

    validated_slides = []

    for index, slide in enumerate(
        slides,
        start=1
    ):

        if not isinstance(slide, dict):
            continue

        slide_number = slide.get(
            "slide_number",
            index
        )

        try:
            slide_number = int(
                slide_number
            )
        except (TypeError, ValueError):
            slide_number = index

        source_chunks = slide.get(
            "source_chunks",
            []
        )

        if not isinstance(
            source_chunks,
            list
        ):
            source_chunks = []

        # Only allow real chunk IDs.
        source_chunks = [
            chunk_id
            for chunk_id in source_chunks
            if chunk_id in valid_chunk_ids
        ]

        # Derive source pages ourselves.
        source_pages = set()

        chunk_lookup = {
            chunk.get("chunk_id"): chunk
            for chunk in selected_chunks
        }

        for chunk_id in source_chunks:

            chunk = chunk_lookup.get(
                chunk_id
            )

            if not chunk:
                continue

            for page in chunk.get(
                "source_pages",
                []
            ):
                source_pages.add(page)

        key_points = slide.get(
            "key_points",
            []
        )

        if not isinstance(
            key_points,
            list
        ):
            key_points = []

        key_points = [
            str(point).strip()
            for point in key_points
            if str(point).strip()
        ]

        validated_slides.append(
            {
                "slide_number": slide_number,

                "slide_type": slide.get(
                    "slide_type",
                    "concept"
                ),

                "title": str(
                    slide.get(
                        "title",
                        ""
                    )
                ).strip(),

                "subtitle": str(
                    slide.get(
                        "subtitle",
                        ""
                    )
                ).strip(),

                "purpose": str(
                    slide.get(
                        "purpose",
                        ""
                    )
                ).strip(),

                "key_points": key_points,

                "source_chunks": source_chunks,

                "source_pages": sorted(
                    source_pages
                ),

                "visual": {
                    "type": "none"
                }
            }
        )

    # Sort slides
    validated_slides.sort(
        key=lambda x: x[
            "slide_number"
        ]
    )

    # Renumber if necessary
    for index, slide in enumerate(
        validated_slides,
        start=1
    ):
        slide["slide_number"] = index

    result["slides"] = validated_slides

    return result


# ============================================================================
# STEP 4: VALIDATE SLIDE PLAN
# ============================================================================

def validate_slide_plan(
    slide_plan: Dict[str, Any],
    selected_chunks: List[Dict[str, Any]],
    num_slides: int = DEFAULT_NUM_SLIDES,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Run a second AI pass over the generated slide plan.

    This is important because the first model call may:

        - repeat concepts
        - omit important content
        - overload a slide
        - create unsupported claims
        - produce weak sequencing

    The validator does not rewrite everything automatically.

    It returns:

        {
            "valid": true,
            "score": 0.91,
            "issues": [],
            "recommendations": []
        }
    """

    prompt = f"""
You are validating an educational PowerPoint presentation.

The presentation must contain exactly {num_slides} slides.

Check the presentation against the supplied source chunks.

Evaluate:

1. Does the presentation cover the most important source ideas?

2. Are there unnecessary or repetitive slides?

3. Does each content slide have a clear primary idea?

4. Are the source chunks appropriate for each slide?

5. Are source pages consistent with the source chunks?

6. Are any claims unsupported by the source?

7. Is the sequence logical?

8. Are slides overloaded with text?

9. Does the conclusion synthesize rather than introduce new
   information?

10. Does the presentation appear coherent as a whole?

Return ONLY valid JSON.

FORMAT:

{{
    "valid": true,
    "score": 0.92,
    "issues": [
        {{
            "severity": "medium",
            "slide_number": 7,
            "issue": "...",
            "recommendation": "..."
        }}
    ],
    "missing_important_topics": [],
    "repeated_topics": []
}}

SOURCE CHUNKS:

{json.dumps(selected_chunks, indent=2, ensure_ascii=False)}

SLIDE PLAN:

{json.dumps(slide_plan, indent=2, ensure_ascii=False)}
"""

    result = llm_json(
        prompt,
        model=model,
        temperature=0.1
    )

    return result


# ============================================================================
# FULL PRESENTATION PLANNING PIPELINE
# ============================================================================

def create_presentation_plan(
    processed_document: Dict[str, Any],
    num_slides: int = DEFAULT_NUM_SLIDES,
    model: str = DEFAULT_MODEL,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    minimum_importance: float = 0.45,
    select_visuals: bool = True,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Run the entire AI presentation planning pipeline.

    Input:
        Output from pdf_to_embedded_chunks()

    Output:

        {
            "presentation": {...},
            "rankings": [...],
            "selected_chunks": [...],
            "slides": [...],
            "validation": {...}
        }
    """

    document = processed_document.get(
        "document",
        {}
    )

    chunks = processed_document.get(
        "chunks",
        []
    )

    images = processed_document.get(
        "images",
        []
    )

    if not chunks:
        raise ValueError(
            "Processed document contains no chunks."
        )

    print("\n===================================")
    print("AI PRESENTATION PLANNER")
    print("===================================")

    # -------------------------------------------------------
    # Step 1
    # -------------------------------------------------------

    print(
        "\n[1/5] Ranking source chunks..."
    )

    rankings = rank_chunks_for_presentation(
        chunks,
        model=model,
        max_chunks=max_chunks
    )

    print(
        f"Ranked {len(rankings)} chunks."
    )

    # -------------------------------------------------------
    # Step 2
    # -------------------------------------------------------

    print(
        "\n[2/5] Selecting important content..."
    )

    selected_chunks = select_important_chunks(
        chunks,
        rankings,
        max_chunks=max_chunks,
        minimum_score=minimum_importance
    )

    print(
        f"Selected {len(selected_chunks)} chunks."
    )

    # -------------------------------------------------------
    # Step 3
    # -------------------------------------------------------

    print(
        f"\n[3/5] Generating {num_slides}-slide plan..."
    )

    slide_plan = generate_slide_plan(
        document,
        selected_chunks,
        num_slides=num_slides,
        model=model
    )

    print(
        f"Generated "
        f"{len(slide_plan.get('slides', []))} slides."
    )

    # -------------------------------------------------------
    # Step 4
    # -------------------------------------------------------

    if select_visuals:

        print(
            "\n[4/5] Selecting PDF visuals..."
        )

        slide_plan = select_visuals_for_slides(
            slide_plan,
            images,
            model=model
        )

    else:

        print(
            "\n[4/5] Visual selection skipped."
        )

    # -------------------------------------------------------
    # Step 5
    # -------------------------------------------------------

    if validate:

        print(
            "\n[5/5] Validating presentation..."
        )

        validation = validate_slide_plan(
            slide_plan,
            selected_chunks,
            num_slides=num_slides,
            model=model
        )

    else:

        validation = {
            "valid": None,
            "score": None,
            "issues": [],
            "missing_important_topics": [],
            "repeated_topics": []
        }

    # -------------------------------------------------------
    # Final result
    # -------------------------------------------------------

    return {
        "presentation": {
            "title": slide_plan.get(
                "presentation_title",
                document.get(
                    "title",
                    "Presentation"
                )
            ),

            "objective": slide_plan.get(
                "presentation_objective",
                ""
            ),

            "num_slides": num_slides
        },

        "rankings": rankings,

        "selected_chunks": selected_chunks,

        "slides": slide_plan.get(
            "slides",
            []
        ),

        "validation": validation
    }


# ============================================================================
# SAVE PRESENTATION PLAN
# ============================================================================

def save_presentation_plan(
    presentation_plan: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save presentation plan as JSON.

    This is useful for inspecting the AI output before
    generating the PowerPoint.
    """

    output_directory = os.path.dirname(
        os.path.abspath(output_path)
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
            presentation_plan,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved presentation plan:\n"
        f"{os.path.abspath(output_path)}"
    )


# ============================================================================
# PRINT HUMAN-READABLE PLAN
# ============================================================================

def print_presentation_plan(
    presentation_plan: Dict[str, Any]
) -> None:
    """
    Print a readable version of the generated presentation.
    """

    presentation = presentation_plan.get(
        "presentation",
        {}
    )

    slides = presentation_plan.get(
        "slides",
        []
    )

    validation = presentation_plan.get(
        "validation",
        {}
    )

    print("\n")
    print("===================================")
    print(
        presentation.get(
            "title",
            "Presentation"
        )
    )
    print("===================================")

    print(
        "\nObjective:"
    )

    print(
        presentation.get(
            "objective",
            ""
        )
    )

    print(
        f"\nSlides: {len(slides)}"
    )

    for slide in slides:

        print("\n-----------------------------------")

        print(
            f"SLIDE {slide['slide_number']}: "
            f"{slide['title']}"
        )

        print(
            f"Type: "
            f"{slide['slide_type']}"
        )

        print(
            f"Purpose: "
            f"{slide['purpose']}"
        )

        print(
            f"Source pages: "
            f"{slide['source_pages']}"
        )

        print(
            f"Source chunks: "
            f"{slide['source_chunks']}"
        )

        visual = slide.get(
            "visual",
            {}
        )

        print(
            f"Visual: "
            f"{visual.get('type', 'none')}"
        )

        for point in slide[
            "key_points"
        ]:

            print(
                f"  • {point}"
            )

    print("\n===================================")
    print("VALIDATION")
    print("===================================")

    print(
        f"Valid: "
        f"{validation.get('valid')}"
    )

    print(
        f"Score: "
        f"{validation.get('score')}"
    )

    issues = validation.get(
        "issues",
        []
    )

    if issues:

        print("\nIssues:")

        for issue in issues:

            print(
                f"- Slide "
                f"{issue.get('slide_number')}: "
                f"{issue.get('issue')}"
            )

            print(
                f"  Recommendation: "
                f"{issue.get('recommendation')}"
            )


# ============================================================================
# TEST / COMMAND LINE
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate an AI-powered presentation "
            "plan from processed PDF chunks."
        )
    )

    parser.add_argument(
        "json_file",
        help=(
            "JSON produced by chunking.py"
        )
    )

    parser.add_argument(
        "--slides",
        type=int,
        default=20,
        help="Number of slides"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model"
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Maximum chunks given to planner"
    )

    parser.add_argument(
        "--min-importance",
        type=float,
        default=0.45,
        help="Minimum importance score"
    )

    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip visual selection"
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation"
    )

    parser.add_argument(
        "--output",
        default="presentation_plan.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Load processed PDF
    # -------------------------------------------------------

    with open(
        args.json_file,
        "r",
        encoding="utf-8"
    ) as file:

        processed_document = json.load(
            file
        )

    # -------------------------------------------------------
    # Generate presentation
    # -------------------------------------------------------

    presentation_plan = create_presentation_plan(
        processed_document,
        num_slides=args.slides,
        model=args.model,
        max_chunks=args.max_chunks,
        minimum_importance=args.min_importance,
        select_visuals=not args.no_visuals,
        validate=not args.no_validation
    )

    # -------------------------------------------------------
    # Print
    # -------------------------------------------------------

    print_presentation_plan(
        presentation_plan
    )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    save_presentation_plan(
        presentation_plan,
        args.output
    )