import json
from pathlib import Path

from pdf_extractor import extract_pdf
from chunking import pdf_to_embedded_chunks
from presentation_ai import create_presentation_plan
from ppt_generator import generate_presentation


PDF_PATH = "input/osi_model.pdf"
TEMPLATE_PATH = "templates/downloadTestFile1.pptx"

OUTPUT_DIR = Path("output")
IMAGE_DIR = OUTPUT_DIR / "osi_images"

EXTRACTION_PATH = OUTPUT_DIR / "osi_extraction.json"
PLAN_PATH = OUTPUT_DIR / "presentation_plan.json"
PPT_PATH = OUTPUT_DIR / "osi_model.pptx"


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STEP 1 — EXTRACT PDF
# ============================================================

print("\n[1/4] Extracting PDF...")

extraction = extract_pdf(
    pdf_path=PDF_PATH,
    image_output_dir=str(IMAGE_DIR),
    extract_pdf_images=True,
)

with open(
    EXTRACTION_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        extraction,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(
    f"Pages: {extraction['page_count']}"
)

print(
    f"Paragraphs: {len(extraction['paragraphs'])}"
)

print(
    f"Images: {len(extraction['images'])}"
)


# ============================================================
# STEP 2 — CREATE CHUNKS
# ============================================================

print("\n[2/4] Creating semantic chunks...")

# Adapt this call to the exact signature of your
# current chunking.py implementation.

chunks = pdf_to_embedded_chunks(
    extraction
)

print(
    f"Chunks: {len(chunks)}"
)


# ============================================================
# STEP 3 — CREATE PRESENTATION PLAN
# ============================================================

print("\n[3/4] Creating presentation plan...")

# The exact arguments should match your presentation_ai.py.

plan = create_presentation_plan(
    chunks=chunks,
    images=extraction["images"],
)


with open(
    PLAN_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        plan,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(
    f"Slides planned: {len(plan['slides'])}"
)


# ============================================================
# STEP 4 — GENERATE POWERPOINT
# ============================================================

print("\n[4/4] Generating PowerPoint...")

generate_presentation(
    plan_path=str(PLAN_PATH),
    output_path=str(PPT_PATH),
    template_path=TEMPLATE_PATH,
    image_root=str(IMAGE_DIR),
)


print("\n================================")
print("PIPELINE COMPLETE")
print("================================")
print(f"PowerPoint: {PPT_PATH}")