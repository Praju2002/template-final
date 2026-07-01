"""
HTTP wrapper around the DocuFind confidence-scoring pipeline.

Run from this directory with:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000

POST an image and query word to /detect.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np

import preProcessing as pp
import wordExtract as we
import templateOperation as tempOp


app = FastAPI(title="DocuFind Confidence API")


def _box_to_dict(box):
    """Convert a matcher box to the JSON shape exposed by this API."""
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        left, top, right, bottom = box
    else:
        left, top = box[0]
        right, bottom = box[1]

    return {
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(
    image_file: UploadFile = File(...),
    query_word: str = Form(...),
    min_confidence: float = Form(0.0),
):
    """Detect query-word matches and return their confidence metadata."""
    raw_bytes = await image_file.read()
    npimg = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode uploaded image."},
        )

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = pp.backgroundBlackForegroundWhite(image=grey_image)
    average_height, tallest_text_coords = pp.averageHeightOfLetters(image=binary_image)

    if (
        average_height <= 0
        or not isinstance(tallest_text_coords, tuple)
        or len(tallest_text_coords) != 2
        or tallest_text_coords[1] <= tallest_text_coords[0]
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Could not detect valid text lines in the image."},
        )

    tallest_line_image = binary_image[tallest_text_coords[0]:tallest_text_coords[1], :]
    average_gap = pp.averageGapOfLetter(image=tallest_line_image)
    iterations = pp.requiredNumberOfIterations(
        averageGap=average_gap,
        is_nepali_text=tempOp.is_nepali(query_word),
    )
    smudged = pp.prepareImageForWordExtraction(
        image=binary_image,
        iteration=iterations,
    )

    words_property = we.wordExtract(
        image=smudged,
        averageHeight=average_height,
        smudgedIteration=iterations,
    )
    if not words_property:
        return {
            "query_word": query_word,
            "total_candidates": 0,
            "matches_found": 0,
            "results": [],
        }

    template = tempOp.createTemplate(query_word, fontSize=int(average_height))
    if template is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Failed to create template for query word."},
        )
    print(f"average_height={average_height}, tallest_text_coords={tallest_text_coords}")
    print(f"average_gap={average_gap}, iterations={iterations}")
    print(f"template shape={template.shape}")
    matches = tempOp.templateMatching(
        image=binary_image,
        template=template,
        wordsProperty=words_property,
        matching_mode="auto",
    )
    filtered = [
        result for result in matches
        if result["confidence"] >= min_confidence
    ]

    return {
        "query_word": query_word,
        "total_candidates": len(words_property),
        "matches_found": len(filtered),
        "results": [
            {
                "box": _box_to_dict(result["box"]),
                "score": result["score"],
                "stage": result["stage"],
                "confidence": result["confidence"],
                "needs_review": result["confidence"] < 0.7,
            }
            for result in filtered
        ],
    }
