"""
routers/keywords.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/keywords/broaden

Called by n8n Phase 7 when Pexels returns zero results for a
scene's visual_keyword. Returns 3 alternative keywords ordered
from specific to broad for sequential retry against Pexels API.

n8n usage:
  Method: POST
  URL:    http://localhost:8001/api/v1/keywords/broaden
  Body:   { keyword: "...", mood: "eerie", scene_id: 7 }

n8n then retries Pexels with each keyword in sequence:
  keywords[0] → most specific, try first
  keywords[1] → moderate, try if [0] fails
  keywords[2] → broadest fallback, last resort
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from fastapi import APIRouter
from app.models.keywords import KeywordsRequest, KeywordsResponse
from app.services.provider_router import route_llm_call
from app.services.prompts import KEYWORDS_SYSTEM, build_keywords_prompt
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/keywords", tags=["Keywords"])


@router.post(
    "/broaden",
    response_model=KeywordsResponse,
    summary="Generate alternative stock footage keywords",
    description=(
        "Called when a scene's visual keyword returns zero results on Pexels. "
        "Returns 3 alternative keywords ordered specific → broad. "
        "n8n retries Pexels with each until a clip is found."
    )
)
async def broaden_keywords(request: KeywordsRequest) -> KeywordsResponse:

    settings = get_settings()
    max_tokens = settings.get_max_tokens_for_task("keywords")

    logger.info(
        f"Keywords request — "
        f"failed keyword: '{request.keyword}' | "
        f"mood: {request.mood} | "
        f"scene_id: {request.scene_id}"
    )

    # ── Build prompt ──────────────────────────────────────────────
    prompt = build_keywords_prompt(
        keyword=request.keyword,
        mood=request.mood
    )

    # ── Call LLM ──────────────────────────────────────────────────
    result = await route_llm_call(
        prompt=prompt,
        system=KEYWORDS_SYSTEM,
        task="keywords",
        max_tokens=max_tokens
    )

    if not result["success"]:
        logger.error(f"Keywords LLM call failed: {result['error']}")
        return KeywordsResponse(
            success=False,
            original_keyword=request.keyword,
            scene_id=request.scene_id,
            error=result["error"]
        )

    # ── Parse response ────────────────────────────────────────────
    # Keywords prompt returns a JSON array, not an object
    # e.g: ["dark forest path night", "misty woodland", "forest"]
    raw_content = result["content"]

    try:
        cleaned = _clean_array_response(raw_content)
        keywords = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            f"Keywords: invalid JSON — {str(e)}\n"
            f"Raw: {raw_content[:200]}"
        )
        # Try to extract keywords from plain text as last resort
        keywords = _extract_keywords_from_text(raw_content)
        if not keywords:
            return KeywordsResponse(
                success=False,
                original_keyword=request.keyword,
                scene_id=request.scene_id,
                error=f"Could not parse keywords from LLM response: {str(e)}"
            )

    # ── Validate we have a list of strings ───────────────────────
    if not isinstance(keywords, list):
        return KeywordsResponse(
            success=False,
            original_keyword=request.keyword,
            scene_id=request.scene_id,
            error=f"Expected JSON array, got {type(keywords).__name__}"
        )

    # Clean and filter empty strings
    keywords = [str(k).strip() for k in keywords if str(k).strip()]

    # Ensure exactly 3 — pad with broader fallbacks if needed
    keywords = _ensure_three_keywords(
        keywords=keywords,
        original=request.keyword,
        mood=request.mood
    )

    logger.info(
        f"Keywords generated — "
        f"alternatives: {keywords} | "
        f"provider: {result['provider_used']}"
    )

    return KeywordsResponse(
        success=True,
        keywords=keywords,
        original_keyword=request.keyword,
        scene_id=request.scene_id,
        provider_used=result["provider_used"],
        model_used=result["model_used"]
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_array_response(raw: str) -> str:
    """
    Strips LLM artifacts before parsing a JSON array response.
    Keywords prompt returns [...] not {...} so we look for [ not {
    """
    cleaned = raw.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Find array boundaries
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    if first_bracket == -1 or last_bracket == -1:
        # No array found — return as-is and let caller handle
        return cleaned

    return cleaned[first_bracket:last_bracket + 1].strip()


def _extract_keywords_from_text(raw: str) -> list[str]:
    """
    Last-resort extraction when LLM returns plain text instead of JSON.
    Tries to extract quoted strings or line-separated keywords.

    e.g. "Try: 'dark forest', 'misty trees', 'woodland path'"
    → ["dark forest", "misty trees", "woodland path"]
    """
    import re

    # Try to find quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', raw)
    if len(quoted) >= 2:
        return quoted[:3]

    # Try line-by-line (LLM sometimes returns numbered list)
    lines = [
        re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
        for line in raw.split('\n')
        if line.strip() and len(line.strip()) > 3
    ]
    if len(lines) >= 2:
        return lines[:3]

    return []


def _ensure_three_keywords(
    keywords: list[str],
    original: str,
    mood: str
) -> list[str]:
    """
    Guarantees exactly 3 keywords are returned.
    If LLM returned fewer, pads with mood-based generic fallbacks.
    These are last-resort options that will almost always find Pexels results.
    """
    mood_fallbacks = {
        "tense":       ["dark city street night", "empty corridor shadows", "dramatic clouds"],
        "eerie":       ["fog misty landscape", "abandoned building exterior", "dark forest"],
        "dramatic":    ["dramatic sky clouds", "cinematic landscape wide", "dramatic lighting"],
        "calm":        ["peaceful nature landscape", "calm water reflection", "sunset horizon"],
        "shocking":    ["breaking news screen", "dramatic reveal moment", "crowd reaction"],
        "haunting":    ["empty room window light", "old photograph texture", "misty graveyard"],
        "suspenseful": ["silhouette dark background", "door opening darkness", "shadows wall"],
        "urgent":      ["running feet pavement", "flashing lights police", "city traffic night"]
    }

    fallbacks = mood_fallbacks.get(mood, ["dark atmospheric scene", "cinematic footage", "dramatic landscape"])

    # Fill up to 3
    result = keywords[:3]
    for fallback in fallbacks:
        if len(result) >= 3:
            break
        if fallback not in result:
            result.append(fallback)

    return result[:3]