"""
routers/research.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/research/aggregate

Called by n8n Phase 2 after collecting YouTube + Reddit trending data.
Sends combined data to LLM and returns 5 ranked video ideas.

n8n usage:
  Method: POST
  URL:    http://localhost:8001/api/v1/research/aggregate
  Body:   { youtube_posts: [...], reddit_posts: [...] }
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from fastapi import APIRouter
from app.models.research import ResearchRequest, ResearchResponse, VideoIdea
from app.services.provider_router import route_llm_call
from app.services.prompts import (
    RESEARCH_SYSTEM,
    build_research_prompt
)
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/research", tags=["Research"])


@router.post(
    "/aggregate",
    response_model=ResearchResponse,
    summary="Aggregate trending data into 5 video ideas",
    description=(
        "Receives YouTube and Reddit trending posts from n8n Phase 2. "
        "Sends combined data to LLM. Returns 5 ranked video ideas "
        "with titles, hooks, angles, and content type classification."
    )
)
async def aggregate_research(request: ResearchRequest) -> ResearchResponse:

    settings = get_settings()

    # ── Guard: need at least some data to work with ───────────────
    total_posts = len(request.youtube_posts) + len(request.reddit_posts)
    if total_posts == 0:
        logger.warning("Research request received with zero posts from both sources")
        return ResearchResponse(
            success=False,
            error=(
                "No posts received from YouTube or Reddit. "
                "Check that at least one research source returned data."
            )
        )

    logger.info(
        f"Research request: {len(request.youtube_posts)} YouTube posts, "
        f"{len(request.reddit_posts)} Reddit posts"
    )

    # ── Build prompt ──────────────────────────────────────────────
    prompt = build_research_prompt(
        youtube_posts=[p.model_dump() for p in request.youtube_posts],
        reddit_posts=[p.model_dump() for p in request.reddit_posts]
    )

    max_tokens = settings.get_max_tokens_for_task("research")

    # ── Call LLM via provider router ──────────────────────────────
    result = await route_llm_call(
        prompt=prompt,
        system=RESEARCH_SYSTEM,
        task="research",
        max_tokens=max_tokens
    )

    if not result["success"]:
        logger.error(f"Research LLM call failed: {result['error']}")
        return ResearchResponse(
            success=False,
            error=result["error"]
        )

    # ── Parse LLM response ────────────────────────────────────────
    raw_content = result["content"]

    try:
        cleaned = _clean_json_response(raw_content)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            f"Research: LLM returned invalid JSON — {str(e)}\n"
            f"Raw content (first 300 chars): {raw_content[:300]}"
        )
        # Attempt retry with stricter prompt
        logger.info("Research: retrying with strict JSON-only instruction")
        strict_prompt = prompt + (
            "\n\nCRITICAL: Your previous response could not be parsed as JSON. "
            "Return ONLY the raw JSON object. "
            "No markdown. No backticks. No explanation. Start with { and end with }."
        )
        retry_result = await route_llm_call(
            prompt=strict_prompt,
            system=RESEARCH_SYSTEM,
            task="research",
            max_tokens=max_tokens
        )
        if not retry_result["success"]:
            return ResearchResponse(
                success=False,
                error="LLM returned invalid JSON on both attempts. "
                      "Check provider logs for raw output."
            )
        try:
            cleaned = _clean_json_response(retry_result["content"])
            parsed = json.loads(cleaned)
            result = retry_result  # use retry metadata
        except json.JSONDecodeError as retry_e:
            return ResearchResponse(
                success=False,
                error=f"LLM returned invalid JSON after retry: {str(retry_e)}"
            )

    # ── Validate and build ideas list ─────────────────────────────
    raw_ideas = parsed.get("ideas", [])

    if not raw_ideas:
        return ResearchResponse(
            success=False,
            error="LLM returned empty ideas list. "
                  "Verify research prompt is reaching the model correctly."
        )

    ideas = []
    for raw in raw_ideas:
        try:
            idea = VideoIdea(**raw)
            ideas.append(idea)
        except Exception as e:
            # Skip malformed ideas but log them
            logger.warning(f"Skipping malformed idea: {str(e)} — raw: {raw}")
            continue

    if not ideas:
        return ResearchResponse(
            success=False,
            error="All ideas failed validation. LLM output did not match expected structure."
        )

    # Sort by id to ensure correct ranking order
    ideas.sort(key=lambda x: x.id)

    logger.info(
        f"Research complete: {len(ideas)} ideas generated "
        f"via {result['provider_used']} ({result['model_used']})"
    )

    return ResearchResponse(
        success=True,
        ideas=ideas,
        provider_used=result["provider_used"],
        model_used=result["model_used"]
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    """
    Strips common LLM response artifacts before JSON parsing.
    Handles:
      - ```json ... ``` code fences
      - ``` ... ``` plain fences
      - Leading/trailing whitespace
      - Occasional "Here is the JSON:" preamble lines
    """
    cleaned = raw.strip()

    # Remove markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Remove preamble lines before the first {
    first_brace = cleaned.find("{")
    if first_brace > 0:
        cleaned = cleaned[first_brace:]

    # Remove anything after the last }
    last_brace = cleaned.rfind("}")
    if last_brace != -1 and last_brace < len(cleaned) - 1:
        cleaned = cleaned[:last_brace + 1]

    return cleaned.strip()