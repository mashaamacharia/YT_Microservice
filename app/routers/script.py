"""
routers/script.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/script/generate

Called by n8n Phase 4 after operator selects a topic via Telegram.
Generates full 5-minute video script with scenes, SEO metadata,
and complete narration string.

n8n usage:
  Method: POST
  URL:    http://localhost:8001/api/v1/script/generate
  Body:   { topic: "...", regenerate: false }

On success, n8n extracts:
  script.full_narration     → sent to ElevenLabs (Phase 5)
  script.scenes             → sent to FFmpeg assembly (Phase 8)
  script.title              → YouTube upload metadata (Phase 11)
  script.description        → YouTube upload metadata (Phase 11)
  script.tags               → YouTube upload metadata (Phase 11)
  script.chapters           → YouTube description (Phase 11)
  script.thumbnail_text     → FFmpeg thumbnail overlay (Phase 9)
  script.short_clip_scenes  → Shorts extraction (Phase 10)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from fastapi import APIRouter
from pydantic import ValidationError
from app.models.script import ScriptRequest, ScriptResponse, Script
from app.services.provider_router import route_llm_call
from app.services.prompts import (
    SCRIPT_SYSTEM,
    build_script_prompt
)
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/script", tags=["Script"])

# Maximum number of generation attempts before giving up
MAX_ATTEMPTS = 2


@router.post(
    "/generate",
    response_model=ScriptResponse,
    summary="Generate full 5-minute video script",
    description=(
        "Generates a complete production-ready script for a faceless "
        "mystery YouTube video. Returns 20-25 scenes with narration, "
        "visual keywords, mood, and clip markers. Includes full SEO metadata."
    )
)
async def generate_script(request: ScriptRequest) -> ScriptResponse:

    settings = get_settings()
    max_tokens = settings.get_max_tokens_for_task("script")

    logger.info(
        f"Script generation request — "
        f"topic: '{request.topic[:60]}' | "
        f"regenerate: {request.regenerate}"
    )

    # ── Attempt loop — retries once on validation failure ─────────
    last_error = None
    last_validation_errors = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        # On attempt 2, always treat as regenerate to force
        # a different approach from the LLM
        is_regenerate = request.regenerate or (attempt > 1)

        prompt = build_script_prompt(
            topic=request.topic,
            regenerate=is_regenerate
        )

        # Add failure context on retry so LLM knows what went wrong
        if attempt > 1 and last_validation_errors:
            failure_context = (
                "\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n"
                + "\n".join(f"- {e}" for e in last_validation_errors)
                + "\n\nFix ALL of the above issues in this attempt. "
                "Pay special attention to scene count, total duration, "
                "and clip_marker placement."
            )
            prompt += failure_context

        logger.info(f"Script attempt {attempt}/{MAX_ATTEMPTS}")

        # ── LLM call ──────────────────────────────────────────────
        result = await route_llm_call(
            prompt=prompt,
            system=SCRIPT_SYSTEM,
            task="script",
            max_tokens=max_tokens
        )

        if not result["success"]:
            logger.error(f"Script LLM call failed: {result['error']}")
            last_error = result["error"]
            continue

        # ── Parse JSON ────────────────────────────────────────────
        raw_content = result["content"]

        try:
            cleaned = _clean_json_response(raw_content)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                f"Script attempt {attempt}: invalid JSON — {str(e)}\n"
                f"Raw (first 400 chars): {raw_content[:400]}"
            )
            last_error = f"Invalid JSON on attempt {attempt}: {str(e)}"
            last_validation_errors = [
                "Response was not valid JSON. "
                "Return ONLY the raw JSON object — no markdown, "
                "no backticks, no explanation."
            ]
            continue

        # ── Validate structure via Pydantic ───────────────────────
        try:
            script = Script(**parsed)
        except ValidationError as e:
            validation_errors = _format_validation_errors(e)
            logger.warning(
                f"Script attempt {attempt}: validation failed — "
                f"{len(validation_errors)} error(s):\n"
                + "\n".join(f"  • {err}" for err in validation_errors)
            )
            last_error = f"Script validation failed on attempt {attempt}"
            last_validation_errors = validation_errors
            continue

        # ── Success ───────────────────────────────────────────────
        total_duration = sum(s.duration_seconds for s in script.scenes)

        logger.info(
            f"Script generated successfully — "
            f"{len(script.scenes)} scenes | "
            f"{total_duration}s total | "
            f"provider: {result['provider_used']} ({result['model_used']}) | "
            f"attempt: {attempt}"
        )

        return ScriptResponse(
            success=True,
            script=script,
            total_duration_seconds=total_duration,
            provider_used=result["provider_used"],
            model_used=result["model_used"]
        )

    # ── All attempts exhausted ────────────────────────────────────
    logger.error(
        f"Script generation failed after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )

    return ScriptResponse(
        success=False,
        error=(
            f"Script generation failed after {MAX_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        ),
        validation_errors=last_validation_errors
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    """
    Strips LLM response artifacts before JSON parsing.
    LLMs frequently wrap JSON in markdown code fences
    or add preamble text even when instructed not to.
    """
    cleaned = raw.strip()

    # Strip markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Find the actual JSON object boundaries
    # (handles cases where LLM adds text before/after)
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1:
        # No JSON object found at all
        return cleaned

    return cleaned[first_brace:last_brace + 1].strip()


def _format_validation_errors(e: ValidationError) -> list[str]:
    """
    Converts Pydantic ValidationError into human-readable strings
    suitable for feeding back into the LLM prompt on retry.
    """
    errors = []
    for error in e.errors():
        location = " → ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{location}: {message}")
    return errors