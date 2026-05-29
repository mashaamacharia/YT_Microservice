"""
routers/analytics.py
─────────────────────────────────────────────────────────────────────────────
POST /api/v1/analytics/interpret

Called by n8n Phase 13, 48 hours after a video is published.
Receives YouTube Analytics API metrics and returns a performance
grade, diagnosis, and 3 actionable recommendations for next video.

n8n usage:
  Method: POST
  URL:    http://localhost:8001/api/v1/analytics/interpret
  Body:   {
    title, views, retention, ctr,
    watch_time, subscribers, youtube_video_id
  }
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from fastapi import APIRouter
from app.models.analytics import (
    AnalyticsRequest,
    AnalyticsResponse,
    PerformanceDiagnosis,
    Recommendation
)
from app.services.provider_router import route_llm_call
from app.services.prompts import ANALYTICS_SYSTEM, build_analytics_prompt
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.post(
    "/interpret",
    response_model=AnalyticsResponse,
    summary="Interpret 48-hour YouTube performance metrics",
    description=(
        "Receives early YouTube Analytics data and returns a performance "
        "grade with diagnosis across CTR, hook strength, retention, and "
        "subscriber conversion. Includes 3 prioritized recommendations "
        "for the next video."
    )
)
async def interpret_analytics(request: AnalyticsRequest) -> AnalyticsResponse:

    settings = get_settings()
    max_tokens = settings.get_max_tokens_for_task("analytics")

    logger.info(
        f"Analytics request — "
        f"title: '{request.title[:50]}' | "
        f"views: {request.views} | "
        f"retention: {request.retention}% | "
        f"CTR: {request.ctr}%"
    )

    # ── Build prompt ──────────────────────────────────────────────
    prompt = build_analytics_prompt(
        title=request.title,
        views=request.views,
        retention=request.retention,
        ctr=request.ctr,
        watch_time=request.watch_time,
        subscribers=request.subscribers
    )

    # ── Call LLM ──────────────────────────────────────────────────
    result = await route_llm_call(
        prompt=prompt,
        system=ANALYTICS_SYSTEM,
        task="analytics",
        max_tokens=max_tokens
    )

    if not result["success"]:
        logger.error(f"Analytics LLM call failed: {result['error']}")
        return AnalyticsResponse(
            success=False,
            error=result["error"]
        )

    # ── Parse response ────────────────────────────────────────────
    raw_content = result["content"]

    try:
        cleaned = _clean_json_response(raw_content)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            f"Analytics: invalid JSON — {str(e)}\n"
            f"Raw (first 300 chars): {raw_content[:300]}"
        )
        # Analytics is non-critical — return partial response
        # with raw content rather than failing completely
        return AnalyticsResponse(
            success=False,
            error=(
                f"LLM returned invalid JSON: {str(e)}. "
                f"Raw response logged for manual review."
            )
        )

    # ── Build response objects ────────────────────────────────────
    try:
        diagnosis = PerformanceDiagnosis(
            **parsed.get("diagnosis", {})
        )
    except Exception as e:
        logger.warning(f"Analytics: diagnosis parsing failed — {str(e)}")
        diagnosis = None

    recommendations = []
    for raw_rec in parsed.get("recommendations", []):
        try:
            rec = Recommendation(**raw_rec)
            recommendations.append(rec)
        except Exception as e:
            logger.warning(f"Analytics: recommendation skipped — {str(e)}")
            continue

    # Sort recommendations by priority
    recommendations.sort(key=lambda r: r.priority)

    logger.info(
        f"Analytics complete — "
        f"grade: {parsed.get('performance_grade', 'N/A')} | "
        f"recommendations: {len(recommendations)} | "
        f"provider: {result['provider_used']}"
    )

    return AnalyticsResponse(
        success=True,
        performance_grade=parsed.get("performance_grade"),
        summary=parsed.get("summary"),
        diagnosis=diagnosis,
        recommendations=recommendations if recommendations else None,
        next_video_brief=parsed.get("next_video_brief"),
        provider_used=result["provider_used"],
        model_used=result["model_used"]
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    """Strip LLM response artifacts before JSON parsing."""
    cleaned = raw.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1:
        return cleaned

    return cleaned[first_brace:last_brace + 1].strip()