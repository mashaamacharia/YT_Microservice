"""
models/analytics.py
─────────────────────────────────────────────────────────────────────────────
Pydantic models for the /api/v1/analytics/interpret endpoint.

Request:  Receives YouTube 48-hour metrics from n8n
Response: Returns performance grade, diagnosis, and 3 actionable
          recommendations for the next video
─────────────────────────────────────────────────────────────────────────────
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


# ─── Request models ───────────────────────────────────────────────────────────

class AnalyticsRequest(BaseModel):
    """
    Payload sent by n8n to /api/v1/analytics/interpret.
    Data comes from YouTube Analytics API in Phase 13.
    """
    title: str = Field(description="YouTube video title")
    views: int = Field(ge=0, description="Total views at 48 hours")
    retention: float = Field(
        ge=0.0, le=100.0,
        description="Average view percentage (0-100)"
    )
    ctr: float = Field(
        ge=0.0, le=100.0,
        description="Click-through rate percentage (0-100)"
    )
    watch_time: float = Field(
        ge=0.0,
        description="Total watch time in minutes"
    )
    subscribers: int = Field(
        ge=0,
        description="New subscribers gained from this video"
    )
    youtube_video_id: Optional[str] = Field(
        default=None,
        description="YouTube video ID for reference in logs"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "They Found the Door Locked From the Inside...",
                "views": 1240,
                "retention": 52.3,
                "ctr": 4.8,
                "watch_time": 1085.5,
                "subscribers": 14,
                "youtube_video_id": "dQw4w9WgXcQ"
            }
        }
    }


# ─── Response models ──────────────────────────────────────────────────────────

class PerformanceDiagnosis(BaseModel):
    """Breakdown of performance across the four key YouTube dimensions."""
    ctr_verdict: str = Field(
        description="Assessment of thumbnail + title performance"
    )
    hook_verdict: str = Field(
        description="Assessment of first 30-second retention"
    )
    retention_verdict: str = Field(
        description="Assessment of overall content body retention"
    )
    conversion_verdict: str = Field(
        description="Assessment of subscriber conversion rate"
    )


class Recommendation(BaseModel):
    """A single actionable recommendation for the next video."""
    priority: Literal[1, 2, 3] = Field(
        description="1 = most important fix, 3 = lowest priority"
    )
    area: Literal[
        "thumbnail", "title", "hook",
        "pacing", "ending", "visuals"
    ]
    problem: str = Field(description="Specific problem identified from data")
    fix: str = Field(
        description="Exact change to make in the next video. "
                    "Specific enough for a scriptwriter to act on immediately."
    )


class AnalyticsResponse(BaseModel):
    """
    Response returned to n8n from /api/v1/analytics/interpret.
    n8n formats this into a Telegram message in Phase 13.
    Also logged to Google Sheets / Supabase for trend tracking.
    """
    success: bool
    performance_grade: Optional[Literal["A", "B", "C", "D", "F"]] = None
    summary: Optional[str] = Field(
        default=None,
        description="One sentence verdict on overall performance"
    )
    diagnosis: Optional[PerformanceDiagnosis] = None
    recommendations: Optional[list[Recommendation]] = Field(
        default=None,
        description="Exactly 3 prioritized recommendations"
    )
    next_video_brief: Optional[str] = Field(
        default=None,
        description="One paragraph briefing for the scriptwriter "
                    "on what to do differently next time"
    )
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None