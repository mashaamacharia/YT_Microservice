"""
models/keywords.py
─────────────────────────────────────────────────────────────────────────────
Pydantic models for the /api/v1/keywords/broaden endpoint.

Request:  Receives a failed stock footage keyword + scene mood
Response: Returns 3 alternative keywords ordered specific → broad
─────────────────────────────────────────────────────────────────────────────
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


# ─── Request models ───────────────────────────────────────────────────────────

class KeywordsRequest(BaseModel):
    """
    Payload sent by n8n to /api/v1/keywords/broaden.
    Called in Phase 7 when Pexels returns 0 results for a scene keyword.
    """
    keyword: str = Field(
        description="The original visual keyword that returned no Pexels results"
    )
    mood: Literal[
        "tense", "eerie", "dramatic", "calm",
        "shocking", "haunting", "suspenseful", "urgent"
    ] = Field(
        description="The scene mood — used to keep alternatives atmospherically consistent"
    )
    scene_id: Optional[int] = Field(
        default=None,
        description="Scene ID for logging — helps trace which scene had footage issues"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "keyword": "underwater cave diver with torch 1970s",
                "mood": "eerie",
                "scene_id": 7
            }
        }
    }


# ─── Response models ──────────────────────────────────────────────────────────

class KeywordsResponse(BaseModel):
    """
    Response returned to n8n from /api/v1/keywords/broaden.
    n8n tries each keyword against Pexels in sequence.
    Keyword 1 is most specific, keyword 3 is broadest last-resort fallback.
    """
    success: bool
    keywords: Optional[list[str]] = Field(
        default=None,
        description="Exactly 3 alternative keywords. "
                    "Index 0 = most specific. Index 2 = broadest fallback."
    )
    original_keyword: Optional[str] = Field(
        default=None,
        description="The original keyword that failed — echoed back for logging"
    )
    scene_id: Optional[int] = Field(
        default=None,
        description="Echoed back for n8n to match response to the correct scene"
    )
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_keywords_count(self) -> "KeywordsResponse":
        """Ensure exactly 3 keywords are returned on success."""
        if self.success and self.keywords is not None:
            if len(self.keywords) != 3:
                raise ValueError(
                    f"keywords list must contain exactly 3 items, "
                    f"got {len(self.keywords)}"
                )
            # Strip whitespace from each keyword
            self.keywords = [k.strip() for k in self.keywords if k.strip()]
            if len(self.keywords) != 3:
                raise ValueError(
                    "One or more keywords were empty after stripping whitespace."
                )
        return self