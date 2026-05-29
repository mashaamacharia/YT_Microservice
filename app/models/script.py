"""
models/script.py
─────────────────────────────────────────────────────────────────────────────
Pydantic models for the /api/v1/script/generate endpoint.

Request:  Receives chosen topic + regenerate flag from n8n
Response: Returns full validated script with scenes, SEO metadata,
          and full narration string ready for ElevenLabs
─────────────────────────────────────────────────────────────────────────────
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


# ─── Request models ───────────────────────────────────────────────────────────

class ScriptRequest(BaseModel):
    """
    Payload sent by n8n to /api/v1/script/generate.
    Comes from the Telegram Wait node after operator picks a topic.
    """
    topic: str = Field(
        description="The chosen video topic. Can be a title from research "
                    "ideas or a custom topic typed by the operator."
    )
    regenerate: bool = Field(
        default=False,
        description="If True, a previous script for this topic was rejected. "
                    "LLM will take a completely different approach."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "The Seattle mother and daughter found shot dead "
                         "after a hike — 20 years, no arrests",
                "regenerate": False
            }
        }
    }


# ─── Response models ──────────────────────────────────────────────────────────

class Scene(BaseModel):
    """
    A single scene in the video script.
    Maps directly to one stock footage clip + narration segment.
    """
    scene_id: int = Field(description="Sequential scene number starting at 1")
    narration: str = Field(
        description="Full narration text for this scene. "
                    "Sent to ElevenLabs as part of full_narration."
    )
    duration_seconds: int = Field(
        ge=5, le=25,
        description="How long this scene plays. "
                    "Must match the narration length when spoken aloud."
    )
    visual_keyword: str = Field(
        description="Hyper-specific stock footage search keyword. "
                    "e.g. 'abandoned hospital corridor flickering light' "
                    "not just 'hospital'."
    )
    mood: Literal[
        "tense", "eerie", "dramatic", "calm",
        "shocking", "haunting", "suspenseful", "urgent"
    ]
    is_hook: bool = Field(
        default=False,
        description="True only for scene_id 1 — the opening hook scene."
    )
    clip_marker: Optional[Literal["HOOK", "CLIMAX", "ENDING"]] = Field(
        default=None,
        description="Special marker for Shorts extraction. "
                    "HOOK=scene 1, CLIMAX=most dramatic, ENDING=final scene. "
                    "null for all other scenes."
    )


class Script(BaseModel):
    """
    The complete generated video script.
    Validated before being returned to n8n.
    """
    title: str = Field(
        max_length=60,
        description="Curiosity-gap YouTube title. Max 60 characters."
    )
    description: str = Field(
        description="SEO-rich 150-word video description. "
                    "Naturally includes keywords. Ends with a question."
    )
    tags: list[str] = Field(
        min_length=5,
        max_length=15,
        description="YouTube tags. Mix of broad and specific."
    )
    thumbnail_text: str = Field(
        description="Max 4 words for thumbnail text overlay. "
                    "High contrast, instant curiosity."
    )
    chapters: str = Field(
        description="YouTube chapters string. "
                    "Format: '0:00 Introduction\\n0:30 Chapter Name\\n...'"
    )
    scenes: list[Scene] = Field(
        min_length=15,
        description="Individual scenes. Minimum 15, ideally 20-25."
    )
    full_narration: str = Field(
        description="All scene narrations joined as one continuous string. "
                    "This is what gets sent to ElevenLabs TTS."
    )
    short_clip_scenes: list[int] = Field(
        description="Exactly 3 scene_ids: [HOOK_id, CLIMAX_id, ENDING_id]. "
                    "Used for Shorts extraction in Phase 10."
    )
    estimated_total_seconds: Optional[int] = Field(
        default=None,
        description="Sum of all scene durations. Should be 270-330 seconds."
    )

    @model_validator(mode="after")
    def validate_script_integrity(self) -> "Script":
        """
        Runs after model creation to catch structural problems
        that field-level validators cannot catch.
        """
        scenes = self.scenes
        total_duration = sum(s.duration_seconds for s in scenes)

        # Duration check
        if total_duration < 240:
            raise ValueError(
                f"Script too short: {total_duration}s total. "
                f"Minimum is 240s (4 minutes). "
                f"Check scene durations or add more scenes."
            )
        if total_duration > 420:
            raise ValueError(
                f"Script too long: {total_duration}s total. "
                f"Maximum is 420s (7 minutes). "
                f"Trim scenes or reduce count."
            )

        # Hook check — scene 1 must be the hook
        scene_1 = next((s for s in scenes if s.scene_id == 1), None)
        if not scene_1:
            raise ValueError("Scene with scene_id=1 is missing.")
        if not scene_1.is_hook:
            raise ValueError(
                "Scene 1 must have is_hook=True. "
                "The first scene is always the hook."
            )
        if scene_1.clip_marker != "HOOK":
            raise ValueError(
                "Scene 1 must have clip_marker='HOOK'."
            )

        # Clip markers check — exactly one HOOK, CLIMAX, ENDING
        hook_scenes = [s for s in scenes if s.clip_marker == "HOOK"]
        climax_scenes = [s for s in scenes if s.clip_marker == "CLIMAX"]
        ending_scenes = [s for s in scenes if s.clip_marker == "ENDING"]

        if len(hook_scenes) != 1:
            raise ValueError(
                f"Exactly 1 HOOK scene required, found {len(hook_scenes)}."
            )
        if len(climax_scenes) != 1:
            raise ValueError(
                f"Exactly 1 CLIMAX scene required, found {len(climax_scenes)}."
            )
        if len(ending_scenes) != 1:
            raise ValueError(
                f"Exactly 1 ENDING scene required, found {len(ending_scenes)}."
            )

        # short_clip_scenes must contain exactly 3 valid scene_ids
        if len(self.short_clip_scenes) != 3:
            raise ValueError(
                f"short_clip_scenes must contain exactly 3 scene_ids, "
                f"found {len(self.short_clip_scenes)}."
            )

        valid_scene_ids = {s.scene_id for s in scenes}
        for sid in self.short_clip_scenes:
            if sid not in valid_scene_ids:
                raise ValueError(
                    f"short_clip_scenes references scene_id={sid} "
                    f"which does not exist in scenes list."
                )

        # full_narration must be long enough to be a real script
        if len(self.full_narration.strip()) < 500:
            raise ValueError(
                f"full_narration is too short ({len(self.full_narration)} chars). "
                f"Expected at least 500 characters for a 5-minute script."
            )

        # Set calculated total if not provided
        if self.estimated_total_seconds is None:
            self.estimated_total_seconds = total_duration

        return self


class ScriptResponse(BaseModel):
    """
    Response returned to n8n from /api/v1/script/generate.
    n8n passes script.full_narration to ElevenLabs in Phase 5.
    n8n passes script.scenes to FFmpeg assembly in Phase 8.
    """
    success: bool
    script: Optional[Script] = None
    total_duration_seconds: Optional[int] = None
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    validation_errors: Optional[list[str]] = Field(
        default=None,
        description="List of validation failures if script structure was wrong. "
                    "Used to decide whether to retry generation."
    )