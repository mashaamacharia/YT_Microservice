"""
models/research.py
─────────────────────────────────────────────────────────────────────────────
Pydantic models for the /api/v1/research/aggregate endpoint.

Request:  Receives YouTube + Reddit posts from n8n
Response: Returns 5 ranked video ideas with full metadata
─────────────────────────────────────────────────────────────────────────────
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


# ─── Request models ───────────────────────────────────────────────────────────

class YouTubePost(BaseModel):
    """A single trending YouTube video from Phase 2 research."""
    title: str
    description: Optional[str] = ""
    channel: Optional[str] = ""
    published: Optional[str] = ""
    url: Optional[str] = ""
    source: Optional[str] = "youtube"


class RedditPost(BaseModel):
    """A single trending Reddit post from Phase 2 research."""
    title: str
    description: Optional[str] = ""
    url: Optional[str] = ""
    subreddit: Optional[str] = ""
    engagement_score: Optional[int] = 0
    source: Optional[str] = "reddit"


class ResearchRequest(BaseModel):
    """
    Payload sent by n8n to /api/v1/research/aggregate.

    n8n sends the combined output of the YouTube and Reddit
    HTTP Request nodes as a single payload.
    """
    youtube_posts: list[YouTubePost] = Field(
        default=[],
        description="Trending YouTube videos from Phase 2 YouTube node"
    )
    reddit_posts: list[RedditPost] = Field(
        default=[],
        description="Trending Reddit posts from Phase 2 Reddit node"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "youtube_posts": [
                    {
                        "title": "The Man Who Disappeared From a Locked Room",
                        "channel": "True Crime Daily",
                        "url": "https://youtube.com/watch?v=xxx"
                    }
                ],
                "reddit_posts": [
                    {
                        "title": "20 years ago a woman went hiking. Never came back.",
                        "subreddit": "UnresolvedMysteries",
                        "engagement_score": 4500
                    }
                ]
            }
        }
    }


# ─── Response models ──────────────────────────────────────────────────────────

class VideoIdea(BaseModel):
    """A single generated video idea from the LLM."""
    id: int = Field(description="Rank 1-5, where 1 = highest viral potential")
    title: str = Field(description="Curiosity-gap YouTube title, max 60 chars")
    hook_sentence: str = Field(
        description="The single most shocking sentence. Opens the video."
    )
    topic_summary: str = Field(description="2-3 sentence story context")
    trending_reason: str = Field(description="Why this resonates right now")
    search_volume: Literal["high", "medium", "low"]
    unique_angle: str = Field(description="What makes this take different")
    content_type: Literal[
        "true_crime", "dark_history", "conspiracy",
        "survival", "exposé", "unsolved"
    ]
    estimated_watch_time_minutes: Optional[int] = 5
    source: Literal["youtube", "reddit", "combined"]


class ResearchResponse(BaseModel):
    """
    Response returned to n8n from /api/v1/research/aggregate.
    n8n passes this to the Telegram node to show you the 5 ideas.
    """
    success: bool
    ideas: list[VideoIdea] = Field(default=[])
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None