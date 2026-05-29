"""
provider_router.py
─────────────────────────────────────────────────────────────────────────────
Core LLM routing logic for the YouTube Pipeline.

Each provider has a completely different API structure:

  Anthropic  → system is a top-level field, NOT inside messages[]
               messages: [{ role: "user", content: "..." }]
               response: content[0].text

  OpenAI     → system is the first message with role="system"
               messages: [{ role: "system", content: "..." },
                          { role: "user", content: "..." }]
               response: choices[0].message.content

  Gemini     → system is system_instruction.parts[0].text
               contents: [{ parts: [{ text: "..." }] }]
               response: candidates[0].content.parts[0].text
               uses query param for auth (not header)

This module handles all those differences internally.
Every caller receives and returns the same clean interface.
─────────────────────────────────────────────────────────────────────────────
"""

import httpx
import logging
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Provider registry ────────────────────────────────────────────────────────
# Defines known providers and which .env key holds their API credential.
# To add a new provider: add entry here + implement its caller function below.
PROVIDER_REGISTRY = {
    "anthropic": {
        "env_key": "anthropic_api_key",
        "label": "Anthropic (Claude)"
    },
    "openai": {
        "env_key": "openai_api_key",
        "label": "OpenAI (GPT)"
    },
    "gemini": {
        "env_key": "gemini_api_key",
        "label": "Google (Gemini)"
    }
}


# ─── Availability check ───────────────────────────────────────────────────────

def get_available_providers(task: str) -> list[dict]:
    """
    Returns only providers that have:
      1. A non-empty API key in .env
      2. A model string defined for this specific task

    Ordered by LLM_PROVIDER_ORDER from .env.
    Providers without a key are silently skipped — no network call made.

    Args:
        task: One of "research" | "script" | "analytics" | "keywords"

    Returns:
        List of dicts: [{ name, api_key, model, label }, ...]
    """
    settings = get_settings()
    available = []

    for provider_name in settings.provider_order:
        config = PROVIDER_REGISTRY.get(provider_name)
        if not config:
            logger.warning(
                f"Unknown provider '{provider_name}' in LLM_PROVIDER_ORDER — skipping"
            )
            continue

        # Check API key exists
        api_key = settings.get_api_key_for_provider(provider_name)
        if not api_key:
            logger.debug(f"Provider '{provider_name}' skipped — no API key configured")
            continue

        # Check model is defined for this task
        model = settings.get_model_for_task(task, provider_name)
        if not model:
            logger.warning(
                f"Provider '{provider_name}' skipped for task '{task}' — "
                f"no model defined. Add "
                f"{task.upper()}_MODEL_{provider_name.upper()} to .env"
            )
            continue

        available.append({
            "name": provider_name,
            "api_key": api_key,
            "model": model,
            "label": config["label"]
        })

    return available


# ─── Anthropic caller ─────────────────────────────────────────────────────────

async def call_anthropic(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int
) -> Optional[str]:
    """
    Calls the Anthropic Messages API.

    Payload structure:
    {
      "model": "claude-sonnet-4-6",
      "max_tokens": 4000,
      "system": "...",          ← top-level field, NOT in messages
      "messages": [
        { "role": "user", "content": "..." }   ← user only, no system here
      ]
    }

    Auth: x-api-key header
    Response path: response.content[0].text
    """
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                # Anthropic returns: { content: [{ type: "text", text: "..." }] }
                content_blocks = data.get("content", [])
                text_blocks = [b for b in content_blocks if b.get("type") == "text"]
                if text_blocks:
                    return text_blocks[0]["text"]
                logger.error(f"Anthropic: no text block in response — {data}")
                return None

            elif response.status_code == 429:
                logger.warning(f"Anthropic: rate limit hit (429)")
                return None

            elif response.status_code == 401:
                logger.error(f"Anthropic: invalid API key (401)")
                return None

            elif response.status_code == 529:
                logger.warning(f"Anthropic: API overloaded (529)")
                return None

            else:
                logger.error(
                    f"Anthropic: unexpected status {response.status_code} "
                    f"— {response.text[:200]}"
                )
                return None

    except httpx.TimeoutException:
        logger.warning(f"Anthropic: request timed out after {timeout}s")
        return None
    except httpx.ConnectError:
        logger.error("Anthropic: connection failed — check network")
        return None
    except Exception as e:
        logger.error(f"Anthropic: unexpected error — {str(e)}")
        return None


# ─── OpenAI caller ────────────────────────────────────────────────────────────

async def call_openai(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int
) -> Optional[str]:
    """
    Calls the OpenAI Chat Completions API.

    Payload structure:
    {
      "model": "gpt-4o",
      "max_tokens": 4000,
      "messages": [
        { "role": "system", "content": "..." },   ← system goes INSIDE messages
        { "role": "user",   "content": "..." }
      ]
    }

    Auth: Authorization: Bearer {key} header
    Response path: response.choices[0].message.content
    """
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",    # system is a message with role="system"
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                # OpenAI returns: { choices: [{ message: { content: "..." } }] }
                choices = data.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"]
                logger.error(f"OpenAI: no choices in response — {data}")
                return None

            elif response.status_code == 429:
                logger.warning("OpenAI: rate limit or quota exceeded (429)")
                return None

            elif response.status_code == 401:
                logger.error("OpenAI: invalid API key (401)")
                return None

            elif response.status_code == 503:
                logger.warning("OpenAI: service unavailable (503)")
                return None

            else:
                logger.error(
                    f"OpenAI: unexpected status {response.status_code} "
                    f"— {response.text[:200]}"
                )
                return None

    except httpx.TimeoutException:
        logger.warning(f"OpenAI: request timed out after {timeout}s")
        return None
    except httpx.ConnectError:
        logger.error("OpenAI: connection failed — check network")
        return None
    except Exception as e:
        logger.error(f"OpenAI: unexpected error — {str(e)}")
        return None


# ─── Gemini caller ────────────────────────────────────────────────────────────

async def call_gemini(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int
) -> Optional[str]:
    """
    Calls the Google Gemini generateContent API.

    Payload structure:
    {
      "system_instruction": {           ← system is a nested object
        "parts": [{ "text": "..." }]
      },
      "contents": [
        {
          "parts": [{ "text": "..." }]  ← user message — no "role" needed
        }
      ],
      "generationConfig": {
        "maxOutputTokens": 4000
      }
    }

    Auth: ?key={api_key} query parameter (NOT a header)
    Response path: response.candidates[0].content.parts[0].text
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent"
    )

    # Gemini uses query param for auth, not Authorization header
    params = {"key": api_key}

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "system_instruction": {
            "parts": [
                {"text": system}
            ]
        },
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
                # No "role" field needed for single-turn requests
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,       # balanced creativity vs consistency
            "topP": 0.9
        },
        "safetySettings": [
            # Relax safety filters for true crime / mystery content
            # These block legitimate factual content about crimes/history
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, headers=headers, params=params, json=payload
            )

            if response.status_code == 200:
                data = response.json()

                # Gemini returns:
                # { candidates: [{ content: { parts: [{ text: "..." }] } }] }
                candidates = data.get("candidates", [])
                if not candidates:
                    # Check if it was blocked by safety filters
                    prompt_feedback = data.get("promptFeedback", {})
                    block_reason = prompt_feedback.get("blockReason")
                    if block_reason:
                        logger.warning(
                            f"Gemini: response blocked by safety filter — "
                            f"reason: {block_reason}"
                        )
                    else:
                        logger.error(f"Gemini: no candidates in response — {data}")
                    return None

                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text")

                logger.error(f"Gemini: no text parts in candidate — {data}")
                return None

            elif response.status_code == 429:
                logger.warning("Gemini: quota exceeded (429)")
                return None

            elif response.status_code == 400:
                logger.error(
                    f"Gemini: bad request (400) — "
                    f"possibly invalid model name '{model}' "
                    f"— {response.text[:200]}"
                )
                return None

            elif response.status_code == 403:
                logger.error(
                    "Gemini: API key invalid or API not enabled (403). "
                    "Enable Generative Language API in Google Cloud Console."
                )
                return None

            else:
                logger.error(
                    f"Gemini: unexpected status {response.status_code} "
                    f"— {response.text[:200]}"
                )
                return None

    except httpx.TimeoutException:
        logger.warning(f"Gemini: request timed out after {timeout}s")
        return None
    except httpx.ConnectError:
        logger.error("Gemini: connection failed — check network")
        return None
    except Exception as e:
        logger.error(f"Gemini: unexpected error — {str(e)}")
        return None


# ─── Provider caller map ──────────────────────────────────────────────────────
# Maps provider name → its caller function.
# Add new providers here alongside their caller function above.

CALLERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini
}


# ─── Main router ──────────────────────────────────────────────────────────────

async def route_llm_call(
    prompt: str,
    system: str,
    task: str,
    max_tokens: int
) -> dict:
    """
    Main entry point for all LLM calls.

    Tries providers in order (from .env LLM_PROVIDER_ORDER).
    Returns immediately on first success.
    Logs failures and moves to next provider.
    Returns structured result dict always — never raises.

    Args:
        prompt:     The user message / formatted prompt string.
        system:     The system prompt defining persona and rules.
        task:       Task name — "research" | "script" | "analytics" | "keywords"
        max_tokens: Maximum tokens the LLM should return.

    Returns:
        On success:
        {
            "success": True,
            "content": "raw LLM response string",
            "provider_used": "anthropic",
            "model_used": "claude-sonnet-4-6",
            "error": None
        }

        On failure (all providers failed or none configured):
        {
            "success": False,
            "content": None,
            "provider_used": None,
            "model_used": None,
            "error": "human-readable error message"
        }
    """
    settings = get_settings()
    timeout = settings.llm_timeout_seconds
    available = get_available_providers(task)

    # ── No providers configured at all ───────────────────────────
    if not available:
        msg = (
            "No LLM providers are configured for this task. "
            "Add at least one API key and matching model to .env. "
            f"Task: '{task}'"
        )
        logger.error(msg)
        return {
            "success": False,
            "content": None,
            "provider_used": None,
            "model_used": None,
            "error": msg
        }

    logger.info(
        f"Task '{task}' — trying {len(available)} provider(s): "
        f"{[p['name'] for p in available]}"
    )

    errors = {}

    # ── Try each provider in order ────────────────────────────────
    for provider in available:
        provider_name = provider["name"]
        model = provider["model"]
        api_key = provider["api_key"]
        caller = CALLERS.get(provider_name)

        if not caller:
            logger.warning(f"No caller implemented for provider '{provider_name}' — skipping")
            continue

        logger.info(f"Trying {provider['label']} — model: {model}")

        result = await caller(
            prompt=prompt,
            system=system,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout
        )

        if result:
            logger.info(
                f"✅ Success — provider: {provider_name}, model: {model}, "
                f"response length: {len(result)} chars"
            )
            return {
                "success": True,
                "content": result,
                "provider_used": provider_name,
                "model_used": model,
                "error": None
            }

        # Provider failed — log and try next
        errors[provider_name] = "call returned None"
        logger.warning(
            f"❌ {provider['label']} failed for task '{task}' — "
            f"trying next provider"
        )

    # ── All providers failed ──────────────────────────────────────
    msg = (
        f"All {len(available)} configured provider(s) failed for task '{task}'. "
        f"Failures: {errors}"
    )
    logger.error(msg)
    return {
        "success": False,
        "content": None,
        "provider_used": None,
        "model_used": None,
        "error": msg
    }