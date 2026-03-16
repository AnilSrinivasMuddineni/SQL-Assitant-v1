"""
Central LLM Factory
-------------------
Provides two public functions:
  - get_combined_model_list(ollama_manager)  → list of {provider, model, label} dicts
  - create_llm(provider, model, config_path, ollama_manager) → LLM instance

Design goals:
  * Reuse OllamaManager.get_available_models() — no reimplementation.
  * Enterprise models come entirely from env vars — no secrets in code.
  * Backward-compatible: if nothing is configured, returns the Ollama LLM exactly
    as today.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

def get_combined_model_list(ollama_manager) -> List[Dict[str, str]]:
    """
    Return a unified list of available models for the UI.

    Combines:
      1. Local Ollama models — via ollama_manager.get_available_models() (existing helper).
      2. Enterprise models  — from ENTERPRISE_LLM_MODELS env var (comma-separated).

    Each entry: {"provider": str, "model": str, "label": str}
    """
    combined: List[Dict[str, str]] = []

    # --- 1. Local Ollama models (reuse existing helper) ---
    try:
        ollama_models = ollama_manager.get_available_models()
        for m in ollama_models:
            combined.append({
                "provider": "ollama",
                "model": m,
                "label": f"Ollama \u2013 {m}",
            })
        logger.debug(f"[LLMFactory] Loaded {len(ollama_models)} Ollama model(s).")
    except Exception as e:
        logger.warning(f"[LLMFactory] Could not fetch Ollama models: {e}")

    # --- 2. Enterprise models from env ---
    enterprise_models_raw = os.getenv("ENTERPRISE_LLM_MODELS", "").strip()
    if enterprise_models_raw:
        for m in enterprise_models_raw.split(","):
            m = m.strip()
            if m:
                combined.append({
                    "provider": "enterprise",
                    "model": m,
                    "label": f"Enterprise \u2013 {m}",
                })
        logger.debug(f"[LLMFactory] Loaded enterprise model(s): {enterprise_models_raw}")

    return combined


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def create_llm(
    provider: str,
    model: str,
    config_path: str = "config/database_config.json",
    ollama_manager=None,
) -> Any:
    """
    Return a configured LLM instance for the given provider/model.

    provider == "ollama":
        Delegates entirely to ollama_manager.update_model() and returns
        ollama_manager.llm — no new Ollama code.

    provider == "enterprise":
        Builds a ChatOpenAI-compatible client using env vars:
          ENTERPRISE_LLM_PROVIDER, ENTERPRISE_LLM_API_KEY, ENTERPRISE_LLM_ENDPOINT.
        Supported values of ENTERPRISE_LLM_PROVIDER:
          "openai", "azure_openai", "anthropic" (others treated as OpenAI-compatible).

    Falls back to Ollama if provider is unrecognised.
    """
    provider = (provider or "ollama").lower().strip()

    if provider == "ollama":
        return _create_ollama_llm(model, config_path, ollama_manager)

    if provider == "enterprise":
        return _create_enterprise_llm(model)

    # Unknown provider — fall back to Ollama and warn
    logger.warning(
        f"[LLMFactory] Unknown provider '{provider}', falling back to Ollama."
    )
    return _create_ollama_llm(model, config_path, ollama_manager)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_ollama_llm(model: str, config_path: str, ollama_manager) -> Any:
    """Reuse OllamaManager to update and return the Ollama LLM instance."""
    if ollama_manager is None:
        # Lazy import to avoid circular deps
        from src.ollama_llm import OllamaManager
        ollama_manager = OllamaManager(config_path)

    base_url = ollama_manager.base_url
    model = model.strip() if model else ollama_manager.model

    ollama_manager.update_model(model, base_url)
    logger.info(f"[LLMFactory] Ollama LLM: model={model}, base_url={base_url}")
    return ollama_manager.llm


def _create_enterprise_llm(model: str) -> Any:
    """
    Build a ChatOpenAI-compatible enterprise LLM from env vars.
    Never logs or exposes the API key.
    """
    enterprise_provider = os.getenv("ENTERPRISE_LLM_PROVIDER", "openai").lower().strip()
    api_key = os.getenv("ENTERPRISE_LLM_API_KEY", "")
    endpoint = os.getenv("ENTERPRISE_LLM_ENDPOINT", "").rstrip("/")

    if not api_key:
        raise ValueError(
            "[LLMFactory] ENTERPRISE_LLM_API_KEY is not set. "
            "Cannot create enterprise LLM."
        )

    logger.info(
        f"[LLMFactory] Enterprise LLM: provider={enterprise_provider}, "
        f"model={model}, endpoint={'(default)' if not endpoint else endpoint}"
    )

    if enterprise_provider in ("openai", "azure_openai"):
        from langchain_community.chat_models import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": model,
            "openai_api_key": api_key,
            "temperature": 0.1,
        }
        if endpoint:
            kwargs["openai_api_base"] = endpoint

        return ChatOpenAI(**kwargs)

    if enterprise_provider == "anthropic":
        try:
            from langchain_community.chat_models import ChatAnthropic  # type: ignore

            kwargs = {"model": model, "anthropic_api_key": api_key, "temperature": 0.1}
            if endpoint:
                kwargs["anthropic_api_url"] = endpoint
            return ChatAnthropic(**kwargs)
        except ImportError:
            logger.warning(
                "[LLMFactory] langchain-anthropic not installed; "
                "falling back to ChatOpenAI-compatible endpoint."
            )

    # Generic OpenAI-compatible fallback
    from langchain_community.chat_models import ChatOpenAI

    kwargs = {
        "model": model,
        "openai_api_key": api_key,
        "temperature": 0.1,
    }
    if endpoint:
        kwargs["openai_api_base"] = endpoint
    return ChatOpenAI(**kwargs)
