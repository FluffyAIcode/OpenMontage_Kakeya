"""Text-generation (local/remote LLM) capability tools.

This package hosts the `text_generation` capability family: a selector
(`llm_selector`) plus provider tools that route mechanical, batch text
chores (image-prompt expansion, subtitle translation, caption variants,
alt-text) to a Large Language Model backend.

Design intent (see docs/adr/0001-kakeya-llm-inference-integration.md):
the host agent remains the creative intelligence. These tools exist only
to OFFLOAD high-volume, non-creative text transforms to an optional,
user-run local LLM server — never to replace the agent's reasoning.
"""
