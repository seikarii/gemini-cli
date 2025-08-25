# LLM Module

This directory contains modules for interacting with Large Language Models (LLMs), including connectors, client implementations, and specialized interfaces for different LLM providers or roles (e.g., Cerebellum LLM).

## Core Components

*   `base_llm_connector.py`: Defines the abstract base class `BaseLLMConnector` for integrating LLM models.
*   `cerebellum_connector.py`: A specialized connector for interacting with a low-level, tactical LLM (conceptualized as the "Cerebellum").
*   `strategic_llm_connector.py`: A specialized connector for reasoning, planning, and high-level validation.
*   `fallback_manager.py`: Manages situations where the main LLMs fail, allowing switching to alternative strategies.
*   `independence_tracker.py`: An advanced monitor for tracking the autonomy and self-sufficiency of the LLM.
*   `llm_disconnection_manager.py`: Manages the clean and robust disconnection of LLMs when the system achieves independence.
*   `llm_gateway_orchestrator.py`: The main orchestrator for the brain-cerebellum interaction.
*   `llm_health_monitor.py`: An advanced health and performance monitor for LLMs.
*   `llm_metrics_collector.py`: An advanced monitor for LLM metrics and efficiency.
*   `llm_migration_manager.py`: An advanced orchestrator for LLM transition and migration.
*   `ollama_client.py`: An advanced client for integration with Ollama LLMs.
