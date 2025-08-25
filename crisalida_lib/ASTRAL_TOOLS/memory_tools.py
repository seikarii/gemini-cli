#!/usr/bin/env python3
"""
Enhanced Memory Tool for Prometheus Agent (Crisalida)
=====================================================

Python implementation with semantic search, advanced memory management,
clustering, summarization, and robust diagnostics.

Features:
- Semantic search using embeddings (TF-IDF, ready for upgrade to transformer-based)
- Advanced memory management: filtering, sorting, summarizing, clustering
- Regex-based deletion and preview
- Memory statistics and diagnostics
- Extensible for integration with external vector DBs or LLMs

Based on:
- packages/core/src/tools/memoryTool.ts
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

# --- Pydantic Models ---


class SaveMemoryParams(BaseModel):
    fact: str = Field(
        ...,
        description="The specific fact or piece of information to remember. Should be a clear, self-contained statement.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "fact": "User prefers using Python over JavaScript for backend development"
            }
        }
    }


class RecallMemoryParams(BaseModel):
    query: str = Field(..., description="Search query to find relevant facts in memory")
    search_type: Literal["semantic", "keyword", "hybrid"] = Field(
        "semantic", description="Type of search to perform"
    )
    max_results: int = Field(10, description="Maximum number of results to return")
    min_similarity: float = Field(
        0.1, description="Minimum similarity threshold for semantic search"
    )
    sort_by: Literal["relevance", "recency", "frequency"] = Field(
        "relevance", description="How to sort the results"
    )
    filter_pattern: str | None = Field(
        None, description="Optional regex pattern to filter facts"
    )
    summarize: bool = Field(False, description="Whether to summarize the results")


class ClearMemoryParams(BaseModel):
    fact_pattern: str = Field(
        ..., description="Pattern or exact text of the fact to remove (supports regex)"
    )
    use_regex: bool = Field(False, description="Whether to treat pattern as regex")
    confirm: bool = Field(
        False, description="Confirmation flag - must be True to actually delete"
    )
    preview_only: bool = Field(
        False, description="Only show what would be deleted, don't actually delete"
    )
    max_deletions: int = Field(
        10, description="Maximum number of facts to delete in one operation"
    )


# --- Save Memory Tool ---


class SaveMemoryTool(BaseTool):
    """Save important facts to the agent's long-term memory"""

    def __init__(self, agent_memory_system=None):
        super().__init__()
        self.agent_memory_system = agent_memory_system

    def _get_name(self) -> str:
        return "save_memory"

    def _get_description(self) -> str:
        return (
            "Saves a specific piece of information or fact to your long-term memory.\n"
            "Use for clear, concise facts about the user, their preferences, or environment."
        )

    def _get_pydantic_schema(self):
        return SaveMemoryParams

    def _get_category(self) -> str:
        return "memory"

    def set_agent_memory_system(self, memory_system):
        self.agent_memory_system = memory_system

    async def execute(self, **kwargs) -> ToolCallResult:
        params = SaveMemoryParams(**kwargs)
        start_time = datetime.now()
        try:
            if not self.agent_memory_system:
                return ToolCallResult(
                    command="save_memory",
                    success=False,
                    output="",
                    error_message="Memory system not available - tool not properly initialized",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            principle = f"User fact: {params.fact}"
            # Check if principle already exists before adding
            existing_principles = getattr(
                self.agent_memory_system, "core_principles", []
            )
            if principle in existing_principles:
                return ToolCallResult(
                    command="save_memory",
                    success=True,
                    output=f"Fact already in memory: '{params.fact}'",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,  # Added this line
                )
            self.agent_memory_system.add_core_principle(principle)
            logger.info(f"💾 Saved fact to memory: {params.fact}")
            return ToolCallResult(
                command="save_memory",
                success=True,
                output=f"Successfully saved to long-term memory: '{params.fact}'",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
            )
        except Exception as e:
            logger.error(f"SaveMemoryTool error: {e}")
            return ToolCallResult(
                command="save_memory",
                success=False,
                output="",
                error_message=f"Failed to save memory: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the memory tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_memory_tools import demo_memory_tools

        return await demo_memory_tools()


# --- Recall Memory Tool ---


class RecallMemoryTool(BaseTool):
    """Enhanced memory recall tool with semantic search and advanced filtering capabilities"""

    def __init__(self, agent_memory_system=None):
        super().__init__()
        self.agent_memory_system = agent_memory_system
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        self.memory_vectors = None
        self.memory_facts = []

    def set_agent_memory_system(self, memory_system):
        self.agent_memory_system = memory_system

    def _get_name(self) -> str:
        return "recall_memory"

    def _get_description(self) -> str:
        return (
            "Enhanced memory recall with semantic search and advanced filtering.\n"
            "Find contextually related facts, filter, sort, summarize, and cluster results."
        )

    def _get_pydantic_schema(self):
        return RecallMemoryParams

    def _get_category(self) -> str:
        return "memory"

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()
        params = RecallMemoryParams(**kwargs)
        try:
            if not self.agent_memory_system:
                return ToolCallResult(
                    command="recall_memory",
                    success=False,
                    output="",
                    error_message="Memory system not available - tool not properly initialized",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            core_principles = getattr(self.agent_memory_system, "core_principles", [])
            user_facts = [p for p in core_principles if p.startswith("User fact:")]
            if not user_facts:
                return ToolCallResult(
                    command="recall_memory",
                    success=True,
                    output="No user facts found in long-term memory",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,  # Added this line
                )
            self.memory_facts = [fact.replace("User fact: ", "") for fact in user_facts]
            # Apply regex filter if provided
            if params.filter_pattern:
                try:
                    pattern = re.compile(params.filter_pattern, re.IGNORECASE)
                    self.memory_facts = [
                        fact for fact in self.memory_facts if pattern.search(fact)
                    ]
                except re.error as e:
                    return ToolCallResult(
                        command="recall_memory",
                        success=False,
                        output="",
                        error_message=f"Invalid regex pattern: {e}",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
            if not self.memory_facts:
                return ToolCallResult(
                    command="recall_memory",
                    success=True,
                    output="No facts match the filter criteria",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,  # Added this line
                )
            # Search
            results = []
            if params.search_type == "semantic":
                results = self._semantic_search(params.query, params.min_similarity)
            elif params.search_type == "keyword":
                results = self._keyword_search(params.query)
            elif params.search_type == "hybrid":
                semantic_results = self._semantic_search(
                    params.query, params.min_similarity
                )
                keyword_results = self._keyword_search(params.query)
                combined = {r["fact"]: r for r in semantic_results + keyword_results}
                results = list(combined.values())
            else:
                results = self._keyword_search(params.query)
            results = self._sort_results(results, params.sort_by)[: params.max_results]
            output = (
                (
                    self._format_summarized_output(
                        params.query, results, params.search_type
                    )
                    if params.summarize
                    else self._format_search_output(
                        params.query, results, params.search_type
                    )
                )
                if results
                else f"No facts found matching query: '{params.query}' using {params.search_type} search"
            )
            return ToolCallResult(
                command="recall_memory",
                success=True,
                output=output,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
                metadata={
                    "search_type": params.search_type,
                    "results_count": len(results),
                },
            )
        except Exception as e:
            logger.error(f"RecallMemoryTool error: {e}")
            return ToolCallResult(
                command="recall_memory",
                success=False,
                output="",
                error_message=f"Failed to recall memory: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the memory tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_memory_tools import demo_memory_tools

        return await demo_memory_tools()

    def _semantic_search(
        self, query: str, min_similarity: float
    ) -> list[dict[str, Any]]:
        try:
            if not self.memory_facts:
                return []
            if (
                self.memory_vectors is None
                or len(self.memory_facts) != self.memory_vectors.shape[0]
            ):
                self.memory_vectors = self.vectorizer.fit_transform(self.memory_facts)
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.memory_vectors)[0]
            return [
                {
                    "fact": self.memory_facts[i],
                    "score": float(similarity),
                    "type": "semantic",
                    "relevance": (
                        "high"
                        if similarity > 0.5
                        else "medium" if similarity > 0.3 else "low"
                    ),
                }
                for i, similarity in enumerate(similarities)
                if similarity >= min_similarity
            ]
        except Exception as e:
            logger.warning(
                f"Semantic search failed, falling back to keyword search: {e}"
            )
            return self._keyword_search(query)

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for fact in self.memory_facts:
            fact_lower = fact.lower()
            if query_lower in fact_lower:
                score = fact_lower.count(query_lower) * 0.5
                if fact_lower.startswith(query_lower):
                    score += 0.3
                if query_lower in fact_lower[:50]:
                    score += 0.2
                results.append(
                    {
                        "fact": fact,
                        "score": min(score, 1.0),
                        "type": "keyword",
                        "relevance": (
                            "high"
                            if score > 0.7
                            else "medium" if score > 0.4 else "low"
                        ),
                    }
                )
        return results

    def _sort_results(
        self, results: list[dict[str, Any]], sort_by: str
    ) -> list[dict[str, Any]]:
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x["score"], reverse=True)
        elif sort_by == "recency":
            return sorted(
                results,
                key=lambda x: self.memory_facts.index(x["fact"]),
                reverse=True,
            )
        elif sort_by == "frequency":
            word_counts = {r["fact"]: len(r["fact"].split()) for r in results}
            return sorted(
                results, key=lambda x: word_counts.get(x["fact"], 0), reverse=True
            )
        return results

    def _format_search_output(
        self, query: str, results: list[dict[str, Any]], search_type: str
    ) -> str:
        output = f"🔍 **MEMORY SEARCH RESULTS ({search_type.upper()})**\n\n"
        output += f"**Query:** {query}\n"
        output += f"**Found:** {len(results)} relevant fact(s)\n\n"
        for i, result in enumerate(results, 1):
            output += f"**{i}.** {result['fact']}\n"
            output += f"   • Relevance: {result['relevance']} (score: {result['score']:.3f})\n"
            output += f"   • Search type: {result['type']}\n\n"
        return output.strip()

    def _format_summarized_output(
        self, query: str, results: list[dict[str, Any]], search_type: str
    ) -> str:
        output = f"📋 **MEMORY SUMMARY ({search_type.upper()})**\n\n"
        output += f"**Query:** {query}\n"
        output += f"**Total facts found:** {len(results)}\n\n"
        relevance_groups = defaultdict(list)
        for result in results:
            relevance_groups[result["relevance"]].append(result["fact"])
        for relevance in ["high", "medium", "low"]:
            if relevance_groups[relevance]:
                output += f"**{relevance.title()} Relevance ({len(relevance_groups[relevance])} facts):**\n"
                for fact in relevance_groups[relevance][:3]:
                    output += f"• {fact[:80]}{'...' if len(fact) > 80 else ''}\n"
                if len(relevance_groups[relevance]) > 3:
                    output += f"  ... and {len(relevance_groups[relevance]) - 3} more\n"
                output += "\n"
        fact_themes = self._extract_themes(results)
        if fact_themes:
            output += "**Key Themes:**\n"
            for theme in fact_themes:
                output += f"• {theme}\n"
        return output.strip()

    def _extract_themes(self, results: list[dict[str, Any]]) -> list[str]:
        all_text = " ".join([result["fact"] for result in results])
        words = re.findall(r"\b\w+\b", all_text.lower())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "cannot",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
        }
        word_freq = Counter(w for w in words if w not in stop_words and len(w) > 3)
        themes = [
            f"{word} (mentioned {count} times)"
            for word, count in word_freq.most_common(5)
            if count > 1
        ]
        return themes


# --- Clear Memory Tool ---


class ClearMemoryTool(BaseTool):
    """Enhanced memory clearing tool with regex patterns and advanced filtering"""

    def __init__(self, agent_memory_system=None):
        super().__init__()
        self.agent_memory_system = agent_memory_system

    def _get_name(self) -> str:
        return "clear_memory"

    def _get_description(self) -> str:
        return (
            "Enhanced memory clearing with regex patterns and batch operations.\n"
            "Preview, confirm, and selectively clear facts by pattern."
        )

    def _get_pydantic_schema(self):
        return ClearMemoryParams

    def _get_category(self) -> str:
        return "memory"

    def set_agent_memory_system(self, memory_system):
        self.agent_memory_system = memory_system

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()
        params = ClearMemoryParams(**kwargs)
        try:
            if not self.agent_memory_system:
                return ToolCallResult(
                    command="clear_memory",
                    success=False,
                    output="",
                    error_message="Memory system not available - tool not properly initialized",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            core_principles = getattr(self.agent_memory_system, "core_principles", [])
            user_facts = [p for p in core_principles if p.startswith("User fact:")]
            if not user_facts:
                return ToolCallResult(
                    command="clear_memory",
                    success=True,
                    output="No user facts found in memory to clear",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,  # Added this line
                )
            facts_to_remove = []
            if params.use_regex:
                try:
                    pattern = re.compile(params.fact_pattern, re.IGNORECASE)
                    facts_to_remove = [
                        fact for fact in user_facts if pattern.search(fact)
                    ]
                except re.error as e:
                    return ToolCallResult(
                        command="clear_memory",
                        success=False,
                        output="",
                        error_message=f"Invalid regex pattern: {e}",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
            else:
                pattern_lower = params.fact_pattern.lower()
                facts_to_remove = [
                    fact for fact in user_facts if pattern_lower in fact.lower()
                ]
            if not facts_to_remove:
                return ToolCallResult(
                    command="clear_memory",
                    success=True,
                    output=f"No facts found matching pattern: '{params.fact_pattern}'",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,  # Added this line
                )
            truncated = False
            if len(facts_to_remove) > params.max_deletions:
                facts_to_remove = facts_to_remove[: params.max_deletions]
                truncated = True
            if params.preview_only:
                output = f"🔍 **DELETION PREVIEW ({len(facts_to_remove)} facts)**\n\n"
                output += f"**Pattern:** {params.fact_pattern} ({'regex' if params.use_regex else 'text'})\n\n"
                output += "**Facts that would be deleted:**\n"
                for i, fact in enumerate(facts_to_remove, 1):
                    output += f"{i}. {fact.replace('User fact: ', '')}\n"
                if truncated:
                    output += (
                        f"\n*Note: Only showing first {params.max_deletions} matches*"
                    )
                output += "\n\n*To actually delete these facts, set confirm=True and preview_only=False*"
                return ToolCallResult(
                    command="clear_memory",
                    success=True,
                    output=output,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"preview": True, "matches": len(facts_to_remove)},
                    error_message=None,  # Added this line
                )
            if not params.confirm:
                return ToolCallResult(
                    command="clear_memory",
                    success=False,
                    output="",
                    error_message=f"Found {len(facts_to_remove)} facts to delete. Memory clear operation requires confirmation (set confirm=True)",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            removed_count = 0
            removed_facts = []
            for fact in facts_to_remove:
                try:
                    self.agent_memory_system.core_principles.remove(fact)
                    removed_facts.append(fact)
                    removed_count += 1
                except ValueError:
                    pass
            output = "🗑️ **MEMORY DELETION COMPLETED**\n\n"
            output += f"**Pattern:** {params.fact_pattern} ({'regex' if params.use_regex else 'text'})\n"
            output += f"**Removed:** {removed_count} fact(s)\n\n"
            if removed_count > 0:
                output += "**Deleted facts:**\n"
                for i, fact in enumerate(removed_facts, 1):
                    output += f"{i}. {fact.replace('User fact: ', '')}\n"
                if truncated:
                    output += f"\n*Note: Limited to {params.max_deletions} deletions for safety*"
            logger.info(
                f"🗑️ Cleared {removed_count} facts from memory matching: {params.fact_pattern}"
            )
            return ToolCallResult(
                command="clear_memory",
                success=True,
                output=output.strip(),
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
                metadata={
                    "removed_count": removed_count,
                    "pattern": params.fact_pattern,
                    "regex": params.use_regex,
                },
            )
        except Exception as e:
            logger.error(f"ClearMemoryTool error: {e}")
            return ToolCallResult(
                command="clear_memory",
                success=False,
                output="",
                error_message=f"Failed to clear memory: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the memory tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_memory_tools import demo_memory_tools

        return await demo_memory_tools()


# --- Demo Function ---


# Demo functionality moved to tools.memory to avoid duplication

if __name__ == "__main__":
    # Import demo from the canonical location
    from crisalida_lib.ASTRAL_TOOLS.demos.demo_memory_tools import demo_memory_tools

    asyncio.run(demo_memory_tools())
