import logging
from datetime import datetime  # Import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult
from crisalida_lib.EARTH.dialectical_oracle import (
    ArgumentType,
    DialecticalOracle,
    PhilosophicalSchool,
)

logger = logging.getLogger(__name__)


class DialecticalOracleParameters(BaseModel):
    action: Literal[
        "start_debate",
        "continue_debate",
        "synthesize_debate",
        "get_status",
        "suggest_next",
    ] = Field(..., description="Action to perform")
    topic: str | None = Field(
        None,
        description="Philosophical or architectural topic for debate (required for start_debate)",
    )
    premise: str | None = Field(
        None,
        description="Initial premise or statement to debate (required for start_debate)",
    )
    query_id: str | None = Field(
        None, description="Query ID for continuing or synthesizing debates"
    )
    current_argument: str | None = Field(
        None, description="Current argument to respond to (for continue_debate)"
    )
    complexity_level: int = Field(
        3, description="Complexity level from 1 (basic) to 5 (highly advanced)"
    )
    philosophical_school: str | None = Field(
        None, description="Specific philosophical school to use"
    )
    force_perspective: Literal["thesis", "antithesis", "synthesis"] | None = Field(
        None, description="Force a specific dialectical perspective"
    )


class DialecticalOracleTool(BaseTool):
    """
    Herramienta principal para debates dialécticos y razonamiento filosófico avanzado.
    Permite iniciar, continuar, sintetizar y analizar debates sobre temas complejos.
    """

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return DialecticalOracleParameters

    def _get_category(self) -> str:
        return "dialectical_reasoning"

    def __init__(self, llm_gateway: Any | None = None):
        super().__init__()
        self.oracle = DialecticalOracle(llm_gateway=llm_gateway)

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        try:
            action = kwargs.get("action")
            if action == "start_debate":
                return await self._start_debate(**kwargs)
            elif action == "continue_debate":
                return await self._continue_debate(**kwargs)
            elif action == "synthesize_debate":
                return await self._synthesize_debate(**kwargs)
            elif action == "get_status":
                return await self._get_status(**kwargs)
            elif action == "suggest_next":
                return await self._suggest_next(**kwargs)
            else:
                return ToolCallResult(
                    command="dialectical_oracle",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {action}",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                )
        except Exception as e:
            logger.error(f"Dialectical Oracle tool execution failed: {e}")
            return ToolCallResult(
                command="dialectical_oracle",
                success=False,
                output="",
                error_message=f"Oracle execution error: {str(e)}",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )

    async def _start_debate(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        topic = kwargs.get("topic")
        premise = kwargs.get("premise")
        if not topic or not premise:
            return ToolCallResult(
                command="start_debate",
                success=False,
                output="",
                error_message="Both 'topic' and 'premise' are required to start a debate",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        complexity_level = kwargs.get("complexity_level", 3)
        philosophical_schools = []
        school_param = kwargs.get("philosophical_school")
        if school_param:
            try:
                school = PhilosophicalSchool(school_param)
                philosophical_schools = [school]
            except ValueError:
                logger.warning(f"Unknown philosophical school: {school_param}")
        query = await self.oracle.pose_query(
            topic=topic,
            premise=premise,
            complexity_level=complexity_level,
            philosophical_schools=philosophical_schools,
        )
        response = await self.oracle.get_response(query.query_id)
        output = self._format_debate_start_output(query, response)
        return ToolCallResult(
            command="start_debate",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _continue_debate(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        query_id = kwargs.get("query_id")
        if not query_id:
            return ToolCallResult(
                command="continue_debate",
                success=False,
                output="",
                error_message="query_id is required to continue a debate",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        current_argument = kwargs.get("current_argument")
        force_perspective = kwargs.get("force_perspective")
        target_school = None
        school_param = kwargs.get("philosophical_school")
        if school_param:
            try:
                target_school = PhilosophicalSchool(school_param)
            except ValueError:
                logger.warning(f"Unknown philosophical school: {school_param}")
        response = await self.oracle.get_response(
            query_id=query_id,
            current_argument=current_argument,
            force_perspective=force_perspective,
            target_school=target_school,
        )
        if response.perspective == "error":
            return ToolCallResult(
                command="continue_debate",
                success=False,
                output="",
                error_message=response.argument,
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        output = self._format_debate_continue_output(response)
        return ToolCallResult(
            command="continue_debate",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _synthesize_debate(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        query_id = kwargs.get("query_id")
        if not query_id:
            return ToolCallResult(
                command="synthesize_debate",
                success=False,
                output="",
                error_message="query_id is required to synthesize a debate",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        synthesis = await self.oracle.get_debate_synthesis(query_id)
        if "error" in synthesis:
            return ToolCallResult(
                command="synthesize_debate",
                success=False,
                output="",
                error_message=synthesis["error"],
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        output = self._format_synthesis_output(synthesis)
        return ToolCallResult(
            command="synthesize_debate",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _get_status(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        status = self.oracle.get_oracle_status()
        output = self._format_status_output(status)
        return ToolCallResult(
            command="get_status",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _suggest_next(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        query_id = kwargs.get("query_id")
        if not query_id:
            return ToolCallResult(
                command="suggest_next",
                success=False,
                output="",
                error_message="query_id is required to suggest next inquiry",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        # suggestions = await self.oracle.suggest_next_inquiry(query_id) # Commented out
        suggestions = {"error": "suggest_next_inquiry not implemented"}  # Placeholder
        if "error" in suggestions:
            return ToolCallResult(
                command="suggest_next",
                success=False,
                output="",
                error_message=suggestions["error"],
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        output = self._format_suggestions_output(suggestions)
        return ToolCallResult(
            command="suggest_next",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    def _format_debate_start_output(self, query: Any, response: Any) -> str:
        output = "🗣️ **DIALECTICAL DEBATE INITIATED**\n\n"
        output += f"**Topic:** {query.topic}\n"
        output += f"**Premise:** {query.premise}\n"
        output += f"**Query ID:** {query.query_id}\n"
        output += f"**Complexity Level:** {query.complexity_level}/5\n\n"
        output += f"**Initial Position ({response.perspective.upper()}):**\n"
        output += f"*Philosophical Perspective: {response.philosophical_school.value if response.philosophical_school else 'Mixed'}*\n"
        output += f"*Confidence: {response.confidence_score:.2f}*\n\n"
        output += f"{response.argument}\n\n"
        if response.implications:
            output += "**Key Implications:**\n"
            for impl in response.implications[:3]:
                output += f"• {impl}\n"
        output += f"\n*Use query ID '{query.query_id}' to continue this debate.*"
        return output

    def _format_debate_continue_output(self, response: Any) -> str:
        output = f"🔄 **DIALECTICAL RESPONSE ({response.perspective.upper()})**\n\n"
        output += f"**Philosophical School:** {response.philosophical_school.value if response.philosophical_school else 'Mixed'}\n"
        output += f"**Argument Type:** {response.argument_type.value}\n"
        output += f"**Confidence:** {response.confidence_score:.2f}\n\n"
        output += f"**Argument:**\n{response.argument}\n\n"
        if response.implications:
            output += "**Implications:**\n"
            for impl in response.implications:
                output += f"• {impl}\n"
        if response.counter_vulnerabilities:
            output += "\n**Potential Counter-Arguments:**\n"
            for vuln in response.counter_vulnerabilities[:2]:
                output += f"• {vuln}\n"
        return output

    def _format_synthesis_output(self, synthesis: dict[str, Any]) -> str:
        output = "🎯 **DEBATE SYNTHESIS**\n\n"
        output += f"**Query ID:** {synthesis['query_id']}\n"
        output += f"**Total Exchanges:** {synthesis['total_responses']}\n"
        output += f"**Quality Score:** {synthesis['debate_quality_metrics']['overall_quality']:.2f}/1.0\n\n"
        output += f"**Meta-Synthesis:**\n{synthesis['meta_synthesis']}\n\n"
        if synthesis["key_themes"]:
            output += (
                f"**Key Themes Explored:** {', '.join(synthesis['key_themes'])}\n\n"
            )
        if synthesis["convergence_points"]:
            output += "**Points of Convergence:**\n"
            for point in synthesis["convergence_points"]:
                output += f"• {point}\n"
        if synthesis["divergence_points"]:
            output += "\n**Points of Divergence:**\n"
            for point in synthesis["divergence_points"]:
                output += f"• {point}\n"
        return output

    def _format_status_output(self, status: dict[str, Any]) -> str:
        output = "📊 **DIALECTICAL ORACLE STATUS**\n\n"
        output += f"**Total Queries:** {status['total_queries']}\n"
        output += f"**Total Responses:** {status['total_responses']}\n"
        output += f"**Avg Responses/Query:** {status['average_responses_per_query']}\n"
        output += (
            f"**Avg Debate Quality:** {status['average_debate_quality']:.2f}/1.0\n\n"
        )
        diversity = status["philosophical_diversity"]
        output += "**Philosophical Diversity:**\n"
        output += f"• Schools Engaged: {diversity['schools_engaged']}\n"
        output += f"• Perspectives Explored: {diversity['perspectives_explored']}\n\n"
        output += f"**Reasoning Engine:** {status['reasoning_engine_status']}\n"
        output += f"**LLM Integration:** {status['llm_integration']}\n"
        output += f"**Status:** {status['status']}\n"
        return output

    def _format_suggestions_output(self, suggestions: dict[str, Any]) -> str:
        output = "💡 **SUGGESTED NEXT INQUIRIES**\n\n"
        output += f"**Query ID:** {suggestions['query_id']}\n"
        current_state = suggestions["current_state"]
        output += "**Current State:**\n"
        output += f"• Total Responses: {current_state['total_responses']}\n"
        output += (
            f"• Positions Explored: {', '.join(current_state['positions_explored'])}\n"
        )
        output += f"• Depth Level: {current_state['depth_level']}\n\n"
        if suggestions["suggestions"]:
            output += "**Recommendations:**\n"
            for i, suggestion in enumerate(suggestions["suggestions"], 1):
                output += f"{i}. **{suggestion['type'].replace('_', ' ').title()}:** {suggestion['suggestion']}\n"
        else:
            output += "No specific suggestions at this time. Consider starting a new debate thread.\n"
        return output


class PhilosophicalAnalysisParameters(BaseModel):
    text: str = Field(..., description="Text to analyze philosophically")
    analysis_type: Literal[
        "concept_extraction",
        "relationship_mapping",
        "implication_analysis",
        "school_identification",
    ] = Field("concept_extraction", description="Type of analysis to perform")
    philosophical_school: str | None = Field(
        None, description="Analyze from perspective of specific philosophical school"
    )


class PhilosophicalAnalysisTool(BaseTool):
    """Herramienta para análisis filosófico de conceptos, relaciones e implicaciones."""

    def _get_name(self) -> str:
        return "philosophical_analysis"

    def _get_description(self) -> str:
        return "Analyze philosophical concepts, identify relationships, and explore implications"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return PhilosophicalAnalysisParameters

    def _get_category(self) -> str:
        return "philosophical_analysis"

    def __init__(self) -> None:
        super().__init__()

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        try:
            text = kwargs.get("text")
            analysis_type = kwargs.get("analysis_type", "concept_extraction")
            school_param = kwargs.get("philosophical_school")
            if not text:
                return ToolCallResult(
                    command="philosophical_analysis",
                    success=False,
                    output="",
                    error_message="Text is required for analysis",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                )
            if analysis_type == "concept_extraction":
                concepts = self._extract_philosophical_concepts(text)
                output = "**Philosophical Concepts Identified:**\n"
                for concept in concepts:
                    output += f"• {concept}\n"
            elif analysis_type == "relationship_mapping":
                relationships = self._map_conceptual_relationships(text)
                output = "**Conceptual Relationships:**\n"
                for rel in relationships:
                    output += f"• {rel}\n"
            elif analysis_type == "implication_analysis":
                implications = self._analyze_implications(text, school_param)
                output = "**Philosophical Implications:**\n"
                for impl in implications:
                    output += f"• {impl}\n"
            elif analysis_type == "school_identification":
                schools = self._identify_philosophical_schools(text)
                output = "**Relevant Philosophical Schools:**\n"
                for school in schools:
                    output += f"• {school}\n"
            else:
                output = ""
            return ToolCallResult(
                command="philosophical_analysis",
                success=True,
                output=output,
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
                error_message=None,  # Added this line
            )
        except Exception as e:
            return ToolCallResult(
                command="philosophical_analysis",
                success=False,
                output="",
                error_message=f"Philosophical analysis failed: {str(e)}",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )

    def _extract_philosophical_concepts(self, text: str) -> list[str]:
        text_lower = text.lower()
        concepts = []
        concept_keywords = {
            "being": ["being", "existence", "ontology"],
            "knowledge": ["knowledge", "truth", "epistemology", "understanding"],
            "reality": ["reality", "real", "metaphysics", "nature"],
            "consciousness": ["consciousness", "mind", "awareness", "cognition"],
            "ethics": ["good", "bad", "right", "wrong", "moral", "ethical"],
            "system": ["system", "structure", "organization", "architecture"],
            "emergence": ["emergent", "emergence", "complexity", "self-organization"],
            "dialectical": ["dialectical", "thesis", "antithesis", "synthesis"],
        }
        for concept, keywords in concept_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept.title())
        return concepts if concepts else ["General Philosophical Inquiry"]

    def _map_conceptual_relationships(self, text: str) -> list[str]:
        relationships = []
        text_lower = text.lower()
        if "leads to" in text_lower or "causes" in text_lower:
            relationships.append("Causal relationship identified")
        if "similar to" in text_lower or "like" in text_lower:
            relationships.append("Analogical relationship identified")
        if "opposite" in text_lower or "contrary" in text_lower:
            relationships.append("Oppositional relationship identified")
        if "part of" in text_lower or "component" in text_lower:
            relationships.append("Compositional relationship identified")
        return relationships if relationships else ["Complex conceptual network"]

    def _analyze_implications(
        self,
        text: str,
        school_param: str | None,
    ) -> list[str]:
        implications = []
        text_lower = text.lower()
        if "system" in text_lower:
            implications.append("Systems thinking: Emergence and holistic properties")
        if "consciousness" in text_lower:
            implications.append(
                "Mind-body problem: Relationship between mental and physical"
            )
        if "truth" in text_lower or "knowledge" in text_lower:
            implications.append(
                "Epistemological concerns: Nature and limits of knowledge"
            )
        if school_param:
            school_implications = {
                "pragmatism": "Focus on practical consequences and workability",
                "phenomenology": "Emphasis on lived experience and consciousness",
                "existentialism": "Individual existence and freedom of choice",
                "empiricism": "Knowledge derived from sensory experience",
            }
            school_impl = school_implications.get(school_param)
            if school_impl:
                implications.append(f"From {school_param}: {school_impl}")
        return (
            implications
            if implications
            else ["Requires deeper philosophical investigation"]
        )

    def _identify_philosophical_schools(self, text: str) -> list[str]:
        text_lower = text.lower()
        schools = []
        school_indicators = {
            "Pragmatism": ["practical", "useful", "works", "consequences"],
            "Phenomenology": ["experience", "consciousness", "lived", "subjective"],
            "Empiricism": ["observation", "evidence", "sensory", "data"],
            "Rationalism": ["reason", "logic", "rational", "deductive"],
            "Existentialism": ["existence", "freedom", "choice", "individual"],
            "Systems Thinking": ["system", "holistic", "emergence", "complexity"],
        }
        for school, indicators in school_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                schools.append(school)
        return schools if schools else ["Multiple schools relevant"]


class ArgumentStructureParameters(BaseModel):
    action: Literal["analyze", "construct"] = Field(
        ..., description="Action to perform: 'analyze' or 'construct'"
    )
    text: str | None = Field(
        None, description="Argument text to analyze (for analyze action)"
    )
    conclusion: str | None = Field(
        None, description="Desired conclusion (for construct action)"
    )
    argument_type: str = Field("deductive", description="Type of argument to construct")


class ArgumentStructureTool(BaseTool):
    """Herramienta para analizar y construir estructuras argumentativas."""

    def _get_name(self) -> str:
        return "argument_structure"

    def _get_description(self) -> str:
        return "Analyze logical structure of arguments or construct new arguments"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return ArgumentStructureParameters

    def _get_category(self) -> str:
        return "dialectical_reasoning"

    def __init__(self) -> None:
        super().__init__()

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        try:
            action = kwargs.get("action")
            if action == "analyze":
                text = kwargs.get("text")
                if not text:
                    return ToolCallResult(
                        command="argument_structure",
                        success=False,
                        output="",
                        error_message="Text is required for analysis",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                return await self._analyze_argument_structure(text)
            elif action == "construct":
                conclusion = kwargs.get("conclusion")
                if not conclusion:
                    return ToolCallResult(
                        command="argument_structure",
                        success=False,
                        output="",
                        error_message="Conclusion is required for construction",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                arg_type = ArgumentType(kwargs.get("argument_type", "deductive"))
                return await self._construct_argument(conclusion, arg_type)
            return ToolCallResult(
                command="argument_structure",
                success=False,
                output="",
                error_message=f"Unknown action: {action}",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        except Exception as e:
            return ToolCallResult(
                command="argument_structure",
                success=False,
                output="",
                error_message=f"Argument structure analysis failed: {str(e)}",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )

    async def _analyze_argument_structure(self, text: str) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        premises = []
        conclusions = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(
                word in sentence_lower for word in ["therefore", "thus", "hence", "so"]
            ):
                conclusions.append(sentence)
            else:
                premises.append(sentence)
        if not conclusions and sentences:
            conclusions.append(premises.pop())
        output = "**ARGUMENT STRUCTURE ANALYSIS**\n\n"
        if premises:
            output += "**Premises:**\n"
            for i, premise in enumerate(premises, 1):
                output += f"{i}. {premise}\n"
        if conclusions:
            output += "\n**Conclusion(s):**\n"
            for conclusion in conclusions:
                output += f"• {conclusion}\n"
        validity = (
            "Valid structure" if premises and conclusions else "Structure unclear"
        )
        output += f"\n**Assessment:** {validity}\n"
        suggestions = []
        if len(premises) < 2:
            suggestions.append("Consider adding more supporting premises")
        if not conclusions:
            suggestions.append("Make conclusion more explicit")
        if suggestions:
            output += "\n**Suggestions:**\n"
            for suggestion in suggestions:
                output += f"• {suggestion}\n"
        return ToolCallResult(
            command="analyze_argument_structure",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _construct_argument(
        self,
        conclusion: str,
        arg_type: ArgumentType,
    ) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        output = f"**ARGUMENT CONSTRUCTION ({arg_type.value.upper()})**\n\n"
        if arg_type == ArgumentType.DEDUCTIVE:
            premises = [
                "All systems with property X exhibit behavior Y",
                "The system in question has property X",
            ]
            output += f"**Premises:**\n1. {premises[0]}\n2. {premises[1]}\n\n"
            output += f"**Conclusion:** {conclusion}\n\n"
            output += "**Structure:** Valid deductive reasoning (modus ponens pattern)"
        elif arg_type == ArgumentType.INDUCTIVE:
            premises = [
                "Multiple observed instances support this pattern",
                "No contradictory evidence has been found",
            ]
            output += f"**Premises:**\n1. {premises[0]}\n2. {premises[1]}\n\n"
            output += f"**Conclusion:** {conclusion}\n\n"
            output += "**Structure:** Inductive generalization (probabilistic)"
        elif arg_type == ArgumentType.DIALECTICAL:
            output += "**Thesis:** Initial position supporting the conclusion\n"
            output += "**Antithesis:** Counter-position challenging the thesis\n"
            output += f"**Synthesis:** {conclusion}\n\n"
            output += (
                "**Structure:** Dialectical progression toward higher understanding"
            )
        else:
            output += "**Supporting Evidence:** Relevant evidence and reasoning\n"
            output += f"**Conclusion:** {conclusion}\n\n"
            output += f"**Structure:** {arg_type.value} argumentation"
        return ToolCallResult(
            command="construct_argument",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            metadata={"argument_type": arg_type.value, "conclusion": conclusion},
            error_message=None,  # Added this line
        )
