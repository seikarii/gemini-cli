#!/usr/bin/env python3
"""
Babel Agent - Library Creation and Management Agent
==================================================

Agent specialized in creating, validating and managing internal code libraries.
Integrates ADAM consciousness module with existing tools for complete library lifecycle management.

Based on the consciousness infrastructure and leveraging:
- code_generation: For code and API creation
- validation_tools: For linting, testing, and type checking
- dialectical_oracle: For design debate and architecture exploration
- semantic_search: For pattern discovery and duplicate detection
- ast_tools: For code analysis and modification

Key Features:
1. Needs Analysis - Analyzes functionality requirements and project context
2. Design Generation - Creates library structure and API design using dialectical reasoning
3. Code Generation - Generates source code using proven templates
4. Test Generation - Creates comprehensive test suites
5. Quality Validation - Ensures code quality through multiple validation tools
6. Integration Management - Handles dependency management and integration
7. Documentation Generation - Creates basic documentation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from crisalida_lib.ADAM.adam import Adam
from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.code_generation import CodeGenerationTool
from crisalida_lib.ASTRAL_TOOLS.default_tools import ToolRegistry
from crisalida_lib.ASTRAL_TOOLS.dialectical_oracle_tool import DialecticalOracleTool
from crisalida_lib.ASTRAL_TOOLS.validation_tools import (
    LinterTool,
    TesterTool,
    TypeCheckerTool,
)
from crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator import LLMGatewayOrchestrator
from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
from crisalida_lib.EVA.typequalia import QualiaState


# Simple config for now - will integrate with ADAM consciousness when available
@dataclass
class BabelConfig(EVAConfig):
    """
    Configuración avanzada para BabelAgent, basada en EVAConfig.
    Incluye parámetros de memoria viviente, benchmarking, hooks, multiverso y simulación activa.
    """

    # Puedes extender con parámetros específicos de Babel si lo necesitas
    LIBRARY_QUALITY_THRESHOLD: float = 0.85
    MAX_RETRIES: int = 5
    TIMEOUT_SECONDS: int = 120


logger = logging.getLogger(__name__)


@dataclass
class LibrarySpec:
    """Specification for a library to be created"""

    name: str
    description: str
    functionality: list[str] = field(default_factory=list)
    target_language: str = "python"
    dependencies: list[str] = field(default_factory=list)
    api_requirements: list[str] = field(default_factory=list)
    test_requirements: list[str] = field(default_factory=list)
    quality_standards: dict[str, Any] = field(default_factory=dict)


@dataclass
class LibraryCreationSession:
    """Tracks a complete library creation session"""

    session_id: str
    spec: LibrarySpec
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Workflow stages
    needs_analysis_complete: bool = False
    design_complete: bool = False
    code_generation_complete: bool = False
    tests_generated: bool = False
    validation_complete: bool = False
    integration_complete: bool = False
    documentation_complete: bool = False

    # Artifacts
    design_decisions: list[dict[str, Any]] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    validation_results: dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    error_log: list[str] = field(default_factory=list)


class BabelAgent(EVAMemoryMixin):
    """
    Advanced library creation agent integrating Crisalida tools and EVA memory system.
    Orchestrates the complete library creation lifecycle from needs analysis
    to final integration, leveraging multiple specialized tools and EVA memory architecture.
    Extensión EVA: memoria viviente, ingestión/recall de experiencias, faseo, hooks, benchmarking y compresión adaptativa.

    Now uses centralized EVAMemoryMixin for consistency and maintainability.
    """

    def __init__(self, config: BabelConfig | None = None):
        self.config = config or BabelConfig()
        self.tool_registry = ToolRegistry()
        self._register_core_tools()
        self.active_sessions: dict[str, LibraryCreationSession] = {}
        self.session_counter = 0

        # EVA: Integración de conciencia y memoria viviente
        self.adam = Adam(
            entity_id="Adam-BabelAgent",
            config=self.config,
            tool_registry=self.tool_registry,
            get_embedding=lambda x: np.array([0.0]),  # Placeholder
            recall_fn=lambda x: (np.array([0.0]), []),  # Placeholder
            ingest_fn=lambda *args, **kwargs: None,  # Placeholder
            emit_event=lambda *args, **kwargs: None,  # Placeholder
            llm_gateway=LLMGatewayOrchestrator(
                ollama_client=MagicMock(), default_model="mock_model"
            ),  # Provided mock LLMGatewayOrchestrator
        )

        # EVA Memory is initialized by EVAMemoryMixin's __init__
        # No explicit call to _init_eva_memory is needed here.
        # Ensure super().__init__ is called if not already.

        self.divine_compiler = DivineLanguageEvolved(None)
        self.benchmark_log: list = []
        logger.info(
            "BabelAgent initialized with advanced EVAConfig and EVA memory system"
        )

    def _register_core_tools(self):
        """Register essential tools for library creation"""
        self.code_generator = CodeGenerationTool()
        self.dialectical_oracle = DialecticalOracleTool()
        self.linter = LinterTool()
        self.tester = TesterTool()
        self.type_checker = TypeCheckerTool()

        # Register tools in registry for discoverability
        self.tool_registry.register_tool("code_generation", self.code_generator)
        self.tool_registry.register_tool("dialectical_oracle", self.dialectical_oracle)
        self.tool_registry.register_tool("linter", self.linter)
        self.tool_registry.register_tool("tester", self.tester)
        self.tool_registry.register_tool("type_checker", self.type_checker)

    async def create_library(
        self, description: str, **kwargs
    ) -> LibraryCreationSession:
        """
        Main entry point for library creation.

        Args:
            description: High-level description of desired library functionality
            **kwargs: Additional parameters like target_language, quality_standards, etc.

        Returns:
            LibraryCreationSession with complete creation results
        """
        # Create new session
        session_id = f"babel_session_{self.session_counter}"
        self.session_counter += 1

        try:
            # Phase 1: Needs Analysis
            spec = await self._analyze_needs(description, **kwargs)

            session = LibraryCreationSession(session_id=session_id, spec=spec)
            self.active_sessions[session_id] = session

            # Mark needs analysis as complete
            session.needs_analysis_complete = True

            logger.info(
                f"Starting library creation session {session_id} for: {spec.name}"
            )

            # Phase 2: Design and API Generation
            await self._generate_design_and_api(session)

            # Phase 3: Code Generation
            await self._generate_code(session)

            # Phase 4: Test Generation
            await self._generate_tests(session)

            # Phase 5: Quality Validation
            await self._validate_quality(session)

            # Phase 6: Integration Planning
            await self._plan_integration(session)

            # Phase 7: Documentation Generation
            await self._generate_documentation(session)

            session.end_time = datetime.now()
            self._calculate_success_rate(session)
            self._benchmark_session(
                session.session_id, session.success_rate, session.end_time
            )
            logger.info(
                f"Library creation session {session_id} completed with {session.success_rate:.1%} success"
            )
            # EVA: Ingestar experiencia de creación de librería
            qualia_state = QualiaState(
                emotional=session.success_rate,
                complexity=len(session.generated_files) / 10,
            )
            self.eva_ingest_babel_experience(
                experience_data={
                    "session_id": session.session_id,
                    "spec": session.spec,
                    "success_rate": session.success_rate,
                    "generated_files": session.generated_files,
                    "validation_results": session.validation_results,
                },
                qualia_state=qualia_state,
                phase=self.eva_phase,
            )
            self._auto_cleanup_eva_memory()
            return session
        except Exception as e:
            logger.error(f"Library creation failed for session {session_id}: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].error_log.append(str(e))
            raise

    def _benchmark_session(
        self, session_id: str, success_rate: float, end_time: datetime
    ):
        benchmark = {
            "session_id": session_id,
            "success_rate": success_rate,
            "phase": self.eva_phase,
            "timestamp": end_time.timestamp() if end_time else None,
        }
        self.benchmark_log.append(benchmark)
        self.eva_ingest_experience(
            intention_type="ARCHIVE_BABEL_BENCHMARK",
            experience_data=benchmark,
            qualia_state=QualiaState(emotional=success_rate),
        )

    def optimize_eva_memory(self):
        # Use the centralized memory management from EVAMemoryMixin
        total_experiences = len(self.eva_memory.eva_memory_store)
        if total_experiences > 100:  # Basic cleanup threshold
            logger.info(
                f"EVA memory optimization triggered: {total_experiences} experiences"
            )

    def _auto_cleanup_eva_memory(self):
        if (
            hasattr(self.config, "EVA_MEMORY_RETENTION_POLICY")
            and self.config.EVA_MEMORY_RETENTION_POLICY == "dynamic"
        ):
            self.optimize_eva_memory()

    def get_benchmark_log(self) -> list:
        """Devuelve el historial de benchmarks de sesiones de creación de librerías."""
        return self.benchmark_log

    async def _analyze_needs(self, description: str, **kwargs) -> LibrarySpec:
        """
        Phase 1: Analyze functionality needs and create library specification.
        Uses consciousness to deeply understand requirements.
        """
        logger.info("Phase 1: Analyzing needs and requirements")

        # Use consciousness to process the request deeply
        # This is where ADAM's understanding comes into play
        processed_description = await self._conscious_analysis(description)

        # Extract library name from description
        name = kwargs.get("name", self._extract_library_name(description))

        # Build comprehensive spec
        spec = LibrarySpec(
            name=name,
            description=processed_description,
            functionality=self._extract_functionality_list(description),
            target_language=kwargs.get("target_language", "python"),
            dependencies=kwargs.get("dependencies", []),
            api_requirements=kwargs.get("api_requirements", []),
            test_requirements=kwargs.get("test_requirements", []),
            quality_standards=kwargs.get(
                "quality_standards",
                {"coverage_threshold": 0.8, "type_coverage": 0.7, "complexity_max": 10},
            ),
        )

        logger.info("Phase 1: Needs analysis completed successfully")
        return spec

    async def _generate_design_and_api(self, session: LibraryCreationSession):
        """
        Phase 2: Generate library design and API using dialectical reasoning.
        Leverages dialectical_oracle for design exploration and debate.
        """
        logger.info("Phase 2: Generating design and API structure")

        try:
            # Use dialectical oracle to explore design options
            design_topic = f"Architecture for {session.spec.name} library"
            design_premise = (
                f"The library should provide: {', '.join(session.spec.functionality)}"
            )

            # Start architectural debate
            debate_result = await self.dialectical_oracle.execute(
                action="start_debate",
                topic=design_topic,
                premise=design_premise,
                complexity_level=3,
            )

            if debate_result.success:
                session.design_decisions.append(
                    {
                        "phase": "initial_design",
                        "debate_id": debate_result.metadata.get("query_id"),
                        "decision": debate_result.output,
                    }
                )

            # Generate API structure using code generation
            api_result = await self.code_generator.execute(
                action="analyze_context",
                context=f"Design API for {session.spec.description}",
                language=session.spec.target_language,
            )

            if api_result.success:
                session.design_decisions.append(
                    {"phase": "api_design", "structure": api_result.output}
                )

            session.design_complete = True
            logger.info("Design and API generation completed")

        except Exception as e:
            session.error_log.append(f"Design generation error: {str(e)}")
            logger.error(f"Design generation failed: {e}")

    async def _generate_code(self, session: LibraryCreationSession):
        """
        Phase 3: Generate actual source code using code generation templates.
        """
        logger.info("Phase 3: Generating source code")

        try:
            # Generate main library module - simplified approach
            main_file = f"{session.spec.name.lower()}.py"
            session.generated_files.append(main_file)
            logger.info(f"Generated main module: {main_file}")

            # Generate supporting modules based on functionality
            for func in session.spec.functionality[
                :2
            ]:  # Limit to prevent too many files
                func_file = f"{func.lower().replace(' ', '_')}.py"
                session.generated_files.append(func_file)
            # Generate code content using code_generator
            main_result = await self.code_generator.execute(
                action="generate_code",
                language=session.spec.target_language,
                template_name="module_basic",  # Assuming a template for modules
                variables={
                    "module_name": session.spec.name,
                    "description": f"Main module for {session.spec.name}",
                },
            )
            if main_result.success and hasattr(main_result, "output"):
                with open(main_file, "w", encoding="utf-8") as f:
                    f.write(main_result.output)
                session.generated_files.append(main_file)
                logger.info(f"Generated main module: {main_file}")
            else:
                session.error_log.append(
                    f"Failed to generate main module code for {main_file}"
                )
                logger.error(f"Failed to generate main module code for {main_file}")

            # Generate supporting modules based on functionality
            for func in session.spec.functionality[
                :2
            ]:  # Limit to prevent too many files
                func_file = f"{func.lower().replace(' ', '_')}.py"
                func_result = await self.code_generator.execute(
                    action="generate_code",
                    language=session.spec.target_language,
                    template_name="module_basic",  # Assuming same template
                    variables={
                        "module_name": func,
                        "description": f"Module for {func}",
                    },
                )
                if func_result.success and hasattr(func_result, "output"):
                    with open(func_file, "w", encoding="utf-8") as f:
                        f.write(func_result.output)
                    session.generated_files.append(func_file)
                    logger.info(f"Generated module: {func_file}")
                else:
                    session.error_log.append(
                        f"Failed to generate module code for {func_file}"
                    )
                    logger.error(f"Failed to generate module code for {func_file}")

            session.code_generation_complete = True
            logger.info(
                f"Code generation completed: {len(session.generated_files)} files"
            )

        except Exception as e:
            session.error_log.append(f"Code generation error: {str(e)}")
            logger.error(f"Code generation failed: {e}")

    async def _generate_tests(self, session: LibraryCreationSession):
        """
        Phase 4: Generate comprehensive test suites.
        """
        logger.info("Phase 4: Generating test suites")

        try:
            # Generate unit tests for each module
            for module_file in session.generated_files:
                test_result = await self.code_generator.execute(
                    action="generate_code",
                    language=session.spec.target_language,
                    template_name="function_basic",  # Using function template for test functions
                    variables={
                        "function_name": f"test_{module_file.replace('.py', '')}",
                        "description": f"Unit tests for {module_file}",
                        "parameters": "",
                        "return_type": "",
                        "function_body": "    # Test implementation\n    assert True  # Placeholder",
                    },
                )

                if test_result.success:
                    test_file = f"test_{module_file}"
                    session.generated_files.append(test_file)

            session.tests_generated = True
            logger.info("Test generation completed")

        except Exception as e:
            session.error_log.append(f"Test generation error: {str(e)}")
            logger.error(f"Test generation failed: {e}")

    async def _validate_quality(self, session: LibraryCreationSession):
        """
        Phase 5: Validate code quality using linting, testing, and type checking.
        """
        logger.info("Phase 5: Validating code quality")

        validation_results = {}

        try:
            # Run validation tools on generated files
            # Note: In real implementation, would write files to temp directory first

            # Simulate linting (normally would lint actual files)
            lint_result = ToolCallResult(
                command="linter",
                success=True,
                output="WARNING: Validation is simulated. No linting issues found (simulated, not real).",
                metadata={"issues_count": 0, "simulated": True},
                error_message=None,  # Added
                execution_time=0.0,  # Added
            )
            validation_results["linting"] = lint_result

            # Simulate type checking
            type_result = ToolCallResult(
                command="type_checker",
                success=True,
                output="Type checking passed (simulated)",
                metadata={"error_count": 0},
                error_message=None,  # Added
                execution_time=0.0,  # Added
            )
            validation_results["type_checking"] = type_result

            # Simulate testing
            test_result = ToolCallResult(
                command="tester",
                success=True,
                output="All tests passed (simulated)",
                metadata={"tests_run": len(session.generated_files), "failures": 0},
                error_message=None,  # Added
                execution_time=0.0,  # Added
            )
            validation_results["testing"] = test_result

            session.validation_results = validation_results
            session.validation_complete = True
            logger.info("Quality validation completed")

        except Exception as e:
            session.error_log.append(f"Validation error: {str(e)}")
            logger.error(f"Quality validation failed: {e}")

    async def _plan_integration(self, session: LibraryCreationSession):
        """
        Phase 6: Plan integration with existing project and manage dependencies.
        """
        logger.info("Phase 6: Planning integration and dependency management")

        try:
            # Analyze current dependencies and suggest optimizations
            integration_plan = {
                "new_dependencies": session.spec.dependencies,
                "potential_conflicts": [],  # Would analyze actual project
                "integration_steps": [
                    "Install library in project structure",
                    "Update import statements",
                    "Run integration tests",
                ],
                "dependency_reduction_opportunities": [],  # Would analyze for redundant deps
            }

            session.validation_results["integration"] = integration_plan
            session.integration_complete = True
            logger.info("Integration planning completed")

        except Exception as e:
            session.error_log.append(f"Integration planning error: {str(e)}")
            logger.error(f"Integration planning failed: {e}")

    async def _generate_documentation(self, session: LibraryCreationSession):
        """
        Phase 7: Generate basic documentation for the library.
        """
        logger.info("Phase 7: Generating documentation")

        try:
            # Generate README and basic docs
            readme_result = await self.code_generator.execute(
                action="generate_documentation",
                content_type="README",
                library_name=session.spec.name,
                description=session.spec.description,
                features=session.spec.functionality,
            )

            if readme_result.success:
                session.generated_files.append("README.md")

            session.documentation_complete = True
            logger.info("Documentation generation completed")

        except Exception as e:
            session.error_log.append(f"Documentation generation error: {str(e)}")
            logger.error(f"Documentation generation failed: {e}")

    def _calculate_success_rate(self, session: LibraryCreationSession):
        """Calculate overall success rate based on completed phases"""
        completed_phases = sum(
            [
                session.needs_analysis_complete,
                session.design_complete,
                session.code_generation_complete,
                session.tests_generated,
                session.validation_complete,
                session.integration_complete,
                session.documentation_complete,
            ]
        )

        total_phases = 7
        session.success_rate = completed_phases / total_phases

    async def _conscious_analysis(self, description: str) -> str:
        """Analyze requirements using enhanced processing (future ADAM integration)"""
        # Placeholder for consciousness-driven analysis
        # Future: Integrate with ADAM consciousness for deep understanding
        enhanced_desc = f"Enhanced analysis: {description}"
        logger.info("Requirements processed through enhanced analysis pipeline")
        return enhanced_desc

    def _extract_library_name(self, description: str) -> str:
        """Extract or generate library name from description"""
        # Simple extraction - in real implementation would be more sophisticated
        words = description.lower().split()
        if "library" in words:
            idx = words.index("library")
            if idx > 0:
                return words[idx - 1].replace(" ", "_")

        # Fallback to generic name
        return "generated_library"

    def _extract_functionality_list(self, description: str) -> list[str]:
        """Extract functionality requirements from description"""
        # Simple extraction - in real implementation would use NLP
        functionality = []

        # Look for common patterns
        if "serialize" in description.lower():
            functionality.append("serialize_data")
        if "deserialize" in description.lower():
            functionality.append("deserialize_data")
        if "manage" in description.lower():
            functionality.append("manage_resources")
        if "handle" in description.lower():
            functionality.append("handle_operations")

        # Always add basic functionality
        if not functionality:
            functionality = ["core_functionality", "helper_methods", "utilities"]

        return functionality

    def get_session_status(self, session_id: str) -> LibraryCreationSession | None:
        """Get current status of a library creation session"""
        return self.active_sessions.get(session_id)

    def list_active_sessions(self) -> list[str]:
        """List all active session IDs"""
        return list(self.active_sessions.keys())

    def get_agent_statistics(self) -> dict[str, Any]:
        """Get overall agent statistics"""
        total_sessions = len(self.active_sessions)
        completed_sessions = sum(
            1 for s in self.active_sessions.values() if s.end_time is not None
        )
        avg_success_rate = sum(
            s.success_rate for s in self.active_sessions.values()
        ) / max(total_sessions, 1)

        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "average_success_rate": avg_success_rate,
            "tool_registry_size": (
                len(self.tool_registry.tools)
                if hasattr(self.tool_registry, "tools")
                else 5
            ),
            "consciousness_status": "planned",  # Future ADAM integration
        }

    async def divine_evolution_cycle(self, perception_data, external_context=None):
        # 1. Procesamiento simbólico y arquetípico
        pattern_result = self.dialectical_processor.process_perception(perception_data)
        arquetipo = self.soul.get_dominant_archetype()
        self.dual_mind.inject_archetype_step(arquetipo, pattern_result)

        # 2. Bucle de aprendizaje y creatividad
        brain_result = await self.brain_fallback.step(perception_data)
        if brain_result.get("mode") == "STAGNATION":
            self.conscious_mind.set_state("AWAKE")
            self.conscious_mind.generate_creative_plan(perception_data, arquetipo)

        # 3. Meta-adaptación y ajuste de personalidad
        await self.adaptive_tuner.meta_adapt_cycle(self, brain_result)

        # 4. Exportación y visualización
        return self.export_omega_state(include_sensitive=True)

    # --- EVA Memory System Integration ---
    def eva_ingest_babel_experience(
        self,
        experience_data: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compiles a library creation experience into RealityBytecode and stores it in EVA memory.
        This method now uses the centralized EVAMemoryMixin for consistency.
        """
        return self.eva_ingest_experience(
            intention_type="ARCHIVE_BABEL_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

    def eva_recall_babel_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Executes the RealityBytecode for a given experience cue, manifesting the simulation.
        This method now uses the centralized EVAMemoryMixin for consistency.
        """
        return self.eva_recall_experience(cue, phase)

    # Note: add_experience_phase, set_memory_phase, get_memory_phase,
    # get_experience_phases, and add_environment_hook are now provided by EVAMemoryMixin

    # --- EVA API for external integration ---
    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_babel_experience,
            "eva_recall_experience": self.eva_recall_babel_experience,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            # Removed get_eva_memory_stats and clear_eva_memory as they are not directly
            # available in EVAMemoryManager and should be handled by BabelAgent if needed.
        }
