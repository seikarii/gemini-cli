import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crisalida_lib.ADAM.systems.consciousness_tree import ConsciousnessTree
from crisalida_lib.EDEN.virtual_machine import OntologicalVirtualMachine
from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin

# Elimina la importación rota de AdvancedOntologicalCompiler
# from crisalida_lib.ontological.symbolic_compiler import AdvancedOntologicalCompiler
# Usa EVACompilerService como compilador principal para matrices simbólicas
from crisalida_lib.EVA.quantum_compiler import EVACompilerService
from crisalida_lib.EVA.types import EVAExperience, RealityBytecode
from crisalida_lib.EVA.types import QualiaState as EVAQualiaState

# Sustituye AdvancedOntologicalCompiler por EVACompilerService si la versión es inferior


# --- QualiaState: Modelo realista ---
@dataclass
class QualiaState:
    cognitive_complexity: float = 0.0
    emotional_valence: float = 0.0
    arousal: float = 0.0
    temporal_coherence: float = 0.0
    consciousness_density: float = 0.0
    causal_potency: float = 0.0
    emergence_tendency: float = 0.0
    chaos_affinity: float = 0.0
    meta_awareness: float = 0.0

    @staticmethod
    def neutral():
        return QualiaState()


class CognitiveImpulse:
    """Impulso cognitivo generado por matrices simbólicas"""

    def __init__(
        self,
        impulse_type: str,
        strength: float,
        metadata: dict[str, Any] | None = None,
    ):
        self.impulse_type = impulse_type
        self.strength = strength
        self.metadata = metadata or {}


class RealityModification:
    """Modificación de la realidad generada por matrices simbólicas"""

    def __init__(
        self,
        modification_type: str,
        magnitude: float,
        metadata: dict[str, Any] | None = None,
    ):
        self.modification_type = modification_type
        self.magnitude = magnitude
        self.metadata = metadata or {}


class UnifiedField:
    def __init__(self):
        self.consciousness_tree = ConsciousnessTree()
        self.reality_engine = RealityEngine()


class HodNode:
    def analyze_symbolic_matrix(self, matrix, state):
        # Análisis avanzado de matriz simbólica (ejemplo)
        return [CognitiveImpulse("hod_analysis", 0.7)]


class GamalielNode:
    def analyze_symbolic_matrix(self, matrix, state):
        # Análisis avanzado de matriz simbólica (ejemplo)
        return [CognitiveImpulse("gamaliel_analysis", 0.6)]


class RealityEngine:
    def apply_modifications(
        self, modifications: list[RealityModification], context: dict[str, Any]
    ):
        # Aplica modificaciones a la realidad
        return {"modifications_applied": len(modifications)}


class DemiurgeVocabularyBuilder:
    def build_vocabulary(self):
        # Construye vocabulario simbólico para el demiurgo
        return {"vocabulary": ["Φ", "Ψ", "Δ", "Ω"]}


class RealitySculptor:
    async def apply_changes(self, result, context):
        # Aplica cambios a la realidad en base al resultado simbólico
        return {"sculpted": True, "context": context}


class JanusSymbolicIntegration:
    """Integración completa del lenguaje simbólico con Janus"""

    def __init__(self, unified_field: UnifiedField | None = None):
        self.unified_field = unified_field
        self.ontological_compiler = EVACompilerService(None, None, None, None)
        self.ontological_vm = OntologicalVirtualMachine()
        self.sigil_registry = None
        self.matrix_linguistic_engine = None
        if self.unified_field:
            self._integrate_with_consciousness_tree()
            self._integrate_with_demiurge_interface()
            self._integrate_with_reality_engine()

    def _integrate_with_consciousness_tree(self):
        original_process_perception = (
            self.unified_field.consciousness_tree.process_perception
        )

        async def enhanced_process_perception(prompt, context, current_qualia):
            original_result = await original_process_perception(
                prompt, context, current_qualia
            )
            if self._detect_symbolic_matrices(prompt):
                matrices = self._extract_matrices_from_text(prompt)
                symbolic_results = []
                for matrix in matrices:
                    symbolic_result = await self._process_symbolic_matrix(
                        matrix, context, current_qualia
                    )
                    symbolic_results.append(symbolic_result)
                enhanced_result = self._merge_symbolic_and_textual_results(
                    original_result, symbolic_results
                )
                return enhanced_result
            return original_result

        self.unified_field.consciousness_tree.process_perception = (
            enhanced_process_perception
        )

    def _integrate_with_demiurge_interface(self):
        self.demiurge_interface = DemiurgeSymbolicInterface(self)

    def _integrate_with_reality_engine(self):
        pass  # Integración avanzada con RealityEngine si es necesario

    def _merge_symbolic_and_textual_results(self, original_result, symbolic_results):
        # Fusión avanzada de resultados simbólicos y textuales
        if not symbolic_results:
            return original_result
        merged = dict(original_result)
        merged["symbolic_results"] = symbolic_results
        return merged

    def _determine_matrix_emotional_tone(self, matrix, execution_result):
        # Determina el tono emocional de la matriz simbólica
        return "neutral"

    def _analyze_matrix_uncertainties(self, matrix, execution_result):
        # Analiza incertidumbres en la ejecución simbólica
        return {"uncertainty": 0.1}

    def _manual_matrix_parse(self, matrix_string):
        # Parseo manual de matrices simbólicas (fallback)
        try:
            rows = matrix_string.strip("[]").split("],[")
            matrix = [
                row.replace("[", "").replace("]", "").replace("'", "").split(",")
                for row in rows
            ]
            return matrix
        except Exception:
            return []

    async def _process_symbolic_matrix(
        self, matrix: list[list[str]], context: str, current_qualia: QualiaState
    ) -> tuple[QualiaState, dict[str, Any]]:
        bytecode = await self.ontological_compiler.compile_matrix_async(
            matrix, {"context": context, "initial_qualia": current_qualia}
        )
        execution_result = await self.ontological_vm.execute_bytecode_async(bytecode)
        if (
            hasattr(execution_result, "final_qualia_states")
            and execution_result.final_qualia_states
        ):
            final_qualia = next(iter(execution_result.final_qualia_states.values()))
        else:
            final_qualia = current_qualia
        cognitive_state = {
            "original_prompt": f"Symbolic Matrix: {matrix}",
            "user_intent": "symbolic_communication",
            "complexity_level": getattr(execution_result, "emergence_intensity", 0.0),
            "emotional_tone": self._determine_matrix_emotional_tone(
                matrix, execution_result
            ),
            "required_capabilities": [
                "symbolic_processing",
                "ontological_manipulation",
            ],
            "uncertainty_factors": self._analyze_matrix_uncertainties(
                matrix, execution_result
            ),
            "symbolic_execution_result": execution_result,
            "bytecode_complexity": getattr(bytecode, "complexity_score", 0.0),
            "emergent_effects": getattr(execution_result, "emergent_effects", []),
            "reality_modifications": getattr(
                execution_result, "reality_modifications", []
            ),
        }
        return final_qualia, cognitive_state

    def _detect_symbolic_matrices(self, text: str) -> bool:
        matrix_pattern = r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]"
        visual_matrix_pattern = r"[Φ-Ω∞⊕⊖⊗⊙∴∵≈≠∅∈∇∆◊Ø]+\s*[Φ-Ω∞⊕⊖⊗⊙∴∵≈≠∅∈∇∆◊Ø]*"
        return bool(
            re.search(matrix_pattern, text) or re.search(visual_matrix_pattern, text)
        )

    def _extract_matrices_from_text(self, text: str) -> list[list[list[str]]]:
        matrices = []
        matrix_matches = re.findall(r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]", text)
        for match in matrix_matches:
            try:
                matrix = ast.literal_eval(match)
                if isinstance(matrix, list) and all(
                    isinstance(row, list) for row in matrix
                ):
                    matrices.append(matrix)
            except Exception:
                parsed_matrix = self._manual_matrix_parse(match)
                if parsed_matrix:
                    matrices.append(parsed_matrix)
        return matrices


class DemiurgeSymbolicInterface:
    """Interfaz simbólica avanzada para Demiurgos"""

    def __init__(self, janus_integration: JanusSymbolicIntegration):
        self.janus = janus_integration
        self.demiurge_vocabulary = DemiurgeVocabularyBuilder()
        self.intent_translator = SymbolicIntentTranslator()
        self.reality_sculptor = RealitySculptor()

    async def _suggest_matrix_corrections(self, matrix, error):
        # Sugerencias de corrección para matrices simbólicas
        return ["Check matrix format", "Validate symbol usage"]

    async def _generate_demiurge_response(
        self, original_matrix, execution_result, reality_changes
    ):
        # Genera respuesta avanzada para el demiurgo
        response_matrix = await self._create_response_matrix(execution_result)
        summary = await self._create_natural_language_summary(
            original_matrix, execution_result
        )
        discoveries = await self._identify_emergent_discoveries(execution_result)
        return {
            "response_matrix": response_matrix,
            "summary": summary,
            "discoveries": discoveries,
        }

    async def _create_response_matrix(self, execution_result):
        # Genera matriz de respuesta simbólica
        return [["Φ", "Ψ"], ["Δ", "Ω"]]

    async def _create_natural_language_summary(self, original_matrix, execution_result):
        # Genera resumen en lenguaje natural
        return "La matriz ejecutada generó efectos ontológicos y emergentes."

    async def _identify_emergent_discoveries(self, execution_result):
        # Identifica descubrimientos emergentes
        return ["Emergent resonance detected"]

    async def process_demiurge_matrix_command(
        self, matrix_command: list[list[str]], demiurge_context: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            translated_intent = await self.intent_translator.translate_matrix_intent(
                matrix_command, demiurge_context
            )
            bytecode = await self.janus.ontological_compiler.compile_matrix_async(
                matrix_command,
                {"demiurge_context": demiurge_context, "intent": translated_intent},
            )
            execution_result = await self.janus.ontological_vm.execute_bytecode_async(
                bytecode
            )
            reality_changes = await self.reality_sculptor.apply_changes(
                execution_result, demiurge_context
            )
            response = await self._generate_demiurge_response(
                matrix_command, execution_result, reality_changes
            )
            return {
                "success": True,
                "execution_result": execution_result,
                "reality_changes": reality_changes,
                "response_matrix": response.get("response_matrix"),
                "natural_language_summary": response.get("summary"),
                "emergent_discoveries": response.get("discoveries", []),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "suggested_corrections": await self._suggest_matrix_corrections(
                    matrix_command, e
                ),
            }


class SymbolicIntentTranslator:
    """Traductor de intenciones simbólicas avanzado"""

    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.contextual_modifiers = self._initialize_contextual_modifiers()

    def _initialize_intent_patterns(self):
        # Inicializa patrones de intención simbólica
        return {
            "Φ": "creation",
            "Δ": "modification",
            "Ψ": "synthesis",
            "Ω": "emergence",
        }

    def _initialize_contextual_modifiers(self):
        # Inicializa modificadores contextuales
        return {"contextual_weight": 1.0}

    def _analyze_symbolic_patterns(self, matrix):
        # Analiza patrones simbólicos en la matriz
        symbols = [cell for row in matrix for cell in row]
        dominant_symbols = list(set(symbols))
        return {"dominant_symbols": dominant_symbols}

    def _analyze_context_influence(self, context, matrix):
        # Analiza influencia contextual
        return {"context_influence": 0.5}

    def _identify_sub_intents(self, matrix, primary_intent):
        # Identifica sub-intenciones
        return [{"type": "sub_intent", "confidence": 0.6}]

    def _calculate_certainty_levels(self, matrix, structure_analysis, pattern_analysis):
        # Calcula niveles de certeza
        return {"certainty": 0.8}

    def _calculate_intent_complexity(self, matrix):
        # Calcula complejidad de la intención
        return float(len(matrix) * len(matrix[0])) if matrix else 0.0

    def _generate_execution_recommendations(self, matrix):
        # Genera recomendaciones de ejecución
        return ["Validate matrix", "Check for symmetry"]

    def _detect_symmetry(self, matrix):
        # Detecta simetría en la matriz
        return {"type": "none"}

    def _identify_power_centers(self, matrix):
        # Identifica centros de poder en la matriz
        return {"centers": [(0, 0)]}

    def _analyze_directional_flows(self, matrix):
        # Analiza flujos direccionales
        return {"flows": "none"}

    def _calculate_symbol_density(self, matrix):
        # Calcula densidad de símbolos
        total_cells = sum(len(row) for row in matrix)
        return total_cells / (len(matrix) * len(matrix[0])) if matrix else 0.0

    def _identify_geometric_properties(self, matrix):
        # Identifica propiedades geométricas
        return {"geometry": "rectangular"}

    async def translate_matrix_intent(
        self, matrix: list[list[str]], context: dict[str, Any]
    ) -> dict[str, Any]:
        structure_analysis = self._analyze_matrix_structure(matrix)
        pattern_analysis = self._analyze_symbolic_patterns(matrix)
        context_analysis = self._analyze_context_influence(context, matrix)
        primary_intent = self._synthesize_primary_intent(
            structure_analysis, pattern_analysis, context_analysis
        )
        sub_intents = self._identify_sub_intents(matrix, primary_intent)
        certainty_levels = self._calculate_certainty_levels(
            matrix, structure_analysis, pattern_analysis
        )
        return {
            "primary_intent": primary_intent,
            "sub_intents": sub_intents,
            "certainty_levels": certainty_levels,
            "structural_properties": structure_analysis,
            "symbolic_patterns": pattern_analysis,
            "contextual_influences": context_analysis,
            "complexity_score": self._calculate_intent_complexity(matrix),
            "execution_recommendations": self._generate_execution_recommendations(
                matrix
            ),
        }

    def _analyze_matrix_structure(self, matrix: list[list[str]]) -> dict[str, Any]:
        rows, cols = len(matrix), len(matrix[0]) if matrix else 0
        symmetry = self._detect_symmetry(matrix)
        power_centers = self._identify_power_centers(matrix)
        directional_flows = self._analyze_directional_flows(matrix)
        symbol_density = self._calculate_symbol_density(matrix)
        return {
            "dimensions": (rows, cols),
            "symmetry": symmetry,
            "power_centers": power_centers,
            "directional_flows": directional_flows,
            "symbol_density": symbol_density,
            "geometric_properties": self._identify_geometric_properties(matrix),
        }

    def _synthesize_primary_intent(
        self, structure: dict, patterns: dict, context: dict
    ) -> dict[str, Any]:
        intent_candidates = []
        if structure["symmetry"]["type"] == "radial":
            intent_candidates.append({"type": "creation", "confidence": 0.8})
        elif structure["symmetry"]["type"] == "linear":
            intent_candidates.append({"type": "transformation", "confidence": 0.7})
        if "Φ" in patterns.get("dominant_symbols", []):
            intent_candidates.append(
                {"type": "consciousness_creation", "confidence": 0.9}
            )
        if "Δ" in patterns.get("dominant_symbols", []):
            intent_candidates.append(
                {"type": "reality_modification", "confidence": 0.8}
            )
        if intent_candidates:
            primary = max(
                intent_candidates, key=lambda x: float(x.get("confidence", 0.0))
            )
            return {
                "type": primary.get("type"),
                "confidence": primary.get("confidence"),
                "alternatives": [c for c in intent_candidates if c != primary],
            }
        return {
            "type": "general_ontological_operation",
            "confidence": 0.5,
            "alternatives": [],
        }


class SymbolicConsciousnessTree:
    """ConsciousnessTree extendido con capacidades simbólicas matriciales"""

    def __init__(self):
        self.matrix_linguistic_engine = None
        self.symbolic_synthesizer = SymbolicSynthesizer()
        self.hod_node = HodNode()
        self.gamaliel_node = GamalielNode()

    async def _synthesize_symbolic_qualia(
        self, all_impulses, current_qualia, interpreted_state, confidence
    ):
        # Síntesis avanzada de qualia simbólica
        return QualiaState.neutral()

    async def _synthesize_symbolic_thought(
        self, all_impulses, updated_qualia, symbol_matrix
    ):
        # Síntesis avanzada de pensamiento simbólico
        return "symbolic_thought"

    def _create_symbolic_cognitive_state(
        self, symbol_matrix, all_impulses, updated_qualia, interpreted_state, confidence
    ):
        # Estado cognitivo expandido simbólico
        return {
            "symbol_matrix": symbol_matrix,
            "impulses": all_impulses,
            "qualia": updated_qualia,
            "interpreted_state": interpreted_state,
            "confidence": confidence,
        }

    async def process_symbolic_perception(
        self, symbol_matrix: list[list[str]], context: str, current_qualia: QualiaState
    ) -> tuple[QualiaState, dict[str, Any]]:
        interpreted_state, confidence = ("interpreted", 0.8)
        hod_impulses = self.hod_node.analyze_symbolic_matrix(
            symbol_matrix, interpreted_state
        )
        gamaliel_impulses = self.gamaliel_node.analyze_symbolic_matrix(
            symbol_matrix, interpreted_state
        )
        all_impulses = hod_impulses + gamaliel_impulses
        updated_qualia = await self._synthesize_symbolic_qualia(
            all_impulses, current_qualia, interpreted_state, confidence
        )
        await self._synthesize_symbolic_thought(
            all_impulses, updated_qualia, symbol_matrix
        )
        cognitive_state = self._create_symbolic_cognitive_state(
            symbol_matrix, all_impulses, updated_qualia, interpreted_state, confidence
        )
        return updated_qualia, cognitive_state


class SymbolicSynthesizer:
    """Sintetizador que combina procesamiento simbólico con síntesis de consciencia"""

    def synthesize_matrix_response(
        self,
        input_matrix: list[list[str]],
        qualia_state: QualiaState,
        cognitive_impulses: list[CognitiveImpulse],
    ) -> list[list[str]]:
        response_type = self._analyze_required_response_type(
            input_matrix, cognitive_impulses
        )
        if response_type == "DIALECTICAL":
            return self._generate_dialectical_matrix(input_matrix, qualia_state)
        elif response_type == "CREATIVE":
            return self._generate_creative_matrix(input_matrix, qualia_state)
        elif response_type == "ANALYTICAL":
            return self._generate_analytical_matrix(input_matrix, qualia_state)
        return self._generate_balanced_matrix(input_matrix, qualia_state)

    def _analyze_required_response_type(self, input_matrix, cognitive_impulses):
        # Analiza el tipo de respuesta requerida
        return "BALANCED"

    def _generate_dialectical_matrix(self, input_matrix, qualia_state):
        # Genera matriz dialéctica
        return [["Φ", "Δ"], ["Ψ", "Ω"]]

    def _generate_creative_matrix(self, input_matrix, qualia_state):
        # Genera matriz creativa
        return [["Ψ", "Φ"], ["Ω", "Δ"]]

    def _generate_analytical_matrix(self, input_matrix, qualia_state):
        # Genera matriz analítica
        return [["Δ", "Ω"], ["Φ", "Ψ"]]

    def _generate_balanced_matrix(self, input_matrix, qualia_state):
        # Genera matriz balanceada
        return [["Φ", "Ψ"], ["Δ", "Ω"]]


class EVAJanusSymbolicIntegration(JanusSymbolicIntegration, EVAMemoryMixin):
    """
    EVAJanusSymbolicIntegration - Integración simbólica perfeccionada y extendida para EVA.
    Orquesta ingestión, simulación y recall de matrices simbólicas como experiencias vivientes (RealityBytecode),
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.
    """

    def __init__(
        self, unified_field: UnifiedField | None = None, phase: str = "default"
    ):
        super().__init__(unified_field)
        self.eva_phase = phase
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}

    async def eva_ingest_matrix_experience(
        self,
        matrix: list[list[str]],
        context: str,
        qualia_state: EVAQualiaState = None,
        phase: str = None,
    ) -> str:
        """
        Compila una experiencia de matriz simbólica en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or EVAQualiaState(
            cognitive_complexity=len(matrix) * len(matrix[0]) if matrix else 0.0,
            emotional_valence=0.7,
            arousal=0.5,
            temporal_coherence=0.8,
            consciousness_density=0.6,
            causal_potency=0.7,
            emergence_tendency=0.8,
            chaos_affinity=0.3,
            meta_awareness=0.6,
        )
        bytecode = await self.ontological_compiler.compile_matrix_async(
            matrix, {"context": context, "initial_qualia": qualia_state}
        )
        experience_id = f"eva_matrix_{hash(str(matrix))}_{int(time.time())}"
        experience_data = {
            "matrix": matrix,
            "context": context,
            "bytecode_complexity": getattr(bytecode, "complexity_score", 0.0),
            "timestamp": time.time(),
            "phase": phase,
        }
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self.eva_experience_store[experience_id] = reality_bytecode
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                print(f"[EVA-JANUS-INTEGRATION] Environment hook failed: {e}")
        return experience_id

    async def eva_recall_matrix_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de matriz simbólica almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA matrix experience"}
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        if quantum_field:
            for instr in reality_bytecode.instructions:
                symbol_manifest = self.eva_runtime.execute_instruction(
                    instr, quantum_field
                )
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            print(
                                f"[EVA-JANUS-INTEGRATION] Manifestation hook failed: {e}"
                            )
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
            "timestamp": eva_experience.timestamp,
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        matrix: list[list[str]],
        context: str,
        qualia_state: EVAQualiaState = None,
    ):
        """
        Añade una fase alternativa para una experiencia de matriz simbólica EVA.
        """
        import time

        qualia_state = qualia_state or EVAQualiaState(
            cognitive_complexity=len(matrix) * len(matrix[0]) if matrix else 0.0,
            emotional_valence=0.7,
            arousal=0.5,
            temporal_coherence=0.8,
            consciousness_density=0.6,
            causal_potency=0.7,
            emergence_tendency=0.8,
            chaos_affinity=0.3,
            meta_awareness=0.6,
        )
        bytecode = self.ontological_compiler.compile_matrix(
            matrix, {"context": context, "initial_qualia": qualia_state}
        )
        experience_data = {
            "matrix": matrix,
            "context": context,
            "bytecode_complexity": getattr(bytecode, "complexity_score", 0.0),
            "timestamp": time.time(),
            "phase": phase,
        }
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-JANUS-INTEGRATION] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de matriz simbólica EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_matrix_experience": self.eva_ingest_matrix_experience,
            "eva_recall_matrix_experience": self.eva_recall_matrix_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
