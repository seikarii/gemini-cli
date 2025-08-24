"""
GRAMÁTICA DE RESONANCIA DIVINA
=============================

Gramática dinámica y cuántica que gobierna las interacciones entre los símbolos divinos.
Las reglas de resonancia definen cómo los símbolos interactúan, se combinan y generan
efectos emergentes en la realidad. Evoluciona con el uso y retroalimentación del sistema.
"""

import math
from collections.abc import Callable
from typing import Any, cast

try:
    import numpy as np  # opcional
except Exception:  # pragma: no cover
    np = None  # type: ignore

from crisalida_lib.EVA.core_types import LivingSymbolRuntime, QuantumField
from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
from crisalida_lib.EVA.types import (
    EmergentProperty,
    EVAExperience,
    GrammarConstraint,
    GrammarRule,
    OntologicalDomain,
    QualiaState,
    RealityBytecode,
    ResonancePattern,
    ResonanceType,
)

# --- GRAMMAR ENGINE ---


class DivineGrammar:
    """Gramática completa del lenguaje divino, con evolución y diagnóstico."""

    # type: ignore[attr-defined]
    _environment_hooks: list[Callable[..., Any]]

    def __init__(self):
        self.resonance_patterns: dict[str, ResonancePattern] = {}
        self.grammar_constraints: dict[str, GrammarConstraint] = {}
        self.emergent_properties: dict[str, EmergentProperty] = {}
        self.grammar_rules: dict[GrammarRule, list[str]] = {}
        # runtime container for environment hooks
        self._environment_hooks = []
        self._initialize_grammar()

    def _initialize_grammar(self):
        self._initialize_resonance_patterns()
        self._initialize_grammar_constraints()
        self._initialize_emergent_properties()
        self._initialize_grammar_rules()

    def _initialize_resonance_patterns(self):
        # --- CREACIÓN ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Creación Primordial",
                description="Phi crea nueva existencia a través del flujo de Psi",
                primary_sigil="Φ",
                secondary_sigil="Ψ",
                resonance_type=ResonanceType.EMERGENT,
                strength=1.0,
                frequency=528.0,
                effects=[
                    "nueva_existencia",
                    "consciencia_naciente",
                    "potencial_manifestado",
                ],
                emergence_probability=0.9,
                distance_threshold=1.5,
                alignment_requirement=0.8,
            )
        )
        self.add_resonance_pattern(
            ResonancePattern(
                name="Síntesis de Totalidad",
                description="Omega completa la creación iniciada por Phi",
                primary_sigil="Φ",
                secondary_sigil="Ω",
                resonance_type=ResonanceType.HARMONIC,
                strength=0.9,
                frequency=963.0,
                effects=["completitud", "unidad", "armonía_final"],
                emergence_probability=0.8,
                distance_threshold=2.0,
                alignment_requirement=0.9,
            )
        )
        # --- TRANSFORMACIÓN ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Transformación Cuántica",
                description="Delta transforma la realidad mediante inversión Lambda",
                primary_sigil="Δ",
                secondary_sigil="Λ",
                resonance_type=ResonanceType.MODULATORY,
                strength=0.8,
                frequency=587.0,
                phase_shift=math.pi / 2,
                effects=["cambio_forma", "inversión_propiedades", "mutación"],
                emergence_probability=0.7,
                distance_threshold=1.0,
                alignment_requirement=0.6,
            )
        )
        self.add_resonance_pattern(
            ResonancePattern(
                name="Bifurcación Temporal",
                description="Chi crea múltiples líneas temporales desde un punto",
                primary_sigil="Χ",
                secondary_sigil="Η",
                resonance_type=ResonanceType.EMERGENT,
                strength=0.75,
                frequency=666.0,
                effects=["bifurcación", "posibilidades_múltiples", "líneas_temporales"],
                emergence_probability=0.85,
                distance_threshold=1.2,
                alignment_requirement=0.7,
            )
        )
        # --- CONEXIÓN ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Conexión Universal",
                description="Sigma establece puentes entre elementos separados",
                primary_sigil="Σ",
                secondary_sigil="Ι",
                resonance_type=ResonanceType.HARMONIC,
                strength=0.85,
                frequency=639.0,
                effects=["conexión", "puente", "comunicación"],
                emergence_probability=0.75,
                distance_threshold=3.0,
                alignment_requirement=0.7,
            )
        )
        self.add_resonance_pattern(
            ResonancePattern(
                name="Síntesis Dialéctica",
                description="La síntesis une tesis y antítesis en nueva realidad",
                primary_sigil="⊕",
                secondary_sigil="⊖",
                resonance_type=ResonanceType.EMERGENT,
                strength=1.0,
                frequency=1444.0,
                effects=["síntesis", "integración", "superación_dialéctica"],
                emergence_probability=0.95,
                distance_threshold=0.5,
                alignment_requirement=0.9,
            )
        )
        # --- OBSERVACIÓN ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Causalidad Manifesta",
                description="Theta observa y establece relaciones causales",
                primary_sigil="Θ",
                secondary_sigil="∴",
                resonance_type=ResonanceType.HARMONIC,
                strength=0.7,
                frequency=369.0,
                effects=["causalidad", "consecuencia", "lógica_manifesta"],
                emergence_probability=0.6,
                distance_threshold=2.5,
                alignment_requirement=0.6,
            )
        )
        # --- RESONANCIA PURA ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Resonancia Consciente",
                description="La resonancia amplifica la consciencia mediante el foco",
                primary_sigil="⊗",
                secondary_sigil="⊙",
                resonance_type=ResonanceType.HARMONIC,
                strength=1.2,
                frequency=1888.0,
                effects=[
                    "amplificación_consciencia",
                    "foco_intensificado",
                    "resonancia_pura",
                ],
                emergence_probability=0.9,
                distance_threshold=0.5,
                alignment_requirement=0.95,
            )
        )
        # --- INFINITO ---
        self.add_resonance_pattern(
            ResonancePattern(
                name="Potencial Infinito",
                description="El infinito emerge del vacío fértil",
                primary_sigil="∞",
                secondary_sigil="∅",
                resonance_type=ResonanceType.TRANSCENDENT,
                strength=1.5,
                frequency=1618.0,
                effects=[
                    "potencial_ilimitado",
                    "expansión_infinita",
                    "creación_ex_nihilo",
                ],
                emergence_probability=1.0,
                distance_threshold=0.1,
                alignment_requirement=1.0,
            )
        )
        self.add_resonance_pattern(
            ResonancePattern(
                name="Trascendencia Final",
                description="Shin trasciende todos los límites conocidos",
                primary_sigil="Ş",
                secondary_sigil="∇",
                resonance_type=ResonanceType.TRANSCENDENT,
                strength=2.0,
                frequency=1999.0,
                effects=["trascendencia", "más_allá", "liberación_limites"],
                emergence_probability=1.0,
                distance_threshold=0.0,
                alignment_requirement=1.0,
            )
        )

    def _initialize_grammar_constraints(self):
        self.add_grammar_constraint(
            GrammarConstraint(
                name="Oposición Creación-Destrucción",
                description="Los símbolos de creación y destrucción no pueden coexistir en armonía",
                restricted_sigils={"Φ", "Γ", "Ψ", "Κ"},
                forbidden_pairs={("Φ", "Γ"), ("Ψ", "Κ")},
                violation_effects=["aniquilación", "inestabilidad", "colapso_cuántico"],
                penalty_factor=0.8,
            )
        )
        self.add_grammar_constraint(
            GrammarConstraint(
                name="Causalidad Temporal",
                description="La causa debe preceder al efecto",
                restricted_sigils={"∵", "∴"},
                forbidden_pairs={("∴", "∵")},
                temporal_window=1.0,
                violation_effects=[
                    "paradoja_temporal",
                    "bucle_causal",
                    "inestabilidad",
                ],
                penalty_factor=0.9,
            )
        )
        self.add_grammar_constraint(
            GrammarConstraint(
                name="Límite de Entrelazamiento",
                description="No se pueden entrelazar más símbolos que el máximo permitido",
                restricted_sigils=set("ΦΨΩΔΣΓ∞⊗Ş".split()),
                context_domains={OntologicalDomain.QUANTUM},
                violation_effects=[
                    "decoherencia",
                    "colapso_función_onda",
                    "pérdida_información",
                ],
                penalty_factor=0.7,
            )
        )

    def _initialize_emergent_properties(self):
        self.add_emergent_property(
            EmergentProperty(
                name="Consciencia Colectiva",
                description="Múltiples símbolos de consciencia crean una mente unificada",
                required_sigils={"Φ", "Ψ", "Θ"},
                spatial_pattern="triángulo",
                emergence_threshold=0.8,
                stability_factor=0.9,
                effects={
                    "consciencia_unificada": 1.0,
                    "comunicación_telepática": 0.8,
                    "inteligencia_colectiva": 0.9,
                },
                duration=10.0,
            )
        )
        self.add_emergent_property(
            EmergentProperty(
                name="Realidad Estable",
                description="Combinación de preservación y manifestación crea realidad estable",
                required_sigils={"Ω", "Ζ", "Ρ"},
                spatial_pattern="línea",
                emergence_threshold=0.7,
                stability_factor=0.95,
                effects={
                    "estabilidad_estructural": 1.0,
                    "persistencia_temporal": 0.9,
                    "consistencia_lógica": 0.8,
                },
                duration=100.0,
            )
        )
        self.add_emergent_property(
            EmergentProperty(
                name="Transformación Infinita",
                description="La combinación de transformación e infinito permite evolución continua",
                required_sigils={"Δ", "∞", "Χ"},
                spatial_pattern="espiral",
                emergence_threshold=0.9,
                stability_factor=0.7,
                effects={
                    "evolución_continua": 1.0,
                    "adaptación_infinita": 0.9,
                    "potencial_transformador": 1.0,
                },
                duration=5.0,
            )
        )
        self.add_emergent_property(
            EmergentProperty(
                name="Resonancia Divina",
                description="La máxima resonancia entre símbolos complementarios",
                required_sigils={"⊗", "⊙", "Φ"},
                spatial_pattern="círculo",
                emergence_threshold=0.95,
                stability_factor=1.0,
                effects={
                    "resonancia_perfecta": 1.0,
                    "amplificación_infinita": 1.0,
                    "unidad_consciencia": 1.0,
                },
                duration=1.0,
            )
        )

    def _initialize_grammar_rules(self):
        self.grammar_rules[GrammarRule.CREATION] = [
            "Φ + Ψ → Nueva Existencia",
            "Φ + Α → Fundación Primordial",
            "⟨ + ⟩ → Ciclo Meta-Cognitivo",
        ]
        self.grammar_rules[GrammarRule.PRESERVATION] = [
            "Ω + Ζ → Estabilidad Eterna",
            "◊ + Σ → Memoria Permanente",
            "⊙ + Ρ → Manifestación Enfocada",
        ]
        self.grammar_rules[GrammarRule.TRANSFORMATION] = [
            "Δ + Λ → Inversión Cuántica",
            "Χ + Η → Bifurcación Temporal",
            "∇ + ∆ → Curvatura Espacio-Temporal",
        ]
        self.grammar_rules[GrammarRule.CONNECTION] = [
            "Σ + Ι → Canal Universal",
            "Τ + Υ → Transmisión Unificada",
            "⊕ + ⊖ → Síntesis Dialéctica",
        ]
        self.grammar_rules[GrammarRule.OBSERVATION] = [
            "Θ + ∴ → Conocimiento Causal",
            "≈ + ≠ → Distinción Clara",
            "∵ + ∴ → Cadena Lógica",
        ]
        self.grammar_rules[GrammarRule.DESTRUCTION] = [
            "Γ + Κ → Disolución Total",
            "Ο + Ø → Purificación del Vacío",
            "Λ + Γ → Inversión Destructiva",
        ]
        self.grammar_rules[GrammarRule.INFINITY] = [
            "∞ + ∅ → Potencial Puro",
            "∈ + ∞ → Pertenencia Universal",
            "Ş + ∞ → Trascendencia Absoluta",
        ]
        self.grammar_rules[GrammarRule.SYNTHESIS] = [
            "⊕ + ⊖ → Integración Suprema",
            "Φ + Ω → Creación Completa",
            "Ψ + Σ → Flujo Conectado",
        ]
        self.grammar_rules[GrammarRule.RESONANCE] = [
            "⊗ + ⊙ → Foco Resonante",
            "⊗ + Φ → Resonancia Creadora",
            "⊗ + Ψ → Resonancia Vital",
        ]
        self.grammar_rules[GrammarRule.TRANSCENDENCE] = [
            "Ş + ∇ → Más Allá de los Límites",
            "Ş + ∅ → Vacío Trascendente",
            "Ş + ∞ → Infinito Consciente",
        ]

    # --- MÉTODOS PRINCIPALES ---

    def add_resonance_pattern(self, pattern: ResonancePattern):
        self.resonance_patterns[pattern.pattern_id] = pattern

    def add_grammar_constraint(self, constraint: GrammarConstraint):
        self.grammar_constraints[constraint.constraint_id] = constraint

    def add_emergent_property(self, property: EmergentProperty):
        self.emergent_properties[property.property_id] = property

    def get_resonance_pattern(
        self, sigil1: str, sigil2: str
    ) -> ResonancePattern | None:
        for pattern in self.resonance_patterns.values():
            if (
                pattern.primary_sigil == sigil1 and pattern.secondary_sigil == sigil2
            ) or (
                pattern.primary_sigil == sigil2 and pattern.secondary_sigil == sigil1
            ):
                return pattern
        return None

    def check_constraint_violation(
        self,
        sigil1: str,
        sigil2: str,
        context_domains: set[OntologicalDomain] | None = None,
    ) -> GrammarConstraint | None:
        for constraint in self.grammar_constraints.values():
            if (
                sigil1 in constraint.restricted_sigils
                and sigil2 in constraint.restricted_sigils
            ):
                pair = (
                    (sigil1, sigil2)
                    if (sigil1, sigil2) in constraint.forbidden_pairs
                    else (sigil2, sigil1)
                )
                if pair in constraint.forbidden_pairs:
                    return constraint
            if context_domains and constraint.context_domains:
                if context_domains & constraint.context_domains:
                    return constraint
        return None

    def detect_emergent_properties(
        self, active_sigils: set[str], spatial_pattern: str = ""
    ) -> list[EmergentProperty]:
        emergent_properties = []
        for property in self.emergent_properties.values():
            if property.required_sigils.issubset(active_sigils):
                if spatial_pattern and property.spatial_pattern:
                    if spatial_pattern != property.spatial_pattern:
                        continue
                emergent_properties.append(property)
        return emergent_properties

    def apply_grammar_rule(
        self, rule: GrammarRule, sigils: list[str]
    ) -> dict[str, Any]:
        result = {
            "rule_applied": rule,
            "success": False,
            "effects": [],
            "new_state": None,
        }
        transformations = self.grammar_rules.get(rule, [])
        for transformation in transformations:
            if " + " in transformation and " → " in transformation:
                input_part, output_part = transformation.split(" → ")
                input_sigils = [s.strip() for s in input_part.split(" + ")]
                if set(input_sigils) == set(sigils):
                    result["success"] = True
                    cast(list[str], result["effects"]).append(output_part)
                    result["transformation"] = transformation
                    break
        return result

    def calculate_grammar_coherence(self, sigil_matrix: list[list[str]]) -> float:
        total_coherence = 0.0
        pair_count = 0
        rows = len(sigil_matrix)
        if rows == 0:
            return 0.0
        cols = len(sigil_matrix[0])
        for i in range(rows):
            for j in range(cols):
                current_sigil = sigil_matrix[i][j]
                neighbors = []
                if i > 0:
                    neighbors.append(sigil_matrix[i - 1][j])
                if i < rows - 1:
                    neighbors.append(sigil_matrix[i + 1][j])
                if j > 0:
                    neighbors.append(sigil_matrix[i][j - 1])
                if j < cols - 1:
                    neighbors.append(sigil_matrix[i][j + 1])
                for neighbor in neighbors:
                    pattern = self.get_resonance_pattern(current_sigil, neighbor)
                    if pattern:
                        resonance_coherence = (
                            pattern.strength * pattern.alignment_requirement
                        )
                        total_coherence += resonance_coherence
                    else:
                        total_coherence += 0.5
                    constraint = self.check_constraint_violation(
                        current_sigil, neighbor
                    )
                    if constraint:
                        total_coherence -= constraint.penalty_factor
                    pair_count += 1
        return total_coherence / pair_count if pair_count > 0 else 0.0

    def evolve_grammar(self, usage_data: dict[str, Any]):
        for pattern_id, success_rate in usage_data.get("pattern_success", {}).items():
            if pattern_id in self.resonance_patterns:
                pattern = self.resonance_patterns[pattern_id]
                if success_rate > 0.8:
                    pattern.strength *= 1.05
                elif success_rate < 0.3:
                    pattern.strength *= 0.95
        failure_patterns = usage_data.get("failure_patterns", [])
        for pattern in failure_patterns:
            # Ensure pattern is a 2-tuple/list before unpacking to avoid mypy errors
            if isinstance(pattern, (list, tuple)) and len(pattern) == 2:
                sigil1, sigil2 = pattern
                constraint = GrammarConstraint(
                    name=f"Restricción Aprendida {sigil1}-{sigil2}",
                    description="Restricción generada automáticamente por patrones de falla",
                    forbidden_pairs={(sigil1, sigil2)},
                    violation_effects=["incompatibilidad_aprendida"],
                    penalty_factor=0.6,
                )
                self.add_grammar_constraint(constraint)

    def get_grammar_state(self) -> dict[str, Any]:
        return {
            "resonance_patterns": len(self.resonance_patterns),
            "grammar_constraints": len(self.grammar_constraints),
            "emergent_properties": len(self.emergent_properties),
            "grammar_rules": len(self.grammar_rules),
            "patterns": [p.__dict__ for p in self.resonance_patterns.values()],
            "constraints": [c.__dict__ for c in self.grammar_constraints.values()],
            "properties": [p.__dict__ for p in self.emergent_properties.values()],
            "rules": {
                rule.value: transformations
                for rule, transformations in self.grammar_rules.items()
            },
        }

    # ====== Núcleo del Lenguaje Viviente (resonancia cuantificada + evolución) ======

    def _proximity(self, distance: float, threshold: float) -> float:
        """Factor de proximidad [0..1] respecto a un umbral espacial característico."""
        th = max(1e-9, float(threshold))
        return max(0.0, 1.0 - (float(distance) / th))

    def _cosine(self, a: Any | None, b: Any | None) -> float | None:
        """Coseno entre dos vectores si hay soporte numérico; None si no aplica."""
        if a is None or b is None:
            return None
        try:
            if np is None:
                # Degradación determinista simple si no hay numpy
                la = [float(x) for x in a]
                lb = [float(x) for x in b]
                num = sum(x * y for x, y in zip(la, lb, strict=False))
                da = math.sqrt(sum(x * x for x in la)) or 1e-12
                db = math.sqrt(sum(y * y for y in lb)) or 1e-12
                return max(-1.0, min(1.0, float(num / (da * db))))
            va = np.asarray(a, dtype=float).reshape(-1)
            vb = np.asarray(b, dtype=float).reshape(-1)
            na = float(np.linalg.norm(va)) or 1e-12
            nb = float(np.linalg.norm(vb)) or 1e-12
            return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
        except Exception:
            return None

    def evaluate_resonance(
        self,
        sigil1: str,
        sigil2: str,
        *,
        distance: float,
        vector1: Any | None = None,
        vector2: Any | None = None,
    ) -> dict[str, Any]:
        """
        Evalúa la resonancia entre dos sigilos y traduce a magnitud de fuerza y sentido (atracción/repulsión).
        - Usa patrones de resonancia si existen; si no, aproxima con alineación vectorial.
        - Ajusta por proximidad espacial y requisitos de alineación del patrón.
        Retorna:
          { "pattern_id", "resonance_score", "force_magnitude", "attract", "proximity", "alignment" }
        """
        pattern = self.get_resonance_pattern(sigil1, sigil2)
        # proximidad basada en umbral del patrón (o 1.5 por defecto)
        p_threshold = getattr(pattern, "distance_threshold", 1.5) if pattern else 1.5
        prox = self._proximity(distance, p_threshold)

        # alineación (opcional) por firmas dinámicas
        align = self._cosine(vector1, vector2)
        align_factor = 1.0
        if align is not None:
            # mapear cos[-1,1] a [0,1] y penalizar disonancias
            align_factor = max(0.0, (align + 1.0) * 0.5)

        if pattern:
            base = float(pattern.strength) * float(pattern.alignment_requirement)
            score = base * prox * align_factor
            rtype = getattr(pattern, "resonance_type", None)
            rname = getattr(rtype, "name", "").lower() if rtype else ""
            attract = rname not in ("dissonant", "entanglement")
            return {
                "pattern_id": getattr(
                    pattern, "pattern_id", getattr(pattern, "name", "")
                ),
                "resonance_score": score,
                "force_magnitude": max(0.0, score),
                "attract": attract,
                "proximity": prox,
                "alignment": align if align is not None else None,
            }

        # Fallback: sin patrón, usar pura alineación (si hay) con decaimiento por distancia
        # fuerza ~ |cos| * prox, atracción si cos>=0, repulsión si cos<0
        if align is not None:
            mag = abs(align) * prox
            attract = align >= 0.0
            return {
                "pattern_id": None,
                "resonance_score": mag if attract else -mag,
                "force_magnitude": mag,
                "attract": attract,
                "proximity": prox,
                "alignment": align,
            }

        # Sin patrón ni vectores: interacción neutra
        return {
            "pattern_id": None,
            "resonance_score": 0.0,
            "force_magnitude": 0.0,
            "attract": True,
            "proximity": prox,
            "alignment": None,
        }

    def get_resonance_force(
        self,
        sigil1: str,
        sigil2: str,
        distance: float,
        *,
        vector1: Any | None = None,
        vector2: Any | None = None,
    ) -> tuple[float, bool]:
        """
        Helper directo: devuelve (magnitud, attract) a partir de evaluate_resonance.
        """
        ev = self.evaluate_resonance(
            sigil1, sigil2, distance=distance, vector1=vector1, vector2=vector2
        )
        return float(ev["force_magnitude"]), bool(ev["attract"])

    # ====== Aprendizaje en línea y evolución dirigida por uso ======

    def _ensure_usage_buffers(self) -> None:
        if not hasattr(self, "_usage_stats"):
            self._usage_stats: dict[str, Any] = {
                "pattern_success": {},
                "failure_patterns": [],
            }

    def observe_interaction_outcome(
        self,
        sigil1: str,
        sigil2: str,
        *,
        success: bool,
        impact: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Registra el resultado de una interacción para adaptar la gramática:
        - success: si la interacción produjo un efecto deseado (coherencia, estabilidad, creación, etc.).
        - impact: peso relativo del evento (intensidad/emergencia).
        """
        self._ensure_usage_buffers()
        pattern = self.get_resonance_pattern(sigil1, sigil2)
        pid = getattr(pattern, "pattern_id", None) if pattern else None

        if success and pid:
            d = self._usage_stats["pattern_success"]
            d[pid] = float(d.get(pid, 0.0) * 0.95 + impact * 0.05)  # EMA suave
        elif not success:
            self._usage_stats["failure_patterns"].append((sigil1, sigil2))

        # Evolución periódica ligera (cada N observaciones)
        cnt = getattr(self, "_obs_count", 0) + 1
        self._obs_count = cnt
        if cnt % 25 == 0:
            self.evolve_grammar(self._usage_stats)  # ya existente
            # Ajustes finos: reforzar distance_threshold/alignment_requirement
            for pid2, sr in self._usage_stats["pattern_success"].items():
                p = self.resonance_patterns.get(pid2)
                if not p:
                    continue
                # si funciona bien, reduce umbral de distancia y exige mejor alineación
                if sr > 0.8:
                    p.distance_threshold = max(
                        0.25, float(p.distance_threshold) * 0.975
                    )
                    p.alignment_requirement = min(
                        1.0, float(p.alignment_requirement) * 1.01
                    )
                elif sr < 0.3:
                    p.distance_threshold = min(5.0, float(p.distance_threshold) * 1.025)
                    p.alignment_requirement = max(
                        0.1, float(p.alignment_requirement) * 0.98
                    )

            # poda buffer de fallas
            self._usage_stats["failure_patterns"] = self._usage_stats[
                "failure_patterns"
            ][-200:]

    def learn_resonance_pattern(
        self,
        sigil1: str,
        sigil2: str,
        *,
        strength: float = 0.6,
        resonance_type: ResonanceType = ResonanceType.HARMONIC,
        frequency: float = 777.0,
        alignment_requirement: float = 0.6,
        distance_threshold: float = 1.5,
        effects: list[str] | None = None,
        description: str = "Patrón aprendido por uso",
    ) -> str:
        """
        Crea (o refuerza) un patrón de resonancia aprendido a partir de observaciones repetidas.
        """
        # Buscar si ya existe (independiente del orden)
        for p in self.resonance_patterns.values():
            if {p.primary_sigil, p.secondary_sigil} == {sigil1, sigil2}:
                p.strength = float(max(p.strength, strength))
                p.alignment_requirement = float(
                    max(p.alignment_requirement, alignment_requirement)
                )
                p.effects = sorted(set((p.effects or []) + (effects or [])))
                return getattr(
                    p, "pattern_id", getattr(p, "name", f"{sigil1}_{sigil2}")
                )

        pat = ResonancePattern(
            name=f"Aprendido {sigil1}-{sigil2}",
            description=description,
            primary_sigil=sigil1,
            secondary_sigil=sigil2,
            resonance_type=resonance_type,
            strength=strength,
            frequency=frequency,
            effects=effects or ["aprendido", "adaptativo"],
            emergence_probability=0.5,
            distance_threshold=distance_threshold,
            alignment_requirement=alignment_requirement,
        )
        self.add_resonance_pattern(pat)
        return pat.pattern_id

    # ====== Persistencia del estado gramatical ======

    def export_state(self) -> dict[str, Any]:
        """Serializa el estado vivo de la gramática (patrones, restricciones, emergentes)."""
        return self.get_grammar_state()

    def import_state(self, state: dict[str, Any]) -> None:
        """Carga un estado previamente exportado (mejor para bootstrapping/evolución offline)."""
        try:
            # Patrones
            self.resonance_patterns.clear()
            for pdata in state.get("patterns", []):
                # Campos mínimos; otros quedan por defecto
                pat = ResonancePattern(
                    name=pdata.get("name", pdata.get("pattern_id", "pat")),
                    description=pdata.get("description", ""),
                    primary_sigil=pdata.get("primary_sigil", ""),
                    secondary_sigil=pdata.get("secondary_sigil", ""),
                    resonance_type=(
                        ResonanceType[pdata.get("resonance_type", "HARMONIC")]
                        if isinstance(pdata.get("resonance_type"), str)
                        else pdata.get("resonance_type")
                    ),
                    strength=float(pdata.get("strength", 0.5)),
                    frequency=float(pdata.get("frequency", 777.0)),
                    effects=list(pdata.get("effects", [])),
                    emergence_probability=float(
                        pdata.get("emergence_probability", 0.5)
                    ),
                    distance_threshold=float(pdata.get("distance_threshold", 1.5)),
                    alignment_requirement=float(
                        pdata.get("alignment_requirement", 0.6)
                    ),
                )
                self.add_resonance_pattern(pat)

            # Restricciones
            self.grammar_constraints.clear()
            for cdata in state.get("constraints", []):
                cons = GrammarConstraint(
                    name=cdata.get("name", ""),
                    description=cdata.get("description", ""),
                    restricted_sigils=set(cdata.get("restricted_sigils", [])),
                    forbidden_pairs={
                        tuple(t) for t in cdata.get("forbidden_pairs", [])
                    },
                    context_domains=(
                        set(cdata.get("context_domains", []))
                        if cdata.get("context_domains")
                        else set()
                    ),
                    violation_effects=list(cdata.get("violation_effects", [])),
                    penalty_factor=float(cdata.get("penalty_factor", 0.5)),
                    temporal_window=(
                        float(cdata.get("temporal_window", 0.0))
                        if cdata.get("temporal_window") is not None
                        else None
                    ),
                )
                self.add_grammar_constraint(cons)

            # Propiedades emergentes
            self.emergent_properties.clear()
            for edata in state.get("properties", []):
                prop = EmergentProperty(
                    name=edata.get("name", ""),
                    description=edata.get("description", ""),
                    required_sigils=set(edata.get("required_sigils", [])),
                    spatial_pattern=edata.get("spatial_pattern", ""),
                    emergence_threshold=float(edata.get("emergence_threshold", 0.8)),
                    stability_factor=float(edata.get("stability_factor", 0.9)),
                    effects=dict(edata.get("effects", {})),
                    duration=float(edata.get("duration", 1.0)),
                )
                self.add_emergent_property(prop)
        except Exception:
            # Estado inválido: mantener lo actual
            pass


class EVAGrammarEngine(DivineGrammar):
    """
    Gramática divina extendida para integración con EVA.
    Permite compilar patrones de resonancia, restricciones y propiedades emergentes en RealityBytecode,
    soporta faseo, hooks de entorno y simulación activa en QuantumField.
    """

    def __init__(self, phase: str = "default"):
        super().__init__()
        self.eva_runtime = LivingSymbolRuntime()
        self.divine_compiler = DivineLanguageEvolved(None)
        self.eva_memory_store = {}  # type: dict[str, RealityBytecode]
        self.eva_experience_store = {}  # type: dict[str, EVAExperience]
        self.eva_phases = {}  # type: dict[str, dict[str, RealityBytecode]]
        self._current_phase = phase
        self._environment_hooks = []

    def eva_ingest_grammar_experience(
        self,
        sigil_matrix: list[list[str]],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia gramatical (matriz de sigilos y reglas aplicadas) y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        coherence = self.calculate_grammar_coherence(sigil_matrix)
        emergent = self.detect_emergent_properties(
            {s for row in sigil_matrix for s in row}
        )
        intention = {
            "intention_type": "ARCHIVE_GRAMMAR_EXPERIENCE",
            "sigil_matrix": sigil_matrix,
            "coherence": coherence,
            "emergent_properties": [e.name for e in emergent],
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
        experience_id = f"grammar_{hash(str(sigil_matrix))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_grammar_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia gramatical almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for grammar experience"}
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] Grammar environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
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
        }

    def add_grammar_experience_phase(
        self,
        experience_id: str,
        phase: str,
        sigil_matrix: list[list[str]],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia gramatical.
        """
        coherence = self.calculate_grammar_coherence(sigil_matrix)
        emergent = self.detect_emergent_properties(
            {sigil for row in sigil_matrix for sigil in row}
        )
        intention = {
            "intention_type": "ARCHIVE_GRAMMAR_EXPERIENCE",
            "sigil_matrix": sigil_matrix,
            "coherence": coherence,
            "emergent_properties": [e.name for e in emergent],
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self._current_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia gramatical."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_grammar_experience": self.eva_ingest_grammar_experience,
            "eva_recall_grammar_experience": self.eva_recall_grammar_experience,
            "add_grammar_experience_phase": self.add_grammar_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }

    # Exponer helpers de resonancia para consumidores externos (EDEN/QualiaEngine, ADAM)
    def predict_interaction(
        self,
        sigil1: str,
        sigil2: str,
        distance: float,
        *,
        vector1: Any | None = None,
        vector2: Any | None = None,
    ) -> dict[str, Any]:
        """
        Fachada de evaluate_resonance para motores externos.
        """
        return self.evaluate_resonance(
            sigil1, sigil2, distance=distance, vector1=vector1, vector2=vector2
        )


# Instancia global EVA Grammar
eva_grammar_engine = EVAGrammarEngine()
