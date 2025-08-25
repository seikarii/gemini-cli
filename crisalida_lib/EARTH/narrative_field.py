"""Advanced narrative field system for tracking and managing story events in Crisalida."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from crisalida_lib.EDEN.living_symbol import (
    LivingSymbolRuntime as EDEN_LivingSymbolRuntime,
)
from crisalida_lib.EDEN.living_symbol import QuantumField as EDEN_QuantumField

if TYPE_CHECKING:
    # Import concrete types only for type checking. Keep runtime names untouched.
    from crisalida_lib.EDEN.living_symbol import (
        LivingSymbolRuntime,  # type: ignore
        QuantumField,  # type: ignore
    )
    from crisalida_lib.EVA.core_types import (
        EVAExperience,  # type: ignore
        RealityBytecode,  # type: ignore
    )
    from crisalida_lib.EVA.typequalia import QualiaState  # type: ignore

    # Expose typed aliases for static analysis
    EVAExperience_t = EVAExperience
    RealityBytecode_t = RealityBytecode
    QualiaState_t = QualiaState
    LivingSymbolRuntime_t = LivingSymbolRuntime
    QuantumField_t = QuantumField
else:
    # Runtime conservative fallbacks and EDEN shims
    EVAExperience_t = Any
    RealityBytecode_t = Any
    QualiaState_t = Any
    LivingSymbolRuntime_t = EDEN_LivingSymbolRuntime
    QuantumField_t = EDEN_QuantumField


# Export small enums/types used by narrative_service
class CrisisType(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CATASTROPHIC = "catastrophic"


class EvolutionType(str, Enum):
    SLOW = "slow"
    RAPID = "rapid"
    TRANSFORMATIVE = "transformative"


class EventType(Enum):
    """Types of narrative events in the simulation."""

    ENTITY_BIRTH = "entity_birth"
    ENTITY_DEATH = "entity_death"
    CONSCIOUSNESS_SHIFT = "consciousness_shift"
    ARCHETYPE_ACTIVATION = "archetype_activation"
    REALITY_DISTORTION = "reality_distortion"
    DIMENSIONAL_BREACH = "dimensional_breach"
    QUALIA_SYNTHESIS = "qualia_synthesis"
    MEMORY_FORMATION = "memory_formation"
    SOUL_EVOLUTION = "soul_evolution"
    TRANSCENDENCE = "transcendence"
    NARRATIVE_CAUSE = "narrative_cause"
    NARRATIVE_CONSEQUENCE = "narrative_consequence"
    STELLAR_PURPOSE = "stellar_purpose"


class EventImportance(Enum):
    """Importance levels for narrative events."""

    TRIVIAL = 1
    MINOR = 2
    MODERATE = 3
    SIGNIFICANT = 4
    MAJOR = 5
    EPOCHAL = 6


@dataclass
class NarrativeEvent:
    """A structured narrative event with metadata and relationships."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CONSCIOUSNESS_SHIFT
    timestamp: float = field(default_factory=time.time)
    importance: EventImportance = EventImportance.MODERATE

    # Event content
    title: str = ""
    description: str = ""
    participants: list[str] = field(default_factory=list)  # Entity IDs
    location: tuple[float, float, float] | None = None

    # Narrative metadata
    narrative_tags: list[str] = field(default_factory=list)
    causal_links: list[str] = field(default_factory=list)  # Related event IDs
    emotional_resonance: float = 0.5  # 0-1 scale
    consciousness_impact: float = 0.0  # -1 to 1 scale
    evolutionary_impact: float = 0.0
    dramatic_function: str = ""
    archetype: str = ""
    character_arc: str = ""
    plot_significance: float = 0.0
    consequences: list[str] = field(default_factory=list)

    # Additional data
    raw_data: dict[str, Any] = field(default_factory=dict)

    def get_narrative_weight(self) -> float:
        """Calculate the narrative weight of this event."""
        base_weight = self.importance.value / 6.0  # Normalize to 0-1
        resonance_boost = self.emotional_resonance * 0.3
        impact_boost = abs(self.consciousness_impact) * 0.2
        evo_boost = self.evolutionary_impact * 0.2
        plot_boost = self.plot_significance * 0.1

        return min(
            1.0, base_weight + resonance_boost + impact_boost + evo_boost + plot_boost
        )


@dataclass
class NarrativeThread:
    """A connected sequence of narrative events forming a coherent story thread."""

    def __init__(self, initial_event: NarrativeEvent, thread_id: str | None = None):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.events: list[NarrativeEvent] = [initial_event]
        self.thread_type = initial_event.event_type
        self.active = True
        self.narrative_tension = 0.0
        self.dramatic_functions: set[str] = (
            {initial_event.dramatic_function}
            if initial_event.dramatic_function
            else set()
        )
        self.archetypes: set[str] = (
            {initial_event.archetype} if initial_event.archetype else set()
        )

    def add_event(self, event: NarrativeEvent) -> None:
        """Add an event to this narrative thread."""
        self.events.append(event)
        if event.dramatic_function:
            self.dramatic_functions.add(event.dramatic_function)
        if event.archetype:
            self.archetypes.add(event.archetype)
        self._update_tension()

    def _update_tension(self) -> None:
        """Update narrative tension based on recent events."""
        if len(self.events) < 2:
            return

        recent_events = self.events[-5:]  # Last 5 events
        importance_variance = max(e.importance.value for e in recent_events) - min(
            e.importance.value for e in recent_events
        )
        evo_variance = max(e.evolutionary_impact for e in recent_events) - min(
            e.evolutionary_impact for e in recent_events
        )
        self.narrative_tension = min(
            1.0, (importance_variance / 5.0) + (evo_variance * 0.2)
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of this narrative thread."""
        return {
            "thread_id": self.thread_id,
            "thread_type": self.thread_type.value,
            "event_count": len(self.events),
            "active": self.active,
            "narrative_tension": self.narrative_tension,
            "dramatic_functions": list(self.dramatic_functions),
            "archetypes": list(self.archetypes),
            "participants": list(
                {
                    participant
                    for event in self.events
                    for participant in event.participants
                }
            ),
            "timespan": (
                self.events[-1].timestamp - self.events[0].timestamp
                if self.events
                else 0
            ),
        }


@dataclass
class NarrativeField:
    """Advanced narrative field for managing complex story dynamics in Crisalida."""

    def __init__(self, dimensions: tuple[int, ...] = (100, 100, 100)):
        """Initialize the narrative field with dimensional constraints."""
        self.dimensions = dimensions
        self.events: list[NarrativeEvent] = []
        self.event_index: dict[str, NarrativeEvent] = {}
        self.narrative_threads: dict[str, NarrativeThread] = {}

        # Spatial and temporal organization
        self.spatial_index: dict[tuple[int, int, int], list[str]] = defaultdict(list)
        self.temporal_index: dict[int, list[str]] = defaultdict(list)  # Hour buckets

        # Event type tracking
        self.event_type_counts: dict[EventType, int] = defaultdict(int)

        # Recent events buffer for real-time analysis
        self.recent_events: deque = deque(maxlen=100)

        # Narrative metrics
        self.narrative_coherence = 1.0
        self.overall_tension = 0.0
        self.story_complexity = 0.0
        self.cosmic_themes: dict[str, float] = defaultdict(float)
        self.narrative_archetypes_active: set[str] = set()
        self.narrative_field_intensity: float = 0.5
        self.cosmic_dramatic_tension: float = 0.5

    def emit_event(self, event: Any) -> str:
        """Enhanced event emission with automatic narrative structuring."""
        # Convert raw event to structured NarrativeEvent if needed
        if isinstance(event, NarrativeEvent):
            narrative_event = event
        else:
            narrative_event = self._convert_to_narrative_event(event)

        # Store the event
        self.events.append(narrative_event)
        self.event_index[narrative_event.event_id] = narrative_event
        self.recent_events.append(narrative_event)

        # Update indices
        self._update_spatial_index(narrative_event)
        self._update_temporal_index(narrative_event)
        self.event_type_counts[narrative_event.event_type] += 1

        # Find or create appropriate narrative thread
        self._assign_to_narrative_thread(narrative_event)

        # Update narrative metrics
        self._update_narrative_metrics()
        self._update_cosmic_themes(narrative_event)

        return narrative_event.event_id

    def _convert_to_narrative_event(self, raw_event: Any) -> NarrativeEvent:
        """Convert raw event data to structured NarrativeEvent."""
        event_data = raw_event if isinstance(raw_event, dict) else {"data": raw_event}

        # Extract or infer event properties
        event_type = self._infer_event_type(event_data)
        importance = self._calculate_importance(event_data)

        return NarrativeEvent(
            event_type=event_type,
            importance=importance,
            title=event_data.get(
                "title", f"{event_type.value.replace('_', ' ').title()}"
            ),
            description=event_data.get("description", "Narrative event occurred"),
            participants=event_data.get("participants", []),
            location=event_data.get("location"),
            narrative_tags=event_data.get("tags", []),
            emotional_resonance=event_data.get("emotional_resonance", 0.5),
            consciousness_impact=event_data.get("consciousness_impact", 0.0),
            evolutionary_impact=event_data.get("evolutionary_impact", 0.0),
            dramatic_function=event_data.get("dramatic_function", ""),
            archetype=event_data.get("archetype", ""),
            character_arc=event_data.get("character_arc", ""),
            plot_significance=event_data.get("plot_significance", 0.0),
            consequences=event_data.get("consequences", []),
            raw_data=event_data,
        )

    def _infer_event_type(self, event_data: dict[str, Any]) -> EventType:
        """Infer event type from event data."""
        event_type_hints = {
            "birth": EventType.ENTITY_BIRTH,
            "death": EventType.ENTITY_DEATH,
            "consciousness": EventType.CONSCIOUSNESS_SHIFT,
            "archetype": EventType.ARCHETYPE_ACTIVATION,
            "reality": EventType.REALITY_DISTORTION,
            "dimension": EventType.DIMENSIONAL_BREACH,
            "qualia": EventType.QUALIA_SYNTHESIS,
            "memory": EventType.MEMORY_FORMATION,
            "soul": EventType.SOUL_EVOLUTION,
            "transcend": EventType.TRANSCENDENCE,
            "cause": EventType.NARRATIVE_CAUSE,
            "consequence": EventType.NARRATIVE_CONSEQUENCE,
            "purpose": EventType.STELLAR_PURPOSE,
        }

        event_str = str(event_data).lower()
        for hint, event_type in event_type_hints.items():
            if hint in event_str:
                return event_type

        return EventType.CONSCIOUSNESS_SHIFT  # Default

    def _calculate_importance(self, event_data: dict[str, Any]) -> EventImportance:
        """Calculate event importance based on various factors."""
        importance_score = 0

        # Base importance from explicit setting
        if "importance" in event_data:
            try:
                return EventImportance(event_data["importance"])
            except Exception:
                pass

        # Participants count
        participants = event_data.get("participants", [])
        importance_score += min(2, len(participants))

        # Consciousness impact
        consciousness_impact = abs(event_data.get("consciousness_impact", 0.0))
        importance_score += consciousness_impact * 2

        # Emotional resonance
        emotional_resonance = event_data.get("emotional_resonance", 0.5)
        if emotional_resonance > 0.8:
            importance_score += 1

        # Evolutionary impact
        evo_impact = event_data.get("evolutionary_impact", 0.0)
        importance_score += evo_impact * 1.5

        # Plot significance
        plot_significance = event_data.get("plot_significance", 0.0)
        importance_score += plot_significance * 1.2

        # Map score to importance level
        importance_levels = [
            EventImportance.TRIVIAL,
            EventImportance.MINOR,
            EventImportance.MODERATE,
            EventImportance.SIGNIFICANT,
            EventImportance.MAJOR,
            EventImportance.EPOCHAL,
        ]

        level_index = min(len(importance_levels) - 1, int(importance_score))
        return importance_levels[level_index]

    def _update_spatial_index(self, event: NarrativeEvent) -> None:
        """Update spatial index for location-based event queries."""
        if event.location:
            # Convert to grid coordinates
            x, y, z = event.location
            grid_x = (
                int(x * self.dimensions[0] // 100) if len(self.dimensions) > 0 else 0
            )
            grid_y = (
                int(y * self.dimensions[1] // 100) if len(self.dimensions) > 1 else 0
            )
            grid_z = (
                int(z * self.dimensions[2] // 100) if len(self.dimensions) > 2 else 0
            )

            grid_coord = (grid_x, grid_y, grid_z)
            self.spatial_index[grid_coord].append(event.event_id)

    def _update_temporal_index(self, event: NarrativeEvent) -> None:
        """Update temporal index for time-based event queries."""
        hour_bucket = int(event.timestamp // 3600)  # Group by hour
        self.temporal_index[hour_bucket].append(event.event_id)

    def _assign_to_narrative_thread(self, event: NarrativeEvent) -> None:
        """Assign event to appropriate narrative thread."""
        best_thread = None
        best_score = 0.0

        # Find the best matching existing thread
        for thread in self.narrative_threads.values():
            if not thread.active:
                continue

            score = self._calculate_thread_affinity(event, thread)
            if score > best_score:
                best_score = score
                best_thread = thread

        # Create new thread if no good match found
        if best_score < 0.3:  # Threshold for creating new thread
            new_thread = NarrativeThread(event)
            self.narrative_threads[new_thread.thread_id] = new_thread
        elif best_thread is not None:
            best_thread.add_event(event)

    def _calculate_thread_affinity(
        self, event: NarrativeEvent, thread: NarrativeThread
    ) -> float:
        """Calculate how well an event fits into a narrative thread."""
        score = 0.0

        # Type similarity
        if event.event_type == thread.thread_type:
            score += 0.4

        # Participant overlap
        thread_participants = set()
        for thread_event in thread.events:
            thread_participants.update(thread_event.participants)

        event_participants = set(event.participants)
        if thread_participants and event_participants:
            overlap = len(thread_participants & event_participants)
            total = len(thread_participants | event_participants)
            score += 0.3 * (overlap / total)

        # Temporal proximity (events within last hour get bonus)
        if thread.events:
            last_event_time = thread.events[-1].timestamp
            time_diff = abs(event.timestamp - last_event_time)
            if time_diff < 3600:  # 1 hour
                score += 0.2 * (1.0 - time_diff / 3600)

        # Causal links
        if any(
            link_id in [e.event_id for e in thread.events]
            for link_id in event.causal_links
        ):
            score += 0.1

        # Dramatic function similarity
        if (
            event.dramatic_function
            and event.dramatic_function in thread.dramatic_functions
        ):
            score += 0.1

        # Archetype similarity
        if event.archetype and event.archetype in thread.archetypes:
            score += 0.1

        return score

    def _update_narrative_metrics(self) -> None:
        """Update overall narrative metrics."""
        if not self.events:
            return

        # Calculate narrative coherence based on thread connectivity
        active_threads = [t for t in self.narrative_threads.values() if t.active]
        if active_threads:
            avg_thread_length = sum(len(t.events) for t in active_threads) / len(
                active_threads
            )
            self.narrative_coherence = min(1.0, avg_thread_length / 10.0)  # Normalize

        # Calculate overall tension
        if active_threads:
            self.overall_tension = sum(
                t.narrative_tension for t in active_threads
            ) / len(active_threads)

        # Calculate story complexity
        unique_participants = set()
        for event in self.events[-50:]:  # Recent events
            unique_participants.update(event.participants)

        self.story_complexity = min(1.0, len(unique_participants) / 20.0)  # Normalize

    def _update_cosmic_themes(self, event: NarrativeEvent):
        """Actualiza los temas cósmicos basado en eventos narrativos."""
        theme_mapping = {
            "conflicto": "dualidad",
            "resolución": "unidad",
            "descubrimiento": "conocimiento",
            "transformación": "evolución",
            "creación": "génesis",
            "destrucción": "renacimiento",
            "propósito": "voluntad",
            "consecuencia": "destino",
            "causa": "origen",
        }
        theme = theme_mapping.get(event.dramatic_function, "general")
        self.cosmic_themes[theme] += (
            event.plot_significance * 0.1 + event.evolutionary_impact * 0.1
        )
        if event.archetype:
            self.narrative_archetypes_active.add(event.archetype)
        self.narrative_field_intensity *= 0.995
        self.cosmic_dramatic_tension *= 0.99
        self.narrative_field_intensity = min(1.0, self.narrative_field_intensity)
        self.cosmic_dramatic_tension = min(1.0, self.cosmic_dramatic_tension)

    def get_events_by_location(
        self, center: tuple[float, float, float], radius: float
    ) -> list[NarrativeEvent]:
        """Get events within a spatial radius of a location."""
        matching_events = []

        for event in self.events:
            if event.location:
                distance = (
                    sum(
                        (a - b) ** 2
                        for a, b in zip(event.location, center, strict=False)
                    )
                    ** 0.5
                )
                if distance <= radius:
                    matching_events.append(event)

        return matching_events

    def get_events_by_timerange(
        self, start_time: float, end_time: float
    ) -> list[NarrativeEvent]:
        """Get events within a time range."""
        return [
            event for event in self.events if start_time <= event.timestamp <= end_time
        ]

    def get_narrative_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the narrative field state."""
        return {
            "total_events": len(self.events),
            "active_threads": len(
                [t for t in self.narrative_threads.values() if t.active]
            ),
            "narrative_coherence": self.narrative_coherence,
            "overall_tension": self.overall_tension,
            "story_complexity": self.story_complexity,
            "event_type_distribution": {
                event_type.value: count
                for event_type, count in self.event_type_counts.items()
            },
            "recent_activity": len(self.recent_events),
            "top_threads": [
                thread.get_summary()
                for thread in sorted(
                    self.narrative_threads.values(),
                    key=lambda t: len(t.events),
                    reverse=True,
                )[:5]
            ],
            "cosmic_themes": dict(self.cosmic_themes),
            "active_archetypes": list(self.narrative_archetypes_active),
            "field_intensity": self.narrative_field_intensity,
            "dramatic_tension": self.cosmic_dramatic_tension,
        }

    def analyze_narrative_patterns(self) -> dict[str, Any]:
        """Analyze patterns in the narrative field."""
        if len(self.events) < 10:
            return {"status": "Insufficient data for pattern analysis"}

        # Event frequency analysis
        recent_events = [
            e for e in self.events if e.timestamp > time.time() - 3600
        ]  # Last hour
        event_frequency = len(recent_events)

        # Importance distribution
        importance_dist: dict[str, int] = defaultdict(int)
        for event in self.events[-100:]:  # Recent 100 events
            importance_dist[event.importance.name] += 1

        # Participant activity
        participant_activity: dict[str, int] = defaultdict(int)
        for event in recent_events:
            for participant in event.participants:
                participant_activity[participant] += 1

        most_active = sorted(
            participant_activity.items(), key=lambda x: x[1], reverse=True
        )[:5]

        dict(self.cosmic_themes.items())

        return {
            "event_frequency_last_hour": event_frequency,
            "importance_distribution": dict(importance_dist),
            "most_active_participants": most_active,
            "narrative_trends": {
                "coherence_trend": self.narrative_coherence,
                "tension_trend": self.overall_tension,
                "complexity_trend": self.story_complexity,
            },
        }


class EVANarrativeField(NarrativeField):
    """
    Campo narrativo avanzado extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar eventos narrativos como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking, gestión de memoria viviente EVA y optimización GPU/ECS.
    """

    # Class-level annotations for static checkers
    eva_memory_store: dict[str, Any]
    eva_experience_store: dict[str, Any]
    eva_phases: dict[str, dict[str, Any]]
    _environment_hooks: list[Callable[..., Any]]
    _gpu_enabled: bool
    _ecs_components: dict[str, Any]

    def __init__(
        self, dimensions: tuple[int, ...] = (100, 100, 100), phase: str = "default"
    ):
        """
        Initialize EVANarrativeField with EVA runtime and storage.
        """
        super().__init__(dimensions)
        self.eva_phase = phase
        # instantiate runtime using runtime alias (may be EDEN shim)
        try:
            self.eva_runtime = LivingSymbolRuntime_t()
        except Exception:
            # If runtime cannot be constructed, keep None and allow later injection
            self.eva_runtime = None  # type: ignore

        # EVA runtime-backed storage and runtime configuration (initialized)
        self.eva_memory_store = {}
        self.eva_experience_store = {}
        self.eva_phases = {}

        # Hooks and optional subsystems (initialized)
        self._environment_hooks = []
        self._gpu_enabled = False
        self._ecs_components = {}

    def enable_gpu_optimization(self, enable: bool = True):
        """Activa la optimización GPU para simulación masiva de eventos narrativos."""
        self._gpu_enabled = enable
        # Integrar con gpu_physics_engine si está disponible
        if enable and hasattr(self.eva_runtime, "gpu_physics_engine"):
            self.eva_runtime.gpu_physics_engine.enable_narrative_mode()

    def register_ecs_component(self, name: str, component: Any):
        """Registra un componente ECS para simulación narrativa orientada a datos."""
        self._ecs_components[name] = component

    def eva_ingest_narrative_experience(
        self,
        event: NarrativeEvent,
        qualia_state: Any | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila un evento narrativo en RealityBytecode y lo almacena en la memoria EVA.
        Optimizado para GPU/ECS si está habilitado.
        """
        # Normalize phase to a usable key (avoid None dict indexing)
        phase_key = phase or self.eva_phase

        qualia_state = qualia_state or QualiaState_t(
            emotional_valence=event.emotional_resonance,
            cognitive_complexity=0.7,
            consciousness_density=event.consciousness_impact,
            narrative_importance=event.plot_significance,
            energy_level=1.0,
        )
        experience_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "importance": event.importance.value,
            "title": event.title,
            "description": event.description,
            "participants": event.participants,
            "location": event.location,
            "narrative_tags": event.narrative_tags,
            "causal_links": event.causal_links,
            "emotional_resonance": event.emotional_resonance,
            "consciousness_impact": event.consciousness_impact,
            "evolutionary_impact": event.evolutionary_impact,
            "dramatic_function": event.dramatic_function,
            "archetype": event.archetype,
            "character_arc": event.character_arc,
            "plot_significance": event.plot_significance,
            "consequences": event.consequences,
            "raw_data": event.raw_data,
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
        }
        intention = {
            "intention_type": "ARCHIVE_NARRATIVE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase_key,
        }
        bytecode = []
        runtime = getattr(self, "eva_runtime", None)
        if runtime is not None:
            _dc = getattr(runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(intention)
            except Exception:
                bytecode = []
        experience_id = f"eva_narrative_{event.event_id}"
        reality_bytecode = RealityBytecode_t(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase_key,
            timestamp=event.timestamp,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase_key not in self.eva_phases:
            self.eva_phases[phase_key] = {}
        self.eva_phases[phase_key][experience_id] = reality_bytecode
        # Benchmark: registrar tiempo de ingestión si GPU/ECS activo
        if self._gpu_enabled:
            start = time.time()
            # Simulación masiva en GPU (placeholder)
            if hasattr(self.eva_runtime, "gpu_physics_engine"):
                self.eva_runtime.gpu_physics_engine.simulate_narrative_event(event)
            end = time.time()
            reality_bytecode.simulation_metadata = {"gpu_ingest_time": end - start}
        return experience_id

    def eva_recall_narrative_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de un evento narrativo almacenado, manifestando la simulación.
        Optimizado para GPU/ECS si está habilitado.
        """
        # Normalize phase for safe dict access
        phase_key = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase_key, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA narrative experience"}
        quantum_field = QuantumField_t()
        manifestations = []
        start = time.time()
        runtime = getattr(self, "eva_runtime", None)
        for instr in reality_bytecode.instructions:
            if self._gpu_enabled and runtime and hasattr(runtime, "gpu_physics_engine"):
                symbol_manifest = runtime.gpu_physics_engine.execute_instruction(
                    instr, quantum_field
                )
            elif runtime is not None and hasattr(runtime, "execute_instruction"):
                symbol_manifest = runtime.execute_instruction(instr, quantum_field)
            else:
                # No runtime available to execute this instruction; skip
                continue
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA-NARRATIVE] Environment hook failed: {e}")
        end = time.time()

        eva_experience = EVAExperience_t(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        # Benchmark: registrar tiempo de recall si GPU/ECS activo
        if self._gpu_enabled:
            eva_experience.metadata = {"gpu_recall_time": end - start}
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
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
            "benchmark": (
                eva_experience.metadata if hasattr(eva_experience, "metadata") else {}
            ),
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        event: NarrativeEvent,
        qualia_state: Any,
    ):
        """
        Añade una fase alternativa para una experiencia narrativa EVA.
        """
        experience_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "importance": event.importance.value,
            "title": event.title,
            "description": event.description,
            "participants": event.participants,
            "location": event.location,
            "narrative_tags": event.narrative_tags,
            "causal_links": event.causal_links,
            "emotional_resonance": event.emotional_resonance,
            "consciousness_impact": event.consciousness_impact,
            "evolutionary_impact": event.evolutionary_impact,
            "dramatic_function": event.dramatic_function,
            "archetype": event.archetype,
            "character_arc": event.character_arc,
            "plot_significance": event.plot_significance,
            "consequences": event.consequences,
            "raw_data": event.raw_data,
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
        }
        intention = {
            "intention_type": "ARCHIVE_NARRATIVE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = []
        runtime = getattr(self, "eva_runtime", None)
        if runtime is not None:
            _dc = getattr(runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(intention)
            except Exception:
                bytecode = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=event.timestamp,
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
                print(f"[EVA-NARRATIVE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia narrativa EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_narrative_experience": self.eva_ingest_narrative_experience,
            "eva_recall_narrative_experience": self.eva_recall_narrative_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "enable_gpu_optimization": self.enable_gpu_optimization,
            "register_ecs_component": self.register_ecs_component,
        }
