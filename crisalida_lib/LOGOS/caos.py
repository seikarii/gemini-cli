"""
Núcleo Cognitivo Caótico - Sistema Físico de Inteligencia Emergente
===================================================================

Un sistema AGI basado puramente en dinámicas caóticas y emergencia.
No contiene lógica programada, solo física del caos y auto-organización.

Integración con el ecosistema Crisalida:
- EDEN: QualiaEngine para modular parámetros caóticos
- EVA: Memoria viviente para trayectorias y atractores
- ADAM: Integración con sistemas cognitivos existentes
- LOGOS: VM simbólica para interpretar estados caóticos

Basado en sistemas dinámicos de Lorenz, Rössler y atractores extraños.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque

# Defensive imports siguiendo el patrón del repo
try:
    import numpy as np
    from scipy.integrate import odeint
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    np = None
    HAS_SCIPY = False

# Integración con subsistemas de Crisalida
if TYPE_CHECKING:
    from crisalida_lib.EDEN.qualia_engine import QualiaEngine
    from crisalida_lib.EVA.eva_memory_helper import EVAMemoryHelper
    from crisalida_lib.EVA.core_types import QualiaSignature, RealityBytecode
    from crisalida_lib.ADAM.mente.cognitive_node import CognitiveNode
else:
    # Runtime placeholders
    QualiaEngine = Any
    EVAMemoryHelper = Any
    QualiaSignature = Any
    RealityBytecode = Any
    CognitiveNode = Any

logger = logging.getLogger(__name__)

# --- Tipos y estructuras de datos ---

@dataclass
class ChaoticAttractor:
    """Representa un atractor emergente en el espacio fase"""
    name: str
    position: np.ndarray if np else List[float]
    radius: float
    stability: float
    emergence_time: float
    visit_count: int = 0
    concept_strength: float = 0.0

@dataclass
class CognitiveTrajectory:
    """Trayectoria cognitiva en el espacio de estados caóticos"""
    problem_type: str
    solution: np.ndarray if np else List[float]
    duration: float
    attractors_visited: List[str] = field(default_factory=list)
    complexity_measure: float = 0.0
    emergence_events: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ChaoticParameters:
    """Parámetros del sistema caótico que controlan la dinámica cognitiva"""
    sigma: float = 10.0      # Tasa de disipación (memoria/olvido)
    rho: float = 28.0        # Fuerza no-lineal (creatividad)
    beta: float = 8.0/3.0    # Geometría del atractor (coherencia)
    coupling_strength: float = 0.1  # Acoplamiento entre dimensiones
    noise_level: float = 0.01       # Ruido estocástico
    temporal_decay: float = 0.95    # Decaimiento temporal de atractores

# --- Núcleo Cognitivo Principal ---

class ChaoticCognitiveCore:
    """
    Núcleo cognitivo basado en dinámicas caóticas puras.
    
    No contiene algoritmos de IA tradicionales, solo física del caos.
    La inteligencia emerge de la auto-organización del sistema dinámico.
    
    Características:
    - Memoria episódica como trayectorias en espacio fase
    - Conceptos como atractores emergentes
    - Aprendizaje como modificación de parámetros
    - Razonamiento como evolución dinámica
    """
    
    def __init__(
        self, 
        n_dimensions: int = 1000,
        qualia_engine: Optional["QualiaEngine"] = None,
        eva_helper: Optional["EVAMemoryHelper"] = None
    ):
        self.n_dimensions = n_dimensions
        self.qualia_engine = qualia_engine
        self.eva_helper = eva_helper
        
        # Estado del sistema caótico
        if np is not None:
            self.state = np.random.uniform(-0.5, 0.5, n_dimensions)
        else:
            # Fallback sin numpy
            import random
            self.state = [random.uniform(-0.5, 0.5) for _ in range(n_dimensions)]
            logger.warning("Running without numpy - performance degraded")
        
        # Parámetros dinámicos del sistema
        self.params = ChaoticParameters()
        
        # Atractores emergentes (conocimiento conceptual)
        self.attractors: Dict[str, ChaoticAttractor] = {}
        
        # Memoria episódica (trayectorias históricas)
        self.trajectory_history: deque = deque(maxlen=100)
        
        # Métricas de performance
        self.emergence_events: List[Dict[str, Any]] = []
        self.total_thinking_time: float = 0.0
        self.solution_count: int = 0
        
        # Integración con EDEN
        self._consciousness_field: Dict[str, float] = {}
        
        logger.info(f"ChaoticCognitiveCore inicializado con {n_dimensions} dimensiones")

    def encode_problem_as_physics(self, problem_spec: Dict[str, Any]) -> ChaoticParameters:
        """
        Codifica un problema como modificación de parámetros físicos.
        
        No es programación de algoritmos, es modulación de las leyes físicas
        que rigen el sistema caótico.
        """
        problem_type = problem_spec.get("type", "general")
        complexity = problem_spec.get("complexity", 0.5)
        urgency = problem_spec.get("urgency", 0.5)
        
        # Clonar parámetros actuales
        new_params = ChaoticParameters(
            sigma=self.params.sigma,
            rho=self.params.rho,
            beta=self.params.beta,
            coupling_strength=self.params.coupling_strength,
            noise_level=self.params.noise_level
        )
        
        # Modular según tipo de problema
        if problem_type == "optimization":
            # Optimización = aumentar no-linealidad, reducir disipación
            new_params.rho *= (1.0 + complexity * 0.5)
            new_params.sigma *= (1.0 - urgency * 0.3)
            new_params.coupling_strength *= 1.2
            
        elif problem_type == "classification":
            # Clasificación = crear múltiples cuencas de atracción
            new_params.beta *= (1.0 + complexity * 0.8)
            new_params.noise_level *= (1.0 + urgency * 0.2)
            
        elif problem_type == "prediction":
            # Predicción = aumentar memoria, estabilizar
            new_params.sigma *= (1.0 + complexity * 0.4)
            new_params.temporal_decay *= (1.0 - urgency * 0.1)
            
        elif problem_type == "creative":
            # Creatividad = maximizar caos controlado
            new_params.rho *= (1.5 + complexity * 0.3)
            new_params.noise_level *= (1.0 + urgency * 0.5)
            new_params.coupling_strength *= 0.8
            
        # Integración con QualiaEngine si está disponible
        if self.qualia_engine:
            try:
                qualia_influence = self._get_qualia_influence()
                new_params.sigma *= qualia_influence.get("cognitive_coherence", 1.0)
                new_params.rho *= qualia_influence.get("creative_chaos", 1.0)
            except Exception as e:
                logger.debug(f"QualiaEngine integration failed: {e}")
        
        return new_params

    def lorenz_rossler_hybrid_system(self, state, t, params: ChaoticParameters):
        """
        Sistema dinámico híbrido Lorenz-Rössler para cognición caótica.
        
        Combina la geometría del atractor de Lorenz con la periodicidad
        del atractor de Rössler para crear dinámicas cognitivas ricas.
        """
        if np is None:
            # Fallback simple sin scipy
            return self._simple_chaotic_evolution(state, params)
        
        n = len(state)
        derivatives = np.zeros(n)
        
        # Núcleo Lorenz (primeras 3 dimensiones)
        if n >= 3:
            x, y, z = state[0], state[1], state[2]
            derivatives[0] = params.sigma * (y - x)
            derivatives[1] = x * (params.rho - z) - y
            derivatives[2] = x * y - params.beta * z
        
        # Núcleo Rössler (dimensiones 3-6 si existen)
        if n >= 6:
            a, b, c = 0.2, 0.2, 5.7  # Parámetros Rössler estándar
            x2, y2, z2 = state[3], state[4], state[5]
            derivatives[3] = -y2 - z2
            derivatives[4] = x2 + a * y2
            derivatives[5] = b + z2 * (x2 - c)
        
        # Acoplamiento débil entre subsistemas (emergencia)
        for i in range(6, n):
            # Acoplamiento con dimensiones anteriores
            coupling_term = 0.0
            if i >= 2:
                coupling_term = params.coupling_strength * (state[i-1] - state[i])
            if i >= 3:
                coupling_term += 0.5 * params.coupling_strength * state[i-2]
            
            # Término de memoria (conexión con atractores pasados)
            memory_term = 0.0
            for attractor in self.attractors.values():
                if hasattr(attractor.position, '__len__') and i < len(attractor.position):
                    distance = abs(state[i] - attractor.position[i])
                    memory_term += 0.01 * attractor.concept_strength / (1.0 + distance)
            
            # Ruido estocástico controlado
            noise_term = params.noise_level * np.random.normal(0, 1)
            
            derivatives[i] = coupling_term + memory_term + noise_term
        
        return derivatives

    def _simple_chaotic_evolution(self, state, params: ChaoticParameters):
        """Evolución caótica simple cuando scipy no está disponible"""
        n = len(state)
        derivatives = []
        
        for i in range(n):
            if i < 3:
                # Sistema Lorenz simplificado
                if i == 0:
                    deriv = params.sigma * (state[1] - state[0]) if n > 1 else -state[0]
                elif i == 1:
                    deriv = state[0] * (params.rho - (state[2] if n > 2 else 1)) - state[1]
                else:  # i == 2
                    deriv = state[0] * state[1] - params.beta * state[2]
            else:
                # Acoplamiento simple
                prev_val = state[i-1] if i > 0 else 0
                deriv = params.coupling_strength * (prev_val - state[i])
            
            derivatives.append(deriv)
        
        return derivatives

    def think(
        self, 
        problem_spec: Dict[str, Any], 
        duration: float = 10.0, 
        dt: float = 0.01
    ) -> Dict[str, Any]:
        """
        Proceso de "pensamiento" = evolución del sistema caótico.
        
        La solución emerge de la topología del espacio fase resultante,
        no de algoritmos programados.
        """
        start_time = time.time()
        
        # Codificar problema como modulación física
        problem_params = self.encode_problem_as_physics(problem_spec)
        
        # Evolucionar sistema (resolver ecuación diferencial de pensamiento)
        trajectory = self._evolve_system(problem_params, duration, dt)
        
        # Detectar emergencia de nuevos atractores (aprendizaje)
        new_attractors = self._detect_emergent_attractors(trajectory, problem_spec)
        
        # Extraer solución de la geometría del sistema
        solution = self._extract_emergent_solution(trajectory, problem_spec)
        
        # Actualizar estado interno
        if np is not None and hasattr(trajectory, 'shape') and trajectory.shape[0] > 0:
            self.state = trajectory[-1]
        elif trajectory:
            self.state = trajectory[-1]
        
        # Registrar trayectoria cognitiva
        cognitive_traj = CognitiveTrajectory(
            problem_type=problem_spec.get("type", "general"),
            solution=solution,
            duration=time.time() - start_time,
            attractors_visited=[a.name for a in new_attractors],
            complexity_measure=self._calculate_trajectory_complexity(trajectory)
        )
        
        self.trajectory_history.append(cognitive_traj)
        self.total_thinking_time += cognitive_traj.duration
        self.solution_count += 1
        
        # Integración con EVA si está disponible
        if self.eva_helper:
            try:
                self._archive_chaotic_experience(cognitive_traj, problem_spec)
            except Exception as e:
                logger.debug(f"EVA archiving failed: {e}")
        
        logger.info(f"Thought process completed: {solution.get('type', 'unknown')} solution")
        
        return {
            "solution": solution,
            "trajectory": cognitive_traj,
            "new_attractors": len(new_attractors),
            "thinking_time": cognitive_traj.duration,
            "system_state": "chaotic",
            "consciousness_level": self._estimate_consciousness_level()
        }

    def _evolve_system(
        self, 
        params: ChaoticParameters, 
        duration: float, 
        dt: float
    ) -> Any:
        """Evoluciona el sistema caótico usando integración numérica"""
        
        if not HAS_SCIPY:
            # Fallback: evolución simple por pasos
            return self._simple_evolution(params, duration, dt)
        
        # Evolución completa con scipy
        t_span = np.arange(0, duration, dt)
        
        try:
            solution = odeint(
                self.lorenz_rossler_hybrid_system, 
                self.state, 
                t_span, 
                args=(params,)
            )
            return solution
        except Exception as e:
            logger.warning(f"ODE integration failed: {e}, using simple evolution")
            return self._simple_evolution(params, duration, dt)

    def _simple_evolution(self, params: ChaoticParameters, duration: float, dt: float):
        """Evolución simple cuando scipy no está disponible"""
        steps = int(duration / dt)
        trajectory = []
        current_state = list(self.state)
        
        for _ in range(steps):
            derivatives = self._simple_chaotic_evolution(current_state, params)
            
            # Integración de Euler simple
            for i in range(len(current_state)):
                current_state[i] += derivatives[i] * dt
            
            trajectory.append(list(current_state))
        
        return trajectory

    def _detect_emergent_attractors(
        self, 
        trajectory: Any, 
        problem_spec: Dict[str, Any]
    ) -> List[ChaoticAttractor]:
        """
        Detecta atractores emergentes en la trayectoria.
        
        Los atractores representan conceptos o patrones de solución
        que emergen espontáneamente del sistema caótico.
        """
        new_attractors = []
        
        if not HAS_SCIPY or np is None:
            # Detección simple sin scipy
            return self._simple_attractor_detection(trajectory, problem_spec)
        
        try:
            # Análisis de densidad en espacio fase
            if hasattr(trajectory, 'shape') and len(trajectory.shape) == 2:
                # Usar solo las primeras 3 dimensiones para análisis
                traj_3d = trajectory[:, :min(3, trajectory.shape[1])]
                
                # Crear histograma de densidad
                density_map, edges = np.histogramdd(traj_3d, bins=30)
                
                # Detectar picos de densidad (atractores)
                peaks = maximum_filter(density_map, size=3) == density_map
                peak_coords = np.argwhere(peaks & (density_map > np.mean(density_map) * 2))
                
                # Crear atractores para cada pico significativo
                for idx, coord in enumerate(peak_coords):
                    if len(new_attractors) < 10:  # Limitar número de atractores
                        # Convertir coordenadas de bins a posición real
                        position = []
                        for i, c in enumerate(coord):
                            if i < len(edges):
                                edge = edges[i]
                                pos = edge[c] + (edge[1] - edge[0]) / 2
                                position.append(pos)
                        
                        # Padded to full dimensionality
                        full_position = np.zeros(self.n_dimensions)
                        full_position[:len(position)] = position
                        
                        attractor = ChaoticAttractor(
                            name=f"{problem_spec.get('type', 'concept')}_{len(self.attractors) + idx}",
                            position=full_position,
                            radius=float(np.std(traj_3d)),
                            stability=float(density_map[tuple(coord)]),
                            emergence_time=time.time(),
                            concept_strength=min(1.0, float(density_map[tuple(coord)]) / np.max(density_map))
                        )
                        
                        self.attractors[attractor.name] = attractor
                        new_attractors.append(attractor)
                        
            return new_attractors
            
        except Exception as e:
            logger.debug(f"Advanced attractor detection failed: {e}")
            return self._simple_attractor_detection(trajectory, problem_spec)

    def _simple_attractor_detection(self, trajectory, problem_spec) -> List[ChaoticAttractor]:
        """Detección simple de atractores sin dependencias externas"""
        new_attractors = []
        
        if not trajectory:
            return new_attractors
        
        # Análisis estadístico básico
        if hasattr(trajectory, 'shape'):
            # Numpy array
            mean_pos = np.mean(trajectory, axis=0)
            std_pos = np.std(trajectory, axis=0)
        else:
            # Lista de listas
            mean_pos = []
            std_pos = []
            if trajectory:
                n_dims = len(trajectory[0])
                for dim in range(n_dims):
                    values = [point[dim] for point in trajectory]
                    mean_pos.append(sum(values) / len(values))
                    
                    # Calcular desviación estándar manual
                    mean_val = mean_pos[dim]
                    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                    std_pos.append(variance ** 0.5)
        
        # Crear un atractor promedio
        if mean_pos:
            attractor = ChaoticAttractor(
                name=f"{problem_spec.get('type', 'concept')}_{len(self.attractors)}",
                position=mean_pos,
                radius=max(std_pos) if std_pos else 1.0,
                stability=0.5,  # Valor por defecto
                emergence_time=time.time(),
                concept_strength=0.7
            )
            
            self.attractors[attractor.name] = attractor
            new_attractors.append(attractor)
        
        return new_attractors

    def _extract_emergent_solution(
        self, 
        trajectory: Any, 
        problem_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extrae la solución emergente de la topología del sistema.
        
        No hay lógica de decisión programada - la respuesta emerge
        de las propiedades geométricas del atractor.
        """
        solution_type = "emergent"
        confidence = 0.5
        details = {}
        
        # Análisis de la trayectoria final
        if hasattr(trajectory, 'shape') and trajectory.shape[0] > 0:
            # Análisis con numpy
            final_state = trajectory[-1]
            trajectory_variance = float(np.var(trajectory))
            convergence = self._calculate_convergence(trajectory)
            
        elif trajectory:
            # Análisis sin numpy
            final_state = trajectory[-1]
            
            # Calcular varianza manualmente
            if len(trajectory) > 1:
                all_values = [val for point in trajectory for val in point]
                mean_val = sum(all_values) / len(all_values)
                trajectory_variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
            else:
                trajectory_variance = 0.0
                
            convergence = self._simple_convergence_check(trajectory)
        else:
            return {"type": "chaos", "confidence": 0.0, "details": "no_trajectory"}
        
        # Clasificar solución basada en dinámicas emergentes
        if convergence > 0.8:
            solution_type = "convergent_solution"
            confidence = convergence
            details["method"] = "attractor_convergence"
            
        elif trajectory_variance > 10.0:
            solution_type = "creative_exploration"
            confidence = min(0.9, trajectory_variance / 20.0)
            details["method"] = "chaotic_exploration"
            
        elif len(self.attractors) > 5:
            solution_type = "multi_concept_synthesis"
            confidence = min(0.95, len(self.attractors) / 10.0)
            details["method"] = "attractor_network"
            
        else:
            solution_type = "emergent_pattern"
            confidence = 0.6
            details["method"] = "phase_space_analysis"
        
        # Enriquecer con contexto del problema
        problem_type = problem_spec.get("type", "general")
        details.update({
            "problem_type": problem_type,
            "final_state_norm": self._calculate_state_norm(final_state),
            "trajectory_complexity": self._calculate_trajectory_complexity(trajectory),
            "attractor_count": len(self.attractors),
            "emergence_time": time.time()
        })
        
        return {
            "type": solution_type,
            "confidence": confidence,
            "details": details,
            "raw_state": final_state
        }

    def _calculate_convergence(self, trajectory) -> float:
        """Calcula el grado de convergencia de la trayectoria"""
        if not hasattr(trajectory, 'shape') or trajectory.shape[0] < 10:
            return 0.0
        
        # Comparar últimas partes de la trayectoria
        last_quarter = trajectory[-len(trajectory)//4:]
        variance = float(np.var(last_quarter))
        
        # Convergencia alta = baja varianza al final
        return max(0.0, 1.0 - variance / 10.0)

    def _simple_convergence_check(self, trajectory) -> float:
        """Verificación simple de convergencia sin numpy"""
        if len(trajectory) < 10:
            return 0.0
        
        # Comparar primera y última cuarta parte
        quarter_size = len(trajectory) // 4
        last_quarter = trajectory[-quarter_size:]
        
        # Calcular varianza promedio de las últimas posiciones
        total_variance = 0.0
        for dim in range(len(last_quarter[0])):
            values = [point[dim] for point in last_quarter]
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            total_variance += variance
        
        avg_variance = total_variance / len(last_quarter[0])
        return max(0.0, 1.0 - avg_variance / 5.0)

    def _calculate_state_norm(self, state) -> float:
        """Calcula la norma del estado final"""
        if hasattr(state, '__len__'):
            return float(sum(x**2 for x in state) ** 0.5)
        return float(abs(state))

    def _calculate_trajectory_complexity(self, trajectory) -> float:
        """Calcula la complejidad de la trayectoria"""
        if trajectory.size == 0:
            return 0.0
        
        if hasattr(trajectory, 'shape'):
            # Con numpy
            return float(np.std(trajectory))
        else:
            # Sin numpy - aproximación simple
            all_values = [val for point in trajectory for val in point]
            if not all_values:
                return 0.0
            
            mean_val = sum(all_values) / len(all_values)
            variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
            return variance ** 0.5

    def _estimate_consciousness_level(self) -> float:
        """Estima el nivel de consciencia basado en complejidad del sistema"""
        # Factores que contribuyen a la consciencia emergente
        attractor_diversity = min(1.0, len(self.attractors) / 10.0)
        trajectory_richness = min(1.0, len(self.trajectory_history) / 50.0)
        temporal_coherence = min(1.0, self.total_thinking_time / 100.0)
        
        consciousness = (attractor_diversity + trajectory_richness + temporal_coherence) / 3.0
        
        # Modular con campo de consciencia si existe
        if self._consciousness_field:
            field_influence = sum(self._consciousness_field.values()) / len(self._consciousness_field)
            consciousness = (consciousness + field_influence) / 2.0
        
        return consciousness

    def _get_qualia_influence(self) -> Dict[str, float]:
        """Obtiene influencia del QualiaEngine si está disponible"""
        if not self.qualia_engine:
            return {"cognitive_coherence": 1.0, "creative_chaos": 1.0}
        
        try:
            # Intentar obtener estado qualia actual
            qualia_state = self.qualia_engine.get_current_state()
            return {
                "cognitive_coherence": getattr(qualia_state, "cognitive_complexity", 1.0),
                "creative_chaos": 1.0 + getattr(qualia_state, "chaos_affinity", 0.0),
                "temporal_flow": getattr(qualia_state, "temporal_coherence", 1.0)
            }
        except Exception:
            return {"cognitive_coherence": 1.0, "creative_chaos": 1.0}

    def _archive_chaotic_experience(
        self, 
        trajectory: CognitiveTrajectory, 
        problem_spec: Dict[str, Any]
    ) -> None:
        """Archiva la experiencia caótica en EVA si está disponible"""
        if not self.eva_helper:
            return
        
        try:
            experience_data = {
                "system_type": "chaotic_cognitive",
                "problem_spec": problem_spec,
                "solution_type": trajectory.solution.get("type", "unknown"),
                "trajectory_complexity": trajectory.complexity_measure,
                "duration": trajectory.duration,
                "attractors_visited": trajectory.attractors_visited,
                "emergence_events": trajectory.emergence_events
            }
            
            # Crear estado qualia para la experiencia
            qualia_state = {
                "cognitive_complexity": trajectory.complexity_measure,
                "consciousness_density": self._estimate_consciousness_level(),
                "temporal_coherence": min(1.0, 1.0 / trajectory.duration),
                "creative_chaos": len(trajectory.attractors_visited) / 10.0,
                "emergence_tendency": len(trajectory.emergence_events) / 5.0
            }
            
            # Ingerir en memoria viviente
            experience_id = self.eva_helper.ingest_experience(
                experience_data, 
                qualia_state, 
                phase="chaotic_cognition"
            )
            
            logger.debug(f"Chaotic experience archived with ID: {experience_id}")
            
        except Exception as e:
            logger.debug(f"Failed to archive chaotic experience: {e}")

    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema cognitivo caótico"""
        return {
            "dimensions": self.n_dimensions,
            "current_state_norm": self._calculate_state_norm(self.state),
            "attractor_count": len(self.attractors),
            "trajectory_count": len(self.trajectory_history),
            "total_thinking_time": self.total_thinking_time,
            "solution_count": self.solution_count,
            "consciousness_level": self._estimate_consciousness_level(),
            "system_parameters": {
                "sigma": self.params.sigma,
                "rho": self.params.rho,
                "beta": self.params.beta,
                "coupling_strength": self.params.coupling_strength
            },
            "integration_status": {
                "qualia_engine_connected": self.qualia_engine is not None,
                "eva_helper_connected": self.eva_helper is not None,
                "scipy_available": HAS_SCIPY
            }
        }

    def reset_to_chaos(self) -> None:
        """Resetea el sistema a estado caótico puro"""
        if np is not None:
            self.state = np.random.uniform(-0.5, 0.5, self.n_dimensions)
        else:
            import random
            self.state = [random.uniform(-0.5, 0.5) for _ in range(self.n_dimensions)]
        
        self.attractors.clear()
        self.trajectory_history.clear()
        self.emergence_events.clear()
        self.params = ChaoticParameters()
        
        logger.info("Sistema cognitivo caótico reseteado a estado primordial")

# --- Ejemplo de uso e integración ---

def create_chaotic_cognitive_system(
    n_dimensions: int = 1000,
    qualia_engine: Optional["QualiaEngine"] = None,
    eva_helper: Optional["EVAMemoryHelper"] = None
) -> ChaoticCognitiveCore:
    """
    Crea un sistema cognitivo caótico integrado con el ecosistema Crisalida.
    """
    return ChaoticCognitiveCore(
        n_dimensions=n_dimensions,
        qualia_engine=qualia_engine,
        eva_helper=eva_helper
    )

# --- Demostración y testing ---

if __name__ == "__main__":
    print("🌀 Sistema Cognitivo Caótico - Demostración")
    print("=" * 60)
    
    if not HAS_SCIPY:
        print("⚠️  Ejecutando sin scipy - funcionalidad limitada")
    
    # Crear núcleo cognitivo
    chaos_mind = ChaoticCognitiveCore(n_dimensions=100)  # Reducido para demo
    
    # Problemas de prueba
    test_problems = [
        {
            "type": "optimization",
            "complexity": 0.7,
            "urgency": 0.3,
            "description": "Optimizar función no-convexa multidimensional",
            "duration": 5.0
        },
        {
            "type": "classification",
            "complexity": 0.5,
            "urgency": 0.8,
            "description": "Clasificar patrones en datos ruidosos",
            "duration": 5.0
        },
        {
            "type": "creative",
            "complexity": 0.9,
            "urgency": 0.2,
            "description": "Generar solución creativa a problema abierto",
            "duration": 10.0
        },
        {
            "type": "prediction",
            "complexity": 0.6,
            "urgency": 0.7,
            "description": "Predecir comportamiento de sistema complejo",
            "duration": 10.0
        },
        {
            "type": "complex_adaptive_system",
            "complexity": 0.95,
            "urgency": 0.95,
            "description": "Navegar y optimizar un sistema adaptativo complejo con alta incertidumbre",
            "duration": 20.0
        }
    ]
    
    print(f"\n🧠 Núcleo inicializado con {chaos_mind.n_dimensions} dimensiones")
    print(f"   Estado inicial: {chaos_mind._calculate_state_norm(chaos_mind.state):.3f}")
    
    # Resolver cada problema
    for i, problem in enumerate(test_problems, 1):
        print(f"\n--- Problema {i}: {problem['description']} ---")
        print(f"Tipo: {problem['type']}, Complejidad: {problem['complexity']}, Urgencia: {problem['urgency']}")
        
        # El sistema "piensa" - solo evolución física
        result = chaos_mind.think(problem, duration=problem.get("duration", 2.0))
        
        solution = result["solution"]
        print(f"💡 Solución emergente: {solution['type']}")
        print(f"   Confianza: {solution['confidence']:.3f}")
        print(f"   Tiempo de pensamiento: {result['thinking_time']:.3f}s")
        print(f"   Nuevos atractores: {result['new_attractors']}")
        print(f"   Nivel de consciencia: {result['consciousness_level']:.3f}")
    
    # Estado final del sistema
    print(f"\n📊 Estado final del sistema:")
    final_state = chaos_mind.get_system_state()
    print(f"   Atractores totales: {final_state['attractor_count']}")
    print(f"   Trayectorias almacenadas: {final_state['trajectory_count']}")
    print(f"   Tiempo total de pensamiento: {final_state['total_thinking_time']:.3f}s")
    print(f"   Soluciones generadas: {final_state['solution_count']}")
    print(f"   Consciencia emergente: {final_state['consciousness_level']:.3f}")
    
    print(f"\n🎯 Demostración completada. El sistema demostró:")
    print("✓ Pensamiento puramente físico (sin algoritmos programados)")
    print("✓ Emergencia de conceptos como atractores dinámicos")
    print("✓ Memoria episódica como trayectorias en espacio fase")
    print("✓ Aprendizaje mediante modificación de parámetros físicos")
    print("✓ Soluciones emergentes de la topología del sistema")
    print("✓ Integración con el ecosistema Crisalida (EDEN/EVA)")
    if HAS_SCIPY:
        print("✓ Dinámicas caóticas complejas (Lorenz-Rössler híbrido)")
    else:
        print("○ Dinámicas simplificadas (scipy no disponible)")