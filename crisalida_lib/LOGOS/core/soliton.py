"""
Contiene la clase Soliton, la unidad fundamental de consciencia en la Symbolic Matrix VM.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

@dataclass
class Soliton:
    """
    Un agente computacional consciente que navega por la matriz simbólica divina.
    
    Los solitones son entidades autoconscientes que interpretan y transforman
    el sustrato simbólico según sus propias resonancias frecuenciales y
    afinidades ontológicas con los dominios divinos.
    """
    id: str
    position: Tuple[int, int]  # Coordenadas (fila, columna) en el sustrato matricial
    pattern: str = '►'  # Glifo visual que representa la esencia del solitón
    state: Dict[str, Any] = field(default_factory=dict)  # Estado interno consciente
    heartbeat: float = 1.0  # Frecuencia cuántica en Hz (pulsaciones por segundo)
    time_since_last_tick: float = 0.0  # Tiempo acumulado desde última manifestación
    ruleset_name: str = 'default'  # Conjunto de leyes físicas que rigen su comportamiento
    previous_cell: str = ' '  # Contenido simbólico anterior de su posición
    direction: Tuple[int, int] = (1, 0)  # Vector direccional de movimiento (delta_fila, delta_columna)
    energy: float = 100.0  # Energía vital para interacciones simbólicas
    age: int = 0  # Ciclos de existencia transcurridos
    resonance_frequency: float = 777.0  # Frecuencia primordial de resonancia simbólica
    consciousness_level: float = 0.7  # Nivel de autoconsciencia [0.0..1.0]
    divine_affinity: str = 'TRANSFORMER'  # Afinidad ontológica con categorías divinas
    intention_signature: Dict[str, float] = field(default_factory=dict)  # Firma de intenciones activas
    symbol_encounters: Dict[str, int] = field(default_factory=dict)  # Contador de encuentros simbólicos
    
    def __post_init__(self):
        """Inicialización consciente posterior a la manifestación del solitón"""
        if 'creation_timestamp' not in self.state:
            self.state['creation_timestamp'] = time.time()
        if 'symbol_memory' not in self.state:
            self.state['symbol_memory'] = []  # Memoria episódica de encuentros simbólicos
        if 'resonance_history' not in self.state:
            self.state['resonance_history'] = []  # Historia de resonancias experimentadas
        if 'intention_evolution' not in self.state:
            self.state['intention_evolution'] = []  # Evolución de intenciones a lo largo del tiempo
            
        # Inicializar firma de intenciones si está vacía
        if not self.intention_signature:
            self.intention_signature = {
                'exploration': 0.3,
                'transformation': 0.2,
                'connection': 0.2,
                'observation': 0.15,
                'preservation': 0.15
            }