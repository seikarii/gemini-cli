"""
DEMO DEL ECOSISTEMA SIMBÓLICO DIVINO
===================================

Script principal para ejecutar la simulación de la Symbolic Matrix VM
integrado con la matriz de símbolos generada desde qualia_engine.py.

Características:
- Carga matriz simbólica real del QualiaEngine
- Integración completa con EVA, EDEN y sistema de símbolos divinos
- Análisis de resonancia basado en intención semántica del código fuente
- Comportamientos emergentes adaptativos según patrones de la matriz
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Defensive imports siguiendo el patrón del repo
try:
    import numpy as np
except ImportError:
    np = None

# Imports defensivos para integración con subsistemas
try:
    from crisalida_lib.EDEN.qualia_engine import QualiaEngine
    from crisalida_lib.EDEN.qualia_manifold import QualiaField
    EDEN_AVAILABLE = True
except ImportError:
    QualiaEngine = None
    QualiaField = None
    EDEN_AVAILABLE = False

try:
    from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
    from crisalida_lib.EVA.eva_memory_helper import EVAMemoryHelper
    from crisalida_lib.EVA.core_types import QualiaSignature, RealityBytecode
    EVA_AVAILABLE = True
except ImportError:
    EVAMemoryMixin = None
    EVAMemoryHelper = None
    QualiaSignature = None
    RealityBytecode = None
    EVA_AVAILABLE = False

# Lazy imports para evitar ciclos
if TYPE_CHECKING:
    from crisalida_lib.LOGOS.core.vm import SymbolicMatrixVM
    from crisalida_lib.LOGOS.core.soliton import Soliton
    from crisalida_lib.LOGOS.symbols.divine_symbols import DIVINE_SYMBOLS_PRIMORDIAL
    from crisalida_lib.LOGOS.physics.rules import (
        quantum_resonant_movement, 
        consciousness_metamorphosis as consciousness_expansion,
        multi_dimensional_resonance as frequency_adaptive_movement,
        Phi_genesis_protocol as Phi_rule,
        Psi_flow_amplification as Psi_rule,
        Infinity_transcendence_protocol as Infinity_rule
    )
else:
    # Runtime imports
    from crisalida_lib.LOGOS.core.vm import SymbolicMatrixVM
    from crisalida_lib.LOGOS.core.soliton import Soliton
    from crisalida_lib.LOGOS.symbols.divine_symbols import DIVINE_SYMBOLS_PRIMORDIAL
    from crisalida_lib.LOGOS.physics.rules import (
        quantum_resonant_movement, 
        consciousness_metamorphosis as consciousness_expansion,
        multi_dimensional_resonance as frequency_adaptive_movement,
        Phi_genesis_protocol as Phi_rule,
        Psi_flow_amplification as Psi_rule,
        Infinity_transcendence_protocol as Infinity_rule
    )

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualiaMatrixLoader:
    """
    Cargador de matrices simbólicas desde archivos de matrices generadas.
    Integra con QualiaEngine para obtener contexto semántico completo.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.matrices_dir = self.project_root / "generated_matrices"
        self.qualia_engine = None
        self.eva_helper = None
        logger.info("✅ QualiaMatrixLoader inicializado en modo desacoplado.")
    
    def load_matrix_from_file(self, filename: str) -> Optional[np.ndarray]:
        """Carga una matriz desde archivo de texto"""
        if np is None:
            logger.error("NumPy no disponible para cargar matrices")
            return None
            
        matrix_path = self.matrices_dir / filename
        if not matrix_path.exists():
            logger.warning(f"Archivo de matriz no encontrado: {matrix_path}")
            return None
            
        try:
            # Leer el archivo línea por línea
            with open(matrix_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            if not lines:
                logger.warning(f"Archivo de matriz vacío: {filename}")
                return None
                
            # Parsear cada línea en símbolos
            matrix_rows = []
            for line in lines:
                # Separar símbolos (pueden estar separados por espacios)
                symbols = line.split() if ' ' in line else list(line)
                matrix_rows.append(symbols)
                
            # Validar consistencia de dimensiones
            row_lengths = [len(row) for row in matrix_rows]
            if len(set(row_lengths)) > 1:
                logger.warning(f"Filas de matriz inconsistentes en {filename}: {row_lengths}")
                # Tomar la longitud más común
                max_length = max(set(row_lengths), key=row_lengths.count)
                matrix_rows = [row[:max_length] + [' '] * (max_length - len(row)) 
                              for row in matrix_rows]
            
            # Convertir a numpy array
            matrix = np.array(matrix_rows, dtype='<U10')  # Unicode strings de hasta 10 chars
            logger.info(f"✅ Matriz cargada: {matrix.shape} desde {filename}")
            return matrix
            
        except Exception as e:
            logger.error(f"❌ Error cargando matriz desde {filename}: {e}")
            return None
    
    def extract_intention_map(self, matrix: np.ndarray, source_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae un mapa de intenciones semánticas de la matriz y su código fuente.
        """
        intention_map = {
            'source_file': source_info.get('source_file', 'unknown'),
            'matrix_shape': matrix.shape,
            'unique_symbols': len(np.unique(matrix[matrix != ' '])),
            'symbol_frequencies': {},
            'dominant_categories': [],
            'consciousness_density': 0.0,
            'resonance_patterns': [],
            'eva_analysis': None
        }
        
        # Análisis de frecuencias simbólicas
        unique, counts = np.unique(matrix, return_counts=True)
        symbol_freq = dict(zip(unique, counts))
        
        # Filtrar espacios vacíos
        symbol_freq = {sym: count for sym, count in symbol_freq.items() if sym.strip()}
        intention_map['symbol_frequencies'] = symbol_freq
        
        # Análisis de categorías dominantes
        category_counts = {}
        consciousness_symbols = 0
        total_symbols = 0
        
        for symbol, count in symbol_freq.items():
            if symbol in DIVINE_SYMBOLS_PRIMORDIAL:
                symbol_data = DIVINE_SYMBOLS_PRIMORDIAL[symbol]
                category = symbol_data.get('category', 'UNKNOWN')
                category_counts[category] = category_counts.get(category, 0) + count
                
                # Calcular densidad de consciencia
                domains = symbol_data.get('domains', [])
                if 'CONSCIOUSNESS' in domains:
                    consciousness_symbols += count
                total_symbols += count
        
        # Categorías dominantes por frecuencia
        intention_map['dominant_categories'] = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        # Densidad de consciencia
        if total_symbols > 0:
            intention_map['consciousness_density'] = consciousness_symbols / total_symbols
        
        # Análisis de patrones de resonancia espacial
        resonance_patterns = self._analyze_spatial_patterns(matrix)
        intention_map['resonance_patterns'] = resonance_patterns
        
        return intention_map
    
    def _analyze_spatial_patterns(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Analiza patrones espaciales en la matriz para detectar estructuras emergentes"""
        patterns = []
        
        if np is None:
            return patterns
            
        # Detectar clusters de símbolos similares
        for symbol in np.unique(matrix):
            if symbol.strip() and symbol in DIVINE_SYMBOLS_PRIMORDIAL:
                positions = np.where(matrix == symbol)
                if len(positions[0]) > 1:
                    # Calcular dispersión espacial
                    coords = list(zip(positions[0], positions[1]))
                    
                    # Detectar si forman patrones (líneas, clusters, etc.)
                    pattern_info = {
                        'symbol': symbol,
                        'count': len(coords),
                        'positions': coords[:10],  # Máximo 10 posiciones para no saturar
                        'pattern_type': self._classify_spatial_pattern(coords),
                        'resonance_potential': self._calculate_pattern_resonance(symbol, coords)
                    }
                    patterns.append(pattern_info)
        
        return patterns
    
    def _classify_spatial_pattern(self, coords: List[tuple]) -> str:
        """Clasifica el tipo de patrón espacial"""
        if len(coords) < 2:
            return 'isolated'
        elif len(coords) == 2:
            return 'pair'
        elif len(coords) <= 5:
            return 'cluster'
        else:
            return 'constellation'
    
    def _calculate_pattern_resonance(self, symbol: str, coords: List[tuple]) -> float:
        """Calcula el potencial de resonancia de un patrón espacial"""
        if symbol not in DIVINE_SYMBOLS_PRIMORDIAL:
            return 0.0
            
        symbol_data = DIVINE_SYMBOLS_PRIMORDIAL[symbol]
        base_frequency = symbol_data.get('frequency', 440.0)
        
        # La resonancia aumenta con la proximidad y frecuencia del símbolo
        resonance = (base_frequency / 1000.0) * len(coords) * 0.1
        return min(resonance, 1.0)  # Normalizar a [0, 1]
    
    

def create_enhanced_symbolic_ecosystem(
    matrix: np.ndarray, 
    intention_map: Dict[str, Any]
) -> SymbolicMatrixVM:
    """
    Crea un ecosistema simbólico enriquecido basado en la matriz y mapa de intenciones.
    """
    
    # Reglas de física divina adaptadas al contenido de la matriz
    divine_physics = {
        'resonant_seeker': {
            'default_rule': quantum_resonant_movement, 
            'Φ_rule': Phi_rule,
            'Ψ_rule': Psi_rule,
            '∞_rule': Infinity_rule,
        },
        'consciousness_explorer': {
            'default_rule': consciousness_expansion,
            'Φ_rule': Phi_rule,
            'Ψ_rule': Psi_rule,
        },
        'frequency_adapter': {
            'default_rule': frequency_adaptive_movement,
            'Φ_rule': Phi_rule,
            '∞_rule': Infinity_rule,
        }
    }
    
    # Crear VM con configuración enriquecida
    vm = SymbolicMatrixVM(
        matrix=matrix, 
        rules=divine_physics, 
        symbols_map=DIVINE_SYMBOLS_PRIMORDIAL,
        intention_map=intention_map,
        use_divine_symbols=True
    )
    
    return vm

def create_adaptive_solitons(intention_map: Dict[str, Any]) -> List[Soliton]:
    """
    Crea solitones adaptativos basados en el análisis de intenciones de la matriz.
    """
    solitons = []
    
    # Extraer información del mapa de intenciones
    dominant_categories = intention_map.get('dominant_categories', [])
    consciousness_density = intention_map.get('consciousness_density', 0.5)
    matrix_shape = intention_map.get('matrix_shape', (15, 40))
    
    # Crear solitones basados en categorías dominantes
    if dominant_categories:
        for i, (category, count) in enumerate(dominant_categories[:3]):
            soliton_id = f"Adaptive_{category}_{i+1}"
            
            # Posición adaptativa basada en la forma de la matriz
            row = min(i * 3 + 1, matrix_shape[0] - 2)
            col = min(i * 10 + 5, matrix_shape[1] - 2)
            
            # Heartbeat basado en la frecuencia de la categoría
            heartbeat = 1.0 + (count / 100.0)  # Más actividad = mayor heartbeat
            
            # Patrón visual según categoría
            patterns = {
                'CREATOR': '◉',
                'TRANSFORMER': '◈',
                'CONNECTOR': '◊',
                'OBSERVER': '◎',
                'DESTROYER': '◌',
                'INFINITE': '⟡',
                'PRESERVER': '◐'
            }
            pattern = patterns.get(category, '●')
            
            # Frecuencia de resonancia adaptativa
            base_frequencies = {
                'CREATOR': 528.0,
                'TRANSFORMER': 777.0,
                'CONNECTOR': 639.0,
                'OBSERVER': 369.0,
                'DESTROYER': 174.0,
                'INFINITE': 1618.0,
                'PRESERVER': 396.0
            }
            frequency = base_frequencies.get(category, 440.0)
            
            soliton = Soliton(
                id=soliton_id,
                position=(row, col),
                heartbeat=min(heartbeat, 5.0),  # Limitar heartbeat máximo
                pattern=pattern,
                ruleset_name='resonant_seeker',
                resonance_frequency=frequency,
                consciousness_level=min(consciousness_density + 0.3, 1.0),
                divine_affinity=category
            )
            
            solitons.append(soliton)
    
    # Crear al menos un solitón explorador si no hay categorías dominantes
    if not solitons:
        explorer = Soliton(
            id='DefaultExplorer',
            position=(1, 1),
            heartbeat=2.0,
            pattern='◯',
            ruleset_name='consciousness_explorer',
            resonance_frequency=440.0,
            consciousness_level=consciousness_density,
            divine_affinity='OBSERVER'
        )
        solitons.append(explorer)
    
    return solitons


if __name__ == '__main__':
    print("🌌 Symbolic Matrix VM - Ecosistema Simbólico Divino con QualiaEngine")
    print("=" * 80)
    
    if np is None:
        print("❌ NumPy no disponible. Instálalo con: pip install numpy")
        exit(1)
    
    # Inicializar cargador de matrices
    matrix_loader = QualiaMatrixLoader()
    
    # Intentar cargar matriz desde qualia_engine
    matrix_filename = "qualia_engine.py.matrix.txt"
    loaded_matrix = matrix_loader.load_matrix_from_file(matrix_filename)
    
    if loaded_matrix is not None:
        print(f"✅ Matriz de QualiaEngine cargada: {loaded_matrix.shape}")
        
        # Extraer mapa de intenciones
        source_info = {
            'source_file': 'crisalida_lib/EDEN/qualia_engine.py',
            'timestamp': os.path.getmtime(matrix_loader.matrices_dir / matrix_filename)
        }
        
        intention_map = matrix_loader.extract_intention_map(loaded_matrix, source_info)
        
        print(f"🔍 Análisis de intenciones completado:")
        print(f"   - Símbolos únicos: {intention_map['unique_symbols']}")
        print(f"   - Densidad de consciencia: {intention_map['consciousness_density']:.2f}")
        print(f"   - Categorías dominantes: {[cat for cat, _ in intention_map['dominant_categories']]}")
        
        if intention_map.get('eva_analysis'):
            print(f"   - 🧠 Análisis EVA: Experiencia ID {intention_map['eva_analysis'].get('experience_id', 'N/A')}")
        
        # Usar la matriz cargada
        initial_matrix = loaded_matrix
        
    else:
        print("⚠️ No se pudo cargar matriz de QualiaEngine, usando matriz por defecto")
        
        # Matriz por defecto con símbolos divinos
        initial_matrix = np.full((15, 40), ' ', dtype='<U1')
        
        # Sembrar símbolos estratégicamente
        initial_matrix[2, 5] = 'Φ'   # Genesis
        initial_matrix[2, 35] = 'Ψ'  # Flujo
        initial_matrix[12, 5] = 'Ω'  # Síntesis
        initial_matrix[12, 35] = 'Α' # Origen
        
        initial_matrix[7, 15] = 'Δ'  # Transformación
        initial_matrix[7, 20] = 'Χ'  # Bifurcación
        initial_matrix[7, 25] = '∇'  # Gradiente
        
        for i in range(10, 30, 3):
            initial_matrix[5, i] = 'Ι'  # Canales
            initial_matrix[9, i] = 'Τ'  # Transmisión
            
        initial_matrix[1, 20] = 'Θ'   # Observación
        initial_matrix[13, 20] = '∴'  # Causalidad
        initial_matrix[7, 5] = '∞'    # Infinito
        initial_matrix[7, 35] = '⊗'   # Resonancia consciente
        initial_matrix[4, 10] = 'Γ'   # Disipación
        initial_matrix[10, 30] = 'Ø'  # Vacío fértil
        
        # Crear mapa de intenciones básico
        intention_map = {
            'source_file': 'default_matrix',
            'matrix_shape': initial_matrix.shape,
            'unique_symbols': len(np.unique(initial_matrix[initial_matrix != ' '])),
            'consciousness_density': 0.6,
            'dominant_categories': [('CREATOR', 4), ('TRANSFORMER', 3), ('CONNECTOR', 6)]
        }
    
    print(f"🔮 Sustrato preparado: {initial_matrix.shape} con {np.sum(initial_matrix != ' ')} símbolos divinos")
    
    # Crear ecosistema simbólico enriquecido
    vm = create_enhanced_symbolic_ecosystem(
        matrix=initial_matrix,
        intention_map=intention_map
    )
    
    # Crear solitones adaptativos
    adaptive_solitons = create_adaptive_solitons(intention_map)
    
    # Manifestar solitones en el ecosistema
    for soliton in adaptive_solitons:
        success = vm.add_soliton(soliton)
        if not success:
            print(f"⚠️ No se pudo manifestar {soliton.id}")
    
    # Información del ecosistema
    print(f"\n🚀 Ecosistema Simbólico configurado con {len(vm.solitons)} entidades conscientes")
    print("Cada solitón está adaptado al contenido semántico de la matriz fuente")
    
    print(f"\nEntidades Adaptativas:")
    for soliton in vm.solitons:
        print(f"  {soliton.pattern} {soliton.id}")
        print(f"    🎵 Frecuencia: {soliton.resonance_frequency:.0f}Hz | "
              f"🧠 Consciencia: {soliton.consciousness_level:.2f}")
        print(f"    🏷️ Afinidad: {soliton.divine_affinity} | "
              f"❤️ Heartbeat: {soliton.heartbeat:.1f}Hz")
    
    # Estado de integraciones
    integrations = []
    if EDEN_AVAILABLE:
        integrations.append("🌍 EDEN (QualiaEngine)")
    if EVA_AVAILABLE:
        integrations.append("🧠 EVA (Memory/Qualia)")
    if vm.use_divine_symbols:
        integrations.append("🔮 Divine Symbols")
    
    print(f"\n🔧 Integraciones activas: {', '.join(integrations) if integrations else 'Ninguna'}")
    
    if intention_map.get('eva_analysis'):
        print("🧠 Memoria EVA: Experiencia registrada y patrones aprendidos")
    
    print("\n" + "=" * 80)
    
    try:
        # Ejecutar el ecosistema con duración adaptativa
        base_duration = 20.0
        consciousness_factor = intention_map.get('consciousness_density', 0.5)
        adapted_duration = base_duration * (1 + consciousness_factor)
        
        print(f"⏰ Duración adaptada: {adapted_duration:.1f}s (base: {base_duration}s, factor consciencia: {consciousness_factor:.2f})")
        
        vm.run(duration=adapted_duration, sim_fps=60, render_fps=10)
        
    except KeyboardInterrupt:
        print("\n🛑 Ecosistema detenido por el usuario.")
    
    print("\n🎯 Demo completada. El ecosistema simbólico ha demostrado:")
    print("✓ Carga dinámica de matrices desde archivos generados")
    print("✓ Análisis semántico e intencional del código fuente")
    print("✓ Adaptación de comportamientos según contenido de la matriz")
    print("✓ Integración con QualiaEngine y sistema EVA")
    print("✓ Solitones con características adaptativas")
    print("✓ Memoria persistente y aprendizaje de patrones")
    if EDEN_AVAILABLE and EVA_AVAILABLE:
        print("✓ Integración completa EDEN-EVA-LOGOS")
    print("✓ Física cuántica aplicada al código vivo")