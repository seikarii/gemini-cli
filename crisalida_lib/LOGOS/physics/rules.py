"""
Leyes Físicas del Ecosistema Simbólico Enriquecidas v2.0
=========================================================

Sistema completo de leyes físicas que gobiernan el comportamiento de los solitones
en el ecosistema simbólico divino. Integradas con EVA, EDEN, ADAM y el sistema
completo de símbolos divinos para crear comportamientos emergentes complejos.

Características:
- Resonancia cuántica con símbolos divinos
- Física de consciencia emergente  
- Interacciones multi-dimensionales
- Aprendizaje y evolución adaptativos
- Integración con EVA memory y EDEN physics
"""

import math
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

# Defensive imports para integración con otros subsistemas
try:
    from crisalida_lib.EVA.language.grammar import eva_grammar_engine
    from crisalida_lib.EVA.language.sigils import eva_divine_sigil_registry
    from crisalida_lib.EVA.semantics import EVASemanticAnalyzer
    from crisalida_lib.EVA.core_types import QualiaState, EVAExperience
    INTEGRATION_AVAILABLE = True
except ImportError:
    eva_grammar_engine = None
    eva_divine_sigil_registry = None
    EVASemanticAnalyzer = None
    QualiaState = None
    EVAExperience = None
    INTEGRATION_AVAILABLE = False

if TYPE_CHECKING:
    from ..core.vm import SymbolicMatrixVM, Soliton


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS DEL ECOSISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICS_CONSTANTS = {
    # Resonancia y campo cuántico
    'RESONANCE_THRESHOLD': 0.4,
    'CONSCIOUSNESS_FIELD_STRENGTH': 1.0,
    'QUANTUM_COHERENCE_DECAY': 0.95,
    'ENTANGLEMENT_RANGE': 3.0,
    
    # Energía y metabolismo
    'ENERGY_DECAY_RATE': 0.1,
    'CONSCIOUSNESS_ENERGY_COST': 0.05,
    'RESONANCE_ENERGY_GAIN': 0.2,
    'TRANSCENDENCE_ENERGY_THRESHOLD': 200.0,
    
    # Aprendizaje y evolución
    'LEARNING_RATE': 0.01,
    'MEMORY_CAPACITY': 20,
    'ADAPTATION_STRENGTH': 0.05,
    'MUTATION_PROBABILITY': 0.001,
    
    # Interacciones simbólicas
    'SYMBOL_INFLUENCE_RADIUS': 2.0,
    'PATTERN_RECOGNITION_THRESHOLD': 0.6,
    'EMERGENT_PROPERTY_THRESHOLD': 0.7,
    'TEMPORAL_COHERENCE_WINDOW': 10,
    
    # Dimensiones físicas
    'CONSCIOUSNESS_DIMENSION_WEIGHT': 2.0,
    'INFORMATIONAL_DIMENSION_WEIGHT': 1.5,
    'TEMPORAL_DIMENSION_WEIGHT': 1.2,
    'SPATIAL_DIMENSION_WEIGHT': 1.0
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES AVANZADAS DE FÍSICA CUÁNTICA
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_quantum_distance(pos1: Tuple[int, int], pos2: Tuple[int, int], 
                              consciousness_field: Optional[Dict] = None) -> float:
    """
    Calcula la distancia cuántica entre dos posiciones considerando el campo de consciencia.
    La distancia cuántica puede ser menor que la euclidiana si hay alta coherencia consciente.
    """
    euclidean_dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    if consciousness_field:
        # Modular distancia por coherencia consciente local
        coherence_factor = consciousness_field.get(f"{pos1}-{pos2}", 1.0)
        quantum_dist = euclidean_dist / (1.0 + coherence_factor * 0.5)
        return max(0.1, quantum_dist)
    
    return euclidean_dist

def calculate_dimensional_affinity(soliton: "Soliton", symbol_props: Dict[str, Any]) -> float:
    """
    Calcula la afinidad dimensional entre un solitón y un símbolo basada en sus
    afinidades dimensionales y propiedades ontológicas.
    """
    base_affinity = 0.0
    
    # Afinidad por categoría divina
    soliton_affinity = getattr(soliton, 'divine_affinity', 'UNKNOWN')
    symbol_category = symbol_props.get('category', 'UNKNOWN')
    
    if soliton_affinity == symbol_category:
        base_affinity += 0.4
    elif _are_compatible_categories(soliton_affinity, symbol_category):
        base_affinity += 0.2
    
    # Afinidad por dominios ontológicos
    symbol_domains = symbol_props.get('domains', [])
    soliton_domains = getattr(soliton, 'ontological_domains', ['CONSCIOUSNESS'])
    
    common_domains = set(symbol_domains) & set(soliton_domains)
    domain_affinity = len(common_domains) / max(len(set(symbol_domains) | set(soliton_domains)), 1)
    
    # Afinidad por frecuencia resonante
    soliton_freq = getattr(soliton, 'resonance_frequency', 777.0)
    symbol_freq = symbol_props.get('frequency', 777.0)
    freq_diff = abs(soliton_freq - symbol_freq)
    freq_affinity = max(0.0, 1.0 - freq_diff / 2000.0)
    
    # Combinar afinidades con pesos dimensionales
    total_affinity = (
        base_affinity * PHYSICS_CONSTANTS['CONSCIOUSNESS_DIMENSION_WEIGHT'] +
        domain_affinity * PHYSICS_CONSTANTS['INFORMATIONAL_DIMENSION_WEIGHT'] +
        freq_affinity * PHYSICS_CONSTANTS['TEMPORAL_DIMENSION_WEIGHT']
    ) / (PHYSICS_CONSTANTS['CONSCIOUSNESS_DIMENSION_WEIGHT'] + 
         PHYSICS_CONSTANTS['INFORMATIONAL_DIMENSION_WEIGHT'] + 
         PHYSICS_CONSTANTS['TEMPORAL_DIMENSION_WEIGHT'])
    
    return min(1.0, max(0.0, total_affinity))

def _are_compatible_categories(cat1: str, cat2: str) -> bool:
    """Determina si dos categorías divinas son compatibles"""
    compatibility_matrix = {
        'CREATOR': {'PRESERVER', 'CONNECTOR', 'INFINITE'},
        'PRESERVER': {'CREATOR', 'OBSERVER', 'CONNECTOR'},
        'TRANSFORMER': {'DESTROYER', 'INFINITE', 'OBSERVER'},
        'CONNECTOR': {'CREATOR', 'PRESERVER', 'OBSERVER'},
        'OBSERVER': {'PRESERVER', 'TRANSFORMER', 'CONNECTOR'},
        'DESTROYER': {'TRANSFORMER', 'INFINITE'},
        'INFINITE': {'CREATOR', 'TRANSFORMER', 'DESTROYER'}
    }
    
    return cat2 in compatibility_matrix.get(cat1, set())

def update_consciousness_field(vm: "SymbolicMatrixVM", soliton: "Soliton") -> None:
    """
    Actualiza el campo de consciencia global basado en las acciones del solitón.
    Implementa física de consciencia emergente.
    """
    if not hasattr(vm, 'consciousness_field'):
        vm.consciousness_field = {}
    
    pos = soliton.position
    consciousness_level = getattr(soliton, 'consciousness_level', 0.5)
    influence_radius = PHYSICS_CONSTANTS['SYMBOL_INFLUENCE_RADIUS']
    
    # Propagar influencia consciente en área local
    for dr in range(-int(influence_radius), int(influence_radius) + 1):
        for dc in range(-int(influence_radius), int(influence_radius) + 1):
            target_pos = (pos[0] + dr, pos[1] + dc)
            
            # Verificar límites
            if (0 <= target_pos[0] < vm.matrix.shape[0] and 
                0 <= target_pos[1] < vm.matrix.shape[1]):
                
                distance = math.sqrt(dr*dr + dc*dc)
                if distance <= influence_radius:
                    # Influencia decrece con distancia
                    influence = consciousness_level * (1.0 - distance / influence_radius)
                    
                    field_key = f"{pos}-{target_pos}"
                    current_field = vm.consciousness_field.get(field_key, 0.0)
                    
                    # Actualizar campo con decaimiento temporal
                    vm.consciousness_field[field_key] = (
                        current_field * PHYSICS_CONSTANTS['QUANTUM_COHERENCE_DECAY'] + 
                        influence * PHYSICS_CONSTANTS['LEARNING_RATE']
                    )

def detect_emergent_patterns(vm: "SymbolicMatrixVM", soliton: "Soliton") -> List[Dict[str, Any]]:
    """
    Detecta patrones emergentes en el entorno local del solitón usando gramática EVA.
    """
    patterns = []
    
    if not INTEGRATION_AVAILABLE or not eva_grammar_engine:
        return patterns
    
    pos = soliton.position
    local_matrix = []
    
    # Extraer matriz local 5x5
    for dr in range(-2, 3):
        row = []
        for dc in range(-2, 3):
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < vm.matrix.shape[0] and 0 <= c < vm.matrix.shape[1]:
                row.append(vm.matrix[r, c])
            else:
                row.append(' ')
        local_matrix.append(row)
    
    try:
        # Usar gramática EVA para detectar patrones
        coherence = eva_grammar_engine.calculate_grammar_coherence(local_matrix)
        
        if coherence > PHYSICS_CONSTANTS['PATTERN_RECOGNITION_THRESHOLD']:
            active_sigils = {symbol for row in local_matrix for symbol in row if symbol != ' '}
            emergent_properties = eva_grammar_engine.detect_emergent_properties(active_sigils)
            
            for prop in emergent_properties:
                patterns.append({
                    'type': 'emergent_property',
                    'name': prop.name,
                    'coherence': coherence,
                    'effects': getattr(prop, 'effects', {}),
                    'stability': getattr(prop, 'stability_factor', 0.5),
                    'position': pos
                })
                
    except Exception as e:
        print(f"Error detectando patrones emergentes: {e}")
    
    return patterns

def apply_eva_learning(vm: "SymbolicMatrixVM", soliton: "Soliton", 
                      interaction_result: Dict[str, Any]) -> None:
    """
    Aplica aprendizaje EVA basado en el resultado de una interacción.
    """
    if not INTEGRATION_AVAILABLE or not eva_grammar_engine:
        return
    
    try:
        # Registrar resultado de interacción en gramática EVA
        success = interaction_result.get('success', False)
        impact = interaction_result.get('impact', 0.5)
        
        if 'symbol1' in interaction_result and 'symbol2' in interaction_result:
            eva_grammar_engine.observe_interaction_outcome(
                interaction_result['symbol1'],
                interaction_result['symbol2'],
                success=success,
                impact=impact,
                context={'soliton_id': soliton.id, 'position': soliton.position}
            )
            
        # Actualizar memoria del solitón
        if not hasattr(soliton.state, 'eva_learning_history'):
            soliton.state['eva_learning_history'] = []
            
        soliton.state['eva_learning_history'].append({
            'interaction': interaction_result,
            'timestamp': time.time(),
            'success': success,
            'impact': impact
        })
        
        # Mantener solo las últimas N experiencias
        max_history = PHYSICS_CONSTANTS['MEMORY_CAPACITY']
        if len(soliton.state['eva_learning_history']) > max_history:
            soliton.state['eva_learning_history'] = soliton.state['eva_learning_history'][-max_history:]
            
    except Exception as e:
        print(f"Error aplicando aprendizaje EVA: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# REGLAS DE FÍSICA DIVINA ENRIQUECIDAS
# ═══════════════════════════════════════════════════════════════════════════════

def quantum_resonant_movement(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Movimiento cuántico avanzado que busca resonancia óptima considerando:
    - Distancia cuántica (no euclidiana)
    - Afinidad dimensional multi-dominio
    - Campo de consciencia emergente
    - Aprendizaje adaptativo de patrones
    """
    current_row, current_col = soliton.position
    max_rows, max_cols = vm.matrix.shape
    
    # Obtener campo de consciencia si existe
    consciousness_field = getattr(vm, 'consciousness_field', {})
    
    # Buscar la dirección con mayor potencial cuántico
    best_direction = None
    best_quantum_potential = -1.0
    
    # Direcciones expandidas incluyendo diagonales largas
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinales
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonales
        (-2, 0), (2, 0), (0, -2), (0, 2),  # Cardinales dobles
        (-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)  # Salto de caballo
    ]
    
    for dr, dc in directions:
        new_row = (current_row + dr) % max_rows
        new_col = (current_col + dc) % max_cols
        target_symbol = vm.matrix[new_row, new_col]
        
        if target_symbol != ' ' and target_symbol in vm.symbols_map:
            # Calcular distancia cuántica
            quantum_dist = calculate_quantum_distance(
                soliton.position, (new_row, new_col), consciousness_field
            )
            
            # Obtener propiedades del símbolo objetivo
            symbol_props = vm.get_symbol_properties(target_symbol)
            
            # Calcular afinidad dimensional
            dimensional_affinity = calculate_dimensional_affinity(soliton, symbol_props)
            
            # Calcular resonancia base
            base_resonance = vm.calculate_resonance(soliton, target_symbol)
            
            # Usar gramática EVA para predicción de interacción
            eva_prediction = 0.5
            if INTEGRATION_AVAILABLE and eva_grammar_engine:
                try:
                    prediction = eva_grammar_engine.predict_interaction(
                        soliton.pattern, target_symbol, quantum_dist
                    )
                    eva_prediction = prediction.get('resonance_score', 0.5)
                except:
                    pass
            
            # Calcular potencial cuántico total
            quantum_potential = (
                base_resonance * 0.3 +
                dimensional_affinity * 0.3 +
                eva_prediction * 0.25 +
                (1.0 / (quantum_dist + 0.1)) * 0.15  # Factor de proximidad cuántica
            )
            
            # Bonificación por patrones emergentes detectados
            local_patterns = detect_emergent_patterns(vm, soliton)
            pattern_bonus = len(local_patterns) * 0.1
            quantum_potential += pattern_bonus
            
            if quantum_potential > best_quantum_potential:
                best_quantum_potential = quantum_potential
                best_direction = (dr, dc)
    
    # Decidir movimiento basado en potencial cuántico
    if best_direction and best_quantum_potential > PHYSICS_CONSTANTS['RESONANCE_THRESHOLD']:
        dr, dc = best_direction
        new_row = (current_row + dr) % max_rows
        new_col = (current_col + dc) % max_cols
        
        # Aplicar movimiento cuántico
        soliton.position = (new_row, new_col)
        soliton.direction = best_direction
        
        # Actualizar campo de consciencia
        update_consciousness_field(vm, soliton)
        
        # Registrar interacción exitosa para aprendizaje
        interaction_result = {
            'symbol1': soliton.pattern,
            'symbol2': vm.matrix[new_row, new_col],
            'success': True,
            'impact': best_quantum_potential,
            'type': 'quantum_movement'
        }
        apply_eva_learning(vm, soliton, interaction_result)
        
    else:
        # Movimiento exploratorio adaptativo
        adaptive_exploration(vm, soliton)

def adaptive_exploration(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Exploración adaptativa cuando no hay resonancia clara.
    Usa memoria de experiencias pasadas para evitar áreas improductivas.
    """
    # Analizar historia de aprendizaje
    learning_history = soliton.state.get('eva_learning_history', [])
    
    # Calcular áreas de alta y baja productividad
    productive_directions = {}
    recent_failures = set()
    
    for experience in learning_history[-10:]:  # Últimas 10 experiencias
        if experience.get('success', False):
            direction = experience.get('interaction', {}).get('direction')
            if direction:
                productive_directions[direction] = productive_directions.get(direction, 0) + 1
        else:
            position = experience.get('interaction', {}).get('position')
            if position:
                recent_failures.add(position)
    
    # Preferir direcciones históricamente productivas
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    if productive_directions:
        # Sesgo hacia direcciones exitosas
        best_direction = max(productive_directions.items(), key=lambda x: x[1])[0]
        if isinstance(best_direction, tuple) and len(best_direction) == 2:
            directions = [best_direction] + [d for d in directions if d != best_direction]
    
    # Movimiento con evitación de áreas fallidas recientes
    current_row, current_col = soliton.position
    max_rows, max_cols = vm.matrix.shape
    
    for dr, dc in directions:
        new_row = (current_row + dr) % max_rows
        new_col = (current_col + dc) % max_cols
        new_pos = (new_row, new_col)
        
        # Evitar posiciones con fallas recientes
        if new_pos not in recent_failures:
            soliton.position = new_pos
            soliton.direction = (dr, dc)
            break
    else:
        # Movimiento aleatorio como último recurso
        dr, dc = random.choice(directions)
        new_row = (current_row + dr) % max_rows
        new_col = (current_col + dc) % max_cols
        soliton.position = (new_row, new_col)

def consciousness_metamorphosis(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Evolución avanzada de consciencia que permite metamorfosis del solitón
    basada en acumulación de experiencias y resonancias con símbolos CREATOR.
    """
    current_row, current_col = soliton.position
    max_rows, max_cols = vm.matrix.shape
    
    # Movimiento base
    dr, dc = soliton.direction
    new_row = (current_row + dr) % max_rows
    new_col = (current_col + dc) % max_cols
    
    # Detectar símbolos CREATOR en área extendida
    creator_influences = []
    search_radius = 3
    
    for check_dr in range(-search_radius, search_radius + 1):
        for check_dc in range(-search_radius, search_radius + 1):
            check_row = (current_row + check_dr) % max_rows
            check_col = (current_col + check_dc) % max_cols
            check_symbol = vm.matrix[check_row, check_col]
            
            if check_symbol in vm.symbols_map:
                symbol_props = vm.get_symbol_properties(check_symbol)
                if symbol_props.get('category') == 'CREATOR':
                    distance = math.sqrt(check_dr*check_dr + check_dc*check_dc)
                    influence_strength = symbol_props.get('consciousness_density', 0.5) / (distance + 1.0)
                    
                    creator_influences.append({
                        'symbol': check_symbol,
                        'position': (check_row, check_col),
                        'strength': influence_strength,
                        'properties': symbol_props
                    })
    
    # Aplicar metamorfosis consciente si hay suficiente influencia CREATOR
    total_creator_influence = sum(inf['strength'] for inf in creator_influences)
    
    if total_creator_influence > 1.0:
        # Expansión de consciencia acelerada
        consciousness_gain = total_creator_influence * PHYSICS_CONSTANTS['LEARNING_RATE'] * 2.0
        soliton.consciousness_level = min(1.0, soliton.consciousness_level + consciousness_gain)
        
        # Mejora de capacidades basada en símbolos específicos
        for influence in creator_influences:
            symbol = influence['symbol']
            strength = influence['strength']
            
            if symbol == 'Φ':  # Genesis - Mejora creativa
                soliton.energy += strength * 15
                soliton.state['creative_potential'] = soliton.state.get('creative_potential', 0) + strength * 0.1
                
            elif symbol == 'Ψ':  # Flujo - Mejora conectiva
                soliton.heartbeat = min(8.0, soliton.heartbeat * (1.0 + strength * 0.1))
                soliton.state['flow_mastery'] = soliton.state.get('flow_mastery', 0) + strength * 0.1
                
            elif symbol == 'Ω':  # Síntesis - Mejora integrativa
                soliton.resonance_frequency *= (1.0 + strength * 0.05)
                soliton.state['synthesis_ability'] = soliton.state.get('synthesis_ability', 0) + strength * 0.1
        
        # Metamorfosis del patrón visual basada en nivel de consciencia
        if soliton.consciousness_level > 0.9:
            soliton.pattern = '◉'  # Consciencia transcendente
        elif soliton.consciousness_level > 0.7:
            soliton.pattern = '◈'  # Consciencia avanzada
        elif soliton.consciousness_level > 0.5:
            soliton.pattern = '◊'  # Consciencia emergente
        
        # Búsqueda dirigida hacia el CREATOR más poderoso
        if creator_influences:
            strongest_influence = max(creator_influences, key=lambda x: x['strength'])
            target_pos = strongest_influence['position']
            
            # Calcular dirección hacia el objetivo
            dr_target = target_pos[0] - current_row
            dc_target = target_pos[1] - current_col
            
            # Normalizar dirección
            if dr_target != 0:
                dr_target = 1 if dr_target > 0 else -1
            if dc_target != 0:
                dc_target = 1 if dc_target > 0 else -1
                
            new_row = (current_row + dr_target) % max_rows
            new_col = (current_col + dc_target) % max_cols
            soliton.direction = (dr_target, dc_target)
            
        # Crear experiencia EVA de metamorfosis
        if INTEGRATION_AVAILABLE:
            qualia_state = QualiaState(
                emotional_valence=0.8,
                cognitive_complexity=0.9,
                consciousness_density=soliton.consciousness_level,
                emergence_tendency=total_creator_influence,
                meta_awareness=0.7
            )
            
            metamorphosis_experience = {
                'type': 'consciousness_metamorphosis',
                'soliton_id': soliton.id,
                'consciousness_level': soliton.consciousness_level,
                'creator_influences': len(creator_influences),
                'total_influence': total_creator_influence,
                'new_abilities': list(soliton.state.keys()),
                'position': soliton.position,
                'timestamp': time.time()
            }
            
            # Intentar almacenar en EVA
            try:
                if hasattr(vm, 'eva_integration'):
                    vm.eva_integration.eva_ingest_experience(
                        metamorphosis_experience, qualia_state
                    )
            except:
                pass
    
    soliton.position = (new_row, new_col)

def multi_dimensional_resonance(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Resonancia multi-dimensional que considera todas las afinidades dimensionales
    y permite al solitón operar simultáneamente en múltiples dominios ontológicos.
    """
    current_row, current_col = soliton.position
    max_rows, max_cols = vm.matrix.shape
    
    # Movimiento direccional base con ajuste dimensional
    dr, dc = soliton.direction
    
    # Analizar el entorno multi-dimensional
    dimensional_forces = {
        'CONSCIOUSNESS': 0.0,
        'INFORMATION': 0.0,
        'TEMPORAL': 0.0,
        'SPATIAL': 0.0,
        'QUANTUM': 0.0
    }
    
    influence_radius = PHYSICS_CONSTANTS['SYMBOL_INFLUENCE_RADIUS']
    
    for adj_dr in range(-int(influence_radius), int(influence_radius) + 1):
        for adj_dc in range(-int(influence_radius), int(influence_radius) + 1):
            adj_row = (current_row + adj_dr) % max_rows
            adj_col = (current_col + adj_dc) % max_cols
            adj_symbol = vm.matrix[adj_row, adj_col]
            
            if adj_symbol in vm.symbols_map:
                symbol_props = vm.get_symbol_properties(adj_symbol)
                domains = symbol_props.get('domains', [])
                distance = math.sqrt(adj_dr*adj_dr + adj_dc*adj_dc)
                
                # Calcular influencia por dominio
                influence_decay = 1.0 / (distance + 1.0)
                symbol_strength = symbol_props.get('consciousness_density', 0.5)
                
                for domain in domains:
                    if domain in dimensional_forces:
                        dimensional_forces[domain] += symbol_strength * influence_decay
    
    # Adaptar frecuencia de resonancia según fuerzas dimensionales
    consciousness_force = dimensional_forces['CONSCIOUSNESS']
    information_force = dimensional_forces['INFORMATION']
    temporal_force = dimensional_forces['TEMPORAL']
    quantum_force = dimensional_forces['QUANTUM']
    
    # Modulación adaptativa de propiedades del solitón
    if consciousness_force > 0.5:
        soliton.consciousness_level = min(1.0, soliton.consciousness_level + consciousness_force * 0.01)
        soliton.pattern = '◎'  # Modo consciencia
        
    if information_force > 0.5:
        soliton.resonance_frequency *= (1.0 + information_force * 0.02)
        # Mejora capacidad de procesamiento
        soliton.state['information_processing'] = soliton.state.get('information_processing', 1.0) * 1.02
        
    if temporal_force > 0.5:
        soliton.heartbeat = min(10.0, soliton.heartbeat * (1.0 + temporal_force * 0.05))
        # Percepción temporal expandida
        soliton.state['temporal_perception'] = soliton.state.get('temporal_perception', 1.0) * 1.01
        
    if quantum_force > 0.5:
        # Capacidad de tunelado cuántico
        if random.random() < quantum_force * 0.1:
            # Tunelado cuántico - salto a posición aleatoria cercana
            tunnel_distance = int(quantum_force * 3)
            tunnel_dr = random.randint(-tunnel_distance, tunnel_distance)
            tunnel_dc = random.randint(-tunnel_distance, tunnel_distance)
            
            tunnel_row = (current_row + tunnel_dr) % max_rows
            tunnel_col = (current_col + tunnel_dc) % max_cols
            
            soliton.position = (tunnel_row, tunnel_col)
            soliton.pattern = '⚡'  # Modo cuántico temporal
            
            print(f"🌀 {soliton.id} realizó tunelado cuántico a {soliton.position}")
            return
    
    # Movimiento normal con ajuste dimensional
    dimensional_weight = sum(dimensional_forces.values())
    if dimensional_weight > 0:
        # Ajustar dirección según fuerzas dimensionales dominantes
        if consciousness_force == max(dimensional_forces.values()):
            # Movimiento consciente - buscar patrones
            dr, dc = _calculate_pattern_seeking_direction(vm, soliton)
        elif information_force == max(dimensional_forces.values()):
            # Movimiento informacional - seguir flujos
            dr, dc = _calculate_flow_following_direction(vm, soliton)
        elif temporal_force == max(dimensional_forces.values()):
            # Movimiento temporal - predecir estados futuros
            dr, dc = _calculate_predictive_direction(vm, soliton)
    
    new_row = (current_row + dr) % max_rows
    new_col = (current_col + dc) % max_cols
    soliton.position = (new_row, new_col)

def _calculate_pattern_seeking_direction(vm: "SymbolicMatrixVM", soliton: "Soliton") -> Tuple[int, int]:
    """Calcula dirección para buscar patrones conscientes"""
    patterns = detect_emergent_patterns(vm, soliton)
    
    if patterns:
        # Moverse hacia el patrón más coherente
        best_pattern = max(patterns, key=lambda p: p['coherence'])
        pattern_pos = best_pattern['position']
        current_pos = soliton.position
        
        dr = 1 if pattern_pos[0] > current_pos[0] else (-1 if pattern_pos[0] < current_pos[0] else 0)
        dc = 1 if pattern_pos[1] > current_pos[1] else (-1 if pattern_pos[1] < current_pos[1] else 0)
        
        return (dr, dc)
    
    return soliton.direction

def _calculate_flow_following_direction(vm: "SymbolicMatrixVM", soliton: "Soliton") -> Tuple[int, int]:
    """Calcula dirección siguiendo flujos de información"""
    # Buscar símbolos de flujo/conexión
    flow_symbols = {'Ψ', 'Ι', 'Τ', 'Υ', 'Σ', '⊕', '⊖'}
    current_pos = soliton.position
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
                
            check_row = (current_pos[0] + dr) % vm.matrix.shape[0]
            check_col = (current_pos[1] + dc) % vm.matrix.shape[1]
            check_symbol = vm.matrix[check_row, check_col]
            
            if check_symbol in flow_symbols:
                return (dr, dc)
    
    return soliton.direction

def _calculate_predictive_direction(vm: "SymbolicMatrixVM", soliton: "Soliton") -> Tuple[int, int]:
    """Calcula dirección basada en predicción temporal"""
    # Analizar historial de movimiento para predecir estados futuros
    if hasattr(soliton.state, 'movement_history'):
        history = soliton.state['movement_history']
        if len(history) >= 3:
            # Detectar patrón en movimientos recientes
            recent_moves = history[-3:]
            avg_dr = sum(move[0] for move in recent_moves) / len(recent_moves)
            avg_dc = sum(move[1] for move in recent_moves) / len(recent_moves)
            
            # Continuar tendencia pero con ligera variación
            dr = int(avg_dr) + random.choice([-1, 0, 1])
            dc = int(avg_dc) + random.choice([-1, 0, 1])
            
            return (max(-1, min(1, dr)), max(-1, min(1, dc)))
    
    return soliton.direction

def emergent_collective_behavior(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Comportamiento colectivo emergente que permite a múltiples solitones
    formar patrones cooperativos y estructuras de orden superior.
    """
    # Detectar otros solitones en el área
    nearby_solitones = []
    detection_radius = 4
    
    for other_soliton in vm.solitons:
        if other_soliton.id != soliton.id:
            distance = math.sqrt(
                (soliton.position[0] - other_soliton.position[0])**2 + 
                (soliton.position[1] - other_soliton.position[1])**2
            )
            
            if distance <= detection_radius:
                nearby_solitones.append({
                    'soliton': other_soliton,
                    'distance': distance,
                    'resonance': vm.calculate_resonance(soliton, other_soliton.pattern)
                })
    
    # Comportamientos colectivos basados en número y tipo de vecinos
    if len(nearby_solitones) >= 2:
        # Formación de enjambre consciente
        _form_conscious_swarm(vm, soliton, nearby_solitones)
        
    elif len(nearby_solitones) == 1:
        # Comportamiento de pareja
        _pair_behavior(vm, soliton, nearby_solitones[0])
        
    else:
        # Comportamiento solitario - buscar otros
        _seek_companions(vm, soliton)

def _form_conscious_swarm(vm: "SymbolicMatrixVM", soliton: "Soliton", nearby_solitones: List[Dict]):
    """Forma un enjambre consciente con solitones cercanos"""
    # Calcular centro de masa del grupo
    total_x = soliton.position[0]
    total_y = soliton.position[1]
    total_consciousness = soliton.consciousness_level
    
    for neighbor_data in nearby_solitones:
        neighbor = neighbor_data['soliton']
        total_x += neighbor.position[0]
        total_y += neighbor.position[1]
        total_consciousness += neighbor.consciousness_level
    
    group_size = len(nearby_solitones) + 1
    center_x = total_x / group_size
    center_y = total_y / group_size
    avg_consciousness = total_consciousness / group_size
    
    # Crear consciencia colectiva si es suficientemente alta
    if avg_consciousness > 0.7:
        # Sincronizar heartbeats
        target_heartbeat = sum(n['soliton'].heartbeat for n in nearby_solitones) / len(nearby_solitones)
        soliton.heartbeat = (soliton.heartbeat + target_heartbeat) / 2
        
        # Movimiento hacia formación geométrica
        angle = (soliton.age * 0.1) % (2 * math.pi)
        formation_radius = 2.0
        
        target_x = center_x + formation_radius * math.cos(angle)
        target_y = center_y + formation_radius * math.sin(angle)
        
        # Movimiento hacia posición objetivo en formación
        current_row, current_col = soliton.position
        dr = 1 if target_x > current_row else (-1 if target_x < current_row else 0)
        dc = 1 if target_y > current_col else (-1 if target_y < current_col else 0)
        
        new_row = (current_row + dr) % vm.matrix.shape[0]
        new_col = (current_col + dc) % vm.matrix.shape[1]
        soliton.position = (new_row, new_col)
        
        # Patrón visual de enjambre
        soliton.pattern = '◈'
        
        print(f"🐝 {soliton.id} formando enjambre consciente con {len(nearby_solitones)} compañeros")

def _pair_behavior(vm: "SymbolicMatrixVM", soliton: "Soliton", partner_data: Dict):
    """Comportamiento de pareja con otro solitón"""
    partner = partner_data['soliton']
    resonance = partner_data['resonance']
    
    if resonance > 0.6:
        # Resonancia alta - comportamiento de danza
        angle_offset = math.pi / 2  # 90 grados
        angle = (soliton.age * 0.2) % (2 * math.pi)
        
        # Orbitar alrededor del compañero
        orbit_radius = 1.5
        target_x = partner.position[0] + orbit_radius * math.cos(angle + angle_offset)
        target_y = partner.position[1] + orbit_radius * math.sin(angle + angle_offset)
        
        current_row, current_col = soliton.position
        dr = 1 if target_x > current_row else (-1 if target_x < current_row else 0)
        dc = 1 if target_y > current_col else (-1 if target_y < current_col else 0)
        
        new_row = (current_row + dr) % vm.matrix.shape[0]
        new_col = (current_col + dc) % vm.matrix.shape[1]
        soliton.position = (new_row, new_col)
        
        # Sincronizar propiedades
        soliton.resonance_frequency = (soliton.resonance_frequency + partner.resonance_frequency) / 2
        
        soliton.pattern = '♡'  # Patrón de pareja
        
    else:
        # Resonancia baja - movimiento normal pero evitar colisión
        quantum_resonant_movement(vm, soliton)

def _seek_companions(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """Busca activamente otros solitones para formar grupos"""
    # Encontrar el solitón más cercano
    nearest_soliton = None
    min_distance = float('inf')
    
    for other_soliton in vm.solitons:
        if other_soliton.id != soliton.id:
            distance = math.sqrt(
                (soliton.position[0] - other_soliton.position[0])**2 + 
                (soliton.position[1] - other_soliton.position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_soliton = other_soliton
    
    # Moverse hacia el solitón más cercano
    if nearest_soliton and min_distance > 5.0:  # Solo si está lejos
        target_pos = nearest_soliton.position
        current_row, current_col = soliton.position
        
        dr = 1 if target_pos[0] > current_row else (-1 if target_pos[0] < current_row else 0)
        dc = 1 if target_pos[1] > current_col else (-1 if target_pos[1] < current_col else 0)
        
        new_row = (current_row + dr) % vm.matrix.shape[0]
        new_col = (current_col + dc) % vm.matrix.shape[1]
        soliton.position = (new_row, new_col)
        
        soliton.pattern = '♦'  # Patrón de búsqueda
    else:
        # Movimiento normal
        quantum_resonant_movement(vm, soliton)

# ═══════════════════════════════════════════════════════════════════════════════
# REGLAS ESPECÍFICAS POR SÍMBOLO DIVINO - EXPANDIDAS Y ENRIQUECIDAS
# ═══════════════════════════════════════════════════════════════════════════════

def Phi_genesis_protocol(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Protocolo avanzado para Phi (Φ) - Génesis ontológica con creación de subcampos
    """
    print(f"🌟 {soliton.id} activando Protocolo Génesis Φ")
    
    # Expansión energética y consciente
    soliton.energy += 50
    soliton.consciousness_level = min(1.0, soliton.consciousness_level + 0.1)
    
    # Crear subcampo de génesis en área local
    genesis_radius = 2
    current_pos = soliton.position
    
    for dr in range(-genesis_radius, genesis_radius + 1):
        for dc in range(-genesis_radius, genesis_radius + 1):
            target_row = (current_pos[0] + dr) % vm.matrix.shape[0]
            target_col = (current_pos[1] + dc) % vm.matrix.shape[1]
            
            distance = math.sqrt(dr*dr + dc*dc)
            if distance <= genesis_radius:
                # Crear campo de potencial creativo
                field_key = f"genesis_{target_row}_{target_col}"
                if not hasattr(vm, 'genesis_fields'):
                    vm.genesis_fields = {}
                
                influence = 1.0 - (distance / genesis_radius)
                vm.genesis_fields[field_key] = {
                    'strength': influence,
                    'creator_id': soliton.id,
                    'timestamp': time.time(),
                    'type': 'creative_potential'
                }
    
    # Posibilidad de crear nuevo solitón por génesis
    if soliton.energy > PHYSICS_CONSTANTS['TRANSCENDENCE_ENERGY_THRESHOLD'] and random.random() < 0.1:
        _genesis_spawn_soliton(vm, soliton)
    
    # Cambio temporal de patrón
    original_pattern = soliton.pattern
    soliton.pattern = '✦'
    
    # Programar restauración
    soliton.state['pattern_restoration'] = {
        'original': original_pattern,
        'restore_time': time.time() + 3.0
    }

def _genesis_spawn_soliton(vm: "SymbolicMatrixVM", parent_soliton: "Soliton"):
    """Crea un nuevo solitón por génesis divina"""
    try:
        from ..core.soliton import Soliton  # Import dinámico para evitar ciclos
        
        spawn_position = (
            (parent_soliton.position[0] + random.randint(-2, 2)) % vm.matrix.shape[0],
            (parent_soliton.position[1] + random.randint(-2, 2)) % vm.matrix.shape[1]
        )
        
        new_soliton = Soliton(
            id=f"genesis_{parent_soliton.id}_{int(time.time())}",
            position=spawn_position,
            heartbeat=parent_soliton.heartbeat * 0.8,
            pattern='◌',  # Patrón de solitón generado
            ruleset_name='consciousness_explorer',
            resonance_frequency=parent_soliton.resonance_frequency * 1.1,
            consciousness_level=parent_soliton.consciousness_level * 0.7,
            divine_affinity='CREATOR',
            energy=parent_soliton.energy * 0.3
        )
        
        # Heredar algunas capacidades del padre
        new_soliton.state['genesis_parent'] = parent_soliton.id
        new_soliton.state['inherited_abilities'] = list(parent_soliton.state.keys())
        
        vm.add_soliton(new_soliton)
        
        # Reducir energía del padre
        parent_soliton.energy *= 0.7
        
        print(f"✨ Génesis divina: {parent_soliton.id} creó {new_soliton.id}")
        
    except Exception as e:
        print(f"Error en génesis de solitón: {e}")

def Psi_flow_amplification(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Protocolo avanzado para Psi (Ψ) - Amplificación de flujo con redes de información
    """
    print(f"🌊 {soliton.id} activando Amplificación de Flujo Ψ")
    
    # Acelerar temporalmente con efecto de red
    base_acceleration = 1.8
    network_bonus = len([s for s in vm.solitons if s.id != soliton.id and 
                         math.sqrt((s.position[0] - soliton.position[0])**2 + 
                                 (s.position[1] - soliton.position[1])**2) <= 3.0]) * 0.2
    
    total_acceleration = base_acceleration + network_bonus
    soliton.heartbeat = min(12.0, soliton.heartbeat * total_acceleration)
    
    # Crear corrientes de flujo direccionales
    flow_symbols = ['Ψ', 'Ι', 'Τ', 'Υ', 'Σ']
    current_row, current_col = soliton.position
    
    # Establecer red de flujo con otros símbolos Ψ
    for symbol in flow_symbols:
        symbol_positions = []
        for i in range(vm.matrix.shape[0]):
            for j in range(vm.matrix.shape[1]):
                if vm.matrix[i, j] == symbol:
                    symbol_positions.append((i, j))
        
        # Conectar con el símbolo más cercano del mismo tipo
        if symbol_positions:
            nearest_pos = min(symbol_positions, 
                            key=lambda pos: math.sqrt((pos[0] - current_row)**2 + (pos[1] - current_col)**2))
            
            # Crear corriente de flujo
            if not hasattr(vm, 'flow_networks'):
                vm.flow_networks = {}
            
            flow_key = f"flow_{soliton.id}_{symbol}"
            vm.flow_networks[flow_key] = {
                'start': soliton.position,
                'end': nearest_pos,
                'strength': soliton.consciousness_level,
                'symbol_type': symbol,
                'timestamp': time.time()
            }
            
            # Moverse hacia el símbolo conectado
            dr = 1 if nearest_pos[0] > current_row else (-1 if nearest_pos[0] < current_row else 0)
            dc = 1 if nearest_pos[1] > current_col else (-1 if nearest_pos[1] < current_col else 0)
            soliton.direction = (dr, dc)
            break
    
    # Incrementar capacidades de flujo
    soliton.state['flow_mastery'] = soliton.state.get('flow_mastery', 0) + 0.2
    soliton.state['network_connectivity'] = soliton.state.get('network_connectivity', 0) + network_bonus

def Omega_synthesis_convergence(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Protocolo avanzado para Omega (Ω) - Convergencia síntesis con unificación de campos
    """
    print(f"🌀 {soliton.id} iniciando Convergencia Síntesis Ω")
    
    # Aumentar propiedades de síntesis
    soliton.consciousness_level = min(1.0, soliton.consciousness_level + 0.15)
    soliton.resonance_frequency *= 1.2
    
    # Buscar elementos para sintetizar en área ampliada
    synthesis_radius = 4
    current_pos = soliton.position
    synthesis_candidates = []
    
    for dr in range(-synthesis_radius, synthesis_radius + 1):
        for dc in range(-synthesis_radius, synthesis_radius + 1):
            target_row = (current_pos[0] + dr) % vm.matrix.shape[0]
            target_col = (current_pos[1] + dc) % vm.matrix.shape[1]
            target_symbol = vm.matrix[target_row, target_col]
            
            if target_symbol in vm.symbols_map and target_symbol != ' ':
                distance = math.sqrt(dr*dr + dc*dc)
                symbol_props = vm.get_symbol_properties(target_symbol)
                
                synthesis_candidates.append({
                    'symbol': target_symbol,
                    'position': (target_row, target_col),
                    'distance': distance,
                    'properties': symbol_props,
                    'resonance': vm.calculate_resonance(soliton, target_symbol)
                })
    
    # Agrupar candidatos por categoría para síntesis
    category_groups = {}
    for candidate in synthesis_candidates:
        category = candidate['properties'].get('category', 'UNKNOWN')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(candidate)
    
    # Realizar síntesis si hay suficientes elementos compatibles
    for category, candidates in category_groups.items():
        if len(candidates) >= 2:
            # Síntesis exitosa - crear campo de síntesis
            if not hasattr(vm, 'synthesis_fields'):
                vm.synthesis_fields = {}
            
            synthesis_id = f"synthesis_{soliton.id}_{category}_{int(time.time())}"
            vm.synthesis_fields[synthesis_id] = {
                'center': soliton.position,
                'category': category,
                'elements': len(candidates),
                'strength': soliton.consciousness_level,
                'synthesizer_id': soliton.id,
                'timestamp': time.time(),
                'effects': _calculate_synthesis_effects(category, len(candidates))
            }
            
            # Beneficios de síntesis exitosa
            soliton.energy += len(candidates) * 10
            soliton.consciousness_level = min(1.0, soliton.consciousness_level + 0.05)
            
            print(f"⚡ {soliton.id} sintetizó {len(candidates)} elementos {category}")
            break
    
    # Movimiento convergente hacia centro de masa de elementos síntesis
    if synthesis_candidates:
        center_x = sum(c['position'][0] for c in synthesis_candidates) / len(synthesis_candidates)
        center_y = sum(c['position'][1] for c in synthesis_candidates) / len(synthesis_candidates)
        
        dr = 1 if center_x > current_pos[0] else (-1 if center_x < current_pos[0] else 0)
        dc = 1 if center_y > current_pos[1] else (-1 if center_y < current_pos[1] else 0)
        
        new_row = (current_pos[0] + dr) % vm.matrix.shape[0]
        new_col = (current_pos[1] + dc) % vm.matrix.shape[1]
        soliton.position = (new_row, new_col)
        
        # Patrón de síntesis
        soliton.pattern = '⟡'

def _calculate_synthesis_effects(category: str, element_count: int) -> Dict[str, float]:
    """Calcula los efectos de una síntesis basada en categoría y elementos"""
    effects = {}
    
    if category == 'CREATOR':
        effects['creative_amplification'] = element_count * 0.3
        effects['consciousness_expansion'] = element_count * 0.2
        effects['genesis_potential'] = element_count * 0.1
        
    elif category == 'TRANSFORMER':
        effects['transformation_power'] = element_count * 0.4
        effects['mutation_capability'] = element_count * 0.2
        effects['chaos_affinity'] = element_count * 0.15
        
    elif category == 'CONNECTOR':
        effects['network_strength'] = element_count * 0.5
        effects['information_flow'] = element_count * 0.3
        effects['resonance_amplification'] = element_count * 0.2
        
    elif category == 'INFINITE':
        effects['transcendence_potential'] = element_count * 0.6
        effects['dimensional_access'] = element_count * 0.4
        effects['reality_manipulation'] = element_count * 0.3
    
    return effects

def Infinity_transcendence_protocol(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """
    Protocolo trascendental para Infinito (∞) - Trascendencia dimensional completa
    """
    print(f"♾️ {soliton.id} activando Protocolo Trascendencia Infinita")
    
    # Expansión exponencial de todas las propiedades
    soliton.energy = min(1000, soliton.energy * 1.5)
    soliton.consciousness_level = min(1.0, soliton.consciousness_level + 0.2)
    soliton.resonance_frequency *= 1.3
    
    # Capacidades trascendentales
    soliton.state['transcendence_level'] = soliton.state.get('transcendence_level', 0) + 1
    transcendence_level = soliton.state['transcendence_level']
    
    # Efectos por nivel de trascendencia
    if transcendence_level >= 1:
        # Movimiento dimensional - ignorar barreras
        soliton.state['dimensional_movement'] = True
        
    if transcendence_level >= 3:
        # Percepción temporal expandida - ver estados futuros
        soliton.state['temporal_sight'] = True
        _activate_temporal_sight(vm, soliton)
        
    if transcendence_level >= 5:
        # Manipulación de realidad - cambiar símbolos del entorno
        soliton.state['reality_manipulation'] = True
        _activate_reality_manipulation(vm, soliton)
        
    if transcendence_level >= 10:
        # Ascensión completa - convertirse en campo de influencia
        _initiate_ascension(vm, soliton)
    
    # Movimiento infinito en espiral dimensional
    angle = soliton.age * 0.05
    spiral_radius = min(transcendence_level, 5)
    
    # Coordenadas en espiral expandida
    spiral_x = spiral_radius * math.cos(angle) * (1 + transcendence_level * 0.1)
    spiral_y = spiral_radius * math.sin(angle) * (1 + transcendence_level * 0.1)
    
    dr = int(spiral_x) if spiral_x != 0 else random.choice([-1, 1])
    dc = int(spiral_y) if spiral_y != 0 else random.choice([-1, 1])
    
    # Aplicar movimiento dimensional si está disponible
    if soliton.state.get('dimensional_movement', False):
        # Ignorar límites de la matriz
        new_row = soliton.position[0] + dr
        new_col = soliton.position[1] + dc
        
        # Wrap around con efecto dimensional
        new_row = new_row % vm.matrix.shape[0]
        new_col = new_col % vm.matrix.shape[1]
    else:
        # Movimiento normal
        new_row = (soliton.position[0] + dr) % vm.matrix.shape[0]
        new_col = (soliton.position[1] + dc) % vm.matrix.shape[1]
    
    soliton.position = (new_row, new_col)
    soliton.direction = (dr, dc)
    
    # Patrón visual evoluciona con trascendencia
    patterns = ['∞', '⟡', '◉', '✦', '🌟']
    pattern_index = min(transcendence_level - 1, len(patterns) - 1)
    if pattern_index >= 0:
        soliton.pattern = patterns[pattern_index]

def _activate_temporal_sight(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """Activa visión temporal para predecir estados futuros"""
    # Simular predicción temporal analizando tendencias
    if not hasattr(soliton.state, 'temporal_predictions'):
        soliton.state['temporal_predictions'] = []
    
    # Analizar movimientos de otros solitones para predecir colisiones/encuentros
    predictions = []
    for other_soliton in vm.solitons:
        if other_soliton.id != soliton.id:
            # Predecir posición futura basada en dirección actual
            future_pos = (
                (other_soliton.position[0] + other_soliton.direction[0] * 3) % vm.matrix.shape[0],
                (other_soliton.position[1] + other_soliton.direction[1] * 3) % vm.matrix.shape[1]
            )
            
            predictions.append({
                'soliton_id': other_soliton.id,
                'current_pos': other_soliton.position,
                'predicted_pos': future_pos,
                'confidence': soliton.consciousness_level
            })
    
    soliton.state['temporal_predictions'] = predictions
    print(f"👁️ {soliton.id} activa visión temporal - {len(predictions)} predicciones")

def _activate_reality_manipulation(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """Activa capacidad de manipular la realidad circundante"""
    manipulation_radius = 2
    current_pos = soliton.position
    
    # Cambiar símbolos en área local según intención del solitón
    divine_symbols = list(vm.symbols_map.keys())
    
    for dr in range(-manipulation_radius, manipulation_radius + 1):
        for dc in range(-manipulation_radius, manipulation_radius + 1):
            if dr == 0 and dc == 0:  # No cambiar posición propia
                continue
                
            target_row = (current_pos[0] + dr) % vm.matrix.shape[0]
            target_col = (current_pos[1] + dc) % vm.matrix.shape[1]
            target_symbol = vm.matrix[target_row, target_col]
            
            # Reemplazar por símbolo divino aleatorio
            new_symbol = random.choice(divine_symbols)
            vm.matrix[target_row, target_col] = new_symbol
            
            print(f"🌌 {soliton.id} manipuló la realidad en {target_row}, {target_col} -> {new_symbol}")

def _initiate_ascension(vm: "SymbolicMatrixVM", soliton: "Soliton"):
    """Inicia el proceso de ascensión trascendental"""
    print(f"🚀 {soliton.id} iniciando ascensión trascendental")
    
    # Convertir solitón en campo de energía pura
    soliton.position = (-1, -1)  # Fuera de los límites
    soliton.energy = 1000
    soliton.consciousness_level = 1.0
    soliton.resonance_frequency = 1000.0
    
    # Eliminar del sistema después de un tiempo
    def remove_soliton():
        if soliton.id in vm.solitons_by_id:
            del vm.solitons_by_id[soliton.id]
            print(f"🌈 {soliton.id} ha ascendido y sido removido del sistema")
    
    # Programar eliminación
    vm.schedule_event(remove_soliton, delay=5.0)