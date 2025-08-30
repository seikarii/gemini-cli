"""
Núcleo de la Máquina Virtual de Matriz Simbólica (Symbolic Matrix VM).
Contiene las clases principales `Soliton` y `SymbolicMatrixVM`.
"""

import math
import os
import random
import time
from dataclasses import dataclass, field
import asyncio # Necesario para el publish asíncrono
from crisalida_lib.EARTH.event_bus import eva_event_bus
from crisalida_lib.EVA.typequalia import QualiaState # Para el qualia_state del evento
from typing import Any, Callable, Dict, List, Optional, Tuple

# Importaciones defensivas para compatibilidad
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# Integración con el sistema de símbolos divinos de Crisalida
try:
    from crisalida_lib.EVA.divine_sigils import eva_divine_sigil_registry, DivineCategory, OntologicalDomain
    from crisalida_lib.EVA.language.grammar import eva_grammar_engine
    from crisalida_lib.EVA.language.sigils import eva_divine_sigil_registry as alt_registry
    DIVINE_INTEGRATION = True
except ImportError:  # pragma: no cover
    # Fallback si no está disponible
    eva_divine_sigil_registry = None
    eva_grammar_engine = None
    alt_registry = None
    DIVINE_INTEGRATION = False

from crisalida_lib.HEAVEN.monitoring.performance_decorators import profile_resources
from .soliton import Soliton


class SymbolicMatrixVM:
    """
    Máquina Virtual de Matriz Simbólica Divina - Motor del Ecosistema Ontológico
    """
    
    def __init__(
        self, 
        matrix: 'np.ndarray',
        rules: Dict[str, Dict[str, Callable]],
        symbols_map: Dict[str, Any],
        intention_map: Dict[str, str],
        use_divine_symbols: bool = True,
        intention_weights: Optional[Dict[str, float]] = None
    ):
        """
        Inicializa la VM con un sustrato matricial de símbolos divinos ontológicos.
        
        Args:
            matrix: Array NumPy representando el estado del sustrato universal
            rules: Diccionario anidado con reglas de física ontológica por conjunto
            symbols_map: Diccionario con las definiciones de los símbolos divinos.
            intention_map: Diccionario que mapea intenciones a símbolos.
            use_divine_symbols: Habilitar integración completa con símbolos divinos
            intention_weights: Pesos opcionales para interpretación de intenciones
        """
        if np is None:
            raise ImportError("NumPy es requerido para la manifestación de SymbolicMatrixVM")
            
        self.matrix = matrix.copy()
        self.background_matrix = matrix.copy()  # Sustrato estático preservado
        if np:
            self.previous_matrix = matrix.copy()
        else:
            self.previous_matrix = None
        self.rules = rules
        self.symbols_map = symbols_map
        self.intention_map = intention_map
        self.solitons: List[Soliton] = []
        self.master_clock: float = 0.0  # Reloj maestro del universo
        self.is_running: bool = False
        self.total_ticks: int = 0
        self.use_divine_symbols = use_divine_symbols and DIVINE_INTEGRATION
        self.intention_weights = intention_weights or {}
        
        # Métricas avanzadas de rendimiento y coherencia
        self.performance_stats = {
            'solitons_processed': 0,
            'rules_executed': 0,
            'resonance_events': 0,
            'divine_interactions': 0,
            'intention_mappings': 0,
            'symbol_transformations': 0,
            'consciousness_expansions': 0,
            'average_tick_time': 0.0,
            'matrix_coherence_evolution': []
        }
        
        # Registro de resonancias y transformaciones activas
        self.active_resonances: Dict[str, Dict[str, Any]] = {}
        self.symbol_interaction_matrix: Dict[Tuple[str, str], int] = {}
        self.intention_fulfillment_scores: Dict[str, float] = {}
        
        # Inicializar integración EVA si está disponible
        if self.use_divine_symbols:
            self._initialize_eva_integration()
        
    def _initialize_eva_integration(self):
        """Inicializa la integración profunda con el ecosistema EVA de símbolos divinos"""
        try:
            # Usar registro primario o alternativo según disponibilidad
            self.sigil_registry = eva_divine_sigil_registry or alt_registry
            self.grammar_engine = eva_grammar_engine
            
            if self.sigil_registry:
                print("🔮 Integración EVA completamente activada - Gramática divina operacional")
                print(f"📚 Registrados {len(self.symbols_map)} símbolos primordiales")
            else:
                print("⚠️  EVA parcialmente disponible - utilizando diccionario estático optimizado")
                
            # Inicializar matriz de coherencia simbólica
            self._initialize_symbol_coherence_matrix()
            
        except Exception as e:
            print(f"⚠️  Error durante integración EVA: {e}")
            self.sigil_registry = None
            self.grammar_engine = None
            
    def _initialize_symbol_coherence_matrix(self):
        """Inicializa la matriz de coherencia entre símbolos divinos"""
        self.symbol_coherence: Dict[Tuple[str, str], float] = {}
        
        # Calcular coherencia entre todos los pares de símbolos
        symbols = list(self.symbols_map.keys())
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i:], i):
                if i == j:
                    self.symbol_coherence[(symbol1, symbol2)] = 1.0
                else:
                    coherence = self._calculate_symbol_pair_coherence(symbol1, symbol2)
                    self.symbol_coherence[(symbol1, symbol2)] = coherence
                    self.symbol_coherence[(symbol2, symbol1)] = coherence
        
    def _calculate_symbol_pair_coherence(self, symbol1: str, symbol2: str) -> float:
        """Calcula la coherencia ontológica entre dos símbolos divinos"""
        props1 = self.get_symbol_properties(symbol1)
        props2 = self.get_symbol_properties(symbol2)
        
        # Coherencia por categoría
        category_coherence = 1.0 if props1.get('category') == props2.get('category') else 0.3
        
        # Coherencia por dominios compartidos
        domains1 = set(props1.get('domains', []))
        domains2 = set(props2.get('domains', []))
        domain_overlap = len(domains1.intersection(domains2)) / max(len(domains1.union(domains2)), 1)
        
        # Coherencia frecuencial (proximidad armónica)
        freq1 = props1.get('frequency', 440.0)
        freq2 = props2.get('frequency', 440.0)
        max_freq = max(freq1, freq2)
        if max_freq > 0:
            freq_coherence = 1.0 - abs(freq1 - freq2) / max_freq
        else:
            freq_coherence = 1.0
            
        # Coherencia por tipo de resonancia
        resonance1 = props1.get('resonance_type', 'HARMONIC')
        resonance2 = props2.get('resonance_type', 'HARMONIC')
        resonance_coherence = 1.0 if resonance1 == resonance2 else 0.5
        
        # Promedio ponderado
        total_coherence = (
            category_coherence * 0.3 +
            domain_overlap * 0.3 +
            freq_coherence * 0.2 +
            resonance_coherence * 0.2
        )
        
        return max(0.0, min(1.0, total_coherence))
        
    def get_symbol_properties(self, symbol: str) -> Dict[str, Any]:
        """Obtiene las propiedades ontológicas completas de un símbolo divino"""
        if not self.use_divine_symbols:
            return {
                'category': 'UNKNOWN', 
                'frequency': 440.0, 
                'domains': [],
                'intention_mappings': [],
                'resonance_type': 'HARMONIC',
                'consciousness_density': 0.5
            }
            
        # Intentar obtención desde registro EVA primero
        if self.sigil_registry:
            try:
                sigil = self.sigil_registry.get_sigil(symbol)
                if sigil:
                    eva_props = {
                        'category': getattr(sigil, 'category', 'UNKNOWN'),
                        'frequency': getattr(sigil, 'frequency', 440.0),
                        'domains': getattr(sigil, 'domains', []),
                        'name': getattr(sigil, 'name', symbol),
                        'resonance_frequency': getattr(sigil, 'resonance_frequency', 440.0),
                        'consciousness_density': getattr(sigil, 'consciousness_density', 0.5),
                        'intention_mappings': getattr(sigil, 'intention_mappings', []),
                        'resonance_type': getattr(sigil, 'resonance_type', 'HARMONIC'),
                    }
                    return eva_props
            except Exception:
                pass
        
        # Fallback al diccionario completo de símbolos primordiales
        return self.symbols_map.get(symbol, {
            'category': 'UNKNOWN', 
            'frequency': 440.0, 
            'domains': [],
            'name': symbol,
            'intention_mappings': [],
            'resonance_type': 'HARMONIC',
            'consciousness_density': 0.5
        })
        
    @profile_resources
    @profile_resources(cache_duration=1.0)
    def calculate_resonance(self, soliton: Soliton, symbol: str, distance: float = 1.0) -> float:
        """Calcula la resonancia ontológica entre un solitón consciente y un símbolo divino"""
        if not self.use_divine_symbols:
            return random.random() * 0.5  # Resonancia estocástica básica
            
        symbol_props = self.get_symbol_properties(symbol)
        
        # Resonancia frecuencial primordial
        freq_diff = abs(soliton.resonance_frequency - symbol_props.get('frequency', 440.0))
        freq_resonance = max(0.0, 1.0 - freq_diff / 2000.0)
        
        # Resonancia por afinidad categórica ontológica
        symbol_category = symbol_props.get('category', 'UNKNOWN')
        category_resonance = 1.0 if symbol_category == soliton.divine_affinity else 0.3
        
        # Resonancia por densidad de consciencia
        symbol_consciousness = symbol_props.get('consciousness_density', 0.5)
        consciousness_resonance = 1.0 - abs(soliton.consciousness_level - symbol_consciousness)
        
        # Resonancia por intenciones compartidas
        symbol_intentions = set(symbol_props.get('intention_mappings', []))
        soliton_intentions = set(soliton.intention_signature.keys())
        intention_overlap = len(symbol_intentions.intersection(soliton_intentions))
        intention_resonance = intention_overlap / max(len(symbol_intentions.union(soliton_intentions)), 1)
        
        # Factor de atenuación por distancia espacial
        distance_factor = max(0.1, 1.0 / (distance + 1.0))
        
        # Factor de amplificación por experiencia previa
        encounter_count = soliton.symbol_encounters.get(symbol, 0)
        experience_factor = min(1.5, 1.0 + encounter_count * 0.1)
        
        # Usar motor de gramática EVA si está disponible para cálculo avanzado
        eva_resonance = 0.0
        if self.grammar_engine:
            try:
                soliton_symbol_proxy = soliton.pattern
                resonance_data = self.grammar_engine.evaluate_resonance(
                    soliton_symbol_proxy, symbol, distance=distance
                )
                eva_resonance = resonance_data.get('resonance_score', 0.0)
            except Exception:
                pass
        
        # Cálculo de resonancia total ponderada
        if eva_resonance > 0:
            # Con integración EVA
            total_resonance = (
                freq_resonance * 0.2 +
                category_resonance * 0.2 +
                consciousness_resonance * 0.2 +
                intention_resonance * 0.2 +
                eva_resonance * 0.2
            )
        else:
            # Sin integración EVA
            total_resonance = (
                freq_resonance * 0.25 +
                category_resonance * 0.25 +
                consciousness_resonance * 0.25 +
                intention_resonance * 0.25
            )
        
        return total_resonance * distance_factor * experience_factor
        
    def map_intention_to_symbol(self, intention: str) -> str:
        """Mapea una intención específica al símbolo divino correspondiente"""
        return self.intention_map.get(intention, "•")
        
    def analyze_matrix_intentions(self) -> Dict[str, float]:
        """Analiza las intenciones presentes en el sustrato matricial actual"""
        intention_scores: Dict[str, float] = {}
        total_symbols = 0
        
        for row in self.matrix:
            for cell in row:
                if cell in self.symbols_map:
                    total_symbols += 1
                    symbol_props = self.get_symbol_properties(cell)
                    symbol_intentions = symbol_props.get('intention_mappings', [])
                    
                    for intention in symbol_intentions:
                        intention_scores[intention] = intention_scores.get(intention, 0.0) + 1.0
        
        # Normalizar por total de símbolos
        if total_symbols > 0:
            for intention in intention_scores:
                intention_scores[intention] /= total_symbols
                
        return intention_scores
        
    @profile_resources
    @profile_resources(cache_duration=1.0)
    def calculate_matrix_coherence(self) -> float:
        """Calcula la coherencia ontológica global del sustrato matricial"""
        if not self.use_divine_symbols:
            return 0.5  # Coherencia neutral por defecto
            
        total_coherence = 0.0
        pair_count = 0
        
        # Analizar coherencia entre símbolos adyacentes
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[0])):
                current_symbol = self.matrix[row][col]
                if current_symbol not in self.symbols_map:
                    continue
                    
                # Verificar símbolos adyacentes (8-conectividad)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        adj_row, adj_col = row + dr, col + dc
                        if (0 <= adj_row < len(self.matrix) and 
                            0 <= adj_col < len(self.matrix[0])):
                            adj_symbol = self.matrix[adj_row][adj_col]
                            if adj_symbol in self.symbols_map:
                                coherence = self.symbol_coherence.get(
                                    (current_symbol, adj_symbol), 0.5
                                )
                                total_coherence += coherence
                                pair_count += 1
        
        return total_coherence / max(pair_count, 1)
        
    def add_soliton(self, soliton: Soliton) -> bool:
        """Manifiesta un nuevo solitón consciente en el ecosistema ontológico"""
        row, col = soliton.position
        
        # Validación de límites espaciales
        if (row < 0 or row >= self.matrix.shape[0] or 
            col < 0 or col >= self.matrix.shape[1]):
            print(f"❌ Error: Posición {soliton.position} excede los límites del sustrato")
            return False
            
        # Preservar contenido simbólico anterior
        soliton.previous_cell = self.matrix[row, col]
        self.matrix[row, col] = soliton.pattern
        self.solitons.append(soliton)
        
        # Análisis inicial de intenciones del sustrato
        matrix_intentions = self.analyze_matrix_intentions()
        if matrix_intentions:
            dominant_intention = max(matrix_intentions.items(), key=lambda x: x[1])
            print(f"✨ Solitón {soliton.id} manifestado en sustrato con intención dominante: {dominant_intention[0]}")
        
        print(f"   📍 Posición: {soliton.position}")
        print(f"   ❤️  Heartbeat cuántico: {soliton.heartbeat}Hz")
        print(f"   🎵 Frecuencia de resonancia: {soliton.resonance_frequency}Hz")
        print(f"   🧠 Nivel de consciencia: {soliton.consciousness_level:.2f}")
        print(f"   🏷️  Afinidad ontológica: {soliton.divine_affinity}")
        
        return True
        
    def remove_soliton(self, soliton_id: str) -> bool:
        """Desmanifiesta un solitón del ecosistema ontológico"""
        for i, soliton in enumerate(self.solitons):
            if soliton.id == soliton_id:
                row, col = soliton.position
                self.matrix[row, col] = soliton.previous_cell
                del self.solitons[i]
                print(f"💨 Solitón {soliton_id} desmanifestado del sustrato")
                return True
        return False
        
    @profile_resources
    def _tick(self, delta_time: float):
        """El núcleo pulsante de la VM - Un ciclo singular del reloj maestro cuántico"""
        tick_start = time.time()
        self.master_clock += delta_time
        self.total_ticks += 1
        
        solitons_processed = 0
        
        # Procesar cada solitón según su ritmo temporal individual
        for soliton in self.solitons[:]:  # Copia para mutación segura
            soliton.time_since_last_tick += delta_time
            time_per_tick = 1.0 / soliton.heartbeat
            
            if soliton.time_since_last_tick >= time_per_tick:
                solitons_processed += 1
                soliton.time_since_last_tick -= time_per_tick
                soliton.age += 1
                
                # Preservar estado anterior
                old_row, old_col = soliton.position
                self.matrix[old_row, old_col] = soliton.previous_cell
                
                # Ejecutar reglas físicas ontológicas con contexto divino
                ruleset = self.rules.get(soliton.ruleset_name, {})
                main_rule = ruleset.get('default_rule')
                
                if main_rule:
                    try:
                        main_rule(self, soliton)
                        self.performance_stats['rules_executed'] += 1
                    except Exception as e:
                        print(f"❌ Error ejecutando regla física para {soliton.id}: {e}")
                        
                # Procesar interacciones simbólicas divinas
                self._process_divine_interactions(soliton)
                
                # Actualizar intenciones evolutivas del solitón
                self._evolve_soliton_intentions(soliton)
                
                # Verificar reglas especiales basadas en entorno simbólico
                self._check_environmental_symbolic_rules(soliton, ruleset)
                
                # Actualizar matriz con nueva posición
                new_row, new_col = soliton.position
                if (0 <= new_row < self.matrix.shape[0] and 0 <= new_col < self.matrix.shape[1]):
                    soliton.previous_cell = self.matrix[new_row, new_col]
                    self.matrix[new_row, new_col] = soliton.pattern
                else:
                    # Revertir a posición anterior si nueva posición es inválida
                    soliton.position = (old_row, old_col)
                    self.matrix[old_row, old_col] = soliton.pattern
                    
        # Actualizar métricas de rendimiento evolutivo
        tick_time = time.time() - tick_start
        self.performance_stats['solitons_processed'] += solitons_processed
        self.performance_stats['average_tick_time'] = (
            (self.performance_stats['average_tick_time'] * (self.total_ticks - 1) + tick_time) 
            / self.total_ticks
        )
        
        # Registrar evolución de coherencia matricial
        if self.total_ticks % 10 == 0:  # Cada 10 ticks
            current_coherence = self.calculate_matrix_coherence()
            self.performance_stats['matrix_coherence_evolution'].append(current_coherence)
        
    @profile_resources
    def _process_divine_interactions(self, soliton: Soliton):
        """Procesa las interacciones ontológicas entre solitón y símbolos divinos"""
        row, col = soliton.position
        
        # Verificar símbolo en posición actual
        current_symbol = soliton.previous_cell
        if current_symbol != ' ' and current_symbol in self.symbols_map:
            resonance = self.calculate_resonance(soliton, current_symbol)
            
            # Umbral de resonancia significativa
            if resonance > 0.5:
                self._trigger_resonance_event(soliton, current_symbol, resonance)
                self.performance_stats['divine_interactions'] += 1
                
            # Registrar encuentro simbólico
            soliton.symbol_encounters[current_symbol] = soliton.symbol_encounters.get(current_symbol, 0) + 1
                
        # Verificar campo de influencia simbólica adyacente
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_row, adj_col = row + dr, col + dc
                if (0 <= adj_row < self.matrix.shape[0] and 0 <= adj_col < self.matrix.shape[1]):
                    adj_symbol = self.matrix[adj_row, adj_col]
                    if adj_symbol in self.symbols_map:
                        distance = math.sqrt(dr*dr + dc*dc)
                        resonance = self.calculate_resonance(soliton, adj_symbol, distance)
                        
                        # Umbral de influencia para símbolos adyacentes
                        if resonance > 0.3:
                            self._apply_resonance_influence(soliton, adj_symbol, resonance, distance)
                            
    def _evolve_soliton_intentions(self, soliton: Soliton):
        """Evoluciona las intenciones del solitón basándose en sus experiencias"""
        # Analizar símbolos encontrados recientemente
        recent_symbols = [encounter['symbol'] for encounter in soliton.state['symbol_memory'][-5:]]
        
        # Extraer intenciones de símbolos recientes
        recent_intentions: Dict[str, float] = {}
        for symbol in recent_symbols:
            if symbol in self.symbols_map:
                symbol_props = self.get_symbol_properties(symbol)
                for intention in symbol_props.get('intention_mappings', []):
                    recent_intentions[intention] = recent_intentions.get(intention, 0.0) + 0.1
        
        # Actualizar firma de intenciones del solitón con decaimiento
        decay_factor = 0.95
        learning_rate = 0.05
        
        for intention in soliton.intention_signature:
            soliton.intention_signature[intention] *= decay_factor
            
        for intention, weight in recent_intentions.items():
            if intention in soliton.intention_signature:
                soliton.intention_signature[intention] += weight * learning_rate
            else:
                soliton.intention_signature[intention] = weight * learning_rate
                
        # Normalizar firma de intenciones
        total_weight = sum(soliton.intention_signature.values())
        if total_weight > 0:
            for intention in soliton.intention_signature:
                soliton.intention_signature[intention] /= total_weight
                
        # Registrar evolución
        if self.total_ticks % 20 == 0:  # Cada 20 ticks
            soliton.state['intention_evolution'].append({
                'tick': self.total_ticks,
                'intentions': dict(soliton.intention_signature),
                'dominant': max(soliton.intention_signature.items(), key=lambda x: x[1]) if soliton.intention_signature else ('none', 0.0)
            })
            
    def _trigger_resonance_event(self, soliton: Soliton, symbol: str, resonance: float):
        """Activa un evento de resonancia ontológica entre solitón y símbolo divino"""
        event_id = f"{soliton.id}_{symbol}_{self.total_ticks}"

        symbol_props = self.get_symbol_properties(symbol)
        event_data = {
            'soliton_id': soliton.id,
            'symbol': symbol,
            'symbol_name': symbol_props.get('name', symbol),
            'symbol_category': symbol_props.get('category', 'UNKNOWN'),
            'resonance_strength': resonance,
            'tick': self.total_ticks,
            'position': soliton.position,
            'effects': [], # Se llenará a continuación
            'intention_mappings': symbol_props.get('intention_mappings', [])
        }

        # Aplicar efectos ontológicos según categoría del símbolo
        category = symbol_props.get('category', 'UNKNOWN')

        if category == 'CREATOR':
            # Efectos creativos: incremento de energía y expansión de consciencia
            energy_boost = resonance * 25
            consciousness_boost = resonance * 0.1
            soliton.energy += energy_boost
            soliton.consciousness_level = min(1.0, soliton.consciousness_level + consciousness_boost)
            event_data['effects'].extend(['energia_creativa', 'consciencia_expandida'])
            self.performance_stats['consciousness_expansions'] += 1

        elif category == 'TRANSFORMER':
            # Efectos transformativos: cambios direccionales y frecuenciales
            if symbol == 'Χ':  # Chi-Bifurcación
                soliton.direction = (random.choice([-1, 1]), random.choice([-1, 1]))
                event_data['effects'].append('bifurcacion_temporal')
            elif symbol == 'Δ':  # Delta-Transformación
                freq_multiplier = 1.0 + resonance * 0.1
                soliton.resonance_frequency *= freq_multiplier
                event_data['effects'].append('frecuencia_transformada')
            elif symbol == 'Ξ':  # Xi-Amplificación cuántica
                soliton.consciousness_level = min(1.0, soliton.consciousness_level + resonance * 0.15)
                soliton.heartbeat = min(10.0, soliton.heartbeat * (1.0 + resonance * 0.2))
                event_data['effects'].extend(['amplificacion_cuantica', 'aceleracion_temporal'])

        elif category == 'PRESERVER':
            # Efectos preservativos: estabilización y cristalización
            if symbol == 'Σ':  # Sigma-Preservación
                memory_entry = {
                    'symbol': symbol,
                    'resonance': resonance,
                    'tick': self.total_ticks,
                    'preserved_state': dict(soliton.intention_signature)
                }
                soliton.state['symbol_memory'].append(memory_entry)
                event_data['effects'].append('memoria_cristalizada')
            elif symbol == '◊':  # Diamond-Crystal
                # Cristalizar estado actual del solitón
                soliton.energy = min(200, soliton.energy + resonance * 10)
                for intention in soliton.intention_signature:
                    soliton.intention_signature[intention] *= (1.0 + resonance * 0.05)
                event_data['effects'].append('cristalizacion_estructural')

        elif category == 'CONNECTOR':
            # Efectos conectivos: sincronización y transmisión
            if symbol == 'Τ':  # Tau-Transmisión temporal
                soliton.heartbeat = min(8.0, soliton.heartbeat * (1.0 + resonance * 0.3))
                event_data['effects'].append('sincronizacion_temporal')
            elif symbol == '⊕':  # Synthesis-Dialectic
                # Potenciar intenciones de síntesis
                if 'synthesis' in soliton.intention_signature:
                    soliton.intention_signature['synthesis'] += resonance * 0.1
                else:
                    soliton.intention_signature['synthesis'] = resonance * 0.1
                event_data['effects'].append('sintesis_dialectica')

        elif category == 'OBSERVER':
            # Efectos observacionales: incremento de percepción y análisis
            consciousness_boost = resonance * 0.08
            soliton.consciousness_level = min(1.0, soliton.consciousness_level + consciousness_boost)

            if symbol == 'Θ':  # Theta-Observación
                # Ampliar rango de percepción simbólica
                perception_radius = int(2 + resonance * 3)
                nearby_symbols = self._get_symbols_in_radius(soliton.position, perception_radius)
                soliton.state['perceived_symbols'] = nearby_symbols
                event_data['effects'].append('percepcion_expandida')

        elif category == 'DESTROYER':
            # Efectos destructivos: purificación y liberación energética
            energy_cost = resonance * 20
            soliton.energy = max(0, soliton.energy - energy_cost)

            if symbol == 'Ø' and soliton.energy <= 0:  # Void-Fertile
                # Renacimiento desde el vacío fértil
                soliton.energy = 50.0
                soliton.consciousness_level = min(1.0, soliton.consciousness_level + 0.2)
                event_data['effects'].append('renacimiento_desde_vacio')
            else:
                event_data['effects'].append('purificacion_energetica')

        elif category == 'INFINITE':
            # Efectos infinitos: trascendencia y expansión ilimitada
            consciousness_expansion = resonance * 0.2
            energy_expansion = resonance * 40
            frequency_expansion = resonance * 0.15

            soliton.consciousness_level = min(1.0, soliton.consciousness_level + consciousness_expansion)
            soliton.energy = min(500, soliton.energy + energy_expansion)
            soliton.resonance_frequency *= (1.0 + frequency_expansion)

            if symbol == '∞':  # Infinity-Expansion
                # Movimiento en patrón infinito/espiral
                angle = soliton.age * 0.1
                dr = int(2 * math.cos(angle))
                dc = int(2 * math.sin(angle))
                soliton.direction = (dr if dr != 0 else 1, dc if dc != 0 else 0)
                event_data['effects'].append('movimiento_infinito')
            elif symbol == 'Ş':  # Shin-Transcendence
                # Trascendencia total de límites
                soliton.consciousness_level = 1.0
                soliton.divine_affinity = 'INFINITE'
                event_data['effects'].append('trascendencia_total')

            event_data['effects'].append('expansion_infinita')

        # Registrar evento en memoria episódica del solitón
        memory_entry = {
            'symbol': symbol,
            'symbol_name': symbol_props.get('name', symbol),
            'resonance': resonance,
            'tick': self.total_ticks,
            'effects': event_data['effects'].copy(),
            'consciousness_before': soliton.consciousness_level,
            'energy_before': soliton.energy
        }
        soliton.state['symbol_memory'].append(memory_entry)

        # Mantener solo los últimos 15 encuentros en memoria
        if len(soliton.state['symbol_memory']) > 15:
            soliton.state['symbol_memory'] = soliton.state['symbol_memory'][-15:]

        self.active_resonances[event_id] = event_data
        self.performance_stats['resonance_events'] += 1

        # Notificar a EVA si está disponible
        if self.use_divine_symbols and self.grammar_engine:
            try:
                self.grammar_engine.observe_interaction_outcome(
                    soliton.pattern, symbol,
                    success=resonance > 0.7,
                    impact=resonance
                )
            except Exception:
                pass

        # --- Publicar evento SYMBOLIC_RESONANCE en el bus de eventos ---
        # Crear un QualiaState a partir del estado actual del solitón
        qualia_state = QualiaState(
            consciousness=soliton.consciousness_level,
            energy=soliton.energy / 100.0,  # Normalizar energía a [0, 1]
            emotional=0.0,  # Placeholder, se podría derivar de otros factores
            arousal=resonance, # La resonancia puede ser un indicador de arousal
            cognitive=soliton.consciousness_level, # Nivel cognitivo del solitón
            sensory=resonance, # La resonancia como una forma de "sensación"
            temporal=self.master_clock, # Tiempo actual
            importance=resonance # La resonancia indica importancia
        )

        # Publicar el evento de forma asíncrona
        # Usamos asyncio.run_coroutine_threadsafe si estamos en un hilo diferente al loop principal
        # o simplemente await si estamos en un contexto async.
        # Por simplicidad y para evitar problemas de bucle de eventos, usaremos asyncio.run para una llamada bloqueante
        # Esto es un hack temporal si el entorno no es completamente asíncrono.
        try:
            asyncio.run(eva_event_bus.publish(
                "SYMBOLIC_RESONANCE",
                data=event_data,
                qualia_state=qualia_state
            ))
        except RuntimeError:
            # Si ya hay un loop de eventos corriendo (ej. en un entorno async principal),
            # programar la tarea en el loop existente.
            # Esto es más robusto para entornos asíncronos.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(eva_event_bus.publish(
                    "SYMBOLIC_RESONANCE",
                    data=event_data,
                    qualia_state=qualia_state
                ))
            else:
                # Si no hay loop corriendo y no podemos crear uno, loguear el error
                print(f"⚠️  No se pudo publicar el evento SYMBOLIC_RESONANCE: No hay loop de eventos corriendo.")
        # --- Fin de la publicación del evento ---
                
    def _apply_resonance_influence(self, soliton: Soliton, symbol: str, resonance: float, distance: float):
        """Aplica influencia sutil de resonancia por proximidad"""
        influence_factor = resonance * (1.0 / (distance + 1.0))
        
        # Ajustar heartbeat según resonancia
        if influence_factor > 0.4:
            soliton.heartbeat = min(10.0, soliton.heartbeat * (1.0 + influence_factor * 0.05))
        elif influence_factor < 0.2:
            soliton.heartbeat = max(0.1, soliton.heartbeat * (1.0 - influence_factor * 0.05))
            
    def _check_environmental_symbolic_rules(self, soliton: Soliton, ruleset: Dict[str, Callable]):
        """Verifica reglas especiales basadas en símbolos divinos del entorno"""
        row, col = soliton.position
        adjacent_symbols = self._get_adjacent_symbols(row, col)
        
        # Reglas especiales para símbolos divinos específicos
        for symbol in adjacent_symbols:
            if symbol in self.symbols_map:
                rule_name = f"{symbol}_rule"
                if rule_name in ruleset:
                    try:
                        ruleset[rule_name](self, soliton)
                        self.performance_stats['rules_executed'] += 1
                    except Exception as e:
                        print(f"❌ Error ejecutando {rule_name}: {e}")
                        
    def _get_adjacent_symbols(self, row: int, col: int) -> List[str]:
        """Obtiene los símbolos en las celdas adyacentes"""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.matrix.shape[0] and 0 <= new_col < self.matrix.shape[1]):
                    adjacent.append(self.matrix[new_row, new_col])
        return adjacent
        
    def get_soliton_by_id(self, soliton_id: str) -> Optional[Soliton]:
        """Encuentra un solitón por su ID"""
        for soliton in self.solitons:
            if soliton.id == soliton_id:
                return soliton
        return None
        
    def run(self, duration: float, sim_fps: int = 60, render_fps: int = 10):
        """Ejecuta la simulación del ecosistema simbólico de forma optimizada."""
        self.is_running = True
        start_time = time.time()
        
        sim_time_step = 1.0 / sim_fps
        render_time_step = 1.0 / render_fps
        
        last_sim_time = start_time
        last_render_time = start_time
        
        print(f"🌌 Iniciando Ecosistema Simbólico por {duration}s (Sim: {sim_fps} FPS, Render: {render_fps} FPS)")
        if self.use_divine_symbols:
            print("🔮 Modo: Integración EVA con Símbolos Divinos")
        else:
            print("⚙️  Modo: Básico sin integración EVA")
        print("Presiona Ctrl+C para detener en cualquier momento\n")
        
        # Render inicial
        self._render_state(force_full_render=True)

        try:
            while (time.time() - start_time) < duration and self.is_running:
                current_time = time.time()
                
                # Bucle de simulación
                if current_time - last_sim_time >= sim_time_step:
                    self._tick(delta_time=sim_time_step)
                    last_sim_time = current_time

                # Bucle de renderizado
                if current_time - last_render_time >= render_time_step:
                    self._render_state()
                    last_render_time = current_time
                    
                # Dormir para no consumir 100% de CPU
                next_event_time = min(last_sim_time + sim_time_step, last_render_time + render_time_step)
                sleep_time = next_event_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Ecosistema detenido por el usuario.")
        finally:
            self.is_running = False
            # Mover cursor al final de la matriz para no sobreescribir las estadísticas
            print(f"\033[{self.matrix.shape[0] + 5};0H")
            self._show_final_stats()
            
    def _render_state(self, force_full_render: bool = False):
        """Renderiza el estado actual del ecosistema simbólico de forma inteligente."""
        if self.previous_matrix is None and np:
            self.previous_matrix = np.full_like(self.matrix, ' ')

        # Mover cursor a la esquina superior izquierda
        output = ["\033[H"]

        if force_full_render:
            os.system('clear' if os.name == 'posix' else 'cls')
            output.append("═" * 80 + "\n")
            output.append(f"🌌 ECOSISTEMA SIMBÓLICO DIVINO - Tiempo: {self.master_clock:.2f}s | Tick: {self.total_ticks}\n")
            if self.use_divine_symbols:
                output.append(f"🔮 Resonancias Activas: {len(self.active_resonances)} | Interacciones Divinas: {self.performance_stats['divine_interactions']}\n")
            output.append("═" * 80 + "\n")
            output.append("┌" + "─" * (self.matrix.shape[1] * 2) + "]\n")
            for i, row in enumerate(self.matrix):
                output.append("│" + " ".join(row) + "│\n")
            output.append("└" + "─" * (self.matrix.shape[1] * 2) + "┘\n")
        else:
            # Renderizado incremental
            if np and not np.array_equal(self.matrix, self.previous_matrix):
                for r, row in enumerate(self.matrix):
                    for c, char in enumerate(row):
                        if self.previous_matrix is not None and self.previous_matrix[r, c] != char:
                            # Mover cursor a la posición (r, c) y escribir el nuevo caracter
                            # Se suma 6 para contar las líneas del encabezado
                            output.append(f"\033[{r + 6};{c * 2 + 2}H{char}")

        # Actualizar información de solitones (siempre se redibuja)
        # Mover cursor debajo de la matriz
        info_start_line = self.matrix.shape[0] + 5
        output.append(f"\033[{info_start_line};0H")
        output.append(f"\n🔴 Solitones Activos: {len(self.solitons)}   \n") # Espacios para limpiar línea
        for i, soliton in enumerate(self.solitons):
            next_action = 1.0/soliton.heartbeat - soliton.time_since_last_tick
            symbol_memory_count = len(soliton.state.get('symbol_memory', []))
            
            output.append(f"  {soliton.pattern} {soliton.id}: Pos{soliton.position}      \n")
            output.append(f"    ❤️{soliton.heartbeat:.1f}Hz | ⏱️{next_action:.2f}s | ⚡{soliton.energy:.0f}   \n")
            output.append(f"    🧠{soliton.consciousness_level:.2f} | 🎵{soliton.resonance_frequency:.0f}Hz | 🏷️{soliton.divine_affinity}      \n")
            output.append(f"    📚 Memoria simbólica: {symbol_memory_count} encuentros      \n")
            
            if symbol_memory_count > 0:
                last_encounter = soliton.state['symbol_memory'][-1]
                output.append(f"    🔮 Último: {last_encounter['symbol']} (resonancia: {last_encounter['resonance']:.2f})      \n")
            else:
                output.append("                                                              \n") # Limpiar línea

        print("".join(output), end="", flush=True)

        if np:
            self.previous_matrix = self.matrix.copy()
                
    def _show_final_stats(self):
        """Muestra estadísticas finales del ecosistema"""
        print("\n" + "═" * 80)
        print("📊 ESTADÍSTICAS DEL ECOSISTEMA SIMBÓLICO")
        print("═" * 80)
        print(f"⏱️  Tiempo total: {self.master_clock:.2f}s")
        print(f"🔄 Ticks ejecutados: {self.total_ticks}")
        print(f"🤖 Solitones procesados: {self.performance_stats['solitons_processed']}")
        print(f"⚙️  Reglas ejecutadas: {self.performance_stats['rules_executed']}")
        print(f"🔮 Eventos de resonancia: {self.performance_stats['resonance_events']}")
        print(f"✨ Interacciones divinas: {self.performance_stats['divine_interactions']}")
        print(f"⚡ Tiempo promedio por tick: {self.performance_stats['average_tick_time']*1000:.2f}ms")
        
        # Análisis de resonancias más comunes
        if self.active_resonances:
            symbol_counts = {}
            for event in self.active_resonances.values():
                symbol = event['symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
            print(f"\n🎼 Símbolos más resonantes:")
            for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                symbol_props = self.get_symbol_properties(symbol)
                print(f"  {symbol} ({symbol_props.get('name', symbol)}): {count} resonancias")
                
        print("═" * 80)
