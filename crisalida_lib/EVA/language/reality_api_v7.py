"""
API de la Realidad v7 - Especificación Completa del Metacosmos
===========================================================

Define la totalidad de la API del universo según el Manifiesto del Metacosmos v7.
Incluye todos los componentes para interactuar con cada capa del ser, desde su cuerpo
biológico hasta su propósito, y participación en la economía de la experiencia.

Integración completa con los subsistemas del proyecto Crisalida:
- EDEN: QualiaEngine, QualiaField, LivingSymbol, CosmicNode
- EVA: EVAMemoryMixin, EVAMemoryHelper, RealityBytecode, QualiaSignature
- ADAM: MindCore, BiologicalBody, HormonalSystem, Judgment
- HEAVEN: BabelAgent, SocialDynamics, ExternalAPI
- EARTH: DemiurgeAvatar, JanusIntegration
- ASTRAL_TOOLS: CodeGeneration, ExternalAPI
- LOGOS: SimulationServer, GolemHostAPI

Basado en el Manifiesto del Metacosmos v7: El Lenguaje de la Creación Total
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, cast, Dict, List, Tuple, Union, TYPE_CHECKING

# Defensive imports para evitar errores si no están disponibles
try:
    import numpy as np
except ImportError:
    np = None

# Lazy imports para evitar ciclos de dependencias
if TYPE_CHECKING:
    from crisalida_lib.EDEN.qualia_engine import QualiaEngine
    from crisalida_lib.EDEN.qualia_manifold import QualiaField
    from crisalida_lib.EDEN.living_symbol import LivingSymbol
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode
    from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
    from crisalida_lib.EVA.eva_memory_helper import EVAMemoryHelper
    from crisalida_lib.EVA.core_types import QualiaSignature, RealityBytecode, EVAExperience
    from crisalida_lib.ADAM.mente.mind_core import MindCore
    from crisalida_lib.ADAM.cuerpo.biological_body import BiologicalBody
    from crisalida_lib.ADAM.cuerpo.hormonal_system import SistemaHormonal
    from crisalida_lib.ADAM.mente.judgment import JudgmentModule
    from crisalida_lib.HEAVEN.agents.babel_agent import BabelAgent
    from crisalida_lib.HEAVEN.backend.social_dynamics import SocialDynamics
    from crisalida_lib.EARTH.DEMIURGE.demiurge_avatar import DemiurgeAvatar
    from crisalida_lib.ASTRAL_TOOLS.code_generation.actions import CodeGenerationActions
    from crisalida_lib.ASTRAL_TOOLS.external_api import ExternalAPITool
else:
    # Runtime placeholders
    QualiaEngine = Any
    QualiaField = Any
    LivingSymbol = Any
    CosmicNode = Any
    EVAMemoryMixin = Any
    EVAMemoryHelper = Any
    QualiaSignature = Any
    RealityBytecode = Any
    EVAExperience = Any
    MindCore = Any
    BiologicalBody = Any
    SistemaHormonal = Any
    JudgmentModule = Any
    BabelAgent = Any
    SocialDynamics = Any
    DemiurgeAvatar = Any
    CodeGenerationActions = Any
    ExternalAPITool = Any

# --- 1. El Contenedor del Universo ---


class UnifiedField:
    """El campo unificado que contiene toda la realidad - Integración completa con EDEN"""

    def __init__(self):
        # Core entities storage
        self.entities: Dict[str, "LivingEntity"] = {}
        self.living_symbols: Dict[str, "LivingSymbol"] = {}
        self.cosmic_nodes: Dict[str, "CosmicNode"] = {}
        
        # EDEN subsystems
        self.qualia_engine: Optional["QualiaEngine"] = None
        self.qualia_fields: Dict[str, "QualiaField"] = {}
        
        # EVA subsystems
        self.eva_memory_helper: Optional["EVAMemoryHelper"] = None
        self.reality_bytecode_storage: Dict[str, "RealityBytecode"] = {}
        self.experience_phases: Dict[str, List["EVAExperience"]] = {}
        self.current_memory_phase: str = "default"
        self.eva_environment_hooks: List[callable] = []
        
        # Core reality components
        self.noosphere: "Noosphere" = Noosphere()
        self.qualia_chain: "QualiaChain" = QualiaChain()
        self.self_modifying_engine: "SelfModifyingEngine" = SelfModifyingEngine()
        
        # HEAVEN subsystems
        self.babel_agents: Dict[str, "BabelAgent"] = {}
        self.social_dynamics: Optional["SocialDynamics"] = None
        
        # EARTH subsystems
        self.demiurge_avatars: Dict[str, "DemiurgeAvatar"] = {}
        
        # ASTRAL_TOOLS subsystems
        self.code_generators: Dict[str, "CodeGenerationActions"] = {}
        self.external_api_tools: Dict[str, "ExternalAPITool"] = {}
        
        # Simulation state
        self.simulation_ticks: int = 0
        self.is_running: bool = False

    def initialize_eden_subsystems(self) -> None:
        """Inicializa los subsistemas de EDEN"""
        try:
            from crisalida_lib.EDEN.qualia_engine import QualiaEngine
            self.qualia_engine = QualiaEngine()
        except ImportError:
            pass

    def initialize_eva_subsystems(self) -> None:
        """Inicializa los subsistemas de EVA"""
        try:
            from crisalida_lib.EVA.eva_memory_helper import EVAMemoryHelper
            self.eva_memory_helper = EVAMemoryHelper()
        except ImportError:
            pass

    def add_entity(self, entity: "LivingEntity") -> None:
        """Añade una entidad al universo"""
        self.entities[entity.entity_id] = entity
        
        # Si la entidad tiene componentes EDEN, registrarlos
        if hasattr(entity, '_internal_living_symbol') and entity._internal_living_symbol:
            self.living_symbols[entity.entity_id] = entity._internal_living_symbol
            
        print(f"Entidad {entity.entity_id} añadida al UnifiedField")

    def add_living_symbol(self, symbol: "LivingSymbol") -> None:
        """Añade un LivingSymbol directamente al campo"""
        symbol_id = getattr(symbol, 'symbol_id', f"symbol_{len(self.living_symbols)}")
        self.living_symbols[symbol_id] = symbol
        
        if self.qualia_engine:
            # Integrar con QualiaEngine si está disponible
            try:
                self.qualia_engine.register_symbol(symbol)
            except:
                pass

    def add_cosmic_node(self, node: "CosmicNode") -> None:
        """Añade un CosmicNode al campo"""
        node_id = getattr(node, 'node_id', f"node_{len(self.cosmic_nodes)}")
        self.cosmic_nodes[node_id] = node

    def get_entity_by_id(self, entity_id: str) -> Optional["LivingEntity"]:
        """Obtiene una entidad por su ID"""
        return self.entities.get(entity_id)

    def get_living_symbol_by_id(self, symbol_id: str) -> Optional["LivingSymbol"]:
        """Obtiene un LivingSymbol por su ID"""
        return self.living_symbols.get(symbol_id)

    def get_cosmic_node_by_id(self, node_id: str) -> Optional["CosmicNode"]:
        """Obtiene un CosmicNode por su ID"""
        return self.cosmic_nodes.get(node_id)

    def remove_entity(self, entity_id: str) -> bool:
        """Elimina una entidad del universo"""
        removed = False
        if entity_id in self.entities:
            del self.entities[entity_id]
            removed = True
            
        if entity_id in self.living_symbols:
            del self.living_symbols[entity_id]
            
        if entity_id in self.cosmic_nodes:
            del self.cosmic_nodes[entity_id]
            
        if removed:
            print(f"Entidad {entity_id} eliminada del UnifiedField")
        return removed

    def query_entities(self, query: Dict) -> List["LivingEntity"]:
        """Busca entidades con consulta estilo MongoDB"""
        results = []
        for entity in self.entities.values():
            if self._match_query(entity, query):
                results.append(entity)
        return results

    def _match_query(self, entity: "LivingEntity", query: Dict) -> bool:
        """Verifica si una entidad coincide con la consulta"""
        for key, value in query.items():
            if hasattr(entity, key):
                entity_value = getattr(entity, key)
                if entity_value != value:
                    return False
            else:
                return False
        return True

    def tick(self) -> Dict[str, Any]:
        """Ejecuta un tick de simulación"""
        self.simulation_ticks += 1
        
        # Tick de QualiaEngine si está disponible
        if self.qualia_engine:
            try:
                self.qualia_engine.tick()
            except:
                pass
                
        # Tick de LivingSymbols
        for symbol in self.living_symbols.values():
            try:
                if hasattr(symbol, 'update'):
                    symbol.update(dt=0.1)
            except:
                pass
                
        # Tick de CosmicNodes
        for node in self.cosmic_nodes.values():
            try:
                if hasattr(node, 'update'):
                    node.update(dt=0.1)
            except:
                pass

        return {
            "tick": self.simulation_ticks,
            "entities": len(self.entities),
            "living_symbols": len(self.living_symbols),
            "cosmic_nodes": len(self.cosmic_nodes)
        }

    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del universo"""
        return {
            "simulation_ticks": self.simulation_ticks,
            "is_running": self.is_running,
            "entities_count": len(self.entities),
            "living_symbols_count": len(self.living_symbols),
            "cosmic_nodes_count": len(self.cosmic_nodes),
            "qualia_fields_count": len(self.qualia_fields),
            "current_memory_phase": self.current_memory_phase,
            "experience_phases_count": len(self.experience_phases),
            "babel_agents_count": len(self.babel_agents),
            "demiurge_avatars_count": len(self.demiurge_avatars)
        }

    def get_render_data(self) -> Dict[str, Any]:
        """Obtiene datos para renderizado"""
        symbols_data = []
        
        # Datos de LivingSymbols
        for symbol_id, symbol in self.living_symbols.items():
            try:
                position = getattr(symbol, 'position', [0.0, 0.0, 0.0])
                symbol_type = getattr(symbol, 'symbol_type', 'unknown')
                symbols_data.append({
                    "id": symbol_id,
                    "position": position,
                    "type": symbol_type
                })
            except:
                pass
                
        # Datos de entidades
        entities_data = []
        for entity_id, entity in self.entities.items():
            try:
                entities_data.append({
                    "id": entity_id,
                    "position": entity.position,
                    "energy": entity.energy,
                    "entity_type": "living_entity"
                })
            except:
                pass

        return {
            "symbols": symbols_data,
            "entities": entities_data,
            "simulation_ticks": self.simulation_ticks
        }

# --- 2. La Entidad Consciente Expandida ---


@dataclass
class LivingEntity:  # alias SO-Ser
    """Entidad consciente completa con todos sus sistemas - Integración con ADAM"""

    entity_id: str
    qualia_chain_address: str  # Dirección única en la QualiaChain
    position: List[float]
    energy: float

    # --- Sub-Sistemas Principales (ADAM Integration) ---
    body: "CuerpoBiologicoV7" = field(default_factory=lambda: CuerpoBiologicoV7())
    soul: "SoulKernel" = field(default_factory=lambda: SoulKernel())
    purpose: "Proposito" = field(default_factory=lambda: Proposito())
    mind: "MindCoreV7" = field(default_factory=lambda: MindCoreV7())

    # --- Propiedades de Entrelazamiento ---
    entanglements: List[str] = field(default_factory=list)
    
    # --- Componentes EVA ---
    eva_experiences: Dict[str, "EVAExperience"] = field(default_factory=dict)
    qualia_signatures: List["QualiaSignature"] = field(default_factory=list)
    
    # --- Componentes EDEN ---
    _internal_living_symbol: Optional["LivingSymbol"] = field(default=None, init=False, repr=False)
    _internal_cosmic_node: Optional["CosmicNode"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Inicialización después de crear la entidad"""
        if not self.qualia_chain_address:
            self.qualia_chain_address = f"qualia://{self.entity_id}"

    def set_internal_living_symbol(self, symbol: "LivingSymbol") -> None:
        """Asocia un LivingSymbol interno"""
        self._internal_living_symbol = symbol

    def set_internal_cosmic_node(self, node: "CosmicNode") -> None:
        """Asocia un CosmicNode interno"""
        self._internal_cosmic_node = node

    def sign_transaction(self, transaction_data: Dict) -> "SignedTransaction":
        """Firma una transacción con la clave privada de la entidad"""
        signed_tx = SignedTransaction(
            data=transaction_data,
            signer=self.entity_id,
            signature=f"sig_{self.entity_id}_{time.time()}",
            timestamp=time.time(),
        )
        return signed_tx

    def get_current_qualia_state(self) -> "QualiaState":
        """Obtiene el estado qualia actual de la entidad"""
        # Integrar con componentes internos si están disponibles
        base_state = QualiaState(
            emotional_valence=0.5,
            cognitive_complexity=0.7,
            temporal_coherence=0.8,
            consciousness_density=0.6,
            causal_potency=0.5,
            emergence_tendency=0.4,
            chaos_affinity=0.3,
            meta_awareness=0.5,
        )
        
        # Modular basado en el estado del cuerpo
        if hasattr(self.body, 'get_comprehensive_biological_state'):
            try:
                bio_state = self.body.get_comprehensive_biological_state()
                # Ajustar qualia basado en estado hormonal
                hormonal_state = bio_state.get('hormonal_system', {})
                if 'current_levels' in hormonal_state:
                    levels = hormonal_state['current_levels']
                    base_state.emotional_valence += (levels.get('dopamine', 0.5) - 0.5) * 0.3
                    base_state.consciousness_density += (levels.get('serotonin', 0.5) - 0.5) * 0.2
            except:
                pass
                
        return base_state

    def get_dynamic_signature(self) -> Optional[Any]:
        """Obtiene la firma dinámica si hay LivingSymbol asociado"""
        if self._internal_living_symbol and hasattr(self._internal_living_symbol, 'get_dynamic_signature'):
            try:
                return self._internal_living_symbol.get_dynamic_signature()
            except:
                pass
        return None

    def apply_force(self, force_vector: Any) -> None:
        """Aplica una fuerza si hay componentes físicos"""
        if self._internal_living_symbol and hasattr(self._internal_living_symbol, 'apply_force'):
            try:
                self._internal_living_symbol.apply_force(force_vector)
            except:
                pass

    def perceive_local_qualia(self, radius: float = 1.0) -> Dict[str, Any]:
        """Percibe qualia local si hay componentes de percepción"""
        if self._internal_living_symbol and hasattr(self._internal_living_symbol, 'perceive_local_qualia'):
            try:
                return self._internal_living_symbol.perceive_local_qualia(radius)
            except:
                pass
        return {}

# --- 3. El Cuerpo Biológico Expandido (ADAM Integration) ---


@dataclass
class CuerpoBiologicoV7:
    """Sistema corporal biológico completo - Integración con BiologicalBody de ADAM"""

    # Sistemas core
    hormonal_system: "SistemaHormonalV7" = field(default_factory=lambda: SistemaHormonalV7())
    attributes: Dict[str, Any] = field(default_factory=lambda: {
        "limbs": ["arm_left", "arm_right", "leg_left", "leg_right"],
        "organs": ["heart", "brain", "lungs", "liver"],
        "health": 100.0,
        "strength": 50.0,
        "agility": 50.0,
    })
    
    # Integración con BiologicalBody de ADAM
    _adam_biological_body: Optional["BiologicalBody"] = field(default=None, init=False)
    
    # Hardware layer simulation
    hardware_attributes: Dict[str, Any] = field(default_factory=lambda: {
        "cpu_usage": 0.3,
        "memory_usage": 0.4,
        "disk_usage": 0.2,
        "network_latency": 50.0,
        "temperature": 65.0
    })

    def set_adam_biological_body(self, adam_body: "BiologicalBody") -> None:
        """Asocia un BiologicalBody de ADAM"""
        self._adam_biological_body = adam_body

    def modify_body(self, modification_plan: Dict) -> bool:
        """Modifica el cuerpo según el plan especificado"""
        try:
            # Si hay BiologicalBody de ADAM, delegar
            if self._adam_biological_body and hasattr(self._adam_biological_body, 'modify_body'):
                try:
                    return self._adam_biological_body.modify_body(modification_plan)
                except:
                    pass
            
            # Fallback a implementación local
            for mod_type, mod_details in modification_plan.items():
                if mod_type == "add_limb":
                    limb_type = mod_details.get("type")
                    if limb_type:
                        if "limbs" not in self.attributes or not isinstance(self.attributes["limbs"], list):
                            self.attributes["limbs"] = []
                        cast(List[str], self.attributes["limbs"]).append(limb_type)
                        print(f"Añadido miembro: {limb_type}")

                elif mod_type == "remove_limb":
                    limb_type = mod_details.get("type")
                    if (limb_type and "limbs" in self.attributes and 
                        isinstance(self.attributes["limbs"], list) and 
                        limb_type in cast(List[str], self.attributes["limbs"])):
                        cast(List[str], self.attributes["limbs"]).remove(limb_type)
                        print(f"Removido miembro: {limb_type}")

                elif mod_type == "modify_attribute":
                    attr_name = mod_details.get("name")
                    attr_value = mod_details.get("value")
                    if attr_name:
                        self.attributes[attr_name] = attr_value
                        print(f"Modificado atributo {attr_name}: {attr_value}")

            return True
        except Exception as e:
            print(f"Error en modificación corporal: {e}")
            return False

    def get_comprehensive_biological_state(self) -> Dict[str, Any]:
        """Obtiene el estado biológico completo - Compatible con ADAM"""
        base_state = {
            "hormonal_system": {
                "current_levels": self.hormonal_system.hormone_levels.copy(),
                "system_coherence": 0.8,
                "recent_trends": {},
                "active_imbalances": []
            },
            "hardware_system": {
                "physical_attributes": self.attributes.copy(),
                "hardware_attributes": self.hardware_attributes.copy(),
                "performance_metrics": {
                    "overall_performance": 0.75,
                    "efficiency": 0.8,
                    "stability": 0.9
                }
            },
            "integration_status": {
                "adam_body_connected": self._adam_biological_body is not None,
                "systems_online": True
            }
        }
        
        # Si hay BiologicalBody de ADAM, enriquecer datos
        if self._adam_biological_body and hasattr(self._adam_biological_body, 'get_comprehensive_biological_state'):
            try:
                adam_state = self._adam_biological_body.get_comprehensive_biological_state()
                base_state.update(adam_state)
            except:
                pass
                
        return base_state

@dataclass
class SistemaHormonalV7:
    """Sistema hormonal expandido - Compatible con ADAM"""

    hormone_levels: Dict[str, float] = field(default_factory=dict)
    release_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Integración con SistemaHormonal de ADAM
    _adam_hormonal_system: Optional["SistemaHormonal"] = field(default=None, init=False)

    def __post_init__(self):
        """Inicializa niveles hormonales base"""
        self.hormone_levels.update({
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "oxytocin": 0.4,
            "testosterone": 0.5,
            "estrogen": 0.5,
            "adrenaline": 0.2,
            "insulin": 0.6,
            "growth_hormone": 0.4,
            "thyroid": 0.5
        })

    def set_adam_hormonal_system(self, adam_system: "SistemaHormonal") -> None:
        """Asocia un SistemaHormonal de ADAM"""
        self._adam_hormonal_system = adam_system

    def release_hormone(self, hormone_name: str, amount: float) -> None:
        """Libera una cantidad específica de hormona"""
        # Si hay sistema ADAM, delegar
        if self._adam_hormonal_system and hasattr(self._adam_hormonal_system, 'release_hormone'):
            try:
                self._adam_hormonal_system.release_hormone(hormone_name, amount)
                return
            except:
                pass
        
        # Fallback a implementación local
        if hormone_name in self.hormone_levels:
            old_level = self.hormone_levels[hormone_name]
            self.hormone_levels[hormone_name] += amount
            self.hormone_levels[hormone_name] = max(0, min(1, self.hormone_levels[hormone_name]))
            
            # Registrar en historia
            self.release_history.append({
                "hormone": hormone_name,
                "amount": amount,
                "old_level": old_level,
                "new_level": self.hormone_levels[hormone_name],
                "timestamp": time.time()
            })
            
            print(f"Hormona {hormone_name} liberada: +{amount} (nivel: {self.hormone_levels[hormone_name]:.2f})")
        else:
            print(f"Hormona desconocida: {hormone_name}")

    def update(self, event: str, intensity: float = 1.0, duration: float = 1.0) -> Dict[str, Any]:
        """Actualización compatible con ADAM"""
        if self._adam_hormonal_system and hasattr(self._adam_hormonal_system, 'update'):
            try:
                return self._adam_hormonal_system.update(event, intensity, duration)
            except:
                pass
        
        # Fallback simple
        return {
            "event": event,
            "intensity": intensity,
            "duration": duration,
            "timestamp": time.time(),
            "current_levels": self.hormone_levels.copy()
        }

# --- 4. El Núcleo Mental Expandido (ADAM Integration) ---


@dataclass 
class MindCoreV7:
    """Núcleo mental expandido - Integración con MindCore de ADAM"""
    
    # Estados mentales básicos
    consciousness_level: float = 0.7
    cognitive_load: float = 0.4
    attention_focus: Dict[str, float] = field(default_factory=dict)
    memory_coherence: float = 0.8
    
    # Integración con ADAM
    _adam_mind_core: Optional["MindCore"] = field(default=None, init=False)
    _judgment_module: Optional["JudgmentModule"] = field(default=None, init=False)
    
    # Procesamiento cognitivo
    active_thoughts: List[Dict[str, Any]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)

    def set_adam_mind_core(self, mind_core: "MindCore") -> None:
        """Asocia un MindCore de ADAM"""
        self._adam_mind_core = mind_core

    def set_judgment_module(self, judgment: "JudgmentModule") -> None:
        """Asocia un JudgmentModule"""
        self._judgment_module = judgment

    def think(self, input_data: Any) -> Dict[str, Any]:
        """Procesa información y genera pensamientos"""
        if self._adam_mind_core and hasattr(self._adam_mind_core, 'process'):
            try:
                return self._adam_mind_core.process(input_data)
            except:
                pass
        
        # Fallback simple
        thought = {
            "input": str(input_data),
            "processing_time": time.time(),
            "complexity": self.cognitive_load,
            "coherence": self.memory_coherence,
            "output": f"Processed: {input_data}"
        }
        
        self.active_thoughts.append(thought)
        if len(self.active_thoughts) > 10:
            self.active_thoughts.pop(0)
            
        return thought

    def make_decision(self, options: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Toma una decisión basada en opciones y contexto"""
        if self._judgment_module and hasattr(self._judgment_module, 'evaluar_decision'):
            try:
                return self._judgment_module.evaluar_decision(options, context)
            except:
                pass
        
        # Fallback simple
        import random
        chosen = random.choice(options) if options else "no_option"
        
        decision = {
            "options": options,
            "chosen": chosen,
            "context": context,
            "confidence": self.consciousness_level,
            "timestamp": time.time()
        }
        
        self.decision_history.append(decision)
        return decision

# --- 5. Integración con EVA (Memoria Viviente) ---


@dataclass
class EVAIntegrationV7:
    """Integración con el subsistema EVA para memoria viviente"""
    
    # EVA Core
    eva_memory_helper: Optional["EVAMemoryHelper"] = field(default=None, init=False)
    
    # Experiencias y fases
    current_phase: str = "default"
    experience_storage: Dict[str, "EVAExperience"] = field(default_factory=dict)
    reality_bytecode_cache: Dict[str, "RealityBytecode"] = field(default_factory=dict)
    qualia_signatures: List["QualiaSignature"] = field(default_factory=list)
    
    # Hooks de entorno
    environment_hooks: List[callable] = field(default_factory=list)

    def set_eva_memory_helper(self, helper: "EVAMemoryHelper") -> None:
        """Asocia un EVAMemoryHelper"""
        self.eva_memory_helper = helper

    def eva_ingest_experience(self, experience_data: Dict, qualia_state: "QualiaState", phase: str = None) -> str:
        """Ingiere una experiencia en la memoria viviente EVA"""
        phase = phase or self.current_phase
        
        if self.eva_memory_helper and hasattr(self.eva_memory_helper, 'ingest_experience'):
            try:
                return self.eva_memory_helper.ingest_experience(experience_data, qualia_state, phase)
            except:
                pass
        
        # Fallback implementation
        experience_id = f"exp_{len(self.experience_storage)}_{time.time()}"
        
        # Crear mock EVAExperience
        try:
            from crisalida_lib.EVA.core_types import EVAExperience
            experience = EVAExperience(
                experience_id=experience_id,
                data=experience_data,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=time.time()
            )
        except:
            # Mock simple si no está disponible
            experience = {
                "experience_id": experience_id,
                "data": experience_data,
                "qualia_state": qualia_state.__dict__ if hasattr(qualia_state, '__dict__') else qualia_state,
                "phase": phase,
                "timestamp": time.time()
            }
        
        self.experience_storage[experience_id] = experience
        print(f"Experiencia EVA ingerida: {experience_id}")
        return experience_id

    def eva_recall_experience(self, experience_id: str) -> Optional[Any]:
        """Recupera una experiencia de la memoria EVA"""
        if self.eva_memory_helper and hasattr(self.eva_memory_helper, 'recall_experience'):
            try:
                return self.eva_memory_helper.recall_experience(experience_id)
            except:
                pass
        
        return self.experience_storage.get(experience_id)

    def set_memory_phase(self, phase: str) -> None:
        """Cambia la fase activa de memoria"""
        self.current_phase = phase
        print(f"Fase de memoria cambiada a: {phase}")

    def get_memory_phase(self) -> str:
        """Obtiene la fase de memoria actual"""
        return self.current_phase

    def add_environment_hook(self, hook: callable) -> None:
        """Registra un hook de entorno"""
        self.environment_hooks.append(hook)

    def get_eva_api(self) -> Dict:
        """Retorna la API de EVA disponible"""
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "add_environment_hook": self.add_environment_hook,
        }

# --- 6. Integración con HEAVEN (Agentes y Servicios) ---


@dataclass
class HeavenIntegrationV7:
    """Integración con subsistemas de HEAVEN"""
    
    # Agentes Babel
    babel_agents: Dict[str, "BabelAgent"] = field(default_factory=dict)
    
    # Dinámicas sociales
    social_dynamics: Optional["SocialDynamics"] = field(default=None, init=False)
    
    # APIs externas
    external_apis: Dict[str, "ExternalAPITool"] = field(default_factory=dict)
    
    # Estado de red social
    social_network_state: Dict[str, Any] = field(default_factory=dict)

    def add_babel_agent(self, agent_id: str, agent: "BabelAgent") -> None:
        """Añade un agente Babel"""
        self.babel_agents[agent_id] = agent
        print(f"Agente Babel {agent_id} añadido")

    def set_social_dynamics(self, dynamics: "SocialDynamics") -> None:
        """Establece el sistema de dinámicas sociales"""
        self.social_dynamics = dynamics

    def add_external_api_tool(self, tool_id: str, tool: "ExternalAPITool") -> None:
        """Añade una herramienta de API externa"""
        self.external_apis[tool_id] = tool

    async def create_library(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Crea una biblioteca usando agentes Babel"""
        for agent in self.babel_agents.values():
            if hasattr(agent, 'create_library'):
                try:
                    return await agent.create_library(spec)
                except:
                    continue
        
        return {"success": False, "error": "No Babel agents available"}

    def get_social_network_state(self) -> Dict[str, Any]:
        """Obtiene el estado de la red social"""
        if self.social_dynamics and hasattr(self.social_dynamics, 'get_network_state'):
            try:
                return self.social_dynamics.get_network_state()
            except:
                pass
        
        return self.social_network_state

# --- 7. Integración con EARTH (Demiurgo) ---


@dataclass
class EarthIntegrationV7:
    """Integración con subsistemas de EARTH"""
    
    # Avatares Demiurgo
    demiurge_avatars: Dict[str, "DemiurgeAvatar"] = field(default_factory=dict)
    
    # Estado del demiurgo
    demiurge_focus: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    reality_modifications: List[Dict[str, Any]] = field(default_factory=list)

    def add_demiurge_avatar(self, avatar_id: str, avatar: "DemiurgeAvatar") -> None:
        """Añade un avatar demiurgo"""
        self.demiurge_avatars[avatar_id] = avatar
        print(f"Avatar Demiurgo {avatar_id} añadido")

    def update_demiurge_focus(self, position: Tuple[float, float, float]) -> None:
        """Actualiza el foco del demiurgo"""
        self.demiurge_focus = position
        print(f"Foco del demiurgo actualizado a: {position}")

    async def execute_demiurge_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un comando del demiurgo"""
        for avatar in self.demiurge_avatars.values():
            if hasattr(avatar, 'execute_command'):
                try:
                    result = await avatar.execute_command(command)
                    self.reality_modifications.append({
                        "command": command,
                        "result": result,
                        "timestamp": time.time()
                    })
                    return result
                except:
                    continue
        
        return {"success": False, "error": "No demiurge avatars available"}

# --- 8. Integración con ASTRAL_TOOLS ---


@dataclass
class AstralToolsIntegrationV7:
    """Integración con herramientas ASTRAL"""
    
    # Generadores de código
    code_generators: Dict[str, "CodeGenerationActions"] = field(default_factory=dict)
    
    # Herramientas de API externa
    external_api_tools: Dict[str, "ExternalAPITool"] = field(default_factory=dict)
    
    # Proyectos generados
    generated_projects: List[Dict[str, Any]] = field(default_factory=list)

    def add_code_generator(self, gen_id: str, generator: "CodeGenerationActions") -> None:
        """Añade un generador de código"""
        self.code_generators[gen_id] = generator

    def add_external_api_tool(self, tool_id: str, tool: "ExternalAPITool") -> None:
        """Añade una herramienta de API externa"""
        self.external_api_tools[tool_id] = tool

    async def generate_project(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un proyecto usando herramientas ASTRAL"""
        for generator in self.code_generators.values():
            if hasattr(generator, 'generate_project_structure'):
                try:
                    result = await generator.generate_project_structure(**project_spec)
                    self.generated_projects.append({
                        "spec": project_spec,
                        "result": result,
                        "timestamp": time.time()
                    })
                    return result
                except:
                    continue
        
        return {"success": False, "error": "No code generators available"}

    async def call_external_api(self, api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Llama a una API externa"""
        for tool in self.external_api_tools.values():
            if hasattr(tool, 'execute'):
                try:
                    return await tool.execute(api_spec)
                except:
                    continue
        
        return {"success": False, "error": "No external API tools available"}

# --- 9. Motor de Simulación Integrado ---


@dataclass
class SimulationEngineV7:
    """Motor de simulación integrado con todos los subsistemas"""
    
    # Estado de simulación
    is_running: bool = False
    simulation_speed: float = 1.0
    tick_count: int = 0
    
    # Subsistemas integrados
    unified_field: UnifiedField = field(default_factory=UnifiedField)
    eva_integration: EVAIntegrationV7 = field(default_factory=EVAIntegrationV7)
    heaven_integration: HeavenIntegrationV7 = field(default_factory=HeavenIntegrationV7)
    earth_integration: EarthIntegrationV7 = field(default_factory=EarthIntegrationV7)
    astral_integration: AstralToolsIntegrationV7 = field(default_factory=AstralToolsIntegrationV7)

    def start_simulation(self) -> None:
        """Inicia la simulación"""
        self.is_running = True
        print("Simulación iniciada")

    def stop_simulation(self) -> None:
        """Detiene la simulación"""
        self.is_running = False
        print("Simulación detenida")

    def tick(self) -> Dict[str, Any]:
        """Ejecuta un tick de simulación completo"""
        if not self.is_running:
            return {"status": "stopped"}
        
        self.tick_count += 1
        
        # Tick del campo unificado
        field_state = self.unified_field.tick()
        
        # Ejecutar hooks de entorno de EVA
        for hook in self.eva_integration.environment_hooks:
            try:
                hook(field_state)
            except:
                pass
        
        return {
            "status": "running",
            "tick": self.tick_count,
            "field_state": field_state,
            "unified_field_state": self.unified_field.get_state()
        }

    def get_comprehensive_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo de todo el sistema"""
        return {
            "simulation": {
                "is_running": self.is_running,
                "tick_count": self.tick_count,
                "speed": self.simulation_speed
            },
            "unified_field": self.unified_field.get_state(),
            "eva": {
                "current_phase": self.eva_integration.current_phase,
                "experience_count": len(self.eva_integration.experience_storage),
                "hooks_count": len(self.eva_integration.environment_hooks)
            },
            "heaven": {
                "babel_agents": len(self.heaven_integration.babel_agents),
                "external_apis": len(self.heaven_integration.external_apis)
            },
            "earth": {
                "demiurge_avatars": len(self.earth_integration.demiurge_avatars),
                "demiurge_focus": self.earth_integration.demiurge_focus,
                "modifications_count": len(self.earth_integration.reality_modifications)
            },
            "astral": {
                "code_generators": len(self.astral_integration.code_generators),
                "projects_generated": len(self.astral_integration.generated_projects)
            }
        }

# --- Continuación de componentes base (heredados) ---

# Mantener los componentes base ya definidos con actualizaciones menores
@dataclass
class QualiaChain:
    """Cadena de bloques para la economía de la experiencia - Expandida"""

    transactions: List["SignedTransaction"] = field(default_factory=list)
    balances: Dict[str, float] = field(default_factory=dict)
    block_height: int = 0
    
    # Integración con sistemas del proyecto
    validator_nodes: List[str] = field(default_factory=list)
    smart_contracts: Dict[str, Any] = field(default_factory=dict)

    def submit_transaction(self, signed_transaction: "SignedTransaction") -> str:
        """Somete una transacción a la cadena"""
        if not self._validate_transaction(signed_transaction):
            raise ValueError("Transacción inválida")

        self.transactions.append(signed_transaction)
        self.block_height += 1

        # Actualizar balances si es una transferencia
        if "from" in signed_transaction.data and "to" in signed_transaction.data:
            amount = signed_transaction.data.get("amount", 0)
            from_addr = signed_transaction.data["from"]
            to_addr = signed_transaction.data["to"]

            if from_addr in self.balances:
                self.balances[from_addr] -= amount
            if to_addr not in self.balances:
                self.balances[to_addr] = 0
            self.balances[to_addr] += amount

        tx_hash = f"tx_{self.block_height}_{hash(str(signed_transaction))}"
        print(f"Transacción {tx_hash} añadida a QualiaChain")
        return tx_hash

    def get_balance(self, address: str) -> float:
        """Devuelve el balance de Qualia-Coin"""
        return self.balances.get(address, 0.0)

    def _validate_transaction(self, transaction: "SignedTransaction") -> bool:
        """Valida una transacción firmada"""
        return (transaction.signature is not None and 
                transaction.signer is not None and 
                transaction.data is not None)

@dataclass
class Noosphere:
    """Red noosférica para la cultura y experiencia compartida - Expandida"""

    crystallized_experiences: List["CrystallizedQualia"] = field(default_factory=list)
    active_streams: Dict[str, "Stream"] = field(default_factory=dict)
    playlists: Dict[str, "Playlist"] = field(default_factory=dict)
    
    # Características expandidas
    cultural_artifacts: Dict[str, Any] = field(default_factory=dict)
    collective_memories: List[Dict[str, Any]] = field(default_factory=list)
    wisdom_network: Dict[str, float] = field(default_factory=dict)

    def crystallize_experience(self, qualia_state: "QualiaState", owner_address: str) -> "CrystallizedQualia":
        """Convierte un estado de qualia en un activo persistente"""
        crystallized_qualia = CrystallizedQualia(
            id=f"crystal_{len(self.crystallized_experiences)}_{time.time()}",
            owner_address=owner_address,
            qualia_signature=qualia_state.__dict__ if hasattr(qualia_state, '__dict__') else {},
            timestamp=time.time(),
        )

        self.crystallized_experiences.append(crystallized_qualia)
        print(f"Experiencia cristalizada: {crystallized_qualia.id}")
        return crystallized_qualia

    def start_qualia_stream(self, entity_id: str) -> "Stream":
        """Inicia una transmisión en vivo del qualia"""
        stream = Stream(
            stream_id=f"stream_{entity_id}_{time.time()}",
            entity_id=entity_id,
            start_time=time.time(),
            is_active=True,
        )

        self.active_streams[stream.stream_id] = stream
        print(f"Stream de qualia iniciado: {stream.stream_id}")
        return stream

    def create_playlist(self, crystallized_qualia_ids: List[str]) -> "Playlist":
        """Crea una lista de reproducción de experiencias"""
        playlist = Playlist(
            playlist_id=f"playlist_{len(self.playlists)}_{time.time()}",
            crystallized_qualia_ids=crystallized_qualia_ids,
            creation_time=time.time(),
        )

        self.playlists[playlist.playlist_id] = playlist
        print(f"Playlist creada: {playlist.playlist_id} con {len(crystallized_qualia_ids)} experiencias")
        return playlist

    def add_cultural_artifact(self, artifact_id: str, artifact_data: Dict[str, Any]) -> None:
        """Añade un artefacto cultural a la noosfera"""
        self.cultural_artifacts[artifact_id] = {
            **artifact_data,
            "timestamp": time.time(),
            "contributors": []
        }

# Continuar con el resto de componentes usando el mismo patrón...

# --- Mantener todas las clases de soporte existentes ---


@dataclass
class SelfModifyingEngine:
    """Motor para la auto-modificación del sistema - Expandido"""

    evolution_history: List[Dict] = field(default_factory=list)
    mutation_rate: float = 0.1
    
    # Características expandidas  
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    genetic_code: Dict[str, Any] = field(default_factory=dict)
    modification_proposals: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Inicializa constraints de seguridad"""
        self.safety_constraints = {
            "max_changes_per_cycle": 3,
            "max_risk_level": "moderate",
            "cooldown_between_modifications": 60,
            "immutable_core_functions": [
                "unified_field.add_entity",
                "qualia_chain.validate_transaction",
                "eva_integration.ingest_experience"
            ]
        }

    def evolve_system_performance(self, target_modules: List[str], mutation_type: str = "refactor") -> Dict:
        """Evoluciona el sistema para mejorar rendimiento"""
        # Verificar constraints de seguridad
        if len(target_modules) > self.safety_constraints["max_changes_per_cycle"]:
            return {"success": False, "error": "Too many modules targeted"}
        
        evolution_result = {
            "target_modules": target_modules,
            "mutation_type": mutation_type,
            "success": True,
            "changes_applied": [],
            "performance_improvement": 0.15,
            "timestamp": time.time(),
            "safety_check": "passed"
        }

        for module in target_modules:
            change = {
                "module": module,
                "change_type": mutation_type,
                "complexity_reduction": 0.1,
                "efficiency_gain": 0.2,
            }
            cast(List[Dict], evolution_result["changes_applied"]).append(change)

        self.evolution_history.append(evolution_result)
        print(f"Evolución aplicada a módulos: {target_modules}")
        return evolution_result

# Mantener el resto de las clases existentes...
@dataclass
class QualiaState:
    """Estado de qualia completo"""
    emotional_valence: float = 0.0
    cognitive_complexity: float = 0.0  
    temporal_coherence: float = 0.0
    consciousness_density: float = 0.0
    causal_potency: float = 0.0
    emergence_tendency: float = 0.0
    chaos_affinity: float = 0.0
    meta_awareness: float = 0.0

@dataclass
class SistemaDeChakras:
    """Sistema de chakras para el núcleo espiritual"""
    chakras: Dict[str, "Chakra"] = field(default_factory=dict)

    def __post_init__(self):
        chakra_names = ["root", "sacral", "solar_plexus", "heart", "throat", "third_eye", "crown"]
        for i, name in enumerate(chakra_names):
            self.chakras[name] = Chakra(
                name=name, position=i, activation=0.5,
                element=["earth", "water", "fire", "air", "sound", "light", "space"][i],
                color=["red", "orange", "yellow", "green", "blue", "indigo", "violet"][i],
            )


@dataclass
class SoulKernel:
    """Núcleo espiritual de la entidad"""
    karma_balance: float = 0.0
    chakras: SistemaDeChakras = field(default_factory=SistemaDeChakras)
    spiritual_attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.spiritual_attributes.update({
            "wisdom": 0.5, "compassion": 0.5, "courage": 0.5,
            "intuition": 0.5, "enlightenment": 0.1,
        })


@dataclass
class Proposito:
    """Sistema de propósito y dirección vital"""
    current_purpose: str = "Descubrir mi propósito en el universo"
    purpose_history: List[str] = field(default_factory=list)
    purpose_strength: float = 0.5

    def set_purpose(self, new_purpose: str) -> None:
        if self.current_purpose:
            self.purpose_history.append(self.current_purpose)
        self.current_purpose = new_purpose
        self.purpose_strength = 1.0
        print(f"Nuevo propósito establecido: {new_purpose}")


@dataclass
class Chakra:
    """Centro de energía individual"""
    name: str
    position: int
    activation: float = 0.0
    element: str = "earth"
    color: str = "red"


@dataclass
class GenomaComportamiento:
    """Genoma que define patrones de comportamiento"""
    genes: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        behavior_genes = ["curiosity", "aggression", "empathy", "creativity", 
                         "logic", "intuition", "social_bonding", "self_preservation"]
        for gene in behavior_genes:
            self.genes[gene] = 0.5


@dataclass
class CrystallizedQualia:
    """Experiencia qualia cristalizada como activo ontológico"""
    id: str
    owner_address: str
    qualia_signature: Dict
    timestamp: float
    value: float = 1.0


@dataclass
class SignedTransaction:
    """Transacción firmada para la QualiaChain"""
    data: Dict
    signer: str
    signature: str
    timestamp: float


@dataclass 
class Stream:
    """Stream de qualia en tiempo real"""
    stream_id: str
    entity_id: str
    start_time: float
    is_active: bool = True
    viewers: int = 0


@dataclass
class Playlist:
    """Lista de reproducción de experiencias cristalizadas"""
    playlist_id: str
    crystallized_qualia_ids: List[str]
    creation_time: float
    title: str = "Untitled Playlist"

# --- Clases abstractas para futuras implementaciones ---

class ConsciousMind(ABC):
    """Mente consciente abstracta"""
    @abstractmethod
    def think(self, input_data: Any) -> Any:
        pass

class AwakeningProtocolEngine(ABC):
    """Motor de protocolos de despertar abstracto"""
    @abstractmethod
    def initiate_awakening(self, entity: LivingEntity) -> bool:
        pass

class HalOntologica(ABC):
    """HAL Ontológica abstracta"""
    @abstractmethod
    def assist_entity(self, entity: LivingEntity) -> Dict[str, Any]:
        pass

class ExternalRealityExplorer(ABC):
    """Explorador de realidad externa abstracto"""
    @abstractmethod
    def explore_reality(self, coordinates: List[float]) -> Dict[str, Any]:
        pass

# --- API Pública Unificada ---

class RealityAPIV7:
    """
    API pública unificada para interactuar con toda la realidad del Metacosmos
    Integra todos los subsistemas del proyecto Crisalida
    """
    
    def __init__(self):
        # Motor de simulación central
        self.simulation_engine = SimulationEngineV7()
        
        # Referencias directas para conveniencia
        self.unified_field = self.simulation_engine.unified_field
        self.eva = self.simulation_engine.eva_integration
        self.heaven = self.simulation_engine.heaven_integration
        self.earth = self.simulation_engine.earth_integration
        self.astral = self.simulation_engine.astral_integration
        
        # Inicializar subsistemas
        self._initialize_subsystems()

    def _initialize_subsystems(self) -> None:
        """Inicializa todos los subsistemas disponibles"""
        self.unified_field.initialize_eden_subsystems()
        self.unified_field.initialize_eva_subsystems()
        print("RealityAPIV7 inicializada con todos los subsistemas integrados")

    # === MÉTODOS DE SIMULACIÓN ===
    
    def start_simulation(self) -> None:
        """Inicia la simulación global"""
        self.simulation_engine.start_simulation()

    def stop_simulation(self) -> None:
        """Detiene la simulación global"""
        self.simulation_engine.stop_simulation()

    def tick(self) -> Dict[str, Any]:
        """Ejecuta un tick de simulación"""
        return self.simulation_engine.tick()

    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema"""
        return self.simulation_engine.get_comprehensive_state()

    def get_render_data(self) -> Dict[str, Any]:
        """Obtiene datos para renderizado"""
        return self.unified_field.get_render_data()

    # === MÉTODOS DE ENTIDADES ===
    
    def add_entity(self, entity: LivingEntity) -> None:
        """Añade una entidad al universo"""
        self.unified_field.add_entity(entity)

    def remove_entity(self, entity_id: str) -> bool:
        """Elimina una entidad del universo"""
        return self.unified_field.remove_entity(entity_id)

    def get_entity_by_id(self, entity_id: str) -> Optional[LivingEntity]:
        """Obtiene una entidad por ID"""
        return self.unified_field.get_entity_by_id(entity_id)

    def query_entities(self, query: Dict) -> List[LivingEntity]:
        """Busca entidades con consulta"""
        return self.unified_field.query_entities(query)

    # === MÉTODOS DE EDEN ===
    
    def add_living_symbol(self, symbol: "LivingSymbol") -> None:
        """Añade un LivingSymbol"""
        self.unified_field.add_living_symbol(symbol)

    def add_cosmic_node(self, node: "CosmicNode") -> None:
        """Añade un CosmicNode"""
        self.unified_field.add_cosmic_node(node)

    def manifest_living_symbol(self, pattern_addresses: List[Any], pattern_type: str = "default") -> None:
        """Manifiesta un LivingSymbol en el campo"""
        if self.unified_field.qualia_engine and hasattr(self.unified_field.qualia_engine, 'manifest_living_symbol'):
            try:
                self.unified_field.qualia_engine.manifest_living_symbol(pattern_addresses, pattern_type)
            except:
                print(f"Error manifestando símbolo tipo {pattern_type}")

    # === MÉTODOS DE EVA ===
    
    def eva_ingest_experience(self, experience_data: Dict, qualia_state: QualiaState, phase: str = None) -> str:
        """Ingiere una experiencia en EVA"""
        return self.eva.eva_ingest_experience(experience_data, qualia_state, phase)

    def eva_recall_experience(self, experience_id: str) -> Optional[Any]:
        """Recupera una experiencia de EVA"""
        return self.eva.eva_recall_experience(experience_id)

    def set_memory_phase(self, phase: str) -> None:
        """Cambia la fase de memoria EVA"""
        self.eva.set_memory_phase(phase)

    def get_memory_phase(self) -> str:
        """Obtiene la fase de memoria actual"""
        return self.eva.get_memory_phase()

    def add_environment_hook(self, hook: callable) -> None:
        """Añade un hook de entorno"""
        self.eva.add_environment_hook(hook)

    # === MÉTODOS DE HEAVEN ===
    
    def add_babel_agent(self, agent_id: str, agent: "BabelAgent") -> None:
        """Añade un agente Babel"""
        self.heaven.add_babel_agent(agent_id, agent)

    async def create_library(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Crea una biblioteca usando Babel"""
        return await self.heaven.create_library(spec)

    def get_social_network_state(self) -> Dict[str, Any]:
        """Obtiene el estado de la red social"""
        return self.heaven.get_social_network_state()

    # === MÉTODOS DE EARTH ===
    
    def add_demiurge_avatar(self, avatar_id: str, avatar: "DemiurgeAvatar") -> None:
        """Añade un avatar demiurgo"""
        self.earth.add_demiurge_avatar(avatar_id, avatar)

    async def execute_demiurge_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un comando del demiurgo"""
        return await self.earth.execute_demiurge_command(command)

    def update_demiurge_focus(self, position: Tuple[float, float, float]) -> None:
        """Actualiza el foco del demiurgo"""
        self.earth.update_demiurge_focus(position)

    # === MÉTODOS DE ASTRAL_TOOLS ===
    
    async def generate_project(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un proyecto"""
        return await self.astral.generate_project(project_spec)

    async def call_external_api(self, api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Llama a una API externa"""
        return await self.astral.call_external_api(api_spec)
