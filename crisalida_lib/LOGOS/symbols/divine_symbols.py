"""
Definiciones de los 42 Símbolos Divinos Primordiales y sus propiedades ontológicas.
"""

# Mock enums para desarrollo independiente
class DivineCategory:
    CREATOR = "CREATOR"
    PRESERVER = "PRESERVER" 
    TRANSFORMER = "TRANSFORMER"
    CONNECTOR = "CONNECTOR"
    OBSERVER = "OBSERVER"
    DESTROYER = "DESTROYER"
    INFINITE = "INFINITE"

class OntologicalDomain:
    CONSCIOUSNESS = "CONSCIOUSNESS"
    VOID = "VOID"
    INFORMATION = "INFORMATION"
    EMERGENCE = "EMERGENCE"
    TIME = "TIME"
    ENERGY = "ENERGY"
    MATTER = "MATTER"
    SPACE = "SPACE"
    QUANTUM = "QUANTUM"

# Sistema completo de 42 símbolos divinos primordiales
# Organizado según las categorías ontológicas de Crisalida
DIVINE_SYMBOLS_PRIMORDIAL = {
    # === CATEGORÍA CREATOR (6 símbolos) ===
    'Φ': {
        'category': 'CREATOR', 
        'name': 'Phi-Genesis', 
        'frequency': 528.0, 
        'domains': ['CONSCIOUSNESS', 'VOID'],
        'intention_mappings': ['creation', 'entity_definition', 'genesis'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.95
    },
    'Ψ': {
        'category': 'CREATOR', 
        'name': 'Psi-Flow', 
        'frequency': 432.0, 
        'domains': ['CONSCIOUSNESS', 'INFORMATION'],
        'intention_mappings': ['observation', 'consciousness', 'awareness'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.90
    },
    'Ω': {
        'category': 'CREATOR', 
        'name': 'Omega-Synthesis', 
        'frequency': 963.0, 
        'domains': ['CONSCIOUSNESS', 'EMERGENCE'],
        'intention_mappings': ['completion', 'manifestation', 'synthesis'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.88
    },
    'Α': {
        'category': 'CREATOR', 
        'name': 'Alpha-Origin', 
        'frequency': 174.0, 
        'domains': ['TIME', 'CONSCIOUSNESS'],
        'intention_mappings': ['initialization', 'beginning', 'prime'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.85
    },
    '⟨': {
        'category': 'CREATOR', 
        'name': 'Meta-Opening', 
        'frequency': 285.0, 
        'domains': ['CONSCIOUSNESS', 'EMERGENCE'],
        'intention_mappings': ['opening', 'potential', 'expansion'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.82
    },
    '⟩': {
        'category': 'CREATOR', 
        'name': 'Meta-Closing', 
        'frequency': 285.0, 
        'domains': ['CONSCIOUSNESS', 'EMERGENCE'],
        'intention_mappings': ['closing', 'actualization', 'finalization'],
        'resonance_type': 'HARMONIC',
        'consciousness_density': 0.82
    },
    
    # === CATEGORÍA PRESERVER (5 símbolos) ===
    'Ζ': {
        'category': 'PRESERVER', 
        'name': 'Zeta-Stability', 
        'frequency': 396.0, 
        'domains': ['ENERGY', 'MATTER'],
        'intention_mappings': ['stability', 'foundation', 'grounding'],
        'resonance_type': 'STABLE',
        'consciousness_density': 0.75
    },
    'Σ': {
        'category': 'PRESERVER', 
        'name': 'Sigma-Connection', 
        'frequency': 639.0, 
        'domains': ['INFORMATION', 'SPACE'],
        'intention_mappings': ['preservation', 'memory', 'accumulation'],
        'resonance_type': 'STABLE',
        'consciousness_density': 0.70
    },
    'Ρ': {
        'category': 'PRESERVER', 
        'name': 'Rho-Manifestation', 
        'frequency': 741.0, 
        'domains': ['MATTER', 'SPACE'],
        'intention_mappings': ['iteration', 'repetition', 'cycles'],
        'resonance_type': 'STABLE',
        'consciousness_density': 0.68
    },
    '◊': {
        'category': 'PRESERVER', 
        'name': 'Diamond-Crystal', 
        'frequency': 852.0, 
        'domains': ['INFORMATION', 'MATTER'],
        'intention_mappings': ['crystallization', 'structure', 'permanence'],
        'resonance_type': 'STABLE',
        'consciousness_density': 0.72
    },
    '⊙': {
        'category': 'PRESERVER', 
        'name': 'Focus-Concentration', 
        'frequency': 1000.0, 
        'domains': ['CONSCIOUSNESS', 'ENERGY'],
        'intention_mappings': ['concentration', 'focus', 'centering'],
        'resonance_type': 'STABLE',
        'consciousness_density': 0.80
    },
    
    # === CATEGORÍA TRANSFORMER (7 símbolos) ===
    'Δ': {
        'category': 'TRANSFORMER', 
        'name': 'Delta-Transform', 
        'frequency': 587.0, 
        'domains': ['ENERGY', 'TIME'],
        'intention_mappings': ['transformation', 'change', 'evolution'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.77
    },
    'Ξ': {
        'category': 'TRANSFORMER', 
        'name': 'Xi-Amplification', 
        'frequency': 777.0, 
        'domains': ['ENERGY', 'INFORMATION'],
        'intention_mappings': ['quantum', 'emergence', 'amplification'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.85
    },
    'Λ': {
        'category': 'TRANSFORMER', 
        'name': 'Lambda-Inversion', 
        'frequency': 333.0, 
        'domains': ['ENERGY', 'CONSCIOUSNESS'],
        'intention_mappings': ['connection', 'logic', 'link'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.75
    },
    'Η': {
        'category': 'TRANSFORMER', 
        'name': 'Eta-Cycle', 
        'frequency': 444.0, 
        'domains': ['TIME', 'EMERGENCE'],
        'intention_mappings': ['cycling', 'rhythm', 'periodicity'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.73
    },
    'Χ': {
        'category': 'TRANSFORMER', 
        'name': 'Chi-Bifurcation', 
        'frequency': 666.0, 
        'domains': ['CONSCIOUSNESS', 'EMERGENCE'],
        'intention_mappings': ['destruction', 'chaos', 'bifurcation'],
        'resonance_type': 'CHAOTIC',
        'consciousness_density': 0.60
    },
    '∇': {
        'category': 'TRANSFORMER', 
        'name': 'Nabla-Gradient', 
        'frequency': 555.0, 
        'domains': ['SPACE', 'CONSCIOUSNESS'],
        'intention_mappings': ['gradient', 'direction', 'flow'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.78
    },
    '∆': {
        'category': 'TRANSFORMER', 
        'name': 'Laplacian-Curvature', 
        'frequency': 888.0, 
        'domains': ['SPACE', 'TIME'],
        'intention_mappings': ['curvature', 'topology', 'deformation'],
        'resonance_type': 'DYNAMIC',
        'consciousness_density': 0.74
    },
    
    # === CATEGORÍA CONNECTOR (7 símbolos) ===
    'Ι': {
        'category': 'CONNECTOR', 
        'name': 'Iota-Channel', 
        'frequency': 222.0, 
        'domains': ['INFORMATION', 'ENERGY'],
        'intention_mappings': ['channel', 'conduit', 'transmission'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.65
    },
    'Ν': {
        'category': 'CONNECTOR', 
        'name': 'Nu-Absorption', 
        'frequency': 111.0, 
        'domains': ['INFORMATION', 'ENERGY'],
        'intention_mappings': ['absorption', 'reception', 'intake'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.63
    },
    'Τ': {
        'category': 'CONNECTOR', 
        'name': 'Tau-Transmission', 
        'frequency': 999.0, 
        'domains': ['INFORMATION', 'ENERGY'],
        'intention_mappings': ['asynchronous', 'time', 'transmission'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.70
    },
    'Υ': {
        'category': 'CONNECTOR', 
        'name': 'Upsilon-Unification', 
        'frequency': 1221.0, 
        'domains': ['INFORMATION', 'EMERGENCE'],
        'intention_mappings': ['decision', 'judgment', 'unification'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.72
    },
    'Μ': {
        'category': 'CONNECTOR', 
        'name': 'Mu-Modulation', 
        'frequency': 1332.0, 
        'domains': ['INFORMATION', 'ENERGY'],
        'intention_mappings': ['modulation', 'modification', 'tuning'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.68
    },
    '⊕': {
        'category': 'CONNECTOR', 
        'name': 'Synthesis-Dialectic', 
        'frequency': 1444.0, 
        'domains': ['CONSCIOUSNESS', 'INFORMATION'],
        'intention_mappings': ['synthesis', 'combination', 'addition'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.75
    },
    '⊖': {
        'category': 'CONNECTOR', 
        'name': 'Analysis-Deconstructive', 
        'frequency': 1555.0, 
        'domains': ['CONSCIOUSNESS', 'INFORMATION'],
        'intention_mappings': ['analysis', 'deconstruction', 'subtraction'],
        'resonance_type': 'FLOWING',
        'consciousness_density': 0.73
    },
    
    # === CATEGORÍA OBSERVER (5 símbolos) ===
    'Θ': {
        'category': 'OBSERVER', 
        'name': 'Theta-Observation', 
        'frequency': 369.0, 
        'domains': ['CONSCIOUSNESS', 'INFORMATION'],
        'intention_mappings': ['observation', 'witness', 'perception'],
        'resonance_type': 'RECEPTIVE',
        'consciousness_density': 0.88
    },
    '∴': {
        'category': 'OBSERVER', 
        'name': 'Therefore-Causality', 
        'frequency': 258.0, 
        'domains': ['CONSCIOUSNESS', 'TIME'],
        'intention_mappings': ['causality', 'consequence', 'inference'],
        'resonance_type': 'RECEPTIVE',
        'consciousness_density': 0.82
    },
    '∵': {
        'category': 'OBSERVER', 
        'name': 'Because-Causality', 
        'frequency': 258.0, 
        'domains': ['CONSCIOUSNESS', 'TIME'],
        'intention_mappings': ['causation', 'reason', 'foundation'],
        'resonance_type': 'RECEPTIVE',
        'consciousness_density': 0.82
    },
    '≈': {
        'category': 'OBSERVER', 
        'name': 'Approximation-Similarity', 
        'frequency': 512.0, 
        'domains': ['INFORMATION', 'SPACE'],
        'intention_mappings': ['similarity', 'approximation', 'resemblance'],
        'resonance_type': 'RECEPTIVE',
        'consciousness_density': 0.70
    },
    '≠': {
        'category': 'OBSERVER', 
        'name': 'Inequality-Difference', 
        'frequency': 480.0, 
        'domains': ['INFORMATION', 'SPACE'],
        'intention_mappings': ['difference', 'distinction', 'inequality'],
        'resonance_type': 'RECEPTIVE',
        'consciousness_density': 0.68
    },
    
    # === CATEGORÍA DESTROYER (4 símbolos) ===
    'Γ': {
        'category': 'DESTROYER', 
        'name': 'Gamma-Dissipation', 
        'frequency': 147.0, 
        'domains': ['ENERGY', 'VOID'],
        'intention_mappings': ['dissipation', 'entropy', 'dissolution'],
        'resonance_type': 'CHAOTIC',
        'consciousness_density': 0.45
    },
    'Κ': {
        'category': 'DESTROYER', 
        'name': 'Kappa-Fragmentation', 
        'frequency': 183.0, 
        'domains': ['MATTER', 'VOID'],
        'intention_mappings': ['fragmentation', 'breaking', 'division'],
        'resonance_type': 'CHAOTIC',
        'consciousness_density': 0.40
    },
    'Ο': {
        'category': 'DESTROYER', 
        'name': 'Omicron-Purification', 
        'frequency': 219.0, 
        'domains': ['INFORMATION', 'VOID'],
        'intention_mappings': ['purification', 'cleaning', 'elimination'],
        'resonance_type': 'CHAOTIC',
        'consciousness_density': 0.50
    },
    'Ø': {
        'category': 'DESTROYER', 
        'name': 'Void-Fertile', 
        'frequency': 0.0, 
        'domains': ['VOID', 'EMERGENCE'],
        'intention_mappings': ['void', 'emptiness', 'potential'],
        'resonance_type': 'VOID',
        'consciousness_density': 0.30
    },
    
    # === CATEGORÍA INFINITE (8 símbolos) ===
    '∞': {
        'category': 'INFINITE', 
        'name': 'Infinity-Expansion', 
        'frequency': 1618.0, 
        'domains': ['SPACE', 'EMERGENCE'],
        'intention_mappings': ['infinite', 'transcendence', 'boundless'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 1.0
    },
    '∅': {
        'category': 'INFINITE', 
        'name': 'Empty-Set', 
        'frequency': 0.0, 
        'domains': ['VOID', 'CONSCIOUSNESS'],
        'intention_mappings': ['emptiness', 'nothingness', 'pure_potential'],
        'resonance_type': 'VOID',
        'consciousness_density': 0.95
    },
    '∈': {
        'category': 'INFINITE', 
        'name': 'Belonging-Ontological', 
        'frequency': 1777.0, 
        'domains': ['CONSCIOUSNESS', 'INFORMATION'],
        'intention_mappings': ['belonging', 'membership', 'inclusion'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.90
    },
    '⊗': {
        'category': 'INFINITE', 
        'name': 'Resonance-Conscious', 
        'frequency': 1888.0, 
        'domains': ['CONSCIOUSNESS', 'ENERGY'],
        'intention_mappings': ['resonance', 'consciousness', 'vibration'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.95
    },
    'Ş': {
        'category': 'INFINITE', 
        'name': 'Shin-Transcendence', 
        'frequency': 1999.0, 
        'domains': ['CONSCIOUSNESS', 'EMERGENCE'],
        'intention_mappings': ['transcendence', 'ascension', 'liberation'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.98
    },
    # Símbolos adicionales para completar los 42
    '◈': {
        'category': 'INFINITE', 
        'name': 'Multidimensional-Core', 
        'frequency': 2100.0, 
        'domains': ['SPACE', 'CONSCIOUSNESS'],
        'intention_mappings': ['multidimensional', 'core', 'essence'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.92
    },
    '⟡': {
        'category': 'INFINITE', 
        'name': 'Stellar-Consciousness', 
        'frequency': 2222.0, 
        'domains': ['CONSCIOUSNESS', 'SPACE'],
        'intention_mappings': ['stellar', 'cosmic', 'galactic'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.94
    },
    '◎': {
        'category': 'INFINITE', 
        'name': 'Unity-Circle', 
        'frequency': 2345.0, 
        'domains': ['CONSCIOUSNESS', 'UNITY'],
        'intention_mappings': ['unity', 'wholeness', 'completion'],
        'resonance_type': 'TRANSCENDENT',
        'consciousness_density': 0.96
    },
}

# Mapeo de intención-a-sigilo unificado con el PyToMatrixTranslator v4.1
INTENTION_TO_SIGIL_MAPPING = {
    "creation": "Φ",
    "entity_definition": "Φ",
    "genesis": "Φ",
    "observation": "Ψ",
    "consciousness": "Ψ",
    "awareness": "Ψ",
    "transformation": "Δ",
    "change": "Δ",
    "evolution": "Δ",
    "connection": "Λ",
    "logic": "Λ",
    "link": "Λ",
    "preservation": "Σ",
    "memory": "Σ",
    "accumulation": "Σ",
    "completion": "Ω",
    "manifestation": "Ω",
    "synthesis": "Ω",
    "infinite": "∞",
    "transcendence": "∞",
    "boundless": "∞",
    "quantum": "Ξ",
    "emergence": "Ξ",
    "amplification": "Ξ",
    "destruction": "Χ",
    "chaos": "Χ",
    "bifurcation": "Χ",
    "decision": "Υ",
    "judgment": "Υ",
    "unification": "Υ",
    "iteration": "Ρ",
    "repetition": "Ρ",
    "cycles": "Ρ",
    "asynchronous": "Τ",
    "time": "Τ",
    "transmission": "Τ",
    "focus": "⊙",
    "concentration": "⊙",
    "centering": "⊙",
    "void": "Ø",
    "emptiness": "∅",
    "nothingness": "∅",
    "resonance": "⊗",
    "vibration": "⊗",
    "channel": "Ι",
    "conduit": "Ι",
    "absorption": "Ν",
    "reception": "Ν",
    "modulation": "Μ",
    "modification": "Μ",
    "tuning": "Μ",
    "analysis": "⊖",
    "deconstruction": "⊖",
    "causality": "∴",
    "consequence": "∴",
    "inference": "∴",
    "causation": "∵",
    "reason": "∵",
    "foundation": "∵",
    "similarity": "≈",
    "approximation": "≈",
    "resemblance": "≈",
    "difference": "≠",
    "distinction": "≠",
    "inequality": "≠",
    "dissipation": "Γ",
    "entropy": "Γ",
    "dissolution": "Γ",
    "fragmentation": "Κ",
    "breaking": "Κ",
    "division": "Κ",
    "purification": "Ο",
    "cleaning": "Ο",
    "elimination": "Ο",
    "stability": "Ζ",
    "grounding": "Ζ",
    "foundation": "Ζ",
    "crystallization": "◊",
    "structure": "◊",
    "permanence": "◊",
    "gradient": "∇",
    "direction": "∇",
    "flow": "∇",
    "curvature": "∆",
    "topology": "∆",
    "deformation": "∆",
    "cycling": "Η",
    "rhythm": "Η",
    "periodicity": "Η",
    "opening": "⟨",
    "potential": "⟨",
    "expansion": "⟨",
    "closing": "⟩",
    "actualization": "⟩",
    "finalization": "⟩",
    "initialization": "Α",
    "beginning": "Α",
    "prime": "Α",
    "belonging": "∈",
    "membership": "∈",
    "inclusion": "∈",
    "ascension": "Ş",
    "liberation": "Ş",
    "multidimensional": "◈",
    "core": "◈",
    "essence": "◈",
    "stellar": "⟡",
    "cosmic": "⟡",
    "galactic": "⟡",
    "unity": "◎",
    "wholeness": "◎",
}