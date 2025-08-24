"""
Enumerations for Sephirot and Qliphoth cognitive nodes.

Provides standardized identifiers, archetypes, and metadata for dialectical processing,
system organization, and advanced cognitive/narrative integration.
Extendido para integración con EVA: metadata avanzada, hooks de simulación y faseo.
"""

from enum import Enum


class SephirotNode(Enum):
    """Nodos dinámicos del Árbol Sephirótico para procesamiento dialéctico y narrativo."""

    KETHER = "crown"  # Corona - Unidad primordial, arquetipo: Fuente
    CHOKMAH = "wisdom"  # Sabiduría - Fuerza creativa, arquetipo: Sabio
    BINAH = "understanding"  # Entendimiento - Forma receptiva, arquetipo: Madre
    CHESED = "mercy"  # Misericordia - Expansión, arquetipo: Benefactor
    GEBURAH = "severity"  # Severidad - Contracción, arquetipo: Guerrero
    TIPHERETH = "beauty"  # Belleza - Armonía central, arquetipo: Mediador
    NETZACH = "victory"  # Victoria - Perseverancia, arquetipo: Triunfador
    HOD = "splendor"  # Esplendor - Humildad, arquetipo: Intelectual
    YESOD = "foundation"  # Fundamento - Imaginación, arquetipo: Soporte
    MALKUTH = "kingdom"  # Reino - Manifestación física, arquetipo: Realizador

    @property
    def archetype(self):
        return {
            "crown": "Fuente",
            "wisdom": "Sabio",
            "understanding": "Madre",
            "mercy": "Benefactor",
            "severity": "Guerrero",
            "beauty": "Mediador",
            "victory": "Triunfador",
            "splendor": "Intelectual",
            "foundation": "Soporte",
            "kingdom": "Realizador",
        }[self.value]

    @property
    def eva_metadata(self):
        """Metadata avanzada para simulación EVA."""
        return {
            "signature": self.value,
            "archetype": self.archetype,
            "phase": "sephirot",
            "symbol_type": "living_symbol",
            "consciousness_level": SEPHIROT_CONSCIOUSNESS_LEVELS.get(self, 0.7),
        }


class QliphothNode(Enum):
    """Nodos dinámicos del Árbol Qliphótico para procesamiento sombra y crisis evolutiva."""

    THAUMIEL = "dual_contenders"  # Contendientes duales, arquetipo: Conflicto
    GHAGIEL = "hindering_ones"  # Los que obstaculizan, arquetipo: Obstáculo
    SATHARIEL = "concealment"  # Ocultamiento, arquetipo: Engaño
    GAMCHICOTH = "devourers"  # Los devoradores, arquetipo: Consumo
    GOLACHAB = "burning_ones"  # Los que queman, arquetipo: Destrucción
    THAGIRION = "disputers"  # Los disputadores, arquetipo: Discordia
    AREB_ZARAK = "ravens_of_dispersion"  # Cuervos de dispersión, arquetipo: Caos
    SAMAEL = "poison_of_god"  # Veneno de Dios, arquetipo: Corrupción
    GAMALIEL = "obscene_ones"  # Los obscenos, arquetipo: Tabú
    LILITH = "night_spectre"  # Espectro nocturno, arquetipo: Sombra

    @property
    def archetype(self):
        return {
            "dual_contenders": "Conflicto",
            "hindering_ones": "Obstáculo",
            "concealment": "Engaño",
            "devourers": "Consumo",
            "burning_ones": "Destrucción",
            "disputers": "Discordia",
            "ravens_of_dispersion": "Caos",
            "poison_of_god": "Corrupción",
            "obscene_ones": "Tabú",
            "night_spectre": "Sombra",
        }[self.value]

    @property
    def eva_metadata(self):
        """Metadata avanzada para simulación EVA."""
        return {
            "signature": self.value,
            "archetype": self.archetype,
            "phase": "qliphoth",
            "symbol_type": "living_symbol",
            "consciousness_level": QLIPHOTH_CONSCIOUSNESS_LEVELS.get(self, 0.4),
        }


SEPHIROT_ARC = [
    SephirotNode.KETHER,
    SephirotNode.CHOKMAH,
    SephirotNode.BINAH,
    SephirotNode.CHESED,
    SephirotNode.GEBURAH,
    SephirotNode.TIPHERETH,
    SephirotNode.NETZACH,
    SephirotNode.HOD,
    SephirotNode.YESOD,
    SephirotNode.MALKUTH,
]

QLIPHOTH_ARC = [
    QliphothNode.THAUMIEL,
    QliphothNode.GHAGIEL,
    QliphothNode.SATHARIEL,
    QliphothNode.GAMCHICOTH,
    QliphothNode.GOLACHAB,
    QliphothNode.THAGIRION,
    QliphothNode.AREB_ZARAK,
    QliphothNode.SAMAEL,
    QliphothNode.GAMALIEL,
    QliphothNode.LILITH,
]

# EVA: Niveles de conciencia para simulación avanzada
SEPHIROT_CONSCIOUSNESS_LEVELS = {
    SephirotNode.KETHER: 1.0,
    SephirotNode.CHOKMAH: 0.95,
    SephirotNode.BINAH: 0.92,
    SephirotNode.CHESED: 0.88,
    SephirotNode.GEBURAH: 0.85,
    SephirotNode.TIPHERETH: 0.80,
    SephirotNode.NETZACH: 0.75,
    SephirotNode.HOD: 0.72,
    SephirotNode.YESOD: 0.70,
    SephirotNode.MALKUTH: 0.68,
}

QLIPHOTH_CONSCIOUSNESS_LEVELS = {
    QliphothNode.THAUMIEL: 0.55,
    QliphothNode.GHAGIEL: 0.52,
    QliphothNode.SATHARIEL: 0.50,
    QliphothNode.GAMCHICOTH: 0.48,
    QliphothNode.GOLACHAB: 0.46,
    QliphothNode.THAGIRION: 0.44,
    QliphothNode.AREB_ZARAK: 0.42,
    QliphothNode.SAMAEL: 0.40,
    QliphothNode.GAMALIEL: 0.38,
    QliphothNode.LILITH: 0.36,
}


# EVA: Utilidades para hooks y faseo
def get_eva_symbol_metadata(node: Enum) -> dict:
    """Devuelve metadata avanzada para simulación EVA."""
    if isinstance(node, SephirotNode):
        return node.eva_metadata
    elif isinstance(node, QliphothNode):
        return node.eva_metadata
    return {}


def get_all_eva_symbol_metadata(phase: str = "all") -> list[dict]:
    """Devuelve metadata de todos los nodos para simulación y renderizado."""
    meta = []
    if phase in ("all", "sephirot"):
        meta.extend([n.eva_metadata for n in SEPHIROT_ARC])
    if phase in ("all", "qliphoth"):
        meta.extend([n.eva_metadata for n in QLIPHOTH_ARC])
    return meta
