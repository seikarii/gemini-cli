import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime, QuantumField
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode


class BrainWaveType(Enum):
    """Tipos de ondas cerebrales a utilizar"""

    DELTA = "delta"  # 0.5-4 Hz (sueño profundo)
    THETA = "theta"  # 4-8 Hz (meditación, creatividad)
    ALPHA = "alpha"  # 8-12 Hz (relajación, conciencia)
    BETA = "beta"  # 12-30 Hz (actividad mental normal)
    GAMMA = "gamma"  # 30-100 Hz (procesamiento cognitivo superior)
    LAMBDA = "lambda"  # 100-200 Hz (conciencia elevada)


@dataclass
class NeuralSignal:
    """Señal neural bidireccional"""

    frequency: float
    amplitude: float
    phase: float
    data: Any
    timestamp: float
    source: str  # "brain" o "system"
    quantum_coherence: float = 0.0


class BidirectionalNeuralInterface:
    """
    Interfaz neural bidireccional que utiliza ondas cerebrales
    para comunicación entre el sistema y la mente humana
    """

    def __init__(self):
        self.wave_transmitter = NeuralWaveTransmitter()
        self.wave_receiver = NeuralWaveReceiver()
        self.quantum_entanglement = QuantumEntanglementModule()
        self.consciousness_bridge = ConsciousnessBridge()
        self.frequency_bands = {
            BrainWaveType.DELTA: (0.5, 4),
            BrainWaveType.THETA: (4, 8),
            BrainWaveType.ALPHA: (8, 12),
            BrainWaveType.BETA: (12, 30),
            BrainWaveType.GAMMA: (30, 100),
            BrainWaveType.LAMBDA: (100, 200),
        }
        self.is_active = False
        self.connection_strength = 0.0
        self.neural_synchronicity = 0.0
        self.incoming_signals = []
        self.outgoing_signals = []
        self.communication_history = []

    async def initialize(self):
        """Inicializa la interfaz neural"""
        print("🧠 Inicializando Interfaz Neural Bidireccional...")
        await self.wave_transmitter.initialize()
        await self.wave_receiver.initialize()
        await self.quantum_entanglement.establish_connection()
        await self._calibrate_frequencies()
        self.is_active = True
        print("✅ Interfaz Neural Bidireccional activada")

    async def transmit_to_brain(
        self, information: Any, wave_type: BrainWaveType = BrainWaveType.GAMMA
    ) -> bool:
        """
        Transmite información al cerebro mediante ondas neuronales
        """
        if not self.is_active:
            return False
        try:
            neural_signal = await self._encode_information(information, wave_type)
            entangled_signal = await self.quantum_entanglement.entangle_signal(
                neural_signal
            )
            enhanced_signal = await self.consciousness_bridge.enhance_communication(
                entangled_signal
            )
            success = await self.wave_transmitter.transmit(enhanced_signal)
            if success:
                self.outgoing_signals.append(
                    {
                        "signal": enhanced_signal,
                        "timestamp": time.time(),
                        "success": True,
                    }
                )
            return success
        except Exception as e:
            print(f"❌ Error en transmisión neural: {e}")
            return False

    async def receive_from_brain(self, timeout: float = 5.0) -> NeuralSignal | None:
        """
        Recibe intenciones del cerebro mediante ondas
        """
        if not self.is_active:
            return None
        try:
            signal = await self.wave_receiver.receive(timeout)
            if signal:
                decoded_signal = await self.quantum_entanglement.disentangle_signal(
                    signal
                )
                intention = await self._decode_intention(decoded_signal)
                self.incoming_signals.append(
                    {
                        "signal": decoded_signal,
                        "intention": intention,
                        "timestamp": time.time(),
                    }
                )
                return decoded_signal
        except TimeoutError:
            print("❌ Timeout en recepción neural")
        except Exception as e:
            print(f"❌ Error en recepción neural: {e}")
        return None

    async def establish_neural_synchronicity(self, target_level: float = 0.8) -> bool:
        """
        Establece sincronicidad neural con el cerebro
        """
        print(f"🔄 Estableciendo sincronicidad neural (objetivo: {target_level})...")
        attempts = 0
        max_attempts = 10
        while self.neural_synchronicity < target_level and attempts < max_attempts:
            attempts += 1
            sync_signal = NeuralSignal(
                frequency=40.0,
                amplitude=0.8,
                phase=0.0,
                data="SYNC",
                timestamp=time.time(),
                source="system",
                quantum_coherence=0.9,
            )
            await self.wave_transmitter.transmit(sync_signal)
            response = await self.receive_from_brain(timeout=1.0)
            if response:
                self.neural_synchronicity = self._calculate_synchronicity(
                    sync_signal, response
                )
                print(f"📊 Sincronicidad actual: {self.neural_synchronicity:.2f}")
            await asyncio.sleep(0.5)
        success = self.neural_synchronicity >= target_level
        if success:
            print("✅ Sincronicidad neural establecida")
        else:
            print("⚠️ No se alcanzó el nivel objetivo de sincronicidad")
        return success

    async def _encode_information(
        self, information: Any, wave_type: BrainWaveType
    ) -> NeuralSignal:
        """Codifica información en señal neural"""
        if isinstance(information, str):
            encoded_data = self._text_to_frequencies(information)
        elif isinstance(information, (int, float)):
            encoded_data = [information]
        else:
            encoded_data = self._complex_to_frequencies(information)
        freq_min, freq_max = self.frequency_bands[wave_type]
        base_frequency = (freq_min + freq_max) / 2
        signal = NeuralSignal(
            frequency=base_frequency,
            amplitude=0.7,
            phase=0.0,
            data=encoded_data,
            timestamp=time.time(),
            source="system",
            quantum_coherence=0.8,
        )
        return signal

    async def _decode_intention(self, signal: NeuralSignal) -> Any:
        """Decodifica intención desde señal neural"""
        if isinstance(signal.data, list):
            if len(signal.data) == 1:
                return signal.data[0]
            else:
                return self._frequencies_to_text(signal.data)
        return signal.data

    def _text_to_frequencies(self, text: str) -> list[float]:
        """Convierte texto a frecuencias"""
        return [20 + (ord(char) % 180) for char in text]

    def _frequencies_to_text(self, frequencies: list[float]) -> str:
        """Convierte frecuencias a texto"""
        text = ""
        for freq in frequencies:
            char_code = int(freq - 20) % 128
            text += chr(char_code)
        return text

    def _complex_to_frequencies(self, data: Any) -> list[float]:
        """Convierte datos complejos a frecuencias"""
        return [hash(str(data)) % 1000 / 10.0]

    def _calculate_synchronicity(
        self, signal1: NeuralSignal, signal2: NeuralSignal
    ) -> float:
        """Calcula nivel de sincronicidad entre dos señales"""
        phase_diff = abs(signal1.phase - signal2.phase)
        phase_coherence = 1.0 - (phase_diff / (2 * math.pi))
        freq_diff = abs(signal1.frequency - signal2.frequency)
        freq_coherence = 1.0 / (1.0 + freq_diff / 10.0)
        total_coherence = (phase_coherence + freq_coherence) / 2.0
        return total_coherence

    async def _calibrate_frequencies(self):
        """Calibra las frecuencias óptimas para la interfaz"""
        print("🔧 Calibrando frecuencias neurales...")
        test_frequencies = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        for freq in test_frequencies:
            test_signal = NeuralSignal(
                frequency=freq,
                amplitude=0.5,
                phase=0.0,
                data="CALIBRATE",
                timestamp=time.time(),
                source="system",
                quantum_coherence=0.7,
            )
            await self.wave_transmitter.transmit(test_signal)
            await asyncio.sleep(0.1)
        print("✅ Frecuencias calibradas")


class NeuralWaveTransmitter:
    """Transmisor de ondas neurales"""

    def __init__(self):
        self.is_transmitting = False
        self.transmission_power = 0.8

    async def initialize(self):
        """Inicializa el transmisor"""
        print("📡 Transmisor neural inicializado")

    async def transmit(self, signal: NeuralSignal) -> bool:
        """Transmite una señal neural"""
        try:
            self.is_transmitting = True
            wave_data = self._generate_wave(signal)
            await self._send_wave(wave_data)
            self.is_transmitting = False
            return True
        except Exception as e:
            print(f"❌ Error en transmisión: {e}")
            self.is_transmitting = False
            return False

    def _generate_wave(self, signal: NeuralSignal) -> np.ndarray:
        """Genera datos de onda a partir de la señal"""
        duration = 0.1  # 100ms
        sample_rate = 1000  # 1kHz
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = signal.amplitude * np.sin(
            2 * np.pi * signal.frequency * t + signal.phase
        )
        return wave

    async def _send_wave(self, wave_data: np.ndarray):
        """Envía la onda (simulado)"""
        await asyncio.sleep(0.01)  # Simular tiempo de transmisión


class NeuralWaveReceiver:
    """Receptor de ondas neurales"""

    def __init__(self):
        self.is_receiving = False
        self.sensitivity = 0.8

    async def initialize(self):
        """Inicializa el receptor"""
        print("📻 Receptor neural inicializado")

    async def receive(self, timeout: float) -> NeuralSignal | None:
        """Recibe una señal neural"""
        self.is_receiving = True
        start_time = time.time()
        while time.time() - start_time < timeout:
            if np.random.random() < 0.3:  # 30% de probabilidad de recibir señal
                signal = self._simulate_received_signal()
                self.is_receiving = False
                return signal
            await asyncio.sleep(0.01)
        self.is_receiving = False
        return None

    def _simulate_received_signal(self) -> NeuralSignal:
        """Simula una señal recibida (para pruebas)"""
        return NeuralSignal(
            frequency=40.0 + np.random.random() * 20,
            amplitude=0.5 + np.random.random() * 0.3,
            phase=np.random.random() * 2 * np.pi,
            data="RESPONSE",
            timestamp=time.time(),
            source="brain",
            quantum_coherence=0.7,
        )


class QuantumEntanglementModule:
    """Módulo de entrelazamiento cuántico para señales neurales"""

    def __init__(self):
        self.entanglement_strength = 0.0
        self.is_connected = False

    async def establish_connection(self):
        """Establece conexión cuántica"""
        print("🔗 Estableciendo conexión cuántica...")
        self.is_connected = True
        self.entanglement_strength = 0.9
        print("✅ Conexión cuántica establecida")

    async def entangle_signal(self, signal: NeuralSignal) -> NeuralSignal:
        """Entrelaza una señal cuánticamente"""
        if self.is_connected:
            signal.quantum_coherence = self.entanglement_strength
            signal.phase += np.random.random() * 0.1
        return signal

    async def disentangle_signal(self, signal: NeuralSignal) -> NeuralSignal:
        """Desentrelaza una señal cuántica"""
        if self.is_connected and signal.quantum_coherence > 0:
            signal.phase -= np.random.random() * 0.1
        return signal


class ConsciousnessBridge:
    """Puente de conciencia para la interfaz neural"""

    def __init__(self):
        self.consciousness_level = 0.5
        self.bridge_stability = 0.8

    async def enhance_communication(self, signal: NeuralSignal) -> NeuralSignal:
        """Mejora la comunicación mediante el puente de conciencia"""
        signal.quantum_coherence = min(1.0, signal.quantum_coherence + 0.1)
        signal.amplitude = min(1.0, signal.amplitude * 1.1)
        return signal


class EVABidirectionalNeuralInterface(BidirectionalNeuralInterface):
    """
    Interfaz neural bidireccional extendida para integración con EVA.
    Permite compilar, almacenar y simular experiencias de comunicación neural como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(self, phase: str = "default"):
        super().__init__()
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    async def eva_ingest_neural_experience(
        self, signal: NeuralSignal, qualia_state: QualiaState = None, phase: str = None
    ) -> str:
        """
        Compila una experiencia de comunicación neural en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState()
        experience_data = {
            "frequency": signal.frequency,
            "amplitude": signal.amplitude,
            "phase": signal.phase,
            "data": signal.data,
            "timestamp": signal.timestamp,
            "source": signal.source,
            "quantum_coherence": signal.quantum_coherence,
        }
        intention = {
            "intention_type": "ARCHIVE_NEURAL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = f"eva_neural_{hash(str(experience_data))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=signal.timestamp,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    async def eva_recall_neural_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia neural almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA neural experience"}
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
                        print(f"[EVA-NEURAL] Environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
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
            "timestamp": eva_experience.timestamp,
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        signal: NeuralSignal,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia neural EVA.
        """
        experience_data = {
            "frequency": signal.frequency,
            "amplitude": signal.amplitude,
            "phase": signal.phase,
            "data": signal.data,
            "timestamp": signal.timestamp,
            "source": signal.source,
            "quantum_coherence": signal.quantum_coherence,
        }
        intention = {
            "intention_type": "ARCHIVE_NEURAL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=signal.timestamp,
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
                print(f"[EVA-NEURAL] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia neural EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_neural_experience": self.eva_ingest_neural_experience,
            "eva_recall_neural_experience": self.eva_recall_neural_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
