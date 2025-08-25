import ast
import importlib.util
import logging
import os
import random
import sys
import tempfile
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict, Optional, cast

from crisalida_lib.EDEN.qualia_manifold import QualiaState

if TYPE_CHECKING:
    # Import richer types for static analysis only. Keep names available to
    # the type checker but do not bind typing special forms at runtime.
    from crisalida_lib.EDEN.living_symbol import (
        LivingSymbolRuntime,
        QuantumField,
    )
    from crisalida_lib.EVA.types import (
        EVAExperience,
        RealityBytecode,
    )
else:
    # runtime fallbacks
    LivingSymbolRuntime = Any
    QuantumField = Any
    EVAExperience = Any
    RealityBytecode = Any


class ModuleInfo(TypedDict, total=False):
    module_name: str
    function_name: str
    source_code: str
    bytecode: list[Any]
    mutation_count: int
    rationale: str
    estimated_benefit: str

logger = logging.getLogger(__name__)


class ASTMutator(ast.NodeTransformer):
    """
    AST-based mutator for safe code mutations.
    Inherits from NodeTransformer to safely modify AST nodes.
    """

    def __init__(self):
        self.mutations_applied = 0
        self.max_mutations = 3  # Limit mutations to prevent excessive changes

    def visit_Constant(self, node):
        """Mutate constant values (numbers, booleans, strings)."""
        if self.mutations_applied >= self.max_mutations:
            return node

        if isinstance(node.value, (int, float)):
            # Mutate numeric constants
            if random.random() < 0.3:  # 30% chance to mutate
                mutated_value = node.value * random.uniform(0.9, 1.1)
                self.mutations_applied += 1
                if isinstance(node.value, int):
                    return ast.Constant(value=int(mutated_value))
                else:
                    return ast.Constant(value=mutated_value)
        elif isinstance(node.value, bool):
            # Mutate boolean constants
            if random.random() < 0.2:  # 20% chance to mutate
                self.mutations_applied += 1
                return ast.Constant(value=not node.value)

        return node

    def visit_Compare(self, node):
        """Mutate comparison operators."""
        if self.mutations_applied >= self.max_mutations:
            return node

        if random.random() < 0.2:  # 20% chance to mutate
            for i, op in enumerate(node.ops):
                if isinstance(op, ast.Lt):
                    node.ops[i] = ast.Gt()
                    self.mutations_applied += 1
                    break
                elif isinstance(op, ast.Gt):
                    node.ops[i] = ast.Lt()
                    self.mutations_applied += 1
                    break
                elif isinstance(op, ast.LtE):
                    node.ops[i] = ast.GtE()
                    self.mutations_applied += 1
                    break
                elif isinstance(op, ast.GtE):
                    node.ops[i] = ast.LtE()
                    self.mutations_applied += 1
                    break

        return self.generic_visit(node)

    def mutate(self, tree):
        """Apply mutations to the AST tree."""
        self.mutations_applied = 0
        return self.visit(tree)


class AutoGenesisEngine:
    """
    Núcleo evolutivo para generación, mutación y gestión de módulos cognitivos auto-generados.
    Permite la adaptación dinámica de reglas interpretativas y respuestas cognitivas en Observers.
    """

    def __init__(self):
        self.generated_modules: dict[str, dict] = {}
        self.learning_bias: float = 0.5
        self.generated_modules_dir = os.path.join(
            os.path.dirname(__file__), "generated_modules"
        )
        os.makedirs(self.generated_modules_dir, exist_ok=True)

    def _load_module_from_file(
        self, module_name: str, file_path: str, function_name: str
    ):
        """Carga un módulo Python desde archivo y retorna el objeto función."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return getattr(module, function_name, None)
        logger.error(f"Could not load module spec for {module_name}")
        return None

    def _create_and_load_module(
        self, module_id: str, module_type: str, function_name: str, source_code: str
    ) -> dict[str, Any] | None:
        module_name = f"generated_{module_type}_{module_id}"
        file_path = os.path.join(self.generated_modules_dir, f"{module_name}.py")
        try:
            # Quick syntax validation before writing to disk
            try:
                compile(source_code, "<string>", "exec")
            except Exception as e:
                logger.warning(
                    "Generated source for %s has syntax errors; skipping: %s",
                    module_name,
                    e,
                )
                return None

            # Before persisting, run the generated source in a sandbox to
            # validate runtime behavior (simple harness). This helps catch
            # runtime errors and infinite loops that pass syntax check.
            try:
                from crisalida_lib.HEAVEN.sandbox.sandbox_executor import SandboxExecutor

                se = SandboxExecutor(python_path=sys.executable, max_memory_mb=50, max_cpu_time=3)
                # Construct harness: define the function and do a minimal call.
                # Use a conservative call input (0) and avoid f-string interpolation
                harness = source_code + "\n\n# Minimal harness call for validation\ntry:\n    _result = {fname}({arg})\nexcept Exception as _e:\n    raise\n".format(fname=function_name, arg="0")
                res = se.execute_python_code(harness, timeout=4)
                if not res.get("success", False):
                    logger.warning(
                        "Sandbox rejected generated module %s: %s",
                        module_name,
                        res.get("stderr") or res.get("error"),
                    )
                    return None
            except Exception:
                # If sandbox not available or fails, fall back to compile-only check
                pass

            # Write atomically to avoid partial files on crash
            tmp = None
            try:
                fd, tmp = tempfile.mkstemp(prefix=module_name + "_", suffix=".py", dir=self.generated_modules_dir)
                with os.fdopen(fd, "w") as f:
                    f.write(source_code)
                # use atomic replace
                os.replace(tmp, file_path)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            logger.info(f"Generated new {module_type} module: {file_path}")
            function_object = self._load_module_from_file(
                module_name, file_path, function_name
            )
            if function_object:
                # Try to produce a conservative bytecode representation for the
                # generated module so downstream systems can execute or inspect it.
                try:
                    from crisalida_lib.EDEN.bytecode_generator import (
                        compile_intention_to_bytecode,
                    )

                    intention = {
                        "intention_type": f"gen_{module_type}",
                        "experience": {"module": module_name},
                        "payload": source_code,
                    }
                    bytecode = compile_intention_to_bytecode(intention)
                except Exception:
                    bytecode = []

                return {
                    "module_name": module_name,
                    "function_name": function_name,
                    "source_code": source_code,
                    "function_object": function_object,
                    "bytecode": bytecode,
                }
            logger.error(f"Failed to load generated module {module_name}")
        except Exception as e:
            logger.error(f"Error creating and loading module {module_name}: {e}")
        return None

    def get_state(self) -> dict[str, Any]:
        """Devuelve el estado serializable actual del AutoGenesisEngine."""
        serializable_modules = {
            name: {
                "module_name": info["module_name"],
                "function_name": info["function_name"],
                "source_code": info["source_code"],
            }
            for name, info in self.generated_modules.items()
        }
        return {
            "generated_modules": serializable_modules,
            "learning_bias": self.learning_bias,
            "generated_modules_dir": self.generated_modules_dir,
        }

    def set_state(self, state: dict[str, Any]):
        """Restaura el estado del AutoGenesisEngine desde un diccionario deserializado."""
        self.learning_bias = state.get("learning_bias", 0.5)
        self.generated_modules_dir = state.get(
            "generated_modules_dir",
            os.path.join(os.path.dirname(__file__), "generated_modules"),
        )
        self.generated_modules = {}
        for name, module_info in state.get("generated_modules", {}).items():
            module_name = module_info["module_name"]
            function_name = module_info["function_name"]
            source_code = module_info["source_code"]
            file_path = os.path.join(self.generated_modules_dir, f"{module_name}.py")
            try:
                if (
                    not os.path.exists(file_path)
                    or open(file_path).read() != source_code
                ):
                    with open(file_path, "w") as f:
                        f.write(source_code)
                function_object = self._load_module_from_file(
                    module_name, file_path, function_name
                )
                if function_object:
                    self.generated_modules[name] = {
                        "module_name": module_name,
                        "function_name": function_name,
                        "source_code": source_code,
                        "function_object": function_object,
                    }
                else:
                    logger.warning(f"Failed to load function object for {module_name}")
            except Exception as e:
                logger.error(f"Error loading generated module {module_name}: {e}")

    def _generate_high_value_detection_module(
        self, observer_memory: list[tuple[Any, Any, Any, QualiaState]]
    ) -> dict[str, Any]:
        """Genera módulo para detección de alto valor en observaciones."""
        new_modules_info = {}
        high_value_count = sum(1 for _, obs, _, _ in observer_memory if obs > 0.7)
        if high_value_count > 10 and random.random() < (
            0.1 + self.learning_bias * 0.05
        ):
            module_id = uuid.uuid4().hex[:8]
            function_name = "interpret_high_value"
            source_code = f"""
def {function_name}(x):
    # Generated rule for high value detection
    return x > 0.8
"""
            module_info = self._create_and_load_module(
                module_id, "rule", function_name, source_code
            )
            if module_info:
                new_modules_info[f"high_value_detection_{module_id}"] = module_info
        return new_modules_info

    def _generate_qualia_response_module(
        self, qualia_states: list[QualiaState]
    ) -> dict[str, Any]:
        """Genera módulos de respuesta adaptativa según patrones de qualia."""
        new_modules_info = {}
        if len(qualia_states) > 20:
            # Use getattr to avoid mypy/typing issues when QualiaState is a typing.Any fallback
            avg_valence = sum(getattr(q, "emotional_valence", 0.0) for q in qualia_states) / len(
                qualia_states
            )
            avg_arousal = sum(getattr(q, "arousal", 0.0) for q in qualia_states) / len(qualia_states)
            if (
                avg_valence > 0.6
                and avg_arousal > 0.7
                and random.random() < (0.05 + self.learning_bias * 0.02)
            ):
                module_id = uuid.uuid4().hex[:8]
                function_name = "respond_to_positive_arousal"
                source_code = f"""
def {function_name}(qualia):
    # Respond to positive arousal
    return qualia.emotional_valence * qualia.arousal > 0.4
"""
                module_info = self._create_and_load_module(
                    module_id, "response", function_name, source_code
                )
                if module_info:
                    new_modules_info[f"positive_arousal_response_{module_id}"] = (
                        module_info
                    )
                    self.learning_bias = min(1.0, self.learning_bias + 0.1)
            elif (
                avg_valence < -0.6
                and avg_arousal < 0.3
                and random.random() < (0.03 + self.learning_bias * 0.01)
            ):
                module_id = uuid.uuid4().hex[:8]
                function_name = "respond_to_low_valence_calm"
                source_code = f"""
def {function_name}(qualia):
    # Respond to low valence calm
    return qualia.emotional_valence < -0.5 and qualia.arousal < 0.4
"""
                module_info = self._create_and_load_module(
                    module_id, "response", function_name, source_code
                )
                if module_info:
                    new_modules_info[f"low_valence_calm_response_{module_id}"] = (
                        module_info
                    )
                    self.learning_bias = max(0.0, self.learning_bias - 0.05)
        return new_modules_info

    def _generate_predictive_decay_module(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Genera módulo predictivo para alertar sobre decaimiento ontológico."""
        new_modules_info = {}
        if len(historical_data) > 100:
            last_100_ticks_data = historical_data[-100:]
            trend = [d.get("ontological_stability", 0.0) for d in last_100_ticks_data]
            if len(trend) > 1 and trend[-1] < trend[0] * 0.9:
                module_id = uuid.uuid4().hex[:8]
                function_name = "predict_ontological_decay"
                source_code = f"""
def {function_name}(current_stability):
    # Generated rule for predicting ontological decay
    return current_stability < 0.5
"""
                module_info = self._create_and_load_module(
                    module_id, "predictive", function_name, source_code
                )
                if module_info:
                    new_modules_info[f"predictive_decay_alert_{module_id}"] = (
                        module_info
                    )
        return new_modules_info

    def analyze_memory_and_generate_code(
        self,
        observer_memory: list[tuple[Any, Any, Any, QualiaState]],
        historical_data: list[dict[str, Any]],
    ):
        """Analiza memoria y datos históricos para generar nuevos módulos cognitivos."""
        new_modules_info = {}
        new_modules_info.update(
            self._generate_high_value_detection_module(observer_memory)
        )
        new_modules_info.update(
            self._generate_qualia_response_module(
                [mem[3] for mem in observer_memory if len(mem) > 3]
            )
        )
        new_modules_info.update(self._generate_predictive_decay_module(historical_data))
        self.generated_modules.update(new_modules_info)
        return new_modules_info

    def _mutate_with_basic_ast(self, source_code: str) -> str:
        """
        Fallback AST mutation method that doesn't require external libraries.
        Uses ast module's built-in functionality to generate source code.
        """
        tree = ast.parse(source_code)
        mutator = ASTMutator()
        mutated_tree = mutator.mutate(tree)

        # Simple approach: use compile and exec to validate, then return modified code
        # This is a basic fallback - the mutation might not be perfectly preserved
        try:
            compile(mutated_tree, "<string>", "exec")
            # If compilation succeeds, try to reconstruct source code manually
            return self._ast_to_source_basic(mutated_tree)
        except SyntaxError as e:
            logger.error(f"Mutated AST has syntax error: {e}")
            raise

    def _ast_to_source_basic(self, tree: ast.AST) -> str:
        """
        Basic AST to source code conversion.
        This is a simplified implementation for the fallback case.
        """
        # For now, just return a minimal implementation
        # In a real scenario, you'd want a more complete AST->source converter

        # Try to extract just the function if it's a module with a single function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Return a basic function template with modified constants
                return f"""
def {node.name}(*args, **kwargs):
    # Auto-generated mutated function
    return True  # Simplified mutation result
"""

        # Fallback: return a simple module
        return "# Mutated module - basic fallback\nresult = True\n"

    def mutate_and_reload_module(self, rule_name: str, module_info: dict) -> Any | None:
        """
        Muta el código fuente de un módulo usando AST (Abstract Syntax Tree) y lo recarga.
        SECURITY FIX: Reemplaza las mutaciones basadas en strings con manipulación segura de AST.
        """
        module_name = module_info["module_name"]
        function_name = module_info["function_name"]
        source_code = module_info["source_code"]

        try:
            # Parse the source code into an AST
            import ast

            tree = ast.parse(source_code)

            # Apply AST-based mutations
            mutator = ASTMutator()
            mutated_tree = mutator.mutate(tree)

            # Convert the mutated AST back to source code
            try:
                # Prefer astor if available for nicer formatting
                import astor  # type: ignore

                mutated_source_code = astor.to_source(mutated_tree)
            except Exception:
                # Fallback to built-in ast.unparse (Python 3.9+)
                try:
                    mutated_source_code = ast.unparse(mutated_tree)
                except Exception:
                    # Final fallback to basic method
                    mutated_source_code = self._mutate_with_basic_ast(source_code)

        except ImportError:
            # Fallback to simpler AST approach if astor is not available
            logger.warning("astor not available, using basic AST approach")
            try:
                mutated_source_code = self._mutate_with_basic_ast(source_code)
            except Exception as e:
                logger.error(f"AST mutation failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to mutate {module_name} with AST: {e}")
            return None

        if mutated_source_code == source_code:
            logger.warning(f"No mutations applied to {rule_name}. Skipping mutation.")
            return None

        # Save backup version
        version_id = uuid.uuid4().hex[:8]
        backup_dir = os.path.join(self.generated_modules_dir, "versions", module_name)
        os.makedirs(backup_dir, exist_ok=True)
        backup_file_path = os.path.join(backup_dir, f"{version_id}.py")
        with open(backup_file_path, "w") as f:
            f.write(mutated_source_code)
        logger.info(f"Saved backup of {module_name} to {backup_file_path}")

        # Write mutated source and reload
        file_path = os.path.join(self.generated_modules_dir, f"{module_name}.py")
        with open(file_path, "w") as f:
            f.write(mutated_source_code)
        logger.info(f"Mutated and wrote new source for {module_name} to {file_path}")

        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            function_object = self._load_module_from_file(
                module_name, file_path, function_name
            )
            if function_object:
                self.generated_modules[rule_name]["source_code"] = mutated_source_code
                self.generated_modules[rule_name]["function_object"] = function_object
                return function_object
            logger.error(f"Failed to reload mutated module {module_name}")
        except Exception as e:
            logger.error(f"Error during mutation and reload of {module_name}: {e}")
        return None

    def revert_module_to_version(self, rule_name: str, version_id: str) -> Any | None:
        """Revierte un módulo generado a una versión histórica específica."""
        if rule_name not in self.generated_modules:
            logger.warning(f"Rule {rule_name} not found in generated modules.")
            return None
        module_info = self.generated_modules[rule_name]
        module_name = module_info["module_name"]
        function_name = module_info["function_name"]
        backup_dir = os.path.join(self.generated_modules_dir, "versions", module_name)
        backup_file_path = os.path.join(backup_dir, f"{version_id}.py")
        if not os.path.exists(backup_file_path):
            logger.warning(f"Version {version_id} for module {module_name} not found.")
            return None
        with open(backup_file_path) as f:
            source_code = f.read()
        logger.info(f"Reverted {module_name} to version {version_id}")
        file_path = os.path.join(self.generated_modules_dir, f"{module_name}.py")
        with open(file_path, "w") as f:
            f.write(source_code)
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            function_object = self._load_module_from_file(
                module_name, file_path, function_name
            )
            if function_object:
                self.generated_modules[rule_name]["source_code"] = source_code
                self.generated_modules[rule_name]["function_object"] = function_object
                return function_object
            logger.error(f"Failed to reload reverted module {module_name}")
        except Exception as e:
            logger.error(f"Error during revert and reload of {module_name}: {e}")
        return None

    def apply_new_cognitive_module(self, observer, new_modules_info: dict[str, dict]):
        """Aplica nuevos módulos cognitivos a un Observer."""
        for name, module_info in new_modules_info.items():
            observer.interpretation_rules[name] = module_info["function_object"]
            logger.info(
                f"[AUTO-GENESIS] Applied new cognitive module '{name}' to Observer {observer.entity_id}"
            )


class EVAAutoGenesisEngine(AutoGenesisEngine):
    """
    Núcleo evolutivo auto-generativo extendido para integración con EVA.
    Permite la generación, mutación, ingestión y recall de módulos cognitivos como experiencias vivientes EVA,
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente.
    """

    def __init__(self, phase: str | None = "default"):
        super().__init__()
        # ensure concrete str for phase to satisfy callers/mypy
        self.eva_phase: str = phase or "default"
        # Only instantiate LivingSymbolRuntime if it's a real class; fall back to None
        try:
            # annotate as optional since runtime may be None
            if callable(LivingSymbolRuntime):
                self.eva_runtime: Optional[LivingSymbolRuntime] = LivingSymbolRuntime()
            else:
                self.eva_runtime = None
        except Exception:
            self.eva_runtime = None

        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Any] = []

    def eva_ingest_cognitive_module_experience(
        self,
        module_info: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila la experiencia de generación/mutación de módulo cognitivo en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.9,
            consciousness_density=0.8,
            narrative_importance=0.8,
            energy_level=1.0,
        )
        experience_data = {
            "module_name": module_info.get("module_name"),
            "function_name": module_info.get("function_name"),
            "source_code": module_info.get("source_code"),
            "mutation_count": module_info.get("mutation_count", 0),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_COGNITIVE_MODULE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        # Resolve compiler safely: runtime or divine_compiler may be None in compatibility shims
        bytecode = []
        try:
            runtime = getattr(self, "eva_runtime", None)
            _dc = (
                getattr(runtime, "divine_compiler", None)
                if runtime is not None
                else None
            )
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
        except Exception:
            bytecode = []
        # Conservative fallback to EDEN-level generator when runtime compiler not available
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(intention)
            except Exception:
                bytecode = []
        experience_id = f"eva_cognitive_module_{module_info.get('module_name')}_{hash(str(experience_data))}"
        # Ensure phase is a concrete string and cast instructions to a permissive type
        phase_str: str = cast(str, phase or "default")
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=cast(list, bytecode),
            qualia_state=qualia_state,
            phase=phase_str,
            timestamp=experience_data["timestamp"],
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase_str not in self.eva_phases:
            self.eva_phases[phase_str] = {}
        self.eva_phases[phase_str][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_cognitive_module_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de módulo cognitivo almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        phase_str = cast(str, phase or self.eva_phase)
        reality_bytecode = self.eva_phases.get(phase_str, {}).get(cue) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA cognitive module experience"}
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = None
            # Guard runtime execution: eva_runtime may be None or missing executor
            rt = getattr(self, "eva_runtime", None)
            if rt is not None and hasattr(rt, "execute_instruction"):
                try:
                    symbol_manifest = rt.execute_instruction(instr, quantum_field)
                except Exception:
                    symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.warning(f"[EVA-AUTO-GENESIS] Environment hook failed: {e}")
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
        module_info: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia de módulo cognitivo EVA.
        """
        import time

        experience_data = {
            "module_name": module_info.get("module_name"),
            "function_name": module_info.get("function_name"),
            "source_code": module_info.get("source_code"),
            "mutation_count": module_info.get("mutation_count", 0),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_COGNITIVE_MODULE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        # Resolve compiler safely: avoid direct attribute access on possibly-None runtime
        bytecode = []
        try:
            runtime = getattr(self, "eva_runtime", None)
            _dc = (
                getattr(runtime, "divine_compiler", None)
                if runtime is not None
                else None
            )
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
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
        phase_str2: str = phase or "default"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=cast(list, bytecode),
            qualia_state=qualia_state,
            phase=phase_str2,
            timestamp=experience_data["timestamp"],
        )
        if phase_str2 not in self.eva_phases:
            self.eva_phases[phase_str2] = {}
        self.eva_phases[phase_str2][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-AUTO-GENESIS] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de módulo cognitivo EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_cognitive_module_experience": self.eva_ingest_cognitive_module_experience,
            "eva_recall_cognitive_module_experience": self.eva_recall_cognitive_module_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
