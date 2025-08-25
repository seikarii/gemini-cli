from collections.abc import Callable
from typing import Any


class MessageHandlerMeta(type):
    """
    Metaclase para registro automático y seguro de handlers de mensajes.
    Permite diagnóstico extendido, trazabilidad y prevención de duplicados.
    """

    _message_handlers: dict[str, Callable] = {}

    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)
        for _attr_name, attr_value in attrs.items():
            if hasattr(attr_value, "_message_type"):
                message_type = attr_value._message_type
                if message_type in mcs._message_handlers:
                    raise ValueError(
                        f"Duplicate message handler for type: {message_type} in {name}"
                    )
                mcs._message_handlers[message_type] = attr_value
        new_class._message_handlers = mcs._message_handlers.copy()
        return new_class

    @classmethod
    def get_handler(cls, message_type: str) -> Callable | None:
        """Obtiene el handler registrado para un tipo de mensaje."""
        return cls._message_handlers.get(message_type)

    @classmethod
    def list_handlers(cls) -> dict[str, Callable]:
        """Lista todos los handlers registrados."""
        return cls._message_handlers.copy()


def message_handler(message_type: str):
    """
    Decorador para registrar métodos como handlers de mensajes.
    Añade trazabilidad y diagnóstico.
    """

    def decorator(func: Callable):
        func._message_type = message_type
        func._is_message_handler = True
        return func

    return decorator


class BaseMessageHandler(metaclass=MessageHandlerMeta):
    """
    Clase base para handlers de mensajes.
    Permite dispatch, diagnóstico y extensión segura.
    """

    _message_handlers: dict[str, Callable] = {}

    def dispatch(self, message_type: str, *args, **kwargs) -> Any:
        handler = self._message_handlers.get(message_type)
        if not handler:
            raise ValueError(f"No handler registered for message type: {message_type}")
        return handler(self, *args, **kwargs)

    @classmethod
    def get_registered_handlers(cls) -> dict[str, Callable]:
        """Devuelve todos los handlers registrados en la clase."""
        return cls._message_handlers.copy()


class EVAMessageHandlerMeta(MessageHandlerMeta):
    """
    Metaclase extendida para registro y diagnóstico de handlers EVA.
    Permite gestión de mensajes EVA, faseo, hooks de entorno y benchmarking.
    """

    _eva_message_handlers: dict[str, Callable] = {}

    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)
        for _attr_name, attr_value in attrs.items():
            if hasattr(attr_value, "_eva_message_type"):
                message_type = attr_value._eva_message_type
                if message_type in mcs._eva_message_handlers:
                    raise ValueError(
                        f"Duplicate EVA message handler for type: {message_type} in {name}"
                    )
                mcs._eva_message_handlers[message_type] = attr_value
        new_class._eva_message_handlers = mcs._eva_message_handlers.copy()
        return new_class

    @classmethod
    def get_eva_handler(cls, message_type: str) -> Callable | None:
        """Obtiene el handler EVA registrado para un tipo de mensaje."""
        return cls._eva_message_handlers.get(message_type)

    @classmethod
    def list_eva_handlers(cls) -> dict[str, Callable]:
        """Lista todos los handlers EVA registrados."""
        return cls._eva_message_handlers.copy()


def eva_message_handler(message_type: str):
    """
    Decorador para registrar métodos como handlers de mensajes EVA.
    Añade trazabilidad, diagnóstico y soporte de faseo.
    """

    def decorator(func: Callable):
        func._eva_message_type = message_type
        func._is_eva_message_handler = True
        return func

    return decorator


class EVABaseMessageHandler(BaseMessageHandler, metaclass=EVAMessageHandlerMeta):
    """
    Handler base extendido para mensajes EVA.
    Permite dispatch, diagnóstico, hooks de entorno y gestión de fase.
    """

    _eva_message_handlers: dict[str, Callable] = {}
    _eva_phase: str = "default"
    _eva_environment_hooks: list = []

    def dispatch_eva(self, message_type: str, *args, **kwargs) -> Any:
        handler = self._eva_message_handlers.get(message_type)
        if not handler:
            raise ValueError(
                f"No EVA handler registered for message type: {message_type}"
            )
        result = handler(self, *args, **kwargs)
        for hook in self._eva_environment_hooks:
            try:
                hook(
                    {
                        "message_type": message_type,
                        "result": result,
                        "phase": self._eva_phase,
                    }
                )
            except Exception as e:
                print(f"[EVA-MESSAGE-HANDLER] Environment hook failed: {e}")
        return result

    @classmethod
    def get_registered_eva_handlers(cls) -> dict[str, Callable]:
        """Devuelve todos los handlers EVA registrados en la clase."""
        return cls._eva_message_handlers.copy()

    def set_eva_phase(self, phase: str):
        """Cambia la fase/timeline activa para mensajes EVA."""
        self._eva_phase = phase
        for hook in self._eva_environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-MESSAGE-HANDLER] Phase hook failed: {e}")

    def get_eva_phase(self) -> str:
        """Devuelve la fase/timeline activa para mensajes EVA."""
        return self._eva_phase

    def add_eva_environment_hook(self, hook: Callable):
        """Registra un hook para eventos de mensajes EVA (ej. renderizado, benchmarking)."""
        self._eva_environment_hooks.append(hook)
