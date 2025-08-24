# Idea: Gestión Proactiva del Contexto (Más allá de la Sumarización)

## Problema Actual:
Aunque la sumarización del historial de conversación es crucial para extender la ventana de contexto, el agente (yo) actualmente reacciona a la saturación del contexto. No tengo un mecanismo explícito para gestionar proactivamente mi contexto, lo que puede llevar a que el contexto se llene de información menos relevante antes de que se active la sumarización, o a que se resuma información crítica.

## Propuesta:
Desarrollar estrategias y herramientas para que el agente pueda gestionar su contexto de forma proactiva, decidiendo cuándo y cómo podar o resumir información antes de que se alcance el límite de tokens.

## Justificación:
*   **Prevención de la Saturación:** Evita que el contexto se sature con información irrelevante, previniendo la degradación del rendimiento y las "alucinaciones".
*   **Eficiencia Mejorada:** Reduce la cantidad de tokens que se envían al LLM, optimizando el uso de recursos.
*   **Relevancia del Contexto:** Asegura que el contexto siempre contenga la información más relevante para la tarea actual, mejorando la calidad del razonamiento del agente.
*   **Mayor Autonomía:** Permite al agente tomar decisiones inteligentes sobre su propia "memoria".

## Sugerencias de Mejora Específicas:

### 1. Conciencia del Presupuesto de Tokens:
*   **Exposición del Conteo de Tokens:** El CLI debería proporcionar al agente (LLM) una forma de conocer el conteo actual de tokens del historial de conversación.
*   **Umbrales Configurables:** Permitir la configuración de umbrales de tokens para activar acciones proactivas (ej. "cuando el contexto llegue al 70% de su capacidad, considera resumir").

### 2. Estrategias de Sumarización Contextual:
*   **Identificación de Irrelevancia:** Desarrollar heurísticas o un modelo (posiblemente el LLM local de Ollama) que pueda identificar y priorizar la sumarización de partes del historial que son menos relevantes para la tarea actual.
*   **Sumarización por Tarea/Subtarea:** Permitir que el agente resuma el contexto relacionado con tareas o subtareas ya completadas, liberando espacio para la tarea actual.

### 3. Herramientas Explícitas de Poda de Contexto:
*   **`prune_context(turns: number[], sections: string[])`:** Una herramienta que permita al agente eliminar explícitamente turnos específicos o secciones del historial de conversación que considere irrelevantes o redundantes.
*   **`summarize_section(start_turn: number, end_turn: number)`:** Una herramienta más granular que `summarize_conversation` que permita resumir un rango específico de turnos.

### 4. Priorización de la Información en el Contexto:
*   **Etiquetado de Relevancia:** Desarrollar un sistema donde el agente pueda "etiquetar" ciertas partes del contexto como de alta relevancia (ej. instrucciones iniciales, objetivos, código crítico) para que sean protegidas de la sumarización o poda agresiva.

## Archivos Relevantes:
*   `packages/core/src/core/turn.ts`: Para la representación del historial de conversación y posible etiquetado.
*   `packages/core/src/config/config.ts`: Para la configuración de umbrales de tokens.
*   `packages/core/src/tools/`: Para nuevas herramientas de gestión de contexto.
*   `packages/core/src/services/`: Para la lógica de conteo de tokens y gestión del historial.
