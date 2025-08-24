# Idea: Procesamiento de Input de Usuario Durante Acciones Pendientes

## Problema Actual:
Cuando una acción del agente (ej. una llamada a herramienta que requiere confirmación) está pendiente, cualquier input del usuario suele cancelar la acción en curso. Esto interrumpe el flujo de trabajo y frustra al usuario que desea aportar información o modificar la acción sin tener que cancelarla y empezar de nuevo.

## Propuesta:
Modificar el comportamiento del CLI para que, cuando una acción esté pendiente de confirmación o procesamiento, el input del usuario no cancele automáticamente dicha acción. En su lugar, el input del usuario debería ser procesado como un nuevo prompt o una instrucción que influya en la acción pendiente.

## Justificación:
*   **Mejora de la Fluidez:** Permite una interacción más dinámica y menos interruptiva. El usuario puede guiar o corregir al agente en tiempo real.
*   **Optimización del Flujo de Trabajo:** Evita la necesidad de cancelar y reiniciar acciones, lo que ahorra tiempo y reduce la frustración.
*   **Control del Usuario:** Da al usuario un mayor control sobre el agente, permitiéndole intervenir de forma más granular.
*   **Reducción de la Carga Cognitiva del LLM:** Al recibir información más precisa y oportuna, el LLM podría necesitar menos "turnos de razonamiento" para llegar a una solución.

## Consideraciones para el Ingeniero:

### 1. Gestión del Estado de la Acción Pendiente:
*   El CLI necesitaría un estado interno más granular para saber en qué fase del procesamiento se encuentra el agente (ej. esperando resultado de `read_file`, procesando output de `run_shell_command`).
*   El input del usuario debe ser capaz de interactuar con este estado.

### 2. Interpretación del Input del Usuario:
*   **Nuevo Prompt:** Si el input es un comando o una pregunta general, debería tratarse como un nuevo prompt que podría llevar a una nueva ronda de razonamiento.
*   **Modificación de Acción Pendiente:** Si el input es una respuesta a la acción pendiente (ej. "sí", "no", "cambia X"), debería aplicarse a esa acción.
*   **Clarificación/Contexto:** Si el input es una clarificación, debería añadirse al contexto del LLM para que reevalúe la acción pendiente.

### 3. Interfaz de Usuario (UX/UI):
*   **Indicación Visual:** La interfaz debería indicar claramente que hay una acción pendiente y que el input del usuario será procesado en relación con ella.
*   **Feedback al Usuario:** El agente debería reconocer el input del usuario y cómo lo ha interpretado en relación con la acción pendiente.

### 4. Manejo de Conflictos:
*   **Responsabilidad del Usuario:** Si el input suplementario es ambiguo o contradictorio, se asume que es responsabilidad del usuario cancelar la acción pendiente (ej. presionando Escape) antes de proporcionar un nuevo prompt claro. El agente no intentará resolver ambigüedades complejas en este modo.
*   Qué ocurre si el input del usuario es ambiguo o contradictorio con la acción pendiente.

## Archivos Relevantes:
*   `packages/core/src/core/coreToolScheduler.ts`: Para la gestión del flujo de ejecución de herramientas y la interacción con el LLM.
*   `packages/core/src/tools/tools.ts`: Donde se definen las interfaces de confirmación de herramientas.
*   `packages/cli/src/ui/`: Para la implementación de la interfaz de usuario y la gestión del input.
