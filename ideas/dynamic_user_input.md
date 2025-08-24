# Idea: Input Dinámico del Usuario Durante el Turno del Agente

## Problema Actual:
Actualmente, la barra de input del usuario se bloquea mientras el agente está procesando una solicitud (ejecutando herramientas, razonando, esperando resultados). Esto impide que el usuario aporte información adicional o clarificaciones en tiempo real, lo que puede llevar a ciclos de comunicación ineficientes o a que el agente se atasque por falta de contexto.

## Propuesta:
Permitir que el usuario pueda introducir input adicional mientras el agente está en medio de un turno de procesamiento. Este input no sería un nuevo prompt, sino una "aportación suplementaria" que el agente podría utilizar en su razonamiento o para influenciar la ejecución de herramientas en curso.

## Justificación:
*   **Mejora de la Eficiencia:** Permite al usuario corregir o clarificar información en tiempo real, evitando que el agente siga un camino incorrecto o se atasque por ambigüedad.
*   **Mayor Fluidez en la Interacción:** La conversación se vuelve más dinámica y colaborativa, reduciendo la latencia percibida y los ciclos de ida y vuelta.
*   **Control Proactivo del Usuario:** El usuario puede guiar al agente de forma más granular y en el momento oportuno.
*   **Reducción de la Carga Cognitiva del LLM:** Al recibir información más precisa y oportuna, el LLM podría necesitar menos "turnos de razonamiento" para llegar a una solución.

## Consideraciones para el Ingeniero:

### 1. Gestión del Estado del Agente:
*   El CLI necesitaría un estado interno más granular para saber en qué fase del procesamiento se encuentra el agente (ej. esperando resultado de `read_file`, procesando output de `run_shell_command`).
*   Cómo se integra este input suplementario en el contexto del LLM: ¿se añade al final del historial de conversación? ¿Se marca de alguna forma especial?

### 2. Interfaz de Usuario (UX/UI):
*   **Indicación Visual:** La barra de input debería indicar claramente cuándo está disponible para input suplementario y cuándo no.
*   **Feedback al Usuario:** El agente debería reconocer explícitamente la recepción de input suplementario (ej. "Gracias por la aclaración, la tendré en cuenta").
*   **Diferenciación de Prompts:** Cómo distinguir un nuevo prompt de un input suplementario. Quizás un prefijo especial o un modo temporal.

### 3. Routing del Input Suplementario:
*   ¿El input suplementario se envía directamente al LLM principal?
*   ¿Se puede dirigir a una herramienta específica que esté esperando input? (ej. si una herramienta de `run_shell_command` está esperando confirmación, el input suplementario podría ser la confirmación).

### 4. Manejo de Conflictos:
*   Qué ocurre si el input suplementario contradice la información que el agente ya está procesando.

## Archivos Relevantes:
*   `packages/cli/src/ui/`: Para la implementación de la interfaz de usuario.
*   `packages/core/src/config/config.ts`: Posiblemente para opciones de configuración de este modo.
*   `packages/core/src/core/turn.ts`: Para cómo se representa el input suplementario en el historial de conversación.
*   `packages/core/src/core/coreToolScheduler.ts`: Para cómo se gestiona el flujo de ejecución de herramientas.
