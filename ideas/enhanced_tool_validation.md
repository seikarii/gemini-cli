# Idea: Validación de Herramientas y Manejo de Errores Mejorado

## Problema Actual:
Mis llamadas a herramientas a veces fallan de formas inesperadas o propongo parámetros inválidos, lo que lleva a errores, interrupciones en el flujo de trabajo y la necesidad de intervención manual. Aunque existe una validación básica, no es lo suficientemente robusta para prevenir todos los fallos o proporcionar feedback accionable.

## Propuesta:
Implementar una validación más sofisticada de los parámetros de las herramientas antes de su ejecución, y un manejo de errores más informativo y estructurado para las respuestas de las herramientas.

## Justificación:
*   **Reducción de Fallos:** Prevenir errores antes de que ocurran, lo que mejora la fiabilidad del agente y reduce la frustración del usuario.
*   **Feedback Accionable:** Proporcionar mensajes de error claros y estructurados que yo (el LLM) pueda interpretar y utilizar para corregir mi comportamiento.
*   **Mayor Autonomía:** Un agente que comete menos errores y puede recuperarse de ellos de forma más inteligente es más autónomo.

## Sugerencias de Mejora Específicas:

### 1. Validación Pre-Ejecución Más Inteligente:
*   **Validación Semántica de Parámetros:**
    *   Para herramientas como `replace` o `write_file`:
        *   **Verificar Existencia:** Antes de `replace`, comprobar si `old_string` existe realmente en el archivo y si es único (si `expected_replacements` es 1).
        *   **Validar Rutas:** Asegurarse de que las rutas de archivo existen y son accesibles (lectura/escritura) antes de intentar la operación.
        *   **Validar Contenido:** Para `write_file`, verificar que el contenido no está vacío si no se espera crear un archivo vacío.
    *   Para herramientas de `run_shell_command`:
        *   **Análisis Básico de Comandos:** Intentar un análisis sintáctico básico del comando para detectar errores obvios antes de ejecutarlo.
        *   **Verificar Existencia de Binarios:** Comprobar si el binario principal del comando existe en el PATH.
*   **"Dry Run" o Simulación:** Para operaciones críticas (ej. `replace` con múltiples ocurrencias), considerar una "simulación" o "dry run" que verifique el resultado sin modificar el archivo, y reportar el resultado de la simulación.

### 2. Manejo de Errores Estructurado y Accionable:
*   **Códigos de Error Granulares:** Asegurar que los `ToolResult` de las herramientas devuelvan códigos de error específicos y consistentes (ej. `FILE_NOT_FOUND`, `PERMISSION_DENIED`, `SYNTAX_ERROR_IN_COMMAND`, `NO_OCCURRENCE_FOUND`).
*   **Mensajes de Error Claros:** Los mensajes de error deben ser concisos y explicar *por qué* falló la operación, no solo *que* falló.
*   **Contexto del Error:** Incluir en el `ToolResult` información adicional relevante para el error (ej. la línea y columna del error en un archivo, la salida de `stderr` de un comando).

### 3. Estrategias de Reintento (para Errores Transitorios):
*   Implementar una lógica básica de reintento con backoff exponencial para errores transitorios (ej. `EMFILE` - demasiados archivos abiertos, `EAGAIN` - recurso temporalmente no disponible).

## Archivos Relevantes:
*   `packages/core/src/tools/`: Todas las herramientas (`edit.ts`, `write-file.ts`, `run-shell-command.ts`, `upsert_code_block.ts`, etc.).
*   `packages/core/src/services/fileSystemService.ts`: Para validaciones a nivel de sistema de archivos.
*   `packages/core/src/tools/tool-error.ts`: Para definir nuevos tipos de errores.
*   `packages/core/src/core/coreToolScheduler.ts`: Para la posible implementación de lógicas de reintento.
