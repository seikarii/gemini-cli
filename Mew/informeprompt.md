# Informe sobre la Interacción LLM-Herramientas: Prompts y JSON

Este informe analiza la interacción entre el LLM (Gemini 2.5) y el sistema de herramientas, centrándose en el formato de los prompts que se envían a las herramientas y la estructura de los datos que se reciben.

## 1. Comprensión de la Interacción Actual

### Lo que el LLM envía (al sistema de ejecución de herramientas):

El LLM genera prompts en un formato estructurado como:
```
[tool_call: nombre_herramienta for 'argumentos_en_cadena']
```
Internamente, este string es parseado y convertido a una representación de objeto, principalmente la interfaz `ToolCallRequestInfo` (definida en `core/src/core/turn.ts`).

**Estructura de `ToolCallRequestInfo`:**
```typescript
export interface ToolCallRequestInfo {
  name: string; // Nombre de la herramienta (ej. 'run_shell_command', 'read_file')
  args: object; // ¡Crucial! Los argumentos son un objeto, no solo una cadena.
  callId: string; // Identificador único para la llamada a la herramienta
  prompt_id?: string; // Opcional: ID del prompt asociado
}
```
El campo `args: object` confirma que los argumentos de las herramientas se transmiten como objetos estructurados (es decir, JSON), lo que permite el paso de múltiples parámetros con tipos definidos.

### Lo que el LLM recibe (del sistema de ejecución de herramientas):

Después de la ejecución de una herramienta, el LLM recibe una respuesta, cuya estructura principal es `ToolCallResponseInfo` (también en `core/src/core/turn.ts`). Esta interfaz contiene el resultado de la operación de la herramienta.

**Estructura de `ToolCallResponseInfo` (ejemplo simplificado):**
```typescript
export interface ToolCallResponseInfo {
  callId: string; // ID de la llamada a la herramienta a la que responde
  responseParts: Part[]; // Array de partes de la respuesta (ej. { text: '...' }, { image: '...' })
  // Otros campos como 'error', 'status', etc.
}
```
Las respuestas también se basan en estructuras de datos (JSON), lo que permite un feedback detallado sobre el éxito, el error y el contenido generado por la herramienta.

### Uso de JSON y Esquemas:

La presencia de `args: object` y las múltiples interfaces `ToolCall...` (como `ValidatingToolCall`, `ExecutingToolCall`, `CompletedToolCall` en `core/src/core/coreToolScheduler.ts`) que contienen `ToolCallRequestInfo`, junto con las definiciones de esquema en `cli/src/zed-integration/schema.ts` (usando Zod), demuestran una fuerte dependencia de JSON para la comunicación estructurada entre el LLM y el sistema de herramientas.

## 2. Áreas de Mejora Potenciales

El sistema actual es robusto y bien definido para llamadas a herramientas estructuradas. Las mejoras se centrarían en optimizar la eficiencia, la robustez y la capacidad de auto-corrección del LLM.

### a. Definiciones/Esquemas de Herramientas más Ricos:

*   **Problema:** Aunque `args: object` permite datos estructurados, la comprensión del LLM sobre los argumentos esperados por una herramienta (tipos, restricciones, valores enumerados) proviene de las "declaraciones de función" de la herramienta. Si estas declaraciones son insuficientes, el LLM puede "alucinar" argumentos incorrectos.
*   **Mejora:** Asegurar que las definiciones de herramientas proporcionadas al LLM sean lo más completas posible. Esto incluye descripciones detalladas, información de tipo (con ejemplos), reglas de validación, valores enumerados permitidos y descripciones de cada parámetro. Un esquema JSON robusto para cada herramienta ayudaría a esto.

### b. Manejo de Errores y Bucle de Retroalimentación Mejorado:

*   **Problema:** Si una llamada a una herramienta falla debido a argumentos inválidos (ej. tipo incorrecto, campo requerido faltante), la retroalimentación actual podría ser un mensaje de error genérico.
*   **Mejora:** Proporcionar mensajes de error más granulares y accionables desde la ejecución de la herramienta de vuelta al LLM. Por ejemplo, "Error: el argumento 'file_path' es requerido para la herramienta 'write_file'" o "Error: el valor de 'mode' debe ser 'overwrite' o 'append'". Esto permitiría al LLM corregirse de manera más efectiva.

### c. Tipos de Argumentos Complejos:

*   **Problema:** Aunque `args: object` es bueno, algunas herramientas podrían requerir estructuras de datos más complejas (ej. una lista de objetos, objetos anidados con esquemas específicos).
*   **Mejora:** Asegurar que el sistema pueda manejar y validar estos tipos de argumentos complejos, y que el LLM sea capaz de generarlos correctamente basándose en las definiciones de las herramientas. Esto podría implicar una validación de esquema JSON más sofisticada.

### d. Encadenamiento/Orquestación de Herramientas (Mini-Lenguaje de Scripting y `ActionSystem`):

*   **Problema:** La interacción actual es por turnos, con el LLM generando una llamada a la herramienta a la vez. Aunque el LLM puede encadenar acciones a través de un razonamiento iterativo, la sintaxis explícita de encadenamiento no está soportada en el formato de prompt que genera.
*   **Propuesta de Mejora:**
    1.  **Traducción e Integración de `ActionSystem`:** Se propone traducir el `ActionSystem` de Python (visto en `crisalida_lib/HEAVEN/agents/core/action_system.py`) a TypeScript. Este sistema proporciona una robusta infraestructura para la gestión de acciones: cola, priorización, ejecución concurrente y seguimiento del estado.
        *   **Ubicación:** El `ActionSystem` traducido debería ubicarse en `packages/core/src/core/action-system.ts` para mantener la coherencia con la lógica central del sistema.
        *   **Archivos a Modificar/Integrar:**
            *   `packages/core/src/core/turn.ts`: Podría necesitar una nueva interfaz para representar un "script" de multi-acción que el LLM generaría, o una modificación de `ToolCallRequestInfo` para permitir listas de acciones.
            *   `packages/core/src/core/coreToolScheduler.ts`: Este módulo sería reemplazado o refactorizado para delegar la programación y ejecución de herramientas al nuevo `ActionSystem`.
            *   `packages/core/src/core/nonInteractiveToolExecutor.ts`: Su lógica de ejecución de herramientas se integraría dentro del `ActionSystem`.
            *   `packages/core/src/core/client.ts`: El cliente principal que interactúa con el LLM y el sistema de herramientas necesitaría ser actualizado para enviar los "scripts" de multi-acción al `ActionSystem` y procesar sus resultados.
            *   `packages/core/src/tools/tool-registry.ts`: El `ActionSystem` interactuaría con el `ToolRegistry` para obtener las funciones de las herramientas registradas.
            *   `packages/cli/src/nonInteractiveCli.ts` y `packages/cli/src/ui/hooks/useGeminiStream.ts`: Estos archivos, que manejan la interacción con el LLM y la ejecución de herramientas en el lado del cliente/UI, necesitarían adaptarse para enviar los nuevos "scripts" de multi-acción al backend y procesar sus resultados de forma asíncrona.
    2.  **Mini-Lenguaje de Scripting para el LLM:** Para aprovechar el `ActionSystem`, se necesita introducir un formato de prompt más avanzado que permita al LLM expresar una secuencia de llamadas a herramientas, lógica condicional o ejecución paralela dentro de una única salida generada. Este "mini-lenguaje" sería parseado por el sistema para crear las `Action`s que el `ActionSystem` gestionaría. Esto reduciría el número de turnos para tareas complejas y mejoraría la eficiencia.

### e. Salida en Vivo/Streaming para Herramientas de Larga Duración:

*   **Problema:** Para comandos de shell de larga duración u operaciones de archivo, el LLM podría recibir solo la salida final.
*   **Mejora:** Implementar un mecanismo para que las herramientas transmitan la salida intermedia de vuelta al LLM. Esto permitiría al LLM monitorear el progreso, reaccionar a resultados parciales o incluso cancelar operaciones de larga duración si fuera necesario. (Se observó `liveOutput` en `ExecutingToolCall`, lo que sugiere que esto ya se está considerando o implementando para algunas herramientas).

---
**Conclusión:**

El sistema actual de interacción LLM-Herramientas es funcional y se basa en un intercambio de datos estructurados (JSON). Las mejoras futuras deberían centrarse en enriquecer las definiciones de las herramientas, mejorar la retroalimentación de errores, manejar tipos de argumentos más complejos, y, crucialmente, **integrar un `ActionSystem` traducido a TypeScript junto con un mini-lenguaje de scripting para el LLM** para habilitar capacidades de multi-acción y orquestación. Finalmente, considerar la transmisión de salida en vivo para mejorar la experiencia y la eficiencia del LLM.
