# Roadmap: Hacia un Agente Autónomo y Fiable en Gemini CLI

## Visión General:
Transformar Gemini CLI en un agente de software altamente autónomo y fiable, capaz de ejecutar misiones complejas sin supervisión constante, con una "memoria" de contexto extendida y mecanismos robustos de seguridad. El objetivo es que el agente pueda operar durante periodos prolongados (ej. 1 hora) sin necesidad de un `git restore` manual debido a errores.

## Problemas Actuales a Resolver:
*   **Fragilidad del Contexto:** La ventana de contexto limitada del LLM principal lleva a la pérdida de información y a la incoherencia en conversaciones largas o tareas complejas.
*   **Riesgo de Destrucción de Código:** La falta de mecanismos de "deshacer" robustos y automáticos genera desconfianza y requiere supervisión constante.
*   **Limitaciones en la Manipulación de Código:** Las herramientas actuales de manipulación de archivos son de "fuerza bruta" (basadas en texto plano), lo que las hace frágiles y propensas a errores.
*   **Dificultad para Procesar Grandes Volúmenes de Datos:** El CLI actual "intenta no" procesar repositorios enteros o grandes volúmenes de archivos, limitando la capacidad del agente para comprender el contexto global del proyecto.

## Pilares de Mejora:

### 1. Fiabilidad y Seguridad (Mecanismos de "Deshacer" y Checkpointing)
*   **Checkpointing por Defecto:**
    *   **Propuesta:** Habilitar el checkpointing automático por defecto para todas las sesiones.
    *   **Justificación:** Es una característica de seguridad fundamental que previene la pérdida de datos y fomenta la experimentación segura.
    *   **Archivos Relevantes:** `packages/core/src/config/config.ts`, `~/.gemini/settings.json`.
*   **Política de Retención de Checkpoints:**
    *   **Propuesta:** Implementar una política configurable para limpiar checkpoints antiguos (ej. conservar los últimos N, o los de los últimos X días).
    *   **Justificación:** Gestiona el uso de disco y mantiene la lista de checkpoints manejable.
    *   **Archivos Relevantes:** `packages/core/src/config/config.ts`, `packages/core/src/services/gitService.ts`.
*   **Respaldo Automático del Último Cambio (Backup Last Change):**
    *   **Propuesta:** Antes de que cualquier herramienta modifique un archivo, crear una copia de seguridad timestamped de ese archivo.
    *   **Justificación:** Proporciona un "deshacer" inmediato y granular para operaciones individuales, crucial para la confianza del usuario.
    *   **Archivos Relevantes:** `packages/core/src/utils/backupUtils.ts`, `packages/core/src/tools/edit.ts`, `packages/core/src/tools/write-file.ts`, `packages/core/src/tools/upsert_code_block.ts`.
*   **Mejora de la Usabilidad de `/restore`:**
    *   **Propuesta:** Hacer que la salida del comando `/restore` sea más descriptiva y fácil de navegar, mostrando detalles legibles de cada checkpoint.
    *   **Justificación:** Facilita la recuperación de estados anteriores.

### 2. Contexto Extendido e Inteligencia (Sistema de "Doble LLM" y RAG)
*   **Integración de Ollama como LLM Local:**
    *   **Propuesta:** Utilizar Ollama (ej. Qwen3:4b con 256k de contexto) como un LLM local especializado para tareas de procesamiento de contexto.
    *   **Justificación:** Proporciona un potente motor de procesamiento de lenguaje local, superando las limitaciones de contexto del LLM principal y reduciendo la dependencia de la API remota para ciertas tareas.
    *   **Consideraciones:** Implica la gestión de un runtime local y comunicación inter-proceso.
*   **Herramienta `summarize_conversation` (usando Ollama):**
    *   **Propuesta:** Crear una herramienta que resuma partes del historial de conversación y reemplace el contenido original por el resumen.
    *   **Justificación:** Extiende la ventana de contexto efectiva del LLM principal, permitiendo conversaciones más largas y complejas.
*   **Herramienta `summarize_repo` (usando Ollama):**
    *   **Propuesta:** Crear una herramienta que lea y resuma el contenido de todo el repositorio (o partes especificadas), proporcionando un resumen de alto nivel del proyecto.
    *   **Justificación:** Permite al agente comprender el contexto global del proyecto sin saturar la ventana de contexto con archivos individuales. Aborda la dificultad actual de "leer todos los archivos del repo".
*   **Herramientas AST (Análisis y Manipulación Estructurada de Código):
    *   **Propuesta:** Desarrollar un conjunto de herramientas basadas en AST para la manipulación de código estructurado.
    *   **Justificación:** Proporciona operaciones de código más robustas, precisas y seguras que las basadas en texto plano.
    *   **Ejemplos:** `ast_find` (ya iniciada), `ast_modify`, `ast_generate`, `ast_refactor`.

### 3. Visión Futura (Orquestación Multi-Agente)
*   **Paralelización de Tareas y Flujos de Trabajo Multi-Agente:**
    *   **Visión:** Explorar la capacidad de que el agente no solo actúe como una CLI para un único LLM, sino que orqueste múltiples tareas en paralelo o incluso coordine varios "sub-agentes" para abordar misiones más grandes y complejas.
    *   **Justificación:** Llevaría la capacidad del agente a un nivel de competición superior, permitiendo flujos de trabajo verdaderamente autónomos.
    *   **Nota:** Esto es un objetivo a largo plazo, a abordar una vez que el agente único sea extremadamente fiable.

## Próximo Paso Sugerido:
Continuar con la implementación del **respaldo automático del último cambio** (Backup Last Change), ya que es la medida de seguridad más inmediata y crucial para permitir una mayor autonomía y confianza en el agente.
