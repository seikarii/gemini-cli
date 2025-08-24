# Idea: Área de Pre-Edición / Staging para Modificaciones de Código

## Problema Actual:
Las herramientas de modificación de código actuales (ej. `replace`, `write_file`) operan directamente sobre el sistema de archivos. Esto las hace propensas a introducir errores (ej. errores de sintaxis, cambios parciales) que pueden corromper archivos. La falta de un paso intermedio de validación y revisión antes de la escritura definitiva lleva a modificaciones poco fiables y a la frustración del usuario.

## Propuesta:
Introducir un mecanismo de "pre-edición" o "área de staging" donde el agente pueda realizar y validar modificaciones de código en un contexto aislado y temporal antes de aplicarlas al archivo real. Esto sería similar a un "mini-editor" o "scratchpad" de unas 200-300 líneas.

## Justificación:
*   **Fiabilidad Extrema:** Reduce drásticamente el riesgo de introducir errores de sintaxis o corromper archivos, ya que los cambios se validan antes de ser escritos.
*   **Confianza del Usuario:** Aumenta la confianza en las modificaciones del agente, ya que los cambios son verificados y "escaneados" (como lo haría un IDE) antes de ser comprometidos.
*   **Refinamiento Iterativo:** Permite al agente realizar y verificar múltiples cambios pequeños dentro de un ámbito confinado, mejorando la calidad de la edición.
*   **Alineación con Flujos de Trabajo Profesionales:** Imita el proceso de "staging" y revisión que se utiliza en entornos de desarrollo profesionales.
*   **Prioridad a la Fiabilidad:** Aunque pueda añadir unos segundos al proceso, la fiabilidad es un valor mucho más alto que la velocidad en este contexto.

## Sugerencias de Implementación:

### 1. Herramientas para el Área de Staging:
*   **`load_section_to_staging(file_path: string, start_line: number, end_line: number)`:**
    *   **Propósito:** Cargar una sección relevante del archivo (ej. la función o clase a modificar) en un buffer temporal de staging.
    *   **Parámetros:** Ruta del archivo, línea de inicio y línea de fin.
*   **`edit_staging(old_string: string, new_string: string)` / `delete_insert_staging(...)`:**
    *   **Propósito:** Versiones especializadas de las herramientas de modificación que operan *únicamente* sobre el contenido del buffer de staging.
    *   **Funcionamiento:** El agente realizaría sus ediciones (borrar, insertar, reemplazar) dentro de este contexto aislado.
*   **`validate_staging(linter_command: string, type_checker_command: string)`:**
    *   **Propósito:** Ejecutar validaciones locales (ej. linter, type-checker) sobre el contenido del buffer de staging.
    *   **Funcionamiento:** El agente podría invocar herramientas de análisis de código (como `ruff`, `tsc`, `eslint`) sobre el contenido temporal para detectar errores.
*   **`apply_staging_to_file()`:**
    *   **Propósito:** Aplicar los cambios validados del buffer de staging al archivo original.
    *   **Funcionamiento:** Solo se ejecutaría si las validaciones en staging han sido exitosas.

### 2. Flujo de Trabajo del Agente:
1.  Identificar la sección del archivo a modificar.
2.  Cargar esa sección en el área de staging (`load_section_to_staging`).
3.  Realizar las ediciones necesarias dentro del staging (`edit_staging` / `delete_insert_staging`).
4.  Validar los cambios en staging (`validate_staging`).
5.  Si la validación es exitosa, aplicar los cambios al archivo real (`apply_staging_to_file`).
6.  Si la validación falla, el agente podría intentar corregir los errores dentro del staging o pedir ayuda al usuario.

### 3. Interfaz de Usuario (UX/UI):
*   El CLI necesitaría una forma de mostrar el contenido del área de staging al usuario, quizás como un diff o un archivo temporal.
*   Indicar claramente cuándo el agente está operando en el área de staging.

## Archivos Relevantes:
*   `packages/core/src/services/fileSystemService.ts`: Para operaciones de lectura/escritura de archivos temporales.
*   `packages/core/src/tools/`: Para las nuevas herramientas de staging.
*   `packages/cli/src/ui/`: Para la visualización del área de staging.
*   `packages/core/src/core/`: Para la orquestación del flujo de trabajo del agente.
