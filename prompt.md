# Problema de Selección de Herramientas para Edición de Código

## Descripción del Problema

Durante las tareas de edición de código, se ha observado una tendencia a utilizar la herramienta `replace` (basada en texto) para realizar modificaciones. Si bien `replace` es útil para sustituciones de texto genéricas, su naturaleza textual la hace frágil y propensa a errores en el contexto del código fuente. Problemas como diferencias sutiles en espacios en blanco, saltos de línea (`\n` vs `\r\n`) o la necesidad de una coincidencia exacta de cadenas pueden llevar a fallos en la edición o a resultados inesperados.

Existen herramientas más robustas y adecuadas para la manipulación de código, como `ast_edit` (que opera a nivel del Árbol de Sintaxis Abstracta - AST) y `upsert_code_block` (que inserta o actualiza bloques de código completos de forma inteligente). Estas herramientas son inmunes a los problemas de formato y operan sobre la estructura lógica del código, no solo sobre su representación textual. La cuestión principal es la selección subóptima de la herramienta para la tarea de manipulación de código.

## Soluciones Propuestas

Se han planteado dos posibles soluciones para abordar este problema:

### 1. Backend para la Herramienta `edit`

*   **Descripción**: Desarrollar un servicio o módulo de backend que actúe como una capa de abstracción para las operaciones de edición. Cuando se solicite una edición, este backend analizaría el contexto (tipo de archivo, naturaleza de la modificación) y seleccionaría automáticamente la herramienta de edición más adecuada (por ejemplo, `ast_edit` para cambios estructurales en código, `replace` para sustituciones de texto simples).
*   **Ventajas**: Abstrae la complejidad de la selección de herramientas del LLM, asegurando el uso de las mejores prácticas de forma consistente.
*   **Desventajas**: Requiere un esfuerzo de desarrollo significativo y el LLM perdería el control directo sobre el método de edición específico.

### 2. Mejora del Prompt del LLM

*   **Descripción**: Refinar el prompt del sistema que se me envía para incluir instrucciones más explícitas y detalladas sobre cuándo y cómo utilizar las herramientas de edición de código. Esto incluiría:
    *   Priorizar el uso de herramientas basadas en AST (`ast_edit`, `upsert_code_block`) para modificaciones de código.
    *   Instrucciones para analizar los errores de las operaciones de `replace` y, en caso de fallo, intentar la edición con una herramienta más robusta (basada en AST).
    *   Fomentar un análisis previo del código objetivo (por ejemplo, usando `read_file` con `include_ast=true`) para informar la elección de la herramienta.
*   **Ventajas**: Aprovecha la capacidad de razonamiento del LLM para aprender y adaptarse, es más flexible y generalizable a otros tipos de tareas.
*   **Desventajas**: Depende de la fiabilidad del LLM para interpretar y aplicar las instrucciones, y la ingeniería del prompt puede ser un desafío.

## Soluciones Alternativas o Complementarias

Además de las propuestas anteriores, se pueden considerar las siguientes alternativas o enfoques complementarios:

### 3. Enfoque Híbrido y Mejoras Internas de Herramientas

*   **Descripción**: Combinar la mejora del prompt (mi preferencia inmediata) con posibles mejoras internas en las herramientas existentes. Por ejemplo, la herramienta `replace` podría ser mejorada para que, si detecta que está operando sobre un archivo de código y la cadena a reemplazar corresponde a un nodo AST válido, intente internamente una sustitución basada en AST antes de recurrir a la sustitución de texto puro.
*   **Ventajas**: Influencia directamente mi comportamiento a través del prompt y, a largo plazo, hace que las herramientas sean inherentemente más inteligentes y robustas.

### 4. Herramientas de "Refactorización de Alto Nivel"

*   **Descripción**: Desarrollar herramientas especializadas de más alto nivel que encapsulen patrones de refactorización comunes (por ejemplo, "renombrar_variable(archivo, nombre_antiguo, nombre_nuevo)", "extraer_funcion(archivo, inicio, fin, nombre_funcion)"). Estas herramientas de alto nivel se encargarían internamente de realizar las manipulaciones AST necesarias.
*   **Ventajas**: Simplifica mi tarea al proporcionarme abstracciones de refactorización, reduciendo la necesidad de que yo orqueste manipulaciones AST de bajo nivel.
*   **Desventajas**: Requiere un desarrollo significativo de nuevas herramientas especializadas y seguiria necesitando de mejorar la capacidad del llm.

La mejora del prompt es el camino más directo para influir en mi comportamiento actual y es donde se puede lograr un impacto inmediato en la calidad de las ediciones de código.



