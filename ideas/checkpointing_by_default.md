# Idea: Activar Checkpointing por Defecto

## Problema Actual:
El sistema de checkpointing (respaldo automático del estado del proyecto antes de modificaciones por herramientas) es una característica de seguridad fundamental, pero está deshabilitada por defecto. Esto lleva a situaciones de frustración y posible pérdida de trabajo cuando los usuarios no son conscientes de la necesidad de activarlo manualmente, como hemos experimentado. La analogía de "escribir a fuego" es muy pertinente aquí.

## Propuesta:
Habilitar el checkpointing por defecto para todas las sesiones del Gemini CLI.

## Justificación:
El checkpointing es una funcionalidad básica de "deshacer" y una red de seguridad esencial para cualquier herramienta que modifique archivos. Su activación por defecto:
*   Mejora drásticamente la experiencia del usuario al proporcionar un mecanismo de recuperación fiable sin configuración previa.
*   Previene la pérdida accidental de datos y la corrupción de archivos.
*   Fomenta la experimentación segura con las herramientas del agente.
*   Alinea el comportamiento del CLI con las expectativas de herramientas de edición modernas (donde el "deshacer" es omnipresente).

## Consideraciones para el Ingeniero:
*   **Impacto en el Rendimiento/Recursos:** Evaluar el impacto real en el rendimiento y el uso de disco en proyectos grandes. Si es significativo, considerar optimizaciones o una notificación clara al usuario.
*   **Mecanismo de Desactivación:** Asegurar que los usuarios puedan deshabilitar fácilmente el checkpointing si lo desean (ej. mediante una opción en `settings.json` o un flag de línea de comandos).
*   **Notificación al Usuario:** Al iniciar el CLI con checkpointing activado por defecto, podría ser útil una notificación sutil informando al usuario de esta funcionalidad y cómo deshabilitarla.

## Archivos Relevantes:
*   `packages/core/src/config/config.ts`: Donde se lee y se establece la configuración de `checkpointing`.
*   `~/.gemini/settings.json`: Archivo de configuración del usuario donde se puede sobrescribir el valor por defecto.
