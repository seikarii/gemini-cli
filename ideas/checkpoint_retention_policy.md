# Idea: Política de Retención de Checkpoints

## Problema Actual:
El sistema de checkpointing actual crea un nuevo punto de restauración antes de cada operación de herramienta que modifica archivos. Si bien esto es excelente para la seguridad, con el tiempo puede acumular un gran número de checkpoints, consumiendo espacio en disco y haciendo que la lista de checkpoints sea difícil de manejar.

## Propuesta:
Implementar una política de retención configurable para los checkpoints, permitiendo a los usuarios definir cuántos checkpoints desean conservar o por cuánto tiempo.

## Justificación:
*   **Gestión de Recursos:** Evita la acumulación ilimitada de checkpoints, liberando espacio en disco.
*   **Usabilidad:** Mantiene la lista de checkpoints relevante y manejable, facilitando la navegación y restauración.
*   **Control del Usuario:** Permite a los usuarios equilibrar la seguridad con el uso de recursos según sus necesidades.

## Opciones de Configuración Sugeridas:
*   **`max_checkpoints_per_project` (Número):** El número máximo de checkpoints a conservar por proyecto. Cuando se excede este límite, los checkpoints más antiguos se eliminan automáticamente.
*   **`max_checkpoint_age_days` (Número):** El número máximo de días que un checkpoint debe conservarse. Los checkpoints más antiguos que este límite se eliminan automáticamente.
*   **`retain_last_n_checkpoints` (Número):** Siempre conservar los últimos N checkpoints, independientemente de su antigüedad.

## Consideraciones para el Ingeniero:
*   **Proceso de Limpieza:** Definir cuándo se ejecuta el proceso de limpieza (ej. al iniciar el CLI, periódicamente, o al crear un nuevo checkpoint).
*   **Impacto en el Rendimiento:** Asegurar que el proceso de limpieza sea eficiente y no afecte negativamente el rendimiento del CLI.
*   **Notificación al Usuario:** Considerar notificar al usuario cuando se eliminan checkpoints automáticamente.

## Archivos Relevantes:
*   `packages/core/src/config/config.ts`: Para añadir las nuevas opciones de configuración.
*   `packages/core/src/tools/tool-registry.ts` o `packages/core/src/services/gitService.ts`: Posibles lugares para implementar la lógica de limpieza de checkpoints, ya que el `GitService` gestiona el repositorio en la sombra.
*   `docs/checkpointing.md`: Para documentar la nueva política de retención.
