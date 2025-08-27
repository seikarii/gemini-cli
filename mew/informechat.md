# Informe sobre el Problema de Contexto en el Chat

Este documento analiza el problema de "olvido" y "alucinaciones" detectado en el comportamiento del LLM y propone una solución basada en la reordenación del contexto enviado al modelo.

## El Problema: La Relevancia Queda al Final

El análisis del `chatRecordingService.ts` ha revelado que el problema no es que los mensajes se envíen en orden "inverso", sino algo más sutil y problemático: **el orden de construcción del contexto prioriza el historial completo sobre la relevancia reciente.**

1.  **Contexto Inflado**: El historial de la conversación se infla enormemente, no por los mensajes del usuario (que son pequeños), sino por las respuestas del asistente, y muy especialmente, por el **contenido de los ficheros que lee** a través de las herramientas. Un solo `read_file` puede añadir decenas de miles de tokens al historial.

2.  **Orden de Ensamblaje**: El sistema construye el prompt para el LLM en el siguiente orden:
    *   Primero, todos los mensajes del usuario (`allUserMessages`).
    *   Después, todos los mensajes del asistente, incluyendo el contenido de los ficheros leídos (`allAssistantMessages`).

3.  **La Consecuencia**: Esto provoca que los mensajes más recientes y críticos (el plan de acción del LLM, el código que acaba de generar, etc.) queden **al final de un contexto extremadamente largo**.

4.  **El Cuello de Botella**: Cuando este contexto masivo se envía al LLM, si excede la ventana máxima de tokens del modelo, el proveedor lo trunca. Debido al orden de ensamblaje, **lo que se corta es el final del prompt, que es precisamente la información más reciente y relevante**. El LLM pierde de vista su propio plan y actúa basándose en información antigua e incompleta que sí ha logrado "ver".

Esto explica por qué a veces escribe ficheros incompletos o parece haber olvidado lo que iba a hacer.

## La Solución Propuesta: Invertir el Orden del Contexto

La solución más directa y efectiva es cambiar la estrategia de cómo se presenta el contexto al LLM, priorizando la información más reciente.

**Propuesta:**

Modificar el servicio para que, al enviar el contexto al LLM, **invierta el orden de los mensajes**. El prompt final debería tener esta estructura:

1.  **[Mensaje Más Reciente]** (ya sea de usuario o asistente)
2.  **[Segundo Mensaje Más Reciente]**
3.  **[...]**
4.  **[Mensaje Más Antiguo]**
5.  **[Contexto Comprimido (si lo hay)]** (resumen de lo muy antiguo)

### Ventajas de este enfoque:

*   **Preservación de la Relevancia**: Si el proveedor trunca el contexto, cortará los mensajes más antiguos y menos relevantes, no los más recientes.
*   **Foco en la Tarea Actual**: El LLM verá inmediatamente la información más pertinente para su siguiente acción, ya que estará al principio del prompt.
*   **Mitigación de Errores**: Se reduce drásticamente la probabilidad de que el modelo "olvide" su plan o el código que acaba de generar, solucionando el cuello de botella actual.

Este cambio no requiere modificar la lógica de compresión, solo la forma en que se ordena el array final de mensajes antes de ser enviado al modelo. Es una solución quirúrgica que ataca directamente la raíz del problema.
