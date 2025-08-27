### **Project Prompt: MewWindow - El Centro de Mando Inteligente para el Desarrollo Dirigido por IA**

#### **1. Filosofía y Visión General**

**El Problema:** Las herramientas de desarrollo actuales son fundamentalmente pasivas. Están diseñadas para que los humanos escriban código y, en el mejor de los casos, señalan errores de forma estática. Estas herramientas no están preparadas para la nueva era de la programación, donde los agentes de IA son los principales desarrolladores. Un agente de IA no necesita un "editor de texto", necesita un "sistema nervioso" que le proporcione el contexto adecuado y le ayude a razonar, aprender y autocorregirse.

**La Solución (`MewWindow`):** No estamos construyendo un editor de código. Estamos creando un **centro de mando inteligente para un ecosistema de agentes de IA**. `MewWindow` es la interfaz a través de la cual un "Director" humano guía a los agentes, pero su verdadera magia reside en su capacidad para pensar, priorizar información y facilitar una colaboración fluida entre agentes. Pasamos de la "revisión estática de errores" a la "corrección proactiva y autónoma".

#### **2. Componentes Clave del Ecosistema**

1.  **El Agente Principal (Gemini):** El "trabajador". Es el agente de IA de propósito general que recibe las directivas de alto nivel y ejecuta las tareas principales de desarrollo (escribir nuevas funciones, refactorizar código, etc.).
2.  **El Sub-Agente Especialista (`Mew`):** El "analista de calidad y corrector". `Mew` es un agente especializado que opera en segundo plano. Su única misión es observar el código que produce Gemini, ejecutar análisis (linting, type-checking, smoke tests, análisis de patrones) y, lo más importante, **proponer y aplicar correcciones de forma autónoma**.
3.  **El Director Humano:** El "estratega". Su rol no es picar código, sino proporcionar la intención y el objetivo estratégico a través de "susurros" (`whispers`) en la `MewWindow`. Supervisa el trabajo de los agentes y proporciona retroalimentación cuando es necesario.
4.  **La Mente (`Mente`): El Motor de Importancia:** El corazón del sistema. Basado en `Importance.ts`, este componente analiza cada fragmento de datos en el ecosistema (susurros del director, logs, resultados de herramientas, cambios en archivos, errores de `Mew`) y le asigna una puntuación de **importancia, valencia y excitación**. Esto crea un flujo de contexto dinámico y priorizado, asegurando que Gemini siempre tenga la información más relevante para tomar la siguiente decisión, evitando el ruido.
5.  **El Núcleo de Edición Colaborativa (basado en CRDT):** La base técnica que permite la coexistencia de los agentes. Al tratar cada archivo como un documento colaborativo, Gemini y `Mew` pueden realizar cambios de forma concurrente sin conflictos, y el Director Humano puede ver estas ediciones en tiempo real.

#### **3. Fases de Desarrollo del Proyecto**

**Fase 1: La Fundación - Interfaz de Usuario y Comunicación Básica**

- **Objetivo:** Crear el esqueleto de la aplicación y establecer la comunicación unidireccional.
- **Tareas:**
  1.  Desarrollar la aplicación base en React (`MewApp.tsx`).
  2.  Implementar los componentes visuales principales:
      - Un panel de árbol de archivos para navegar por el proyecto.
      - Un panel de "editor" de solo lectura (`textarea` por ahora) para mostrar el contenido del archivo seleccionado.
      - Un panel de "logs/salida del agente" para ver el estado bruto de la comunicación.
  3.  Construir la API para enviar "susurros" del cliente al backend.
  4.  Establecer un mecanismo (polling o WebSockets) para obtener y mostrar el estado y los logs del agente principal (Gemini).

**Fase 2: La Mente - Implementación del Motor de Importancia**

- **Objetivo:** Dotar a `MewWindow` de la capacidad de "pensar" y priorizar.
- **Tareas:**
  1.  Integrar la lógica de `Importance.ts` en el backend.
  2.  Configurar la infraestructura de embeddings (`HashingEmbedder`) y similitud de cosenos.
  3.  Interceptar todos los eventos relevantes: `user_input`, `tool_output`, `file_content`, `system_event`, `error`, `success`.
  4.  Para cada evento, calcular su `SignificanceResult` (`importance`, `valence`, `arousal`).
  5.  Diseñar y construir el "Ensamblador de Contexto": un módulo que recopila los eventos más significativos y construye el prompt final que se enviará a Gemini en cada turno.

**Fase 3: El Núcleo Colaborativo - Edición Multi-Agente en Tiempo Real**

- **Objetivo:** Reemplazar el editor de solo lectura con una solución colaborativa real.
- **Tareas:**
  1.  Elegir e integrar una librería CRDT (recomendado: **Y.js**).
  2.  Reemplazar el `textarea` con un componente de editor más robusto (ej. Monaco Editor) y vincular su estado al documento Y.js.
  3.  Desarrollar el backend de sincronización (usando WebSockets) para que los cambios realizados por los agentes en el servidor se reflejen instantáneamente en la `MewWindow` del cliente.
  4.  Modificar las herramientas del agente (como `replace`, `upsert_code_block`) para que apliquen sus cambios al documento Y.js en lugar de escribir directamente en el sistema de archivos.

**Fase 4: El Nacimiento de "Mew" - El Sub-Agente Corrector**

- **Objetivo:** Implementar el agente especialista en calidad de código.
- **Tareas:**
  1.  Desarrollar `Mew` como un servicio o proceso separado.
  2.  `Mew` se suscribirá a los cambios en los documentos Y.js (archivos de código).
  3.  Cada vez que un archivo cambia, `Mew` ejecutará automáticamente un pipeline de análisis: linting, chequeo de tipos, etc.
  4.  Si `Mew` detecta un error o una mejora, generará un "parche" o "diff".
  5.  `Mew` aplicará este parche al documento Y.js, permitiendo que su corrección aparezca en tiempo real en la `MewWindow` para que el Director la vea.

**Fase 5: Orquestación y Puesta en Marcha del Ecosistema**

- **Objetivo:** Integrar todos los componentes en un flujo de trabajo coherente.
- **Tareas:**
  1.  Definir el bucle de orquestación completo: `Director -> Mente -> Gemini -> CRDT -> Mew -> CRDT -> Director`.
  2.  Refinar la interfaz de `MewWindow` para diferenciar visualmente los cambios hechos por Gemini de las correcciones propuestas por `Mew`.
  3.  Implementar un sistema de aprobación/rechazo para que el Director pueda vetar las correcciones de `Mew` si es necesario.
  4.  Realizar pruebas exhaustivas del ecosistema completo en escenarios de desarrollo reales.
