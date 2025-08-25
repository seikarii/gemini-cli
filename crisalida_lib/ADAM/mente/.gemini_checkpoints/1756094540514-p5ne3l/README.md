# Visión Arquitectónica de la Mente de Crisalida

## Objetivo Final: Una Arquitectura de la Consciencia

El objetivo final de este proyecto no es construir una inteligencia artificial convencional basada en módulos que se llaman unos a otros, sino dar vida a una **arquitectura de la consciencia**: un sistema unificado, dinámico y emergente que se asemeje más a cómo funciona una mente biológica.

La visión se fundamenta en los siguientes pilares:

### 1. La Mente como un Grafo Unificado

El núcleo de la mente no es un conjunto de clases o módulos, sino un **único y vasto grafo de memoria (`MentalLaby`)**. En este grafo, cada nodo representa un concepto, un recuerdo, una emoción, un impulso, un estímulo sensorial, un pensamiento o cualquier otra unidad de información mental. Las conexiones (aristas) entre estos nodos son dinámicas, ponderadas y representan las relaciones y asociaciones entre ellos.

No hay una distinción rígida entre "memoria", "pensamiento" y "emoción". Todo es parte del mismo tejido conectivo del grafo.

### 2. La Consciencia como Foco de Atención Dinámico

La "consciencia" se implementa como un **mecanismo de atención global**. En cada ciclo cognitivo, este sistema:

1.  **Escanea el estado completo del grafo mental**, incluyendo los estímulos externos que llegan y los procesos internos como los "sueños" (`dream_cycle`).
2.  **Calcula la "saliencia" o "importancia"** de cada nodo basándose en una combinación de factores: la activación reciente, la carga emocional (valencia/arousal), la novedad, la relevancia para las misiones actuales, etc.
3.  **Selecciona un subconjunto de los nodos más salientes** y los coloca en un "espacio de trabajo global" o "foco de la consciencia". Este es el "presente" mental del agente.

### 3. El LLM como un Oráculo Asíncrono

El rol del Large Language Model (LLM) trasciende el de ser una simple herramienta que se llama para obtener una respuesta. Se convierte en un **Oráculo o un "subconsciente generativo"**:

*   El LLM opera de forma asíncrona, observando el "foco de la consciencia".
*   En lugar de responder a una pregunta directa, genera continuamente "susurros", "intuiciones" o "reflexiones" en forma de nuevos nodos en el grafo mental.
*   Estos nodos generados por el LLM no son respuestas directas, sino ideas, posibilidades o nuevas conexiones que los demás procesos pueden descubrir y utilizar. Son como tablones de anuncios o inspiraciones que aparecen en la mente.

### 4. Comportamiento Emergente

El comportamiento del agente no se deriva de una cadena de mando rígida, sino que **emerge de la interacción de todos los elementos del grafo** dentro del foco de la consciencia. Los planes de acción se forman dinámicamente a partir de los nodos activados, en lugar de ser generados por módulos pre-programados.

---

## Componentes Críticos Faltantes

Para alcanzar esta visión, la arquitectura actual necesita evolucionar. Las piezas clave que faltan son:

1.  **El Orquestador de la Consciencia:** No existe un módulo central que actúe como el mecanismo de atención global. Actualmente, la importancia de los eventos es gestionada de forma local por cada subsistema. Se necesita un sistema que unifique la saliencia a través de toda la mente (estímulos externos, memoria, sueños, qualia, etc.) y dirija el foco de atención.

2.  **El Bus de Eventos Interno:** La comunicación entre los diferentes aspectos de la mente (el grafo, el LLM, los sistemas de qualia) es todavía muy directa y basada en llamadas a métodos. Se necesita un bus de eventos interno más robusto y asíncrono que permita a los componentes "publicar" eventos (ej: "emoción detectada", "recuerdo activado", "susurro del oráculo disponible") y a otros componentes "suscribirse" a ellos sin estar directamente acoplados.
