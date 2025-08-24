# Plan de Optimización Revisado para Symbolic Matrix VM

## Objetivo Principal Inmediato
Eliminar los cuellos de botella críticos de I/O (Entrada/Salida) y la sobrecarga del renderizado para mejorar drásticamente el rendimiento de la simulación. El objetivo secundario es prepararse para futuras optimizaciones de cálculo.

---

### **Fase 1: Optimización Crítica de I/O y Renderizado (Prioridad Máxima)**

*   **Diagnóstico:** El perfilado (`Data.txt`) revela que `_io.TextIOWrapper.write` es el cuello de botella más significativo (~29 segundos), causado por la función `_render_state` en `vm.py` que redibuja toda la pantalla en cada frame.

*   **Estrategia:**
    1.  **Desacoplar Renderizado y Simulación:** Modificar el método `run()` para que la lógica de la simulación (`_tick()`) no esté limitada por la tasa de refresco de la pantalla. Se introducirán dos parámetros: `sim_fps` (para los ticks de simulación) y `render_fps` (para las actualizaciones visuales).
    2.  **Implementar Renderizado Inteligente (Incremental):**
        *   **Técnica:** En lugar de usar `os.system('clear')`, se mantendrá una copia de la matriz del frame anterior (`self.previous_matrix`). En `_render_state`, se comparará la matriz actual con la anterior y se usarán secuencias de escape ANSI para mover el cursor de la terminal y sobreescribir únicamente los símbolos que han cambiado.
        *   **Beneficio:** Esto reducirá las operaciones de escritura de miles de caracteres por frame a solo unos pocos, eliminando el cuello de botella de I/O.
    3.  **Reducir Frecuencia de Renderizado:** Permitir que `render_fps` sea considerablemente más bajo que `sim_fps` (ej. `sim_fps=100`, `render_fps=10`).

### **Fase 2: Optimización del Cálculo (Prioridad Media)**

*Una vez resueltos los problemas de I/O, las siguientes optimizaciones del plan original serán relevantes.*

*   **1. Re-evaluación con Profiling:** Con el cuello de botella de I/O eliminado, se ejecutará `cProfile` de nuevo para identificar los nuevos cuellos de botella, que ahora sí serán las funciones de cálculo en `rules.py` y `vm.py`.

*   **2. Vectorización con NumPy:**
    *   **Diagnóstico:** Muchas operaciones en `vm.py` y `rules.py` se realizan con bucles de Python.
    *   **Estrategia:** Reemplazar estos bucles con operaciones vectorizadas de NumPy para acelerar masivamente los cálculos matriciales.

*   **3. Paralelización en CPU (`multiprocessing`):**
    *   **Estrategia:** Utilizar `multiprocessing` para distribuir el procesamiento de los solitones en múltiples núcleos de CPU. El principal desafío será la gestión y sincronización del estado de la matriz compartida.

*   **4. Aceleración por GPU (CuPy/Numba):**
    *   **Estrategia:** Como paso final para el máximo rendimiento, migrar las operaciones de NumPy a CuPy o usar Numba para compilar las funciones más críticas a código optimizado para GPU.

---

## Pasos de Ejecución Inmediatos (Fase 1)

1.  **Modificar `SymbolicMatrixVM.__init__`:**
    *   Añadir `self.previous_matrix = self.matrix.copy()`.
2.  **Modificar `SymbolicMatrixVM.run`:**
    *   Cambiar la firma a `run(self, duration: float, sim_fps: int = 60, render_fps: int = 10)`.
    *   Ajustar el bucle principal para que `_tick()` se llame según `sim_fps` y `_render_state()` según `render_fps`.
3.  **Reimplementar `SymbolicMatrixVM._render_state`:**
    *   Eliminar `os.system('clear')`.
    *   Añadir la lógica para comparar `self.matrix` y `self.previous_matrix`.
    *   Generar e imprimir solo las secuencias de escape ANSI necesarias para actualizar los caracteres modificados.
    *   Actualizar `self.previous_matrix = self.matrix.copy()` al final del método.
