# Action Script Prompts para LLM

## Instrucciones para Generar Action Scripts

Los Action Scripts permiten a los LLMs expresar secuencias complejas de acciones, lógica condicional y ejecución paralela dentro de una única respuesta estructurada.

### Formato de Action Script

```json
{
  "id": "script_001",
  "rootNode": {
    "type": "sequence",
    "nodes": [
      // Lista de acciones a ejecutar en secuencia
    ],
    "description": "Descripción opcional del script"
  }
}
```

### Tipos de Nodos Disponibles

#### 1. Acción Básica (`action`)

```json
{
  "type": "action",
  "toolName": "read_file",
  "parameters": {
    "file_path": "/path/to/file.txt",
    "encoding": "utf-8"
  },
  "priority": "normal",
  "description": "Leer el contenido de un archivo"
}
```

#### 2. Secuencia (`sequence`)

```json
{
  "type": "sequence",
  "nodes": [
    {
      /* acción 1 */
    },
    {
      /* acción 2 */
    },
    {
      /* acción 3 */
    }
  ],
  "description": "Ejecutar acciones en orden secuencial"
}
```

#### 3. Paralelo (`parallel`)

```json
{
  "type": "parallel",
  "nodes": [
    {
      /* acción 1 */
    },
    {
      /* acción 2 */
    }
  ],
  "maxConcurrency": 3,
  "description": "Ejecutar acciones en paralelo"
}
```

#### 4. Condicional (`condition`)

```json
{
  "type": "condition",
  "condition": "file_exists('/path/to/file.txt')",
  "thenNode": {
    /* acción si condición es verdadera */
  },
  "elseNode": {
    /* acción si condición es falsa */
  },
  "description": "Ejecutar acción basada en condición"
}
```

#### 5. Bucle (`loop`)

```json
{
  "type": "loop",
  "variable": "item",
  "iterable": ["item1", "item2", "item3"],
  "body": {
    /* acción a repetir */
  },
  "description": "Repetir acción para cada elemento"
}
```

#### 6. Variable (`variable`)

```json
{
  "type": "variable",
  "name": "my_var",
  "value": "some_value",
  "description": "Definir una variable para usar en otras acciones"
}
```

### Prioridades de Acción

- `low`: Para acciones no críticas
- `normal`: Prioridad estándar (por defecto)
- `high`: Para acciones importantes
- `critical`: Para acciones críticas que deben ejecutarse inmediatamente

### Ejemplos de Action Scripts

#### Ejemplo 1: Análisis de Código Simple

```json
{
  "id": "code_analysis_001",
  "rootNode": {
    "type": "sequence",
    "nodes": [
      {
        "type": "action",
        "toolName": "list_dir",
        "parameters": { "path": "." },
        "priority": "normal",
        "description": "Listar archivos en el directorio actual"
      },
      {
        "type": "action",
        "toolName": "read_file",
        "parameters": { "file_path": "package.json" },
        "priority": "high",
        "description": "Leer el archivo package.json"
      }
    ],
    "description": "Análisis básico del proyecto"
  }
}
```

#### Ejemplo 2: Procesamiento Paralelo

```json
{
  "id": "parallel_processing_001",
  "rootNode": {
    "type": "parallel",
    "nodes": [
      {
        "type": "action",
        "toolName": "run_shell_command",
        "parameters": { "command": "npm test" },
        "priority": "high",
        "description": "Ejecutar tests"
      },
      {
        "type": "action",
        "toolName": "run_shell_command",
        "parameters": { "command": "npm run lint" },
        "priority": "normal",
        "description": "Ejecutar linter"
      }
    ],
    "maxConcurrency": 2,
    "description": "Ejecutar tests y linting en paralelo"
  }
}
```

#### Ejemplo 3: Lógica Condicional

```json
{
  "id": "conditional_build_001",
  "rootNode": {
    "type": "condition",
    "condition": "file_exists('dist/')",
    "thenNode": {
      "type": "action",
      "toolName": "run_shell_command",
      "parameters": { "command": "npm run build:incremental" },
      "priority": "normal",
      "description": "Build incremental"
    },
    "elseNode": {
      "type": "action",
      "toolName": "run_shell_command",
      "parameters": { "command": "npm run build:full" },
      "priority": "normal",
      "description": "Build completo"
    },
    "description": "Elegir tipo de build basado en existencia del directorio dist"
  }
}
```

### Instrucciones para el LLM

1. **Usa Action Scripts cuando**: La tarea requiera múltiples herramientas, tenga dependencias entre acciones, o pueda beneficiarse de ejecución paralela.

2. **Prioriza apropiadamente**: Usa `critical` para acciones que bloqueen otras, `high` para acciones importantes, `normal` para la mayoría, y `low` para acciones opcionales.

3. **Sé específico**: Incluye descripciones claras para cada acción y script para facilitar la comprensión y debugging.

4. **Optimiza la concurrencia**: Usa `parallel` cuando las acciones sean independientes y puedan ejecutarse simultáneamente.

5. **Maneja errores**: Considera usar condicionales para manejar casos donde las acciones podrían fallar.

6. **Mantén la simplicidad**: No anides estructuras demasiado profundamente; prefiere secuencias lineales cuando sea posible.

### Función Especial para Action Scripts

Para generar un Action Script, usa la siguiente función especial:

```javascript
executeActionScript({
  script: {
    id: 'script_001',
    rootNode: {
      // Tu script aquí
    },
  },
});
```

O simplemente incluye el JSON del script directamente en tus parámetros si usas una función genérica de ejecución de scripts.</content>
<parameter name="filePath">/media/seikarii/Nvme/gemini-cli/packages/core/src/core/action-script-prompts.md
