# Action System - Guía Técnica para Desarrolladores

## Resumen

El Action System es una infraestructura avanzada para la orquestación inteligente de herramientas en el Gemini CLI. Permite a los LLMs expresar secuencias complejas de acciones con control de prioridades, ejecución concurrente y manejo robusto de errores.

## Arquitectura

### Componentes Principales

#### 1. Action System (`action-system.ts`)

- **Propósito**: Gestiona la cola de acciones, prioridades y ejecución concurrente
- **Características**:
  - Cola de prioridades adaptable
  - Ejecución concurrente con límites configurables
  - Sistema de métricas y estadísticas
  - Manejo de timeouts y reintentos
  - Historial de acciones completadas

#### 2. Action Script Parser (`action-script.ts`)

- **Propósito**: Parsea y valida scripts JSON que expresan lógica compleja
- **Características**:
  - Validación de estructura JSON
  - Soporte para nodos anidados
  - Conversión de scripts a acciones individuales
  - Sistema de tipos TypeScript completo

#### 3. Core Tool Scheduler (`coreToolScheduler.ts`)

- **Propósito**: Punto de entrada para ejecutar tanto tool calls individuales como action scripts
- **Características**:
  - Integración con Action System
  - Compatibilidad con sistema existente
  - Manejo de aprobaciones y confirmaciones
  - Conversión entre formatos

#### 4. Turn Handler (`turn.ts`)

- **Propósito**: Detecta y procesa action scripts en respuestas del LLM
- **Características**:
  - Detección automática de action scripts
  - Emisión de eventos especializados
  - Integración con flujo de conversación existente

## Uso Programático

### Ejecutar un Action Script

```typescript
import {
  CoreToolScheduler,
  ActionScriptRequestInfo,
} from '@google/gemini-cli-core';

const scheduler = new CoreToolScheduler({
  config: myConfig,
  getPreferredEditor: () => 'vscode',
  onEditorClose: () => {},
  onAllToolCallsComplete: (results) => {
    console.log('Action Script completado:', results);
  },
});

const actionScriptRequest: ActionScriptRequestInfo = {
  callId: 'script_001',
  script: {
    id: 'script_001',
    rootNode: {
      type: 'sequence',
      nodes: [
        {
          type: 'action',
          toolName: 'read_file',
          parameters: { file_path: 'package.json' },
          priority: 'high',
        },
      ],
    },
  },
  prompt_id: 'prompt_123',
};

const results = await scheduler.executeActionScript(
  actionScriptRequest,
  abortSignal,
);
```

### Crear un Action Script Programáticamente

```typescript
import { ActionScriptBuilder } from '@google/gemini-cli-core';

const builder = new ActionScriptBuilder();

// Crear un script de secuencia
const script = builder
  .sequence('Análisis de proyecto')
  .action('list_dir', { path: '.' }, 'normal', 'Listar archivos')
  .action(
    'read_file',
    { file_path: 'package.json' },
    'high',
    'Leer package.json',
  )
  .parallel('Procesamiento paralelo')
  .action(
    'run_shell_command',
    { command: 'npm test' },
    'normal',
    'Ejecutar tests',
  )
  .action(
    'run_shell_command',
    { command: 'npm run lint' },
    'normal',
    'Ejecutar linter',
  )
  .build();

console.log(JSON.stringify(script, null, 2));
```

## Formato de Action Scripts

### Estructura Básica

```json
{
  "id": "string",
  "rootNode": {
    "type": "sequence|parallel|condition|loop|action|variable",
    "nodes": [...], // Para sequence/parallel
    "toolName": "string", // Para action
    "parameters": {...}, // Para action
    "condition": "string", // Para condition
    "thenNode": {...}, // Para condition
    "elseNode": {...}, // Para condition (opcional)
    "variable": "string", // Para loop
    "iterable": [...], // Para loop
    "body": {...}, // Para loop
    "name": "string", // Para variable
    "value": "any", // Para variable
    "priority": "low|normal|high|critical", // Para action
    "description": "string" // Opcional para todos
  }
}
```

### Tipos de Nodos

#### Action Node

```json
{
  "type": "action",
  "toolName": "read_file",
  "parameters": {
    "file_path": "/path/to/file.txt",
    "encoding": "utf-8"
  },
  "priority": "normal",
  "description": "Leer contenido de archivo"
}
```

#### Sequence Node

```json
{
  "type": "sequence",
  "nodes": [
    {
      /* action 1 */
    },
    {
      /* action 2 */
    }
  ],
  "description": "Ejecutar en orden"
}
```

#### Parallel Node

```json
{
  "type": "parallel",
  "nodes": [
    {
      /* action 1 */
    },
    {
      /* action 2 */
    }
  ],
  "maxConcurrency": 3,
  "description": "Ejecutar en paralelo"
}
```

#### Condition Node

```json
{
  "type": "condition",
  "condition": "file_exists('/path/to/file.txt')",
  "thenNode": {
    /* action si true */
  },
  "elseNode": {
    /* action si false */
  },
  "description": "Ejecutar condicionalmente"
}
```

#### Loop Node

```json
{
  "type": "loop",
  "variable": "item",
  "iterable": ["a", "b", "c"],
  "body": {
    /* action usando variable */
  },
  "description": "Iterar sobre elementos"
}
```

#### Variable Node

```json
{
  "type": "variable",
  "name": "myVar",
  "value": "someValue",
  "description": "Definir variable"
}
```

## Configuración del Action System

```typescript
const actionSystem = new ActionSystem({
  maxConcurrentActions: 5, // Máximo de acciones concurrentes
  maxQueueSize: 100, // Tamaño máximo de cola
  enableActionHistory: true, // Habilitar historial
  maxHistorySize: 1000, // Tamaño máximo de historial
  autoCleanupCompleted: true, // Limpiar automáticamente completadas
  cleanupInterval: 300, // Intervalo de limpieza (segundos)
  enablePriorityScheduling: true, // Habilitar scheduling por prioridad
  defaultTimeout: 300, // Timeout por defecto (segundos)
  adaptivePriority: true, // Adaptar prioridades dinámicamente
});
```

## Manejo de Eventos

### En Turn.ts

```typescript
// Los action scripts se detectan automáticamente y emiten:
{
  type: 'action_script_request',
  value: ActionScriptRequestInfo
}
```

### En useGeminiStream.ts

```typescript
// Se procesan automáticamente usando CoreToolScheduler
if (actionScriptRequests.length > 0) {
  await executeActionScripts(actionScriptRequests, signal);
}
```

## Métricas y Monitoreo

El Action System proporciona métricas detalladas:

```typescript
const stats = actionSystem.getStats();
console.log('Total acciones:', stats.totalActionsCreated);
console.log('Tasa de éxito:', stats.successRate);
console.log('Tiempo promedio:', stats.averageExecutionTime);
console.log('Uso de herramientas:', stats.toolsUsage);
```

## Integración con Sistema Existente

### Compatibilidad

- ✅ Funciona con tool calls individuales existentes
- ✅ Mantiene interfaz `ToolCallResponseInfo`
- ✅ Compatible con sistema de aprobaciones
- ✅ Integrado con logging y telemetry

### Migración

- Los tool calls individuales siguen funcionando sin cambios
- Action Scripts se detectan automáticamente
- Sistema de prioridades es opcional (default: normal)
- Manejo de errores mantiene compatibilidad

## Debugging y Troubleshooting

### Logs Útiles

```typescript
// Habilitar logs detallados
console.log('Action Script parseado:', parsedScript);
console.log('Acciones creadas:', actions.length);
console.log('Resultados de ejecución:', results);
```

### Errores Comunes

1. **Script JSON inválido**: Verificar sintaxis JSON
2. **Tool no encontrado**: Asegurar que la herramienta esté registrada
3. **Parámetros incorrectos**: Verificar esquema de herramienta
4. **Timeout**: Ajustar `defaultTimeout` en configuración

### Testing

```typescript
// Crear tests unitarios
import { ActionScriptParser } from '@google/gemini-cli-core';

const parser = new ActionScriptParser();
const script = parser.parse(validScriptJson);
// Verificar estructura...
```

## Próximos Pasos

1. **Prompt Engineering**: Crear prompts específicos para diferentes tipos de tareas
2. **Optimización**: Implementar caching y optimizaciones de rendimiento
3. **UI Integration**: Mejorar visualización en interfaz de usuario
4. **Testing**: Crear suite completa de tests
5. **Documentación**: Expandir ejemplos y casos de uso

---

**Nota**: Esta documentación es para desarrolladores que integran o extienden el Action System. Para uso de Action Scripts por parte de LLMs, ver `action-script-prompts.md`.</content>
<parameter name="filePath">/media/seikarii/Nvme/gemini-cli/packages/core/src/core/action-system-developer-guide.md
