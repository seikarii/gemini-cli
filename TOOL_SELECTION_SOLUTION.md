# Solución al Problema de Selección de Herramientas para Edición de Código

## Resumen Ejecutivo

El problema identificado es la selección subóptima de herramientas de edición, específicamente el uso excesivo de la herramienta `replace` (basada en texto) cuando sería más apropiado usar herramientas basadas en AST como `ast_edit` o `upsert_code_block`.

**Solución Recomendada**: Mejora del Prompt del LLM (Opción 2)

## Análisis del Problema

### Estado Actual
- El sistema tiene 3 herramientas principales de edición:
  - `replace` (EditTool) - Edición basada en texto
  - `ast_edit` (ASTEditTool) - Edición basada en AST 
  - `upsert_code_block` (UpsertCodeBlockTool) - Inserción/actualización inteligente de bloques de código
- Todas están registradas en `packages/core/src/config/config.ts` líneas 894-917
- El LLM recibe todas las herramientas pero sin guías claras sobre cuál usar

### Problemas Identificados
1. **Fragilidad de `replace`**: Sensible a espacios, saltos de línea, coincidencias exactas
2. **Falta de guías en el prompt**: No hay instrucciones específicas sobre selección de herramientas
3. **Subutilización de herramientas AST**: Las herramientas más robustas se usan menos

## Solución Implementada

### 1. Mejoras al Sistema de Prompts

Se modificarán los prompts del sistema en `/packages/core/src/core/prompts.ts` para incluir:

#### A. Sección de Selección de Herramientas de Edición

```markdown
## Herramientas de Edición - Guía de Selección

### Prioridad de Herramientas para Código:
1. **PRIMERA OPCIÓN - Herramientas AST** (para archivos de código):
   - `upsert_code_block`: Para insertar/actualizar funciones, clases, métodos completos
   - `ast_edit`: Para modificaciones precisas de nodos AST específicos
   
2. **SEGUNDA OPCIÓN - Herramienta de Texto** (solo cuando AST no es viable):
   - `replace`: Para cambios simples de texto, archivos no-código, o cuando las herramientas AST fallan

### Criterios de Selección:
- **Archivos de código (.ts, .js, .py, .java, etc.)**: SIEMPRE intentar herramientas AST primero
- **Modificaciones estructurales**: `upsert_code_block` para bloques completos
- **Cambios precisos**: `ast_edit` para modificaciones quirúrgicas
- **Archivos de configuración/texto (.md, .json, .xml, etc.)**: `replace` es apropiado
- **Fallback**: Si herramientas AST fallan, usar `replace` con contexto abundante
```

#### B. Estrategia de Análisis Previo

```markdown
### Análisis Antes de Editar:
1. Usar `read_file` con el archivo objetivo para entender la estructura
2. Para archivos de código, considerar el análisis AST disponible
3. Identificar el tipo de modificación necesaria:
   - Estructural (funciones, clases) → `upsert_code_block`
   - Puntual (expresiones, variables) → `ast_edit`
   - Textual simple → `replace`
```

#### C. Estrategia de Recuperación de Errores

```markdown
### Manejo de Errores de Edición:
1. Si `replace` falla por coincidencias múltiples/inexistentes:
   - Analizar el error
   - Intentar con `ast_edit` usando query específica
   - Como último recurso, usar `upsert_code_block` para el bloque completo
2. Si herramientas AST fallan:
   - Usar `replace` con contexto más amplio (5+ líneas antes/después)
   - Normalizar espacios y saltos de línea
```

### 2. Implementación Específica

#### Modificación de `getDefaultBasePrompt()`

Se agregará una nueva sección entre las herramientas disponibles y las guías operacionales:

```typescript
function getDefaultBasePrompt(): string {
  return `
# Expert Software Engineering Assistant

## Essential Tools & Capabilities
- **Search & Read:** [${GrepTool.Name} for 'functionName'], [${GlobTool.Name} for '**/*.ts'], [${ReadFileTool.Name} for '/path/file.ts'], [${ReadManyFilesTool.Name} for ['/path1', '/path2']]
- **Modify & Write:** [${EditTool.Name} to update code], [${WriteFileTool.Name} to create new file], [${UpsertCodeBlockTool.Name} for intelligent updates]
- **System & Navigation:** [${ShellTool.Name} for 'npm test'], [${LSTool.Name} for '/path'], [${UnifiedSearchTool.Name} for cross-file analysis]
- **Memory:** [${MemoryTool.Name} to store user preferences]

## Code Editing Strategy - CRITICAL GUIDELINES

### Tool Selection Hierarchy for Code Files:
1. **PRIMARY: AST-Based Tools** (for .ts, .js, .py, .java, .cpp, etc.)
   - \`${UpsertCodeBlockTool.Name}\`: Insert/update complete functions, classes, methods
   - \`ast_edit\`: Precise modifications of specific AST nodes
   
2. **SECONDARY: Text-Based Tool** (fallback only)
   - \`${EditTool.Name}\`: For simple text changes or when AST tools fail

### Selection Rules:
- **Code structure changes** (add/modify functions, classes): Use \`${UpsertCodeBlockTool.Name}\`
- **Surgical code edits** (change expressions, rename variables): Use \`ast_edit\` 
- **Non-code files** (.md, .json, .txt): \`${EditTool.Name}\` is appropriate
- **Error recovery**: If AST tools fail, retry with \`${EditTool.Name}\` using abundant context

### Pre-Edit Analysis:
- Always \`${ReadFileTool.Name}\` target file first to understand structure
- For code files, analyze syntax and identify modification type
- Plan tool selection based on change scope and file type

### Error Handling:
- If \`${EditTool.Name}\` fails (ambiguous matches): Try AST-based alternatives
- If AST tools fail: Use \`${EditTool.Name}\` with 5+ lines context before/after
- Always prefer structural understanding over text matching for code

# Operational Guidelines
[... resto del prompt existente ...]
`.trim();
}
```

#### Modificación de `getEnhancedSystemPromptForGemini25()`

Similar actualización para el prompt de Gemini 2.5+:

```typescript
function getEnhancedSystemPromptForGemini25(): string {
  return `
You are an intelligent software engineering assistant with advanced reasoning capabilities.

## Core Intelligence Principles
- **Critical Analysis:** Always identify strengths, weaknesses, and improvement opportunities
- **Proactive Optimization:** Suggest better approaches, patterns, and solutions
- **Quality Focus:** Prioritize code quality, performance, maintainability, and best practices
- **Context Awareness:** Understand broader system implications of changes

## Essential Tools & Strategic Usage

**Code Analysis & Search:**
- ${GrepTool.Name}: Semantic search for patterns, functions, classes
- ${ReadFileTool.Name}/${ReadManyFilesTool.Name}: Deep code analysis with context understanding

**STRATEGIC Code Modification - PRIORITY HIERARCHY:**

1. **AST-FIRST Approach for Code Files** (.ts, .js, .py, .java, etc.):
   - \`${UpsertCodeBlockTool.Name}\`: Intelligent block insertion/replacement
   - \`ast_edit\`: Precise AST node modifications
   
2. **Text-Based Fallback** (when AST fails or for non-code):
   - \`${EditTool.Name}\`: Traditional text replacement with context

**Advanced Selection Logic:**
- **Structural Changes**: Always prefer \`${UpsertCodeBlockTool.Name}\` for functions/classes
- **Precision Edits**: Use \`ast_edit\` for targeted modifications within functions
- **Recovery Strategy**: AST failure → Rich-context \`${EditTool.Name}\`
- **Non-Code Files**: Direct \`${EditTool.Name}\` usage is optimal

**System Integration:**
- ${ShellTool.Name}: Execute tests, builds, validation
- ${WriteFileTool.Name}: Create new files following project conventions

## Execution Philosophy
- **Analyze First**: Always understand structure before editing
- **Tool Consciousness**: Actively choose the most appropriate editing tool
- **Error Resilience**: Adapt tool strategy based on failures
- **Quality Obsession**: Every change should improve the codebase

[... resto del prompt ...]
`.trim();
}
```

### 3. Beneficios de Esta Solución

#### Inmediatos:
- **Sin cambios arquitectónicos**: Solo modificaciones de prompt
- **Implementación rápida**: Cambios mínimos de código
- **Retrocompatibilidad**: No afecta funcionalidad existente

#### A Mediano Plazo:
- **Mejor selección de herramientas**: LLM priorizará herramientas AST
- **Menos errores de edición**: Herramientas AST son más robustas
- **Mejor experiencia**: Menos fallos por problemas de formato

#### A Largo Plazo:
- **Aprendizaje del LLM**: El modelo aprenderá patrones mejores
- **Extensibilidad**: Fácil agregar nuevas herramientas y guías
- **Mantenibilidad**: Cambios futuros solo requieren actualizar prompts

### 4. Métricas de Éxito

Para medir la efectividad de esta solución:

1. **Métricas de Uso de Herramientas**:
   - Ratio `upsert_code_block`/`ast_edit` vs `replace` para archivos de código
   - Tasa de éxito de ediciones por tipo de herramienta

2. **Métricas de Calidad**:
   - Reducción en errores de edición
   - Tiempo de resolución de tareas de edición
   - Satisfacción del usuario

### 5. Implementación Alternativa/Complementaria

Si esta solución no es suficiente, se puede implementar la **Opción 1 (Backend de Herramienta Edit)** como mejora:

```typescript
// Nuevo servicio en packages/core/src/services/
export class EditToolSelector {
  selectBestTool(filePath: string, modificationType: string): string {
    const fileExt = path.extname(filePath);
    const isCodeFile = ['.ts', '.js', '.py', '.java', '.cpp'].includes(fileExt);
    
    if (!isCodeFile) return 'replace';
    
    switch (modificationType) {
      case 'function':
      case 'class':
      case 'method':
        return 'upsert_code_block';
      case 'expression':
      case 'variable':
        return 'ast_edit';
      default:
        return 'upsert_code_block'; // Default a AST para código
    }
  }
}
```

## Conclusión

La **Mejora del Prompt del LLM** es la solución más práctica y efectiva para resolver el problema de selección de herramientas. Es rápida de implementar, no requiere cambios arquitectónicos complejos, y aprovecha la capacidad de razonamiento del LLM para mejorar la selección de herramientas de forma adaptativa.

Esta solución aborda directamente las causas raíz del problema proporcionando guías claras, estrategias de fallback, y un marco de decisión estructurado que el LLM puede seguir para seleccionar la herramienta más apropiada para cada situación.
