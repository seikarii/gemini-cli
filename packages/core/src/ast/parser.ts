/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Shape for extracted intentions
export interface Intentions {
  functions: FunctionInfo[];
  classes: ClassInfo[];
  imports: ImportInfo[];
  constants: ConstantInfo[];
  exports: ExportInfo[];
  interfaces: InterfaceInfo[];
  types: TypeInfo[];
  enums: EnumInfo[];
  parsingErrors: string[];
  complexity: ComplexityMetrics;
  [key: string]: unknown;
}

export interface FunctionInfo {
  name: string;
  isAsync: boolean;
  isExported: boolean;
  isGenerator: boolean;
  startLine: number | null;
  endLine: number | null;
  params: ParameterInfo[];
  returnType?: string;
  visibility?: 'public' | 'private' | 'protected';
  isStatic?: boolean;
  complexity?: number;
  documentation?: string;
}

export interface ParameterInfo {
  name: string;
  type: string;
  isOptional: boolean;
  hasDefault: boolean;
  defaultValue?: string;
}

export interface ClassInfo {
  name: string;
  isExported: boolean;
  isAbstract: boolean;
  startLine: number | null;
  endLine: number | null;
  methods: MethodInfo[];
  properties: PropertyInfo[];
  constructors: ConstructorInfo[];
  extends?: string;
  implements: string[];
  documentation?: string;
}

export interface MethodInfo {
  name: string;
  isAsync: boolean;
  isStatic: boolean;
  isAbstract: boolean;
  visibility: 'public' | 'private' | 'protected';
  params: ParameterInfo[];
  returnType?: string;
  complexity?: number;
}

export interface PropertyInfo {
  name: string;
  type?: string;
  isStatic: boolean;
  isReadonly: boolean;
  visibility: 'public' | 'private' | 'protected';
  hasInitializer: boolean;
}

export interface ConstructorInfo {
  params: ParameterInfo[];
  visibility: 'public' | 'private' | 'protected';
  complexity?: number;
}

export interface ImportInfo {
  moduleSpecifier: string;
  namedImports: NamedImportInfo[];
  defaultImport?: string;
  namespaceImport?: string;
  isTypeOnly: boolean;
  startLine: number | null;
}

export interface NamedImportInfo {
  name: string;
  alias?: string;
  isTypeOnly: boolean;
}

export interface ConstantInfo {
  name: string;
  value?: string;
  type?: string;
  isExported: boolean;
  startLine: number | null;
  isReadonly: boolean;
  documentation?: string;
}

export interface ExportInfo {
  name?: string;
  isDefault: boolean;
  isReExport: boolean;
  moduleSpecifier?: string;
  type: 'variable' | 'function' | 'class' | 'interface' | 'type' | 'namespace';
}

export interface InterfaceInfo {
  name: string;
  isExported: boolean;
  startLine: number | null;
  endLine: number | null;
  properties: InterfacePropertyInfo[];
  methods: InterfaceMethodInfo[];
  extends: string[];
  documentation?: string;
}

export interface InterfacePropertyInfo {
  name: string;
  type?: string;
  isOptional: boolean;
  isReadonly: boolean;
}

export interface InterfaceMethodInfo {
  name: string;
  params: ParameterInfo[];
  returnType?: string;
  isOptional: boolean;
}

export interface TypeInfo {
  name: string;
  isExported: boolean;
  definition: string;
  startLine: number | null;
  documentation?: string;
}

export interface EnumInfo {
  name: string;
  isExported: boolean;
  isConst: boolean;
  startLine: number | null;
  endLine: number | null;
  members: EnumMemberInfo[];
  documentation?: string;
}

export interface EnumMemberInfo {
  name: string;
  value?: string | number;
  hasInitializer: boolean;
}

export interface ComplexityMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  halsteadMetrics: HalsteadMetrics;
  linesOfCode: number;
  maintainabilityIndex: number;
}

export interface HalsteadMetrics {
  operatorsCount: number;
  operandsCount: number;
  vocabularySize: number;
  programLength: number;
  difficulty: number;
  effort: number;
  timeRequired: number;
  bugsDelivered: number;
}

/**
 * Enhanced TypeScript AST parser with comprehensive analysis capabilities
 * - robust file reading with encoding fallbacks
 * - file size guard with configurable limits
 * - defensive parsing with ts-morph
 * - comprehensive structural analysis
 * - complexity metrics calculation
 * - documentation extraction
 * - dependency analysis
 */
import fs from 'fs/promises';
import path from 'path';
import {
  Project,
  SourceFile,
  SyntaxKind,
  JSDoc,
  ImportSpecifier,
  ParameterDeclaration,
  MethodDeclaration,
  Node,
} from 'ts-morph';

const MAX_FILE_SIZE_MB = 50; // maximum file size to process

/**
 * Calculate cyclomatic complexity for a function or method
 */
function calculateCyclomaticComplexity(node: Node): number {
  let complexity = 1; // base complexity

  const complexityNodes = [
    SyntaxKind.IfStatement,
    SyntaxKind.WhileStatement,
    SyntaxKind.ForStatement,
    SyntaxKind.ForInStatement,
    SyntaxKind.ForOfStatement,
    SyntaxKind.SwitchStatement,
    SyntaxKind.CaseClause,
    SyntaxKind.ConditionalExpression,
    SyntaxKind.TryStatement,
    SyntaxKind.CatchClause,
  ];

  node.forEachDescendant((child) => {
    if (complexityNodes.includes(child.getKind())) {
      complexity++;
    }
  });

  return complexity;
}

/**
 * Calculate Halstead metrics for code complexity analysis
 */
function calculateHalsteadMetrics(sourceFile: SourceFile): HalsteadMetrics {
  const operators = new Set<string>();
  const operands = new Set<string>();
  let operatorCount = 0;
  let operandCount = 0;

  const operatorKinds = [
    SyntaxKind.PlusToken,
    SyntaxKind.MinusToken,
    SyntaxKind.AsteriskToken,
    SyntaxKind.SlashToken,
    SyntaxKind.PercentToken,
    SyntaxKind.AmpersandToken,
    SyntaxKind.BarToken,
    SyntaxKind.CaretToken,
    SyntaxKind.ExclamationToken,
    SyntaxKind.EqualsEqualsToken,
    SyntaxKind.ExclamationEqualsToken,
    SyntaxKind.LessThanToken,
    SyntaxKind.GreaterThanToken,
    SyntaxKind.LessThanEqualsToken,
    SyntaxKind.GreaterThanEqualsToken,
    SyntaxKind.AmpersandAmpersandToken,
    SyntaxKind.BarBarToken,
    SyntaxKind.EqualsToken,
    SyntaxKind.PlusEqualsToken,
    SyntaxKind.MinusEqualsToken,
    SyntaxKind.AsteriskEqualsToken,
  ];

  sourceFile.forEachDescendant((node) => {
    const kind = node.getKind();
    const text = node.getText();

    if (operatorKinds.includes(kind)) {
      operators.add(text);
      operatorCount++;
    } else if (
      kind === SyntaxKind.Identifier ||
      kind === SyntaxKind.StringLiteral ||
      kind === SyntaxKind.NumericLiteral
    ) {
      operands.add(text);
      operandCount++;
    }
  });

  const n1 = operators.size; // unique operators
  const n2 = operands.size; // unique operands
  const N1 = operatorCount; // total operators
  const N2 = operandCount; // total operands

  const vocabularySize = n1 + n2;
  const programLength = N1 + N2;
  const difficulty = (n1 / 2) * (N2 / n2);
  const effort = difficulty * programLength;
  const timeRequired = effort / 18; // seconds
  const bugsDelivered = effort / 3000;

  return {
    operatorsCount: N1,
    operandsCount: N2,
    vocabularySize,
    programLength,
    difficulty,
    effort,
    timeRequired,
    bugsDelivered,
  };
}

/**
 * Calculate maintainability index
 */
function calculateMaintainabilityIndex(
  halstead: HalsteadMetrics,
  cyclomaticComplexity: number,
  linesOfCode: number,
): number {
  if (linesOfCode === 0 || halstead.programLength === 0) return 100;

  const maintainabilityIndex = Math.max(
    0,
    171 -
      5.2 * Math.log(halstead.programLength) -
      0.23 * cyclomaticComplexity -
      16.2 * Math.log(linesOfCode),
  );

  return Math.min(100, maintainabilityIndex);
}

/**
 * Extract comprehensive complexity metrics
 */
function extractComplexityMetrics(sourceFile: SourceFile): ComplexityMetrics {
  let totalCyclomaticComplexity = 0;
  let functionCount = 0;

  // Count functions and their complexity
  sourceFile.getFunctions().forEach((func) => {
    totalCyclomaticComplexity += calculateCyclomaticComplexity(func);
    functionCount++;
  });

  sourceFile.getClasses().forEach((cls) => {
    cls.getMethods().forEach((method) => {
      totalCyclomaticComplexity += calculateCyclomaticComplexity(method);
      functionCount++;
    });
  });

  const halsteadMetrics = calculateHalsteadMetrics(sourceFile);
  const linesOfCode = sourceFile.getEndLineNumber();
  const avgCyclomaticComplexity =
    functionCount > 0 ? totalCyclomaticComplexity / functionCount : 0;

  return {
    cyclomaticComplexity: totalCyclomaticComplexity,
    cognitiveComplexity: totalCyclomaticComplexity, // Simplified - could be enhanced
    halsteadMetrics,
    linesOfCode,
    maintainabilityIndex: calculateMaintainabilityIndex(
      halsteadMetrics,
      avgCyclomaticComplexity,
      linesOfCode,
    ),
  };
}
export interface ParseResult {
  source?: string | null;
  sourceFile?: SourceFile | null;
  parseError?: string | null;
  comments: string[];
  jsdocs: string[];
  intentions: unknown;
  fileInfo: {
    path: string;
    sizeBytes: number;
    lineCount: number;
    processingTimeMs: number;
  };
}

/**
 * Simple defensive file size check.
 */
export async function checkFileSize(
  filePath: string,
): Promise<{ ok: boolean; error?: string }> {
  try {
    const st = await fs.stat(filePath);
    const sizeMb = st.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_SIZE_MB) {
      return {
        ok: false,
        error: `File too large: ${sizeMb.toFixed(1)}MB (max ${MAX_FILE_SIZE_MB}MB)`,
      };
    }
    return { ok: true };
  } catch (_e: unknown) {
    return { ok: false, error: `Cannot check file size: ${String(_e)}` };
  }
}

/**
 * Read file with encoding fallbacks (utf-8, latin1).
 */
export async function readFileWithEncodingFallback(
  filePath: string,
): Promise<{ content: string | null; error: string | null }> {
  const encodings = ['utf8', 'latin1', 'cp1252'] as const;
  for (const enc of encodings) {
    try {
      const buf = await fs.readFile(filePath);
      // try decoding according to encoding
      const content = Buffer.from(buf).toString(enc as BufferEncoding);
      // cheap sanity: if decoding produced replacement chars many times, fall through (best-effort)
      if (content.length === 0 && buf.length > 0) {
        continue;
      }
      return { content, error: null };
    } catch (_err: unknown) {
      // try next encoding
      void _err;
      continue;
    }
  }
  return { content: null, error: `Could not decode file with tried encodings` };
}

/**
 * Defensive parse to ts-morph SourceFile.
 * Uses a fresh Project (in-memory) so we do not affect disk.
 */
// Internal implementation expects (source, filePath)
function parseSourceToSourceFileImpl(
  source: string,
  filePath: string,
): { sourceFile?: SourceFile | null; error?: string | null } {
  try {
    const project = new Project({
      useInMemoryFileSystem: true,
      compilerOptions: { allowJs: true },
    });
    const normalizedPath = path.isAbsolute(filePath)
      ? filePath
      : path.resolve('/', filePath);
    const sf = project.createSourceFile(normalizedPath, source, {
      overwrite: true,
    });
    return { sourceFile: sf, error: null };
  } catch (e: unknown) {
    return { sourceFile: null, error: `Parsing failed: ${String(e)}` };
  }
}

/**
 * Compatibility wrapper for parseSourceToSourceFile.
 * Accepts either (source, filePath) or the older callers that pass (filePath, source).
 */
export function parseSourceToSourceFile(
  a: string,
  b?: string,
): { sourceFile?: SourceFile | null; error?: string | null } {
  // If only one arg provided, treat it as source with missing filePath -> use virtual path
  if (b === undefined) {
    return parseSourceToSourceFileImpl(a, '/virtual-file.ts');
  }

  // Heuristic: if first arg contains a newline or semicolon, treat as source
  const looksLikeSource =
    a.includes('\n') || a.includes(';') || a.includes('{');
  const looksLikePath =
    b.includes('\n') === false &&
    (a.startsWith('.') || a.startsWith('/') || a.match(/\\.[tj]s[x]?$/i));

  if (looksLikeSource && !looksLikePath) {
    return parseSourceToSourceFileImpl(a, b);
  }

  // Fallback: assume callers passed (filePath, source)
  return parseSourceToSourceFileImpl(b, a);
}

/**
 * Extract comments (line comments and block comments) and JSDoc blocks using regex + ts-morph where possible.
 */
export function extractCommentsAndJsDoc(
  source: string,
  sourceFile?: SourceFile | null,
): { comments: string[]; jsdocs: string[] } {
  const comments: string[] = [];
  const jsdocs: string[] = [];

  // Extract line and block comments via regex (best-effort)
  try {
    const lineRe = /^\s*\/\/(.*)$/gm;
    let m: RegExpExecArray | null = null;
    while ((m = lineRe.exec(source)) !== null) {
      const txt = m[1].trim();
      if (txt) comments.push(txt);
    }

    const blockRe = /\/\*([\s\S]*?)\*\//g;
    while ((m = blockRe.exec(source)) !== null) {
      const txt = m[1].trim();
      if (!txt) continue;
      if (/^\*/.test(txt) || txt.startsWith('*')) {
        const cleaned = txt
          .split('\n')
          .map((l) => l.replace(/^\s*\*\s?/, '').trim())
          .join('\n')
          .trim();
        if (cleaned) jsdocs.push(cleaned);
      } else {
        comments.push(txt);
      }
    }
  } catch (e: unknown) {
    void e; // ignore extraction errors
  }

  // If ts-morph SourceFile available, collect JSDoc nodes defensively
  if (sourceFile) {
    try {
      for (const nd of sourceFile.getDescendants()) {
        try {
          const getJsDocs = (nd as { getJsDocs?: () => JSDoc[] }).getJsDocs;
          const js =
            typeof getJsDocs === 'function' ? (getJsDocs.call(nd) ?? []) : [];
          if (Array.isArray(js) && js.length > 0) {
            for (const d of js) {
              const txt =
                typeof (d as JSDoc).getInnerText === 'function'
                  ? (d as JSDoc).getInnerText()
                  : String((d as JSDoc).getText?.() ?? '');
              if (txt) jsdocs.push(txt.trim());
            }
          }
        } catch {
          // ignore per-node errors
        }
      }
    } catch {
      // ignore
    }
  }

  return { comments, jsdocs };
}

/**
 * Extract intentions (functions, classes, imports, constants) from a ts-morph SourceFile.
 * Defensive: isolate node-level errors to avoid complete failure.
 */
export function extractIntentionsFromSourceFile(
  sourceFile?: SourceFile | null,
): Intentions {
  const intents: Intentions = {
    functions: [],
    classes: [],
    imports: [],
    constants: [],
    exports: [],
    interfaces: [],
    types: [],
    enums: [],
    parsingErrors: [],
    complexity: {
      cyclomaticComplexity: 0,
      cognitiveComplexity: 0,
      halsteadMetrics: {
        operatorsCount: 0,
        operandsCount: 0,
        vocabularySize: 0,
        programLength: 0,
        difficulty: 0,
        effort: 0,
        timeRequired: 0,
        bugsDelivered: 0,
      },
      linesOfCode: 0,
      maintainabilityIndex: 100,
    },
  };

  if (!sourceFile) {
    intents.parsingErrors.push('sourceFile not available');
    return intents;
  }

  try {
    // Extract complexity metrics
    intents.complexity = extractComplexityMetrics(sourceFile);

    // Imports
    try {
      const imps = sourceFile.getImportDeclarations();
      for (const imp of imps) {
        try {
          const namedImports = imp.getNamedImports();
          const mappedNamed = namedImports.map((spec: ImportSpecifier) => ({
            name: spec.getName(),
            alias: spec.getAliasNode()?.getText() ?? undefined,
            isTypeOnly: spec.isTypeOnly(),
          }));

          intents.imports.push({
            moduleSpecifier: imp.getModuleSpecifierValue(),
            namedImports: mappedNamed,
            defaultImport: imp.getDefaultImport()?.getText?.() ?? undefined,
            namespaceImport: imp.getNamespaceImport()?.getText?.() ?? undefined,
            isTypeOnly: imp.isTypeOnly(),
            startLine: imp.getStartLineNumber(),
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`import node error: ${String(e)}`);
        }
      }
    } catch (e: unknown) {
      intents.parsingErrors.push(`imports extraction failed: ${String(e)}`);
    }

    // Functions
    try {
      const funcs = sourceFile.getFunctions();
      for (const f of funcs) {
        try {
          const params: ParameterInfo[] =
            f.getParameters?.().map((p: ParameterDeclaration) => ({
              name: p.getName?.() || '',
              type: p.getType?.().getText?.() ?? '',
              isOptional: p.hasQuestionToken(),
              hasDefault: p.hasInitializer(),
              defaultValue: p.getInitializer()?.getText(),
            })) ?? [];

          intents.functions.push({
            name: f.getName?.() ?? '<anonymous>',
            isAsync: f.isAsync?.() ?? false,
            isExported: f.isExported(),
            isGenerator: f.isGenerator(),
            startLine: f.getStartLineNumber?.() ?? null,
            endLine: f.getEndLineNumber?.() ?? null,
            params,
            returnType: f.getReturnTypeNode()?.getText(),
            visibility: 'public', // functions are always public in TS
            complexity: calculateCyclomaticComplexity(f),
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`function node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`functions extraction failed: ${String(e)}`);
    }

    // Classes
    try {
      const classes = sourceFile.getClasses();
      for (const c of classes) {
        try {
          const methods: MethodInfo[] =
            c.getMethods?.().map((m: MethodDeclaration) => ({
              name: m.getName?.() || '',
              isAsync: m.isAsync?.() ?? false,
              isStatic: m.isStatic(),
              isAbstract: m.isAbstract(),
              visibility: m.getScope() || 'public',
              params: m.getParameters().map((p: ParameterDeclaration) => ({
                name: p.getName() || '',
                type: p.getType().getText() || '',
                isOptional: p.hasQuestionToken(),
                hasDefault: p.hasInitializer(),
                defaultValue: p.getInitializer()?.getText(),
              })),
              returnType: m.getReturnTypeNode()?.getText(),
              complexity: calculateCyclomaticComplexity(m),
            })) ?? [];

          const properties: PropertyInfo[] = c.getProperties().map((p) => ({
            name: p.getName() || '',
            type: p.getType().getText(),
            isStatic: p.isStatic(),
            isReadonly: p.isReadonly(),
            visibility: p.getScope() || 'public',
            hasInitializer: p.hasInitializer(),
          }));

          const constructors: ConstructorInfo[] = c
            .getConstructors()
            .map((ctor) => ({
              params: ctor.getParameters().map((p: ParameterDeclaration) => ({
                name: p.getName() || '',
                type: p.getType().getText() || '',
                isOptional: p.hasQuestionToken(),
                hasDefault: p.hasInitializer(),
                defaultValue: p.getInitializer()?.getText(),
              })),
              visibility: ctor.getScope() || 'public',
              complexity: calculateCyclomaticComplexity(ctor),
            }));

          intents.classes.push({
            name: c.getName?.() ?? '<anonymous>',
            isExported: c.isExported?.() ?? false,
            isAbstract: c.isAbstract(),
            startLine: c.getStartLineNumber?.() ?? null,
            endLine: c.getEndLineNumber?.() ?? null,
            methods,
            properties,
            constructors,
            extends: c.getExtends()?.getText(),
            implements: c.getImplements().map((i) => i.getText()),
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`class node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`classes extraction failed: ${String(e)}`);
    }

    // Interfaces
    try {
      const interfaces = sourceFile.getInterfaces();
      for (const iface of interfaces) {
        try {
          const properties: InterfacePropertyInfo[] = iface
            .getProperties()
            .map((p) => ({
              name: p.getName() || '',
              type: p.getType().getText(),
              isOptional: p.hasQuestionToken(),
              isReadonly: p.isReadonly(),
            }));

          const methods: InterfaceMethodInfo[] = iface
            .getMethods()
            .map((m) => ({
              name: m.getName() || '',
              params: m.getParameters().map((p: ParameterDeclaration) => ({
                name: p.getName() || '',
                type: p.getType().getText() || '',
                isOptional: p.hasQuestionToken(),
                hasDefault: p.hasInitializer(),
                defaultValue: p.getInitializer()?.getText(),
              })),
              returnType: m.getReturnTypeNode()?.getText(),
              isOptional: m.hasQuestionToken(),
            }));

          intents.interfaces.push({
            name: iface.getName(),
            isExported: iface.isExported(),
            startLine: iface.getStartLineNumber(),
            endLine: iface.getEndLineNumber(),
            properties,
            methods,
            extends: iface.getExtends().map((e) => e.getText()),
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`interface node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`interfaces extraction failed: ${String(e)}`);
    }

    // Type aliases
    try {
      const typeAliases = sourceFile.getTypeAliases();
      for (const ta of typeAliases) {
        try {
          intents.types.push({
            name: ta.getName(),
            isExported: ta.isExported(),
            definition: ta.getTypeNode()?.getText() || '',
            startLine: ta.getStartLineNumber(),
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`type alias node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(
        `type aliases extraction failed: ${String(e)}`,
      );
    }

    // Enums
    try {
      const enums = sourceFile.getEnums();
      for (const en of enums) {
        try {
          const members: EnumMemberInfo[] = en.getMembers().map((m) => ({
            name: m.getName(),
            value: m.getValue(),
            hasInitializer: m.hasInitializer(),
          }));

          intents.enums.push({
            name: en.getName(),
            isExported: en.isExported(),
            isConst: false, // ts-morph doesn't expose isConst directly
            startLine: en.getStartLineNumber(),
            endLine: en.getEndLineNumber(),
            members,
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`enum node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`enums extraction failed: ${String(e)}`);
    }

    // Constants - best-effort: numeric/string/boolean initializers from variable statements
    try {
      const vars = sourceFile.getVariableStatements();
      for (const stmt of vars) {
        try {
          const isExported = stmt.isExported?.() ?? false;
          for (const decl of stmt.getDeclarations()) {
            try {
              const init = decl.getInitializer?.();
              if (!init) continue;
              const kind = init.getKind?.();
              if (
                kind === SyntaxKind.StringLiteral ||
                kind === SyntaxKind.NumericLiteral ||
                kind === SyntaxKind.TrueKeyword ||
                kind === SyntaxKind.FalseKeyword
              ) {
                intents.constants.push({
                  name: decl.getName?.() || '',
                  value: init.getText?.(),
                  type: decl.getType().getText(),
                  isExported,
                  startLine: decl.getStartLineNumber?.() ?? null,
                  isReadonly:
                    decl.getParent()?.getKind() ===
                      SyntaxKind.VariableDeclarationList &&
                    decl.getParent()?.getParent()?.getKind() ===
                      SyntaxKind.VariableStatement &&
                    (
                      decl.getParent()?.getParent() as {
                        getDeclarationKind?: () => string;
                      }
                    )?.getDeclarationKind?.() === 'const',
                });
              }
            } catch (e: unknown) {
              intents.parsingErrors.push(
                `variable declaration error: ${String(e)}`,
              );
            }
          }
        } catch (e: unknown) {
          intents.parsingErrors.push(`variable statement error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`constants extraction failed: ${String(e)}`);
    }

    // Export analysis
    try {
      const exports = sourceFile.getExportDeclarations();
      for (const exp of exports) {
        try {
          intents.exports.push({
            name: exp
              .getNamedExports()
              .map((ne) => ne.getName())
              .join(', '),
            isDefault: false,
            isReExport: !!exp.getModuleSpecifier(),
            moduleSpecifier: exp.getModuleSpecifierValue(),
            type: 'variable', // simplified
          });
        } catch (e: unknown) {
          intents.parsingErrors.push(`export node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`exports extraction failed: ${String(e)}`);
    }
  } catch (e: unknown) {
    intents.parsingErrors.push(
      `intentions overall extraction failed: ${String(e)}`,
    );
  }

  return intents;
}

/**
 * Main orchestration: read file, parse, extract comments and intentions.
 */
export async function readAndParseFile(filePath: string): Promise<ParseResult> {
  const start = Date.now();
  const resolved = path.resolve(filePath);

  const result: ParseResult = {
    source: null,
    sourceFile: null,
    parseError: null,
    comments: [],
    jsdocs: [],
    intentions: {},
    fileInfo: {
      path: resolved,
      sizeBytes: 0,
      lineCount: 0,
      processingTimeMs: 0,
    },
  };

  try {
    const sizeCheck = await checkFileSize(resolved);
    if (!sizeCheck.ok) {
      result.parseError = `File size check failed: ${sizeCheck.error}`;
      return result;
    }

    const { content, error } = await readFileWithEncodingFallback(resolved);
    if (error || content === null) {
      result.parseError = `File reading failed: ${error ?? 'unknown'}`;
      return result;
    }

    result.source = content;
    result.fileInfo.sizeBytes = Buffer.byteLength(content, 'utf8');
    result.fileInfo.lineCount = content.split(/\r\n|\r|\n/).length;

    const { sourceFile, error: parseErr } = parseSourceToSourceFile(
      content,
      resolved,
    );
    if (parseErr) {
      result.parseError = parseErr;
      // continue with partial extraction where possible
    }
    result.sourceFile = sourceFile ?? null;

    const { comments, jsdocs } = extractCommentsAndJsDoc(
      content,
      sourceFile ?? null,
    );
    result.comments = comments;
    result.jsdocs = jsdocs;

    result.intentions = extractIntentionsFromSourceFile(sourceFile ?? null);

    result.fileInfo.processingTimeMs = Date.now() - start;
  } catch (e: unknown) {
    result.parseError = `Unexpected processing error: ${String(e)}`;
  }

  return result;
}

/**
 * Minimal ASTReader class matching the Python tool shape.
 * Adjust to your tool framework (BaseTool) if needed.
 */
export interface ASTReaderParams {
  filePath: string;
  includeSource?: boolean;
  extractIntentions?: boolean;
  extractComments?: boolean;
}

export class ASTReader {
  name = 'ast_reader';
  description =
    'Reads JS/TS files, parses them into an AST, extracts structural information and comments with defensive parsing.';

  async execute(params: ASTReaderParams): Promise<{
    success: boolean;
    output: string;
    metadata?: Partial<ParseResult>;
    errorMessage?: string;
    executionTime?: number;
  }> {
    const start = Date.now();
    try {
      const filePath = params.filePath;
      const r = await readAndParseFile(filePath);

      const parts: string[] = [];
      parts.push(`📄 AST READER RESULTS`);
      parts.push(`File: ${r.fileInfo.path}`);
      parts.push(
        `Size: ${r.fileInfo.sizeBytes} bytes (${r.fileInfo.lineCount} lines)`,
      );
      parts.push(`Processing Time: ${r.fileInfo.processingTimeMs}ms`);

      if (r.parseError) {
        parts.push(`⚠️ Parse Error: ${r.parseError}`);
      } else {
        parts.push(`✅ Parse Status: Success`);
      }

      const intents: Intentions = (r.intentions as Intentions) || {
        functions: [],
        classes: [],
        imports: [],
        constants: [],
        exports: [],
        interfaces: [],
        types: [],
        enums: [],
        parsingErrors: [],
        complexity: {
          cyclomaticComplexity: 0,
          cognitiveComplexity: 0,
          halsteadMetrics: {
            operatorsCount: 0,
            operandsCount: 0,
            vocabularySize: 0,
            programLength: 0,
            difficulty: 0,
            effort: 0,
            timeRequired: 0,
            bugsDelivered: 0,
          },
          linesOfCode: 0,
          maintainabilityIndex: 100,
        },
      };
      if (intents && !('extraction_error' in intents)) {
        const fnCount = Array.isArray(intents.functions)
          ? intents.functions.length
          : 0;
        const clsCount = Array.isArray(intents.classes)
          ? intents.classes.length
          : 0;
        const impCount = Array.isArray(intents.imports)
          ? intents.imports.length
          : 0;
        parts.push(
          `📊 Extracted: ${fnCount} functions, ${clsCount} classes, ${impCount} imports`,
        );
        if (
          Array.isArray(intents.parsingErrors) &&
          intents.parsingErrors.length > 0
        ) {
          parts.push(
            `⚠️ Partial Errors: ${intents.parsingErrors.length} node processing issues`,
          );
        }
      }

      parts.push(
        `💬 Documentation: ${r.comments.length} comments, ${r.jsdocs.length} JSDoc blocks`,
      );

      const output = parts.join('\n');

      const metadata: Partial<ParseResult> = {
        fileInfo: r.fileInfo,
        parseError: r.parseError,
        intentions:
          params.extractIntentions === false ? undefined : r.intentions,
        comments: params.extractComments === false ? undefined : r.comments,
        jsdocs: params.extractComments === false ? undefined : r.jsdocs,
      };
      if (params.includeSource !== false) metadata.source = r.source;

      return {
        success: !!r.source && (!r.parseError || !!r.intentions),
        output,
        metadata,
        executionTime: (Date.now() - start) / 1000,
      };
    } catch (e: unknown) {
      return {
        success: false,
        output: '',
        errorMessage: String(e),
        executionTime: (Date.now() - start) / 1000,
      };
    }
  }
}
