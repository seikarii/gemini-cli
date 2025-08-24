/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * TypeScript port of ideas/ASTRAL_TOOLS/ast_tools/reader.py
 * - robust file reading with encoding fallbacks
 * - file size guard
 * - defensive parsing with ts-morph
 * - comment / JSDoc extraction
 * - intention extraction (functions, classes, imports, constants)
 */
import fs from 'fs/promises';
import path from 'path';
import { Project, SourceFile, SyntaxKind } from 'ts-morph';

const MAX_FILE_SIZE_MB = 50; // maximum file size to process

/**
 * Result shape returned by readAndParseFile / ASTReader.execute
 */
export interface ParseResult {
  source?: string | null;
  sourceFile?: SourceFile | null;
  parseError?: string | null;
  comments: string[];
  jsdocs: string[];
  intentions: any;
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
  } catch (e: any) {
    return { ok: false, error: `Cannot check file size: ${String(e)}` };
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
    } catch (e) {
      // try next encoding
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
  } catch (e: any) {
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

  try {
    // line comments //
    const lineRe = /^\s*\/\/(.*)$/gm;
    let m;
    while ((m = lineRe.exec(source)) !== null) {
      const txt = m[1].trim();
      if (txt) comments.push(txt);
    }

    // block comments /* ... */
    const blockRe = /\/\*([\s\S]*?)\*\//g;
    while ((m = blockRe.exec(source)) !== null) {
      const txt = m[1].trim();
      if (txt) {
        // JSDoc-style starts with *
        if (/^\*/.test(txt) || txt.startsWith('*')) {
          // split lines and clean leading '*'
          const cleaned = txt
            .split('\n')
            .map((l) => l.replace(/^\s*\*\s?/, '').trim())
            .join('\n')
            .trim();
          jsdocs.push(cleaned);
        } else {
          comments.push(txt);
        }
      }
    }

    // If ts-morph SourceFile available, collect JSDoc nodes defensively
    if (sourceFile) {
      try {
        for (const nd of sourceFile.getDescendants()) {
          try {
            const js = (nd as any).getJsDocs ? (nd as any).getJsDocs() : [];
            if (Array.isArray(js) && js.length > 0) {
              for (const d of js) {
                const txt =
                  typeof d.getInnerText === 'function'
                    ? d.getInnerText()
                    : String(d.getText?.() ?? '');
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
  } catch {
    // ignore extraction failure; return what we have
  }

  return { comments, jsdocs };
}

/**
 * Extract intentions (functions, classes, imports, constants) from a ts-morph SourceFile.
 * Defensive: isolate node-level errors to avoid complete failure.
 */
export function extractIntentionsFromSourceFile(
  sourceFile?: SourceFile | null,
): any {
  const intents: any = {
    functions: [],
    classes: [],
    imports: [],
    constants: [],
    parsingErrors: [],
  };
  if (!sourceFile) {
    intents.parsingErrors.push('sourceFile not available');
    return intents;
  }

  try {
    // Imports
    try {
      const imps = sourceFile.getImportDeclarations();
      for (const imp of imps) {
        try {
          // small local helper to avoid implicit-`any` callback in map
          const namedImports = imp.getNamedImports();
          const mappedNamed = namedImports.map((spec) => {
            const s = spec as unknown as {
              getName: () => string;
              getAliasNode?: () => { getText?: () => string } | undefined;
            };
            return {
              name: s.getName(),
              alias: s.getAliasNode?.()?.getText?.() ?? undefined,
            };
          });

          intents.imports.push({
            moduleSpecifier: imp.getModuleSpecifierValue(),
            namedImports: mappedNamed,
            defaultImport: imp.getDefaultImport()?.getText?.() ?? undefined,
            namespaceImport: imp.getNamespaceImport()?.getText?.() ?? undefined,
          });
        } catch (e) {
          intents.parsingErrors.push(`import node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`imports extraction failed: ${String(e)}`);
    }

    // Functions
    try {
      const funcs = sourceFile.getFunctions();
      for (const f of funcs) {
        try {
          intents.functions.push({
            name: f.getName?.() ?? '<anonymous>',
            isAsync: f.isAsync?.() ?? false,
            startLine: f.getStartLineNumber?.() ?? null,
            endLine: f.getEndLineNumber?.() ?? null,
            params:
              f.getParameters?.().map((p: any) => ({
                name: p.getName?.(),
                type: p.getType?.().getText?.() ?? '',
              })) ?? [],
          });
        } catch (e: any) {
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
          intents.classes.push({
            name: c.getName?.() ?? '<anonymous>',
            isExported: c.isExported?.() ?? false,
            startLine: c.getStartLineNumber?.() ?? null,
            endLine: c.getEndLineNumber?.() ?? null,
            methods:
              c.getMethods?.().map((m: any) => ({
                name: m.getName?.(),
                isAsync: m.isAsync?.() ?? false,
              })) ?? [],
          });
        } catch (e: any) {
          intents.parsingErrors.push(`class node error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`classes extraction failed: ${String(e)}`);
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
                  name: decl.getName?.(),
                  value: init.getText?.(),
                  isExported,
                  startLine: decl.getStartLineNumber?.(),
                });
              }
            } catch (e: any) {
              intents.parsingErrors.push(
                `variable declaration error: ${String(e)}`,
              );
            }
          }
        } catch (e: any) {
          intents.parsingErrors.push(`variable statement error: ${String(e)}`);
        }
      }
    } catch (e) {
      intents.parsingErrors.push(`constants extraction failed: ${String(e)}`);
    }

    // Additionally: a defensive AST walk to collect some string literal docstrings if any
    try {
      sourceFile.forEachDescendant((n) => {
        try {
          const k = n.getKind?.();
          if (k === SyntaxKind.StringLiteral) {
            const txt = (n as any).getLiteralText?.();
            if (typeof txt === 'string' && txt.length > 20) {
              // heuristics: long string literals may be docstrings/comments
              intents.constants.push({ inferredDocstring: txt.slice(0, 200) });
            }
          }
        } catch {
          // ignore per-node errors
        }
      });
    } catch {
      // ignore
    }
  } catch (e: any) {
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
  } catch (e: any) {
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
    metadata?: any;
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

      const intents = r.intentions || {};
      if (intents && !intents.extraction_error) {
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

      const metadata: any = {
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
    } catch (e: any) {
      return {
        success: false,
        output: '',
        errorMessage: String(e),
        executionTime: (Date.now() - start) / 1000,
      };
    }
  }
}
