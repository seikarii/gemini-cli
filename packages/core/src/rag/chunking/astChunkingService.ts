/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  RAGChunk,
  ChunkType,
  ChunkSource,
  ChunkMetadata,
  RAGConfig,
} from '../types.js';
import { RAGLogger } from '../logger.js';
import { createProject, parseSourceToSourceFileWithProject } from '../../ast/adapter.js';
import { Project, SourceFile, Node, SyntaxKind } from 'ts-morph';
import * as crypto from 'crypto';

/**
 * AST-based intelligent code chunking service.
 * This service parses code into semantically meaningful units
 * using Abstract Syntax Trees for better context preservation.
 */
export class RAGASTChunkingService {
  private project: Project;

  constructor(
    private readonly config: RAGConfig,
    private readonly logger: RAGLogger
  ) {
    this.project = createProject();
  }

  /**
   * Chunk source code into semantic units.
   */
  async chunkSourceCode(
    content: string,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): Promise<RAGChunk[]> {
    const chunks: RAGChunk[] = [];

    try {
      // Parse the source code
      const parseResult = parseSourceToSourceFileWithProject(
        this.project,
        content,
        filePath
      );

      if (parseResult.error || !parseResult.sourceFile) {
        this.logger.warn(`Failed to parse ${filePath}: ${parseResult.error}`);
        // Fallback to simple text chunking
        return this.chunkAsText(content, filePath, language, metadata);
      }

      const sourceFile = parseResult.sourceFile;

      // Extract different types of code units
      chunks.push(...this.extractClasses(sourceFile, filePath, language, metadata));
      chunks.push(...this.extractFunctions(sourceFile, filePath, language, metadata));
      chunks.push(...this.extractInterfaces(sourceFile, filePath, language, metadata));
      chunks.push(...this.extractTypes(sourceFile, filePath, language, metadata));
      chunks.push(...this.extractConstants(sourceFile, filePath, language, metadata));
      chunks.push(...this.extractImportExports(sourceFile, filePath, language, metadata));

      // If no semantic chunks were found, fallback to text chunking
      if (chunks.length === 0) {
        this.logger.debug(`No semantic chunks found in ${filePath}, falling back to text chunking`);
        return this.chunkAsText(content, filePath, language, metadata);
      }

      // Add file-level context chunk if file is large enough
      if (content.length > this.config.chunking.maxChunkSize * 2) {
        chunks.push(this.createFileOverviewChunk(sourceFile, filePath, language, metadata));
      }

      this.logger.debug(`Extracted ${chunks.length} semantic chunks from ${filePath}`);
      return chunks;

    } catch (error) {
      this.logger.error(`Error chunking source code in ${filePath}:`, error);
      // Fallback to simple text chunking
      return this.chunkAsText(content, filePath, language, metadata);
    }
  }

  /**
   * Chunk non-code content (documentation, comments, etc.)
   */
  async chunkTextContent(
    content: string,
    sourceId: string,
    type: ChunkType,
    metadata: Record<string, unknown>
  ): Promise<RAGChunk[]> {
    return this.chunkAsText(content, sourceId, 'text', metadata, type);
  }

  // Private methods for extracting different code elements

  private extractClasses(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const classes = sourceFile.getClasses();

    for (const classNode of classes) {
      const className = classNode.getName() || 'AnonymousClass';
      const classText = classNode.getFullText();
      const { start, end } = this.getNodeLineNumbers(classNode);

      // Extract methods for individual chunks
      const methods = classNode.getMethods();
      for (const method of methods) {
        const methodName = method.getName();
        const methodText = method.getFullText();
        const { start: methodStart, end: methodEnd } = this.getNodeLineNumbers(method);

        chunks.push(this.createChunk({
          content: this.addContextualInfo(methodText, classText, 'method'),
          type: ChunkType.CODE_FUNCTION,
          source: this.createFileSource(filePath, methodStart, methodEnd),
          language,
          metadata: {
            ...metadata,
            code: {
              functionName: methodName,
              className,
              methodName,
              ...this.extractCodeMetadata(method),
            },
          },
        }));
      }

      // Create class-level chunk
      chunks.push(this.createChunk({
        content: classText,
        type: ChunkType.CODE_CLASS,
        source: this.createFileSource(filePath, start, end),
        language,
        metadata: {
          ...metadata,
          code: {
            className,
            methods: methods.map(m => m.getName()),
            ...this.extractCodeMetadata(classNode),
          },
        },
      }));
    }

    return chunks;
  }

  private extractFunctions(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const functions = sourceFile.getFunctions();

    for (const functionNode of functions) {
      const functionName = functionNode.getName() || 'AnonymousFunction';
      const functionText = functionNode.getFullText();
      const { start, end } = this.getNodeLineNumbers(functionNode);

      chunks.push(this.createChunk({
        content: functionText,
        type: ChunkType.CODE_FUNCTION,
        source: this.createFileSource(filePath, start, end),
        language,
        metadata: {
          ...metadata,
          code: {
            functionName,
            ...this.extractCodeMetadata(functionNode),
          },
        },
      }));
    }

    return chunks;
  }

  private extractInterfaces(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const interfaces = sourceFile.getInterfaces();

    for (const interfaceNode of interfaces) {
      const interfaceName = interfaceNode.getName();
      const interfaceText = interfaceNode.getFullText();
      const { start, end } = this.getNodeLineNumbers(interfaceNode);

      chunks.push(this.createChunk({
        content: interfaceText,
        type: ChunkType.CODE_CLASS, // Treat interfaces as class-like structures
        source: this.createFileSource(filePath, start, end),
        language,
        metadata: {
          ...metadata,
          code: {
            className: interfaceName,
            ...this.extractCodeMetadata(interfaceNode),
          },
        },
      }));
    }

    return chunks;
  }

  private extractTypes(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const typeAliases = sourceFile.getTypeAliases();

    for (const typeNode of typeAliases) {
      const typeName = typeNode.getName();
      const typeText = typeNode.getFullText();
      const { start, end } = this.getNodeLineNumbers(typeNode);

      chunks.push(this.createChunk({
        content: typeText,
        type: ChunkType.CODE_SNIPPET,
        source: this.createFileSource(filePath, start, end),
        language,
        metadata: {
          ...metadata,
          code: {
            functionName: typeName,
            ...this.extractCodeMetadata(typeNode),
          },
        },
      }));
    }

    return chunks;
  }

  private extractConstants(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const variableDeclarations = sourceFile.getVariableDeclarations();

    // Group related constants together
    const constantGroups: Array<typeof variableDeclarations> = [];
    let currentGroup: typeof variableDeclarations = [];

    for (const varDecl of variableDeclarations) {
      // Check if it's a constant (const or readonly)
      const isConstant = varDecl.getParent()?.getParent()?.getKind() === SyntaxKind.ConstKeyword;
      
      if (isConstant) {
        currentGroup.push(varDecl);
        
        // If group is getting large or we hit a gap, create a chunk
        if (currentGroup.length >= 5) {
          constantGroups.push([...currentGroup]);
          currentGroup = [];
        }
      } else if (currentGroup.length > 0) {
        // End of constant group
        constantGroups.push([...currentGroup]);
        currentGroup = [];
      }
    }

    // Don't forget the last group
    if (currentGroup.length > 0) {
      constantGroups.push(currentGroup);
    }

    // Create chunks for constant groups
    for (const group of constantGroups) {
      if (group.length === 0) continue;

      const firstNode = group[0];
      const lastNode = group[group.length - 1];
      const groupText = group.map(node => node.getFullText()).join('\n');
      const { start } = this.getNodeLineNumbers(firstNode);
      const { end } = this.getNodeLineNumbers(lastNode);

      chunks.push(this.createChunk({
        content: groupText,
        type: ChunkType.CODE_SNIPPET,
        source: this.createFileSource(filePath, start, end),
        language,
        metadata: {
          ...metadata,
          code: {
            variables: group.map(node => node.getName()),
            ...this.extractCodeMetadata(firstNode),
          },
        },
      }));
    }

    return chunks;
  }

  private extractImportExports(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const imports = sourceFile.getImportDeclarations();
    const exports = sourceFile.getExportDeclarations();

    if (imports.length > 0 || exports.length > 0) {
      const importsText = imports.map(imp => imp.getFullText()).join('\n');
      const exportsText = exports.map(exp => exp.getFullText()).join('\n');
      const combinedText = [importsText, exportsText].filter(text => text.trim()).join('\n\n');

      if (combinedText.trim()) {
        chunks.push(this.createChunk({
          content: combinedText,
          type: ChunkType.CODE_MODULE,
          source: this.createFileSource(filePath, 1, Math.max(imports.length, exports.length) + 1),
          language,
          metadata: {
            ...metadata,
            code: {
              imports: imports.map(imp => imp.getModuleSpecifierValue()),
              exports: exports.map(exp => exp.getModuleSpecifierValue() || 'default'),
            },
          },
        }));
      }
    }

    return chunks;
  }

  private createFileOverviewChunk(
    sourceFile: SourceFile,
    filePath: string,
    language: string,
    metadata: Record<string, unknown>
  ): RAGChunk {
    // Create a high-level overview of the file
    const classes = sourceFile.getClasses().map(c => c.getName());
    const functions = sourceFile.getFunctions().map(f => f.getName());
    const interfaces = sourceFile.getInterfaces().map(i => i.getName());
    
    const overview = `File: ${filePath}
Classes: ${classes.join(', ') || 'None'}
Functions: ${functions.join(', ') || 'None'}
Interfaces: ${interfaces.join(', ') || 'None'}

${sourceFile.getFullText().substring(0, 500)}...`;

    return this.createChunk({
      content: overview,
      type: ChunkType.CODE_MODULE,
      source: this.createFileSource(filePath, 1, sourceFile.getEndLineNumber()),
      language,
      metadata: {
        ...metadata,
        code: {
          overview: true,
          classes,
          functions,
          interfaces,
        },
      },
    });
  }

  // Fallback text chunking for non-parseable content

  private chunkAsText(
    content: string,
    sourceId: string,
    language: string,
    metadata: Record<string, unknown>,
    type: ChunkType = ChunkType.CODE_SNIPPET
  ): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    const maxChunkSize = this.config.chunking.maxChunkSize;
    const overlapSize = Math.floor(maxChunkSize * this.config.chunking.overlapRatio);

    let start = 0;
    let chunkIndex = 0;

    while (start < content.length) {
      const end = Math.min(start + maxChunkSize, content.length);
      let chunkContent = content.substring(start, end);

      // Try to break at natural boundaries (lines, sentences)
      if (end < content.length && this.config.chunking.respectBoundaries) {
        const lastNewline = chunkContent.lastIndexOf('\n');
        const lastSentence = chunkContent.lastIndexOf('.');
        const breakPoint = Math.max(lastNewline, lastSentence);
        
        if (breakPoint > start + this.config.chunking.minChunkSize) {
          chunkContent = content.substring(start, start + breakPoint + 1);
        }
      }

      const chunk = this.createChunk({
        content: chunkContent.trim(),
        type,
        source: this.createTextSource(sourceId, chunkIndex),
        language,
        metadata,
      });

      chunks.push(chunk);

      // Move start position with overlap
      start += chunkContent.length - overlapSize;
      chunkIndex++;
    }

    return chunks;
  }

  // Helper methods

  private createChunk({
    content,
    type,
    source,
    language,
    metadata,
  }: {
    content: string;
    type: ChunkType;
    source: ChunkSource;
    language: string;
    metadata: Record<string, unknown>;
  }): RAGChunk {
    const id = this.generateChunkId(content, source);
    const timestamp = new Date().toISOString();
    const contentHash = crypto.createHash('md5').update(content).digest('hex');

    return {
      id,
      content,
      type,
      language,
      source,
      metadata: this.enrichMetadata(metadata, content),
      timestamp,
      contentHash,
    };
  }

  private createFileSource(filePath: string, startLine: number, endLine: number): ChunkSource {
    return {
      id: filePath,
      type: 'file',
      startLine,
      endLine,
    };
  }

  private createTextSource(sourceId: string, chunkIndex: number): ChunkSource {
    return {
      id: `${sourceId}#${chunkIndex}`,
      type: 'file',
    };
  }

  private generateChunkId(content: string, source: ChunkSource): string {
    const hashInput = `${source.id}:${source.startLine || 0}:${source.endLine || 0}:${content.substring(0, 100)}`;
    return crypto.createHash('sha256').update(hashInput).digest('hex').substring(0, 16);
  }

  private enrichMetadata(
    metadata: Record<string, unknown>,
    content: string
  ): ChunkMetadata {
    const baseMetadata = metadata as ChunkMetadata;
    
    return {
      ...baseMetadata,
      quality: {
        ...baseMetadata.quality,
        readability: this.calculateReadability(content),
        complexity: this.calculateComplexity(content),
        completeness: this.calculateCompleteness(content),
        relevance: baseMetadata.quality?.relevance || 0.5,
      },
    };
  }

  private calculateReadability(content: string): number {
    // Simple readability score based on line length and complexity
    const lines = content.split('\n');
    const avgLineLength = lines.reduce((sum, line) => sum + line.length, 0) / lines.length;
    const complexity = (content.match(/[{}()[\]]/g) || []).length / content.length;
    
    return Math.max(0, Math.min(1, 1 - (avgLineLength / 120) - complexity));
  }

  private calculateComplexity(content: string): number {
    // Simple complexity metric based on nesting and special characters
    const nestingChars = (content.match(/[{}]/g) || []).length;
    const specialChars = (content.match(/[()[\]<>]/g) || []).length;
    const keywords = (content.match(/\b(if|else|for|while|switch|try|catch)\b/g) || []).length;
    
    return Math.min(1, (nestingChars + specialChars + keywords * 2) / content.length);
  }

  private calculateCompleteness(content: string): number {
    // Check if the code chunk appears complete (balanced braces, etc.)
    const openBraces = (content.match(/{/g) || []).length;
    const closeBraces = (content.match(/}/g) || []).length;
    const openParens = (content.match(/\(/g) || []).length;
    const closeParens = (content.match(/\)/g) || []).length;
    
    const braceBalance = openBraces === 0 ? 1 : Math.min(1, closeBraces / openBraces);
    const parenBalance = openParens === 0 ? 1 : Math.min(1, closeParens / openParens);
    
    return (braceBalance + parenBalance) / 2;
  }

  private extractCodeMetadata(node: Node): Record<string, unknown> {
    // Extract relevant metadata from AST node
    try {
      return {
        complexity: this.calculateASTComplexity(node),
        nodeType: node.getKindName(),
        hasComments: node.getLeadingCommentRanges().length > 0,
      };
    } catch {
      return {};
    }
  }

  private calculateASTComplexity(node: Node): number {
    // Calculate cyclomatic complexity based on AST structure
    let complexity = 1; // Base complexity
    
    node.forEachDescendant(child => {
      const kind = child.getKind();
      if ([
        SyntaxKind.IfStatement,
        SyntaxKind.ForStatement,
        SyntaxKind.WhileStatement,
        SyntaxKind.SwitchStatement,
        SyntaxKind.ConditionalExpression,
        SyntaxKind.BinaryExpression, // Use BinaryExpression for logical operations
      ].includes(kind)) {
        complexity++;
      }
    });
    
    return Math.min(1, complexity / 10); // Normalize to 0-1
  }

  private getNodeLineNumbers(node: Node): { start: number; end: number } {
    try {
      return {
        start: node.getStartLineNumber(),
        end: node.getEndLineNumber(),
      };
    } catch {
      return { start: 1, end: 1 };
    }
  }

  private addContextualInfo(methodText: string, classText: string, type: 'method' | 'function'): string {
    // Add minimal class context to method chunks for better understanding
    if (type === 'method') {
      const classDeclaration = classText.split('\n')[0]; // Get class declaration line
      return `${classDeclaration}\n  // ... other class members ...\n${methodText}\n}`;
    }
    return methodText;
  }
}
