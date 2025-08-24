/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as ts from 'typescript';

/**
 * Models for the AST Reader (Intention Map) and Finder/Refactor tools.
 * Based on the Python versions: richer ASTNodeInfo, ComplexQuery, ModificationSpec.
 */

//=================================================================================================
// Intention Map
//=================================================================================================

export interface IntentionMap {
  functions: FunctionInfo[];
  classes: ClassInfo[];
  imports: ImportInfo[];
  exports: ExportInfo[];
  interfaces: InterfaceInfo[];
  variables: VariableInfo[];
}

// Base symbol info augmented with optional AST tracing metadata
export interface BaseSymbolInfo {
  name: string;
  startLine: number;
  endLine: number;
  // Optional debug/tracing fields populated by the parser
  nodePath?: string;
  nodeId?: string;
  depth?: number;
}

export interface FunctionInfo extends BaseSymbolInfo {
  isAsync: boolean;
  isExported: boolean;
  parameters: Array<{ name: string; type: string }>;
  returnType: string;
}

export interface ClassInfo extends BaseSymbolInfo {
  isExported: boolean;
  extendsClause?: string;
  implementsClauses: string[];
  methods: FunctionInfo[];
}

export interface InterfaceInfo extends BaseSymbolInfo {
  isExported: boolean;
  properties: Array<{ name: string; type: string }>;
}

export interface VariableInfo extends BaseSymbolInfo {
  isExported: boolean;
  type: string;
}

export interface ImportInfo {
  moduleSpecifier: string;
  namedImports: Array<{ name: string; alias?: string }>;
  namespaceImport?: string;
  defaultImport?: string;
  nodePath?: string;
  nodeId?: string;
}

export interface ExportInfo {
  namedExports: Array<{ name: string; alias?: string }>;
  moduleSpecifier?: string;
  nodePath?: string;
  nodeId?: string;
}

//=================================================================================================
// Finder / Query Models (TypeScript port of Python QueryOperator / ComplexQuery)
//=================================================================================================

export enum QueryOperator {
  AND = 'and',
  OR = 'or',
  NOT = 'not',
  CONTAINS = 'contains',
  STARTS_WITH = 'starts_with',
  ENDS_WITH = 'ends_with',
  REGEX_MATCH = 'regex_match',
  IN = 'in',
  GT = 'gt',
  LT = 'lt',
  EQ = 'eq',
  NE = 'ne',
  EXISTS = 'exists',
}

export interface QueryCondition {
  field: string;
  operator: QueryOperator;
  value?: unknown;
}

export class ComplexQuery {
  conditions: QueryCondition[] = [];
  logicalOperator: QueryOperator = QueryOperator.AND;
  childQueries: ComplexQuery[] = [];

  constructor(conditions?: QueryCondition[], logicalOperator?: QueryOperator) {
    if (conditions) this.conditions = conditions;
    if (logicalOperator) this.logicalOperator = logicalOperator;
  }

  addCondition(field: string, operator: QueryOperator, value?: unknown): ComplexQuery {
    this.conditions.push({ field, operator, value });
    return this;
  }

  addChildQuery(query: ComplexQuery, operator: QueryOperator = QueryOperator.AND): ComplexQuery {
    query.logicalOperator = operator;
    this.childQueries.push(query);
    return this;
  }
}

//=================================================================================================
// Modification / Refactor Models (TypeScript port of Python ModificationOperation / Spec)
//=================================================================================================

export enum ModificationOperation {
  REPLACE = 'replace',
  INSERT_BEFORE = 'insert_before',
  INSERT_AFTER = 'insert_after',
  DELETE = 'delete',
  MODIFY_ATTRIBUTE = 'modify_attribute',
  WRAP = 'wrap',
  EXTRACT = 'extract',
  REFACTOR = 'refactor',
  RENAME_SYMBOL_SCOPED = 'rename_symbol_scoped',
  EXTRACT_METHOD = 'extract_method',
  SPLIT_BLOCK = 'split_block',
  MERGE_BLOCKS = 'merge_blocks',
  INLINE = 'inline',
  MOVE = 'move',
  ADD_IMPORT = 'add_import',
  ADD_TO_CLASS_BASES = 'add_to_class_bases',
  REMOVE_CLASS_METHODS = 'remove_class_methods',
  UPDATE_METHOD_SIGNATURE = 'update_method_signature',
  INSERT_STATEMENT_INTO_FUNCTION = 'insert_statement_into_function',
  REPLACE_EXPRESSION = 'replace_expression',
  ADD_CLASS = 'add_class',
}

export interface ModificationSpec {
  operation: ModificationOperation;
  targetQuery: string | DictionaryQuery | ComplexQuery;
  newCode?: string | null;
  attribute?: string | null;
  value?: unknown;
  wrapperTemplate?: string | null;
  extractName?: string | null;
  validateBefore?: boolean;
  validateAfter?: boolean;
  metadata?: Record<string, unknown>;
}

//=================================================================================================
// AST Finder helper types
//=================================================================================================

export interface ASTNodeInfo {
  kind: ts.SyntaxKind;
  text: string;
  startLine: number;
  endLine: number;
  nodePath?: string;
  nodeId?: string;
  depth?: number;
}

/**
 * DictionaryQuery kept for backward compatibility with finder.ts usage.
 * More advanced queries should use ComplexQuery or XPath-like strings in the finder.
 */
export interface DictionaryQuery {
  type: 'dictionary';
  conditions: Partial<{
    kind: ts.SyntaxKind | ts.SyntaxKind[];
    name: string;
    parent: Partial<DictionaryQuery['conditions']>;
    child: Partial<DictionaryQuery['conditions']>;
  }>;
}

export type ASTQuery = DictionaryQuery | ComplexQuery | string;
