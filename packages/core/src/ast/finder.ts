/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Node, SourceFile, SyntaxKind } from 'ts-morph';
import { DictionaryQuery } from './models.js';

// Simple in-memory cache for query results (keyed by stringified query)
const queryCache = new Map<string, Node[]>();

/**
 * Enhanced XPath-like parser for simple node tests and predicates.
 * Supports: //Node, /Node, ./Node, ../Node, predicates: [@attr='val'], [contains(@attr,'val')], [starts-with(@attr,'val')], [123]
 */
class EnhancedXPathParser {
  parse(xpath: string) {
    const parts = {
      axis: 'descendant-or-self' as
        | 'descendant-or-self'
        | 'child'
        | 'parent'
        | 'self',
      nodeTest: '' as string,
      conditions: [] as any[],
      position: undefined as number | undefined,
    };

    let expr = xpath.trim();
    if (expr.startsWith('//')) {
      parts.axis = 'descendant-or-self';
      expr = expr.slice(2);
    } else if (expr.startsWith('../')) {
      parts.axis = 'parent';
      expr = expr.slice(3);
    } else if (expr.startsWith('./')) {
      parts.axis = 'self';
      expr = expr.slice(2);
    } else if (expr.startsWith('/')) {
      parts.axis = 'child';
      expr = expr.slice(1);
    }

    // split node test and predicates
    const predicateMatch = expr.match(/^([^[]+)(\[[\s\S]+\])?/);
    if (!predicateMatch) return parts;
    parts.nodeTest = predicateMatch[1].trim();

    const predicates = Array.from(expr.matchAll(/\[([^\]]+)\]/g)).map((m) =>
      m[1].trim(),
    );
    for (const pred of predicates) {
      // position
      if (/^\d+$/.test(pred)) {
        parts.position = Number(pred);
        continue;
      }
      // attribute equality @attr='value'
      const attrEq = pred.match(/^@([\w$]+)\s*=\s*(['"])([\s\S]+)\2$/);
      if (attrEq) {
        parts.conditions.push({
          type: 'attribute',
          field: attrEq[1],
          operator: 'eq',
          value: attrEq[3],
        });
        continue;
      }
      // attribute not equals
      const attrNe = pred.match(/^@([\w$]+)\s*!=\s*(['"])([\s\S]+)\2$/);
      if (attrNe) {
        parts.conditions.push({
          type: 'attribute',
          field: attrNe[1],
          operator: 'ne',
          value: attrNe[3],
        });
        continue;
      }
      // contains(@attr, 'val')
      const contains = pred.match(
        /^contains\(\s*@([\w$]+)\s*,\s*(['"])([\s\S]+)\2\s*\)$/,
      );
      if (contains) {
        parts.conditions.push({
          type: 'attribute',
          field: contains[1],
          operator: 'contains',
          value: contains[3],
        });
        continue;
      }
      // starts-with(@attr, 'val')
      const starts = pred.match(
        /^starts-with\(\s*@([\w$]+)\s*,\s*(['"])([\s\S]+)\2\s*\)$/,
      );
      if (starts) {
        parts.conditions.push({
          type: 'attribute',
          field: starts[1],
          operator: 'starts_with',
          value: starts[3],
        });
        continue;
      }

      // fallback: raw condition
      parts.conditions.push({ type: 'raw', raw: pred });
    }

    return parts;
  }
}

/**
 * Finds nodes in a ts-morph AST that match a given query.
 *
 * Extended behavior:
 * - query can be a DictionaryQuery (existing behavior)
 * - query can be a string (interpreted as an XPath-like expression parsed above)
 * - query can be an object with { xpath: string } or { custom: (node)=>boolean }
 *
 * Caches results for string/xpath queries.
 */
export function findNodes(
  sourceFile: SourceFile,
  query:
    | DictionaryQuery
    | string
    | { xpath?: string; custom?: (n: Node) => boolean },
): Node[] {
  // Support raw xpath string directly
  if (typeof query === 'string') {
    // use cache
    const key = `xpath:${query}`;
    if (queryCache.has(key)) return queryCache.get(key)!;
    const parser = new EnhancedXPathParser();
    const parsed = parser.parse(query);
    const nodes = findByXPath(sourceFile, parsed);
    queryCache.set(key, nodes);
    return nodes;
  }

  // Support object with xpath or custom
  if (typeof query === 'object' && (query as any).xpath) {
    const xpath = (query as any).xpath as string;
    const key = `xpath:${xpath}`;
    if (queryCache.has(key)) return queryCache.get(key)!;
    const parser = new EnhancedXPathParser();
    const parsed = parser.parse(xpath);
    const nodes = findByXPath(sourceFile, parsed);
    queryCache.set(key, nodes);
    return nodes;
  }

  if (
    typeof query === 'object' &&
    (query as any).custom &&
    typeof (query as any).custom === 'function'
  ) {
    const customFn = (query as any).custom as (n: Node) => boolean;
    return findByCustom(sourceFile, customFn);
  }

  // Fallback to original dictionary query behavior
  const results: Node[] = [];
  const descendants = sourceFile.getDescendants();

  for (const node of descendants) {
    if (matchesQuery(node, { conditions: query } as DictionaryQuery)) {
      results.push(node);
    }
  }

  return results;
}

/**
 * Find nodes using a custom function.
 */
function findByCustom(
  sourceFile: SourceFile,
  fn: (n: Node) => boolean,
): Node[] {
  const found: Node[] = [];
  for (const n of sourceFile.getDescendants()) {
    try {
      if (fn(n)) found.push(n);
    } catch {
      // ignore errors in user-provided function
    }
  }
  return found;
}

/**
 * Find nodes by parsed XPath-like expression.
 */
function findByXPath(
  sourceFile: SourceFile,
  parsed: ReturnType<EnhancedXPathParser['parse']>,
): Node[] {
  const candidates: Node[] = [];
  const nodeTest = parsed.nodeTest;
  if (!nodeTest) return [];
  const descendants =
    parsed.axis === 'child'
      ? sourceFile.getChildren()
      : sourceFile.getDescendants();

  for (const node of descendants) {
    // match by kind name (e.g. FunctionDeclaration, ClassDeclaration, VariableStatement...)
    const kindName = (node as any).getKindName
      ? (node as any).getKindName()
      : SyntaxKind[node.getKind()];
    if (kindName === nodeTest || node.getKind().toString() === nodeTest) {
      if (matchesParsedConditions(node, parsed.conditions)) {
        candidates.push(node);
      }
    }
  }

  // if a position was requested
  if (typeof parsed.position === 'number') {
    return candidates.length >= parsed.position
      ? [candidates[parsed.position - 1]]
      : [];
  }

  return candidates;
}

/**
 * Evaluate parsed predicate conditions against a node.
 */
function matchesParsedConditions(node: Node, conditions: Array<any>): boolean {
  if (!conditions || conditions.length === 0) return true;
  for (const cond of conditions) {
    if (cond.type === 'attribute') {
      const val = extractAttributeValue(node, cond.field);
      if (!compareValues(val, cond.value, cond.operator)) return false;
    } else if (cond.type === 'raw') {
      // best-effort: check raw string existence in node text
      if (!node.getText().includes(cond.raw)) return false;
    } else {
      // unknown predicate: fail-safe to false
      return false;
    }
  }
  return true;
}

/**
 * Extracts a best-effort attribute value from a Node.
 * Supports common properties: name, getName(), getText(), getType?.getText()
 */
function extractAttributeValue(node: Node, field: string): string | undefined {
  try {
    // named nodes
    if ((node as any).getName && typeof (node as any).getName === 'function') {
      const name = (node as any).getName();
      if (field === 'name') return name;
    }
    // try direct property
    const prop = (node as any)[field];
    if (typeof prop === 'string') return prop;
    if (prop && typeof prop === 'object' && typeof prop.getText === 'function')
      return prop.getText();
    // fallback to text search
    return node.getText();
  } catch {
    return undefined;
  }
}

/**
 * Compare helper supporting eq, ne, contains, starts_with.
 */
function compareValues(actual: any, expected: any, operator: string) {
  if (operator === 'eq') return String(actual) === String(expected);
  if (operator === 'ne') return String(actual) !== String(expected);
  if (operator === 'contains') return String(actual).includes(String(expected));
  if (operator === 'starts_with')
    return String(actual).startsWith(String(expected));
  return false;
}

/**
 * Checks if a given AST node matches the conditions of a DictionaryQuery.
 */
function matchesQuery(node: Node, query: DictionaryQuery): boolean {
  const { conditions } = query;

  // Check kind
  if (conditions.kind) {
    const kinds = Array.isArray(conditions.kind)
      ? conditions.kind
      : [conditions.kind];
    if (!kinds.includes(node.getKind() as any)) {
      return false;
    }
  }

  // Check name
  if (conditions.name) {
    // Named nodes in ts-morph may have getName()
    const name =
      (node as any).getName && typeof (node as any).getName === 'function'
        ? (node as any).getName()
        : undefined;
    if (name !== conditions.name) return false;
  }

  // Check for parent condition
  if (conditions.parent) {
    let parent = node.getParent();
    let matched = false;
    while (parent) {
      if (
        matchesQuery(parent, {
          type: 'dictionary',
          conditions: conditions.parent,
        } as any)
      ) {
        matched = true;
        break;
      }
      parent = parent.getParent();
    }
    if (!matched) return false;
  }

  // Check for child condition: ensure at least one direct child matches
  if (conditions.child) {
    const children = node.getChildren();
    let hasMatchingChild = false;
    for (const child of children) {
      if (
        matchesQuery(child, {
          type: 'dictionary',
          conditions: conditions.child,
        } as any)
      ) {
        hasMatchingChild = true;
        break;
      }
    }
    if (!hasMatchingChild) return false;
  }

  return true;
}
