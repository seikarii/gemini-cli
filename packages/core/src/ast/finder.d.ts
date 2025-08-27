/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Node, SourceFile } from 'ts-morph';
import { DictionaryQuery } from './models.js';
interface XPathQuery {
    xpath: string;
}
interface CustomQuery {
    custom: (n: Node) => boolean;
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
export declare function findNodes(sourceFile: SourceFile, query: DictionaryQuery | string | XPathQuery | CustomQuery): Node[];
export {};
