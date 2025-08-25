/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import { promises as fs } from 'node:fs';
import { Ignore } from './ignore.js';
import * as cache from './crawlCache.js';

export interface CrawlOptions {
  // The directory to start the crawl from.
  crawlDirectory: string;
  // The project's root directory, for path relativity.
  cwd: string;
  // The fdir maxDepth option.
  maxDepth?: number;
  // A pre-configured Ignore instance.
  ignore: Ignore;
  // Caching options.
  cache: boolean;
  cacheTtl: number;
}

function toPosixPath(p: string) {
  return p.split(path.sep).join(path.posix.sep);
}

export async function crawl(options: CrawlOptions): Promise<string[]> {
  if (options.cache) {
    const cacheKey = cache.getCacheKey(
      options.crawlDirectory,
      options.ignore.getFingerprint(),
      options.maxDepth,
    );
    const cachedResults = cache.read(cacheKey);

    if (cachedResults) {
      return cachedResults;
    }
  }

  const posixCwd = toPosixPath(options.cwd);
  const posixCrawlDirectory = toPosixPath(options.crawlDirectory);

  // Simple recursive crawler to avoid relying on external fdir API differences.
  const dirFilter = options.ignore.getDirectoryFilter();
  const resultsList: string[] = [];

  async function walk(dir: string, depth: number) {
    if (options.maxDepth !== undefined && depth > options.maxDepth) return;
    let entries: string[];
    try {
      entries = await fs.readdir(dir);
    } catch (_e) {
      return;
    }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry);
      let stat;
      try {
        stat = await fs.stat(fullPath);
      } catch (_e) {
        continue;
      }
      const relativeDir = path.posix.relative(posixCrawlDirectory, path.posix.dirname(toPosixPath(fullPath)));
      if (stat.isDirectory()) {
        if (dirFilter(`${relativeDir}/`)) continue;
        await walk(fullPath, depth + 1);
      } else if (stat.isFile()) {
        resultsList.push(toPosixPath(path.posix.relative(posixCrawlDirectory, toPosixPath(fullPath))));
      }
    }
  }

  await walk(options.crawlDirectory, 0);
  const results = resultsList;

  const relativeToCrawlDir = path.posix.relative(posixCwd, posixCrawlDirectory);

  const relativeToCwdResults = results.map((p) =>
    path.posix.join(relativeToCrawlDir, p),
  );

  if (options.cache) {
    const cacheKey = cache.getCacheKey(
      options.crawlDirectory,
      options.ignore.getFingerprint(),
      options.maxDepth,
    );
    cache.write(cacheKey, relativeToCwdResults, options.cacheTtl * 1000);
  }

  return relativeToCwdResults;
}
