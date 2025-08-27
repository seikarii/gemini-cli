#!/usr/bin/env node
/* eslint-env node */
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, 'reports');

const patterns = [
  { label: 'readFileSync', regex: /\breadFileSync\s*\(/ },
  { label: 'writeFileSync', regex: /\bwriteFileSync\s*\(/ },
  { label: 'mkdirSync', regex: /\bmkdirSync\s*\(/ },
  { label: 'unlinkSync', regex: /\bunlinkSync\s*\(/ },
  { label: 'execSync', regex: /\bexecSync\s*\(/ },
  { label: 'spawnSync', regex: /\bspawnSync\s*\(/ },
  { label: 'for-infinite', regex: /for\s*\(\s*;;\s*\)/ },
  { label: 'while-infinite', regex: /while\s*\(\s*true\s*\)/ },
];

const SKIP_DIRS = new Set([
  '.git',
  'node_modules',
  'coverage',
  'dist',
  'build',
  '.cache',
]);
const EXT_WHITELIST = new Set(['.js', '.ts', '.tsx', '.jsx', '.mjs', '.cjs']);

function severityForPath(p) {
  if (
    /\/packages\/(core|cli|vscode-ide-companion)\//.test(p) ||
    /\/packages\/.*\/src\//.test(p)
  )
    return 'high';
  if (
    /\/scripts\//.test(p) ||
    /\/integration-tests\//.test(p) ||
    /(^|\/)test(s)?(\/|$)/.test(p)
  )
    return 'medium';
  if (/\/bundle\//.test(p) || /\/coverage\//.test(p) || /\/dist\//.test(p))
    return 'low';
  return 'medium';
}

async function walk(dir, cb) {
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name)) continue;
      await walk(path.join(dir, e.name), cb);
    } else if (e.isFile()) {
      cb(path.join(dir, e.name));
    }
  }
}

async function scan() {
  const matches = [];
  let fileCount = 0;

  await walk(ROOT, async (filePath) => {
    const rel = path.relative(ROOT, filePath);
    const ext = path.extname(filePath).toLowerCase();
    if (!EXT_WHITELIST.has(ext)) return;
    if (rel === path.relative(ROOT, __filename)) return;
    fileCount++;
    try {
      const content = await fsp.readFile(filePath, 'utf8');
      const lines = content.split(/\r?\n/);
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        for (const p of patterns) {
          if (p.regex.test(line)) {
            matches.push({
              file: rel,
              absolutePath: filePath,
              lineNumber: i + 1,
              line: line.trim(),
              pattern: p.label,
              severity: severityForPath(rel),
            });
          }
        }
      }
    } catch (err) {
      // ignore unreadable files
    }
  });

  const summary = matches.reduce(
    (acc, m) => {
      acc.total++;
      acc.bySeverity[m.severity] = (acc.bySeverity[m.severity] || 0) + 1;
      acc.byPattern[m.pattern] = (acc.byPattern[m.pattern] || 0) + 1;
      return acc;
    },
    { total: 0, bySeverity: {}, byPattern: {} },
  );

  const report = {
    generatedAt: new Date().toISOString(),
    root: ROOT,
    scannedFiles: fileCount,
    patterns: patterns.map((p) => p.label),
    summary,
    matches,
  };

  await fsp.mkdir(OUT_DIR, { recursive: true });
  const jsonPath = path.join(OUT_DIR, 'blocking-patterns-report.json');
  const mdPath = path.join(OUT_DIR, 'blocking-patterns-report.md');
  await fsp.writeFile(jsonPath, JSON.stringify(report, null, 2), 'utf8');

  const md = buildMarkdownReport(report);
  await fsp.writeFile(mdPath, md, 'utf8');

  console.log('Scan complete.');
  console.log(`Files scanned: ${fileCount}`);
  console.log(`Total matches: ${report.summary.total}`);
  console.log(`Report JSON: ${jsonPath}`);
  console.log(`Report MD: ${mdPath}`);
}

function buildMarkdownReport(report) {
  const lines = [];
  lines.push('# Blocking patterns report');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push(`Scanned files: ${report.scannedFiles}`);
  lines.push('');
  lines.push('## Summary');
  lines.push(`- Total matches: ${report.summary.total}`);
  for (const s of Object.keys(report.summary.bySeverity)) {
    lines.push(`- ${s}: ${report.summary.bySeverity[s]}`);
  }
  lines.push('');
  lines.push('## By pattern');
  for (const p of Object.keys(report.summary.byPattern)) {
    lines.push(`- ${p}: ${report.summary.byPattern[p]}`);
  }
  lines.push('');
  lines.push('## Matches (top 200)');
  const max = Math.min(report.matches.length, 200);
  for (let i = 0; i < max; i++) {
    const m = report.matches[i];
    lines.push(`- [${m.severity}] ${m.file}:${m.lineNumber} (${m.pattern})`);
    lines.push('  ```');
    lines.push(`${m.line}`);
    lines.push('  ```');
  }
  if (report.matches.length > max)
    lines.push(
      `\n...and ${report.matches.length - max} more matches. See JSON report for full list.`,
    );
  return lines.join('\n');
}

scan().catch((err) => {
  console.error('Scan failed:', err);
  process.exitCode = 2;
});
