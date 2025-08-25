/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import net from 'net';
import { GeminiAgent } from '../agent/gemini-agent.js';
import fs from 'fs/promises';

type DirEntry =
  | { name: string; type: 'file' }
  | { name: string; type: 'directory'; children: DirEntry[] };

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let activeFilePath: string | null = null;

// Security / hardening: limit recursion depth for directory traversal and
// enforce a maximum file read size to avoid accidental DoS or info leaks.
const DEFAULT_MAX_DIR_DEPTH = 6; // reasonable default for project browsing
const MAX_ALLOWED_FILE_BYTES = 200 * 1024; // 200 KiB

async function getDirectoryTree(
  dirPath: string,
  projectRoot: string,
  currentDepth = 0,
  maxDepth = DEFAULT_MAX_DIR_DEPTH,
): Promise<DirEntry[]> {
  // Enforce max recursion depth
  if (currentDepth >= maxDepth) return [];

  let dirents;
  try {
    dirents = await fs.readdir(dirPath, { withFileTypes: true });
  } catch (e: any) {
    // If we can't read a directory (permissions, etc.), skip it silently.
    return [];
  }

  const children: DirEntry[] = [];

  for (const dirent of dirents) {
    try {
      const candidate = path.resolve(dirPath, dirent.name);

      // Prevent escaping the project root (avoid reading outside the intended scope)
      const rel = path.relative(projectRoot, candidate);
      if (rel.startsWith('..') || path.isAbsolute(rel)) {
        // Skip entries that would escape the project root
        continue;
      }

      // Avoid following symbolic links to arbitrary locations; treat as file.
      const stat = await fs.lstat(candidate);
      if (stat.isSymbolicLink()) {
        // Don't follow symlinks to avoid surprises; show as file entry.
        children.push({ name: dirent.name, type: dirent.isDirectory() ? 'directory' : 'file' } as DirEntry);
        continue;
      }

      if (dirent.isDirectory()) {
        const subChildren = await getDirectoryTree(candidate, projectRoot, currentDepth + 1, maxDepth);
        children.push({ name: dirent.name, type: 'directory', children: subChildren });
      } else {
        children.push({ name: dirent.name, type: 'file' });
      }
    } catch (err) {
      // Best-effort: skip problematic entries instead of failing the whole listing
      continue;
    }
  }

  return children;
}

function findFreePort(startPort: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', (err: any) => {
      if (err.code === 'EADDRINUSE') {
        findFreePort(startPort + 1).then(resolve).catch(reject);
      } else {
        reject(err);
      }
    });
    server.listen(startPort, () => {
      const port = (server.address() as net.AddressInfo).port;
      server.close(() => {
        resolve(port);
      });
    });
  });
}

export async function startWebServer(agent: GeminiAgent) {
  const app = express();
  const port = await findFreePort(3000);

  try {
    const geminiDir = path.resolve(__dirname, '..', '..', '.gemini');
    await fs.mkdir(geminiDir, { recursive: true });
    await fs.writeFile(path.join(geminiDir, 'mew_port.txt'), port.toString(), 'utf8');
  } catch (error) {
    console.error('Failed to write mew_port.txt', error);
  }

  app.use(express.json()); // Middleware to parse JSON request bodies

  app.post('/api/mew/set-active-file', (req, res) => {
    const { filePath } = req.body;
    activeFilePath = filePath;
    res.status(200).send({ message: `Active file set to ${filePath}` });
  });

  // API endpoint for whispering data into the agent's memory
  app.post('/api/whisper', (req, res) => {
    const { data, kind } = req.body;
    if (data) {
      agent.whisper(data, kind);
      res.status(200).send({ message: 'Whisper received and ingested.' });
    } else {
      res.status(400).send({ message: 'Missing data for whisper.' });
    }
  });

  app.get('/api/file-tree', async (req, res) => {
    try {
  const projectRoot = path.resolve(__dirname, '..', '..', '..', '..');
  // Allow callers to request a smaller depth via query param, capped by DEFAULT_MAX_DIR_DEPTH
  const rawDepth = parseInt(((req.query as any)['depth'] as string) || `${DEFAULT_MAX_DIR_DEPTH}`, 10);
  const depth = Number.isFinite(rawDepth) ? Math.min(rawDepth, DEFAULT_MAX_DIR_DEPTH) : DEFAULT_MAX_DIR_DEPTH;
  const tree = await getDirectoryTree(projectRoot, projectRoot, 0, depth);
  return res.status(200).send(tree);
    } catch (error: any) {
      return res.status(500).send({ message: `Error getting file tree: ${error.message}` });
    }
  });

  // Serve static files from the 'public' directory
  app.use(express.static(path.join(__dirname, '..', '..', 'public')));

  app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', '..', 'public', 'index.html'));
  });

  // API endpoint to get file content for the mini-editor
  app.get('/api/file-content', async (req, res) => {
    const filePath = (req.query as any)['path'] as string;
    if (filePath) {
      try {
        const projectRoot = path.resolve(__dirname, '..', '..', '..', '..');

        // Resolve the requested path against projectRoot to avoid absolute path abuses
        const resolved = path.resolve(projectRoot, filePath);
        const rel = path.relative(projectRoot, resolved);
        if (rel.startsWith('..') || path.isAbsolute(rel)) {
          return res.status(403).send({ message: 'Access to the requested path is forbidden.' });
        }

        // Ensure it's a file and not a directory
        const stat = await fs.stat(resolved);
        if (!stat.isFile()) {
          return res.status(400).send({ message: 'Requested path is not a file.' });
        }

        // Enforce a file size limit to avoid large reads
        if (stat.size > MAX_ALLOWED_FILE_BYTES) {
          return res.status(413).send({ message: 'Requested file is too large.' });
        }

  const content = await fs.readFile(resolved, 'utf-8');
  return res.status(200).send({ filePath: path.relative(projectRoot, resolved), content });
      } catch (error: any) { // Added : any for error type
  return res.status(500).send({ message: `Error reading file: ${error.message}` });
      }
    } else {
  return res.status(400).send({ message: 'Missing filePath query parameter.' });
    }
  });

  const server = app.listen(port, () => {
    console.log(`Mew web server listening at http://localhost:${port}`);
  });

  // API endpoint for agent status/logs (MVP: hardcoded status)
  app.get('/api/agent-status', (req, res) => {
    res.status(200).send({
      status: 'Agent is running',
      lastUpdated: new Date().toISOString(),
      thoughts: 'Thinking about the project...',
      activeFilePath: activeFilePath
    });
  });

  // Add signal handlers for graceful shutdown
  process.on('SIGINT', () => {
    console.log('SIGINT signal received: Closing web server.');
    server.close(() => {
      process.exit(0);
    });
  });

  process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: Closing web server.');
    server.close(() => {
      process.exit(0);
    });
  });
}
