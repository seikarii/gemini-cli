/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { GeminiAgent } from '../agent/gemini-agent.js';

export function startWebServer(agent: GeminiAgent) {
  const app = express();
  const port = 3000; // You can change this port

  app.use(express.json()); // Middleware to parse JSON request bodies

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
        // Agent needs a method to read file content
        // For now, let's assume agent has a method like getFileContent
        // This will eventually use the agent's internal memory or read_file tool
        const content = await agent.getFileContent(filePath); // Placeholder call
        res.status(200).send({ filePath, content });
      } catch (error: any) { // Added : any for error type
        res.status(500).send({ message: `Error reading file: ${error.message}` });
      }
    } else {
      res.status(400).send({ message: 'Missing filePath query parameter.' });
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
