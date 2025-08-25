/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';

interface FileTreeNode {
  name: string;
  type: 'file' | 'directory';
  children?: FileTreeNode[];
  isOpen?: boolean;
  path: string;
}

const Directory = ({ node, onFileSelect }: { node: FileTreeNode, onFileSelect: (path: string) => void }) => {
  const [isOpen, setIsOpen] = useState(node.isOpen || false);
  const [children, setChildren] = useState<FileTreeNode[]>(node.children || []);

  const handleToggle = async () => {
    setIsOpen(!isOpen);
    if (!children.length) {
      try {
        const response = await fetch(`/api/file-tree?path=${encodeURIComponent(node.path)}`);
        const data = await response.json();
        setChildren(data.map((child: any) => ({ ...child, path: `${node.path}/${child.name}` })));
      } catch (error) {
        console.error('Error fetching directory contents:', error);
      }
    }
  };

  return (
    <div>
      <div onClick={handleToggle} style={{ cursor: 'pointer' }}>
        {isOpen ? '[-]' : '[+]'} {node.name}
      </div>
      {isOpen && (
        <div style={{ paddingLeft: '20px' }}>
          {children.map(child => (
            <div key={child.name}>
              {child.type === 'directory' ? (
                <Directory node={child} onFileSelect={onFileSelect} />
              ) : (
                <div onClick={() => onFileSelect(child.path)} style={{ cursor: 'pointer' }}>
                  [F] {child.name}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const FileTree = ({ onFileSelect }: { onFileSelect: (path: string) => void }) => {
  const [root, setRoot] = useState<FileTreeNode | null>(null);

  useEffect(() => {
    const fetchRoot = async () => {
      try {
        const response = await fetch('/api/file-tree');
        const data = await response.json();
        setRoot({ name: '/', type: 'directory', children: data.map((child: any) => ({ ...child, path: child.name })), path: '' });
      } catch (error) {
        console.error('Error fetching root directory:', error);
      }
    };
    fetchRoot();
  }, []);

  if (!root) {
    return <div>Loading file tree...</div>;
  }

  return <Directory node={root} onFileSelect={onFileSelect} />;
};

export const MewApp = () => {
  const [agentOutput, setAgentOutput] = useState<string>('Waiting for agent output...');
  const [whisperInput, setWhisperInput] = useState<string>('');
  const [currentFilePath, setCurrentFilePath] = useState<string>('');
  const [fileContent, setFileContent] = useState<string>('// Load a file to see its content');
  const [activeFileFromServer, setActiveFileFromServer] = useState<string | null>(null);

  // Fetch agent status/logs
  useEffect(() => {
    const fetchAgentStatus = async () => {
      try {
        const response = await fetch('/api/agent-status');
        const data = await response.json();
        setAgentOutput(JSON.stringify(data, null, 2)); // Display agent status
        if (data.activeFilePath) {
          setActiveFileFromServer(data.activeFilePath);
        }
      } catch (error) {
        setAgentOutput(`Error fetching agent status: ${error}`);
      }
    };

    const interval = setInterval(fetchAgentStatus, 3000); // Poll every 3 seconds
    fetchAgentStatus(); // Initial fetch
    return () => clearInterval(interval);
  }, []);

  const handleLoadFile = async (path?: string) => {
    const a = path || currentFilePath;
    if (a.trim() === '') return;
    try {
      const response = await fetch(`/api/file-content?path=${encodeURIComponent(a)}`);
      if (response.ok) {
        const data = await response.json();
        setFileContent(data.content);
        setCurrentFilePath(data.filePath);
        console.log(`Loaded file: ${data.filePath}`);
      } else {
        console.error('Failed to load file:', await response.text());
        setFileContent(`Error loading file: ${a}`);
      }
    } catch (error) {
      console.error('Error loading file:', error);
      setFileContent(`Error loading file: ${a}`);
    }
  };

  useEffect(() => {
    if (activeFileFromServer && activeFileFromServer !== currentFilePath) {
      handleLoadFile(activeFileFromServer);
    }
  }, [activeFileFromServer]);

  const handleWhisperSubmit = async () => {
    if (whisperInput.trim() === '') return;
    try {
      const response = await fetch('/api/whisper', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: whisperInput, kind: 'user_input' }), // Default kind for now
      });
      if (response.ok) {
        console.log('Whisper sent successfully!');
        setWhisperInput('');
      } else {
        console.error('Failed to send whisper:', await response.text());
      }
    } catch (error) {
      console.error('Error sending whisper:', error);
    }
  };

  return (
    <div style={{ border: '1px solid grey', padding: '10px', borderRadius: '5px', display: 'flex', flexDirection: 'column', height: '100vh', fontFamily: 'monospace' }}>
      <h1 style={{ fontSize: '1.5em', marginBottom: '10px' }}>Mew Window</h1>
      <div style={{ display: 'flex', flexGrow: 1 }}>
        <div style={{ width: '30%', borderRight: '1px solid lightgrey', overflowY: 'auto' }}>
          <FileTree onFileSelect={handleLoadFile} />
        </div>
        <div style={{ width: '70%', display: 'flex', flexDirection: 'column' }}>
          {/* Agent Output / Logs */}
          <div style={{ flexGrow: 1, border: '1px solid lightgrey', padding: '5px', overflowY: 'auto', marginBottom: '10px', backgroundColor: '#f0f0f0' }}>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{agentOutput}</pre>
          </div>

          {/* Whisper Input */}
          <div style={{ display: 'flex', marginBottom: '10px' }}>
            <input
              type="text"
              value={whisperInput}
              onChange={(e) => setWhisperInput(e.target.value)}
              onKeyPress={(e) => { if (e.key === 'Enter') handleWhisperSubmit(); }}
              placeholder="Whisper to agent..."
              style={{ flexGrow: 1, padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
            <button
              onClick={handleWhisperSubmit}
              style={{ marginLeft: '10px', padding: '8px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
            >
              Whisper
            </button>
          </div>

          {/* Mini-Editor */}
          <div style={{ display: 'flex', marginBottom: '10px' }}>
            <input
              type="text"
              value={currentFilePath}
              onChange={(e) => setCurrentFilePath(e.target.value)}
              onKeyPress={(e) => { if (e.key === 'Enter') handleLoadFile(); }}
              placeholder="Enter file path to load..."
              style={{ flexGrow: 1, padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
            <button
              onClick={() => handleLoadFile()}
              style={{ marginLeft: '10px', padding: '8px 15px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
            >
              Load File
            </button>
          </div>
          <textarea
            value={fileContent}
            readOnly
            style={{ flexGrow: 2, border: '1px solid lightgrey', padding: '5px', overflowY: 'auto', backgroundColor: '#e9ecef', minHeight: '150px' }}
          ></textarea>
        </div>
      </div>
    </div>
  );
};

