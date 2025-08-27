#!/usr/bin/env bash

# Cleanup script for stuck processes
echo "🧹 Cleaning up stuck processes..."

# Kill any hanging node processes related to gemini-cli
pkill -f "node.*gemini-cli" || true
pkill -f "esbuild" || true
pkill -f "tsc" || true

# Wait a moment for processes to terminate
sleep 2

# Check for any remaining processes
REMAINING=$(ps aux | grep -E "(npm|node|tsc)" | grep -v grep | grep gemini-cli | wc -l)
if [ "$REMAINING" -gt 0 ]; then
  echo "⚠️  Some processes still running:"
  ps aux | grep -E "(npm|node|tsc)" | grep -v grep | grep gemini-cli
  echo "Force killing..."
  pkill -9 -f "node.*gemini-cli" || true
  pkill -9 -f "esbuild" || true
  pkill -9 -f "tsc" || true
fi

echo "✅ Cleanup complete!"
