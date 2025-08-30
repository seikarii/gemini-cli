#!/bin/bash

echo "Testing subagent orchestration integration..."

# Navigate to the project directory
cd /media/seikarii/Nvme/backupsmew/gemini-cli

# Quick test to see if tools are registered
echo "Running gemini CLI with a simple query to test tool availability..."

# Test the cognitive orchestration and sequential thinking
echo "Hello, can you help me test the new subagent orchestration functionality?" | npx gemini -m gemini-1.5-flash-latest --include-tools="run_parallel_subagents,delegate_to_subagent,create_analysis_agent" --target-dir=/media/seikarii/Nvme/backupsmew/gemini-cli --model-override-for-config
