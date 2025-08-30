#!/bin/bash

# Test the subagent orchestration with correct model
export GEMINI_MODEL="gemini-1.5-flash"

echo "Testing with model: $GEMINI_MODEL"
echo "Testing subagent orchestration tools..."

echo "Test: Check for cognitive orchestration and sequential thinking features" | npx gemini --max-session-turns 1
