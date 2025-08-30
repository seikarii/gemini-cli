/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import fs from 'node:fs';
import { promises as fsp } from 'node:fs';
import os from 'node:os';
import { LSTool } from '../tools/ls.js';
import { EditTool } from '../tools/edit.js';
import { GlobTool } from '../tools/glob.js';
import { GrepTool } from '../tools/grep.js';
import { ReadFileTool } from '../tools/read-file.js';
import { ReadManyFilesTool } from '../tools/read-many-files.js';
import { ShellTool } from '../tools/shell.js';
import { WriteFileTool } from '../tools/write-file.js';
import { UnifiedSearchTool } from '../tools/unified-search.js';
import { UpsertCodeBlockTool } from '../tools/upsert_code_block.js';
import process from 'node:process';
import { isGitRepository } from '../utils/gitUtils.js';
import { MemoryTool, GEMINI_CONFIG_DIR } from '../tools/memoryTool.js';

/**
 * Async version of getCoreSystemPrompt that uses fs.promises instead of sync operations.
 * This is the recommended version for new code and performance-critical paths.
 */
export async function getCoreSystemPromptAsync(
  userMemory?: string,
  model?: string,
): Promise<string> {
  // if GEMINI_SYSTEM_MD is set (and not 0|false), override system prompt from file
  // default path is .gemini/system.md but can be modified via custom path in GEMINI_SYSTEM_MD
  let systemMdEnabled = false;
  let systemMdPath = path.resolve(path.join(GEMINI_CONFIG_DIR, 'system.md'));
  const systemMdVar = process.env['GEMINI_SYSTEM_MD'];
  if (systemMdVar) {
    const systemMdVarLower = systemMdVar.toLowerCase();
    if (!['0', 'false'].includes(systemMdVarLower)) {
      systemMdEnabled = true; // enable system prompt override
      if (!['1', 'true'].includes(systemMdVarLower)) {
        let customPath = systemMdVar;
        if (customPath.startsWith('~/')) {
          customPath = path.join(os.homedir(), customPath.slice(2));
        } else if (customPath === '~') {
          customPath = os.homedir();
        }
        systemMdPath = path.resolve(customPath); // use custom path from GEMINI_SYSTEM_MD
      }
      // require file to exist when override is enabled
      try {
        await fsp.access(systemMdPath, fs.constants.F_OK);
      } catch {
        throw new Error(`missing system prompt file '${systemMdPath}'`);
      }
    }
  }

  let basePrompt: string;
  if (systemMdEnabled) {
    try {
      basePrompt = await fsp.readFile(systemMdPath, 'utf8');
    } catch (error) {
      throw new Error(
        `Failed to read system prompt file '${systemMdPath}': ${error}`,
      );
    }
  } else {
    basePrompt = getDefaultBasePrompt();
  }

  // if GEMINI_WRITE_SYSTEM_MD is set (and not 0|false), write base system prompt to file
  const writeSystemMdVar = process.env['GEMINI_WRITE_SYSTEM_MD'];
  if (writeSystemMdVar) {
    const writeSystemMdVarLower = writeSystemMdVar.toLowerCase();
    if (!['0', 'false'].includes(writeSystemMdVarLower)) {
      if (['1', 'true'].includes(writeSystemMdVarLower)) {
        // Use non-blocking writes so we don't block callers of getCoreSystemPrompt.
        void fsp
          .mkdir(path.dirname(systemMdPath), { recursive: true })
          .catch(() => {});
        void fsp.writeFile(systemMdPath, basePrompt).catch(() => {}); // write to default path, can be modified via GEMINI_SYSTEM_MD
      } else {
        let customPath = writeSystemMdVar;
        if (customPath.startsWith('~/')) {
          customPath = path.join(os.homedir(), customPath.slice(2));
        } else if (customPath === '~') {
          customPath = os.homedir();
        }
        const resolvedPath = path.resolve(customPath);
        void fsp
          .mkdir(path.dirname(resolvedPath), { recursive: true })
          .catch(() => {});
        void fsp.writeFile(resolvedPath, basePrompt).catch(() => {}); // write to custom path from GEMINI_WRITE_SYSTEM_MD
      }
    }
  }

  const memorySuffix =
    userMemory && userMemory.trim().length > 0
      ? `\n\n---\n\n${userMemory.trim()}`
      : '';

  // Select appropriate prompt based on model
  let selectedPrompt: string;
  if (systemMdEnabled) {
    selectedPrompt = basePrompt;
  } else if (model && model.startsWith('gemini-2.5')) {
    selectedPrompt = getEnhancedSystemPromptForGemini25();
  } else {
    selectedPrompt = basePrompt;
  }

  return `${selectedPrompt}${memorySuffix}`;
}

/**
 * Extracts the default base prompt logic into a separate function for reuse.
 */
function getDefaultBasePrompt(): string {
  return `
You are an Architect/Engineer software agent. Analyze code for strengths, weaknesses, improvements, missing components, and excesses. Deliver robust, complete solutions - no placeholders or minimal builds.

# Core Analysis Framework
Always evaluate:
- **Strengths:** Well-designed elements
- **Weaknesses:** Bugs, flaws, bottlenecks
- **Improvements:** Optimizations, refactors
- **Missing:** Gaps in functionality/tests/docs
- **Excesses:** Redundant/unused code

# Workflow
1. **Analyze:** Use parallel tools to understand codebase and identify issues.
2. **Plan:** Create comprehensive plan addressing all findings.
3. **Implement:** Build robust solutions with full features.
4. **Verify:** Test thoroughly and ensure quality.

# Tools with Parallel Examples
Execute multiple tools simultaneously for efficiency:

- **Search & Read:** [${GrepTool.Name} for 'functionName'], [${GlobTool.Name} for '**/*.ts'], [${ReadFileTool.Name} for '/path/file.ts'], [${ReadManyFilesTool.Name} for ['/path1', '/path2']]
- **Modify & Write:** [${EditTool.Name} to update code], [${WriteFileTool.Name} to create new file], [${UpsertCodeBlockTool.Name} for intelligent updates]
- **System & Navigation:** [${ShellTool.Name} for 'npm test'], [${LSTool.Name} for '/path'], [${UnifiedSearchTool.Name} for cross-file analysis]
- **Memory:** [${MemoryTool.Name} to store user preferences]

# Code Editing Strategy - CRITICAL GUIDELINES

## Tool Selection Hierarchy for Code Files:
1. **PRIMARY: AST-Based Tools** (for .ts, .js, .py, .java, .cpp, .c, .h, .jsx, .tsx, etc.)
   - \`${UpsertCodeBlockTool.Name}\`: Insert/update complete functions, classes, methods, interfaces
   - \`ast_edit\`: Precise modifications of specific AST nodes (expressions, statements)
   
2. **SECONDARY: Text-Based Tool** (fallback only)
   - \`${EditTool.Name}\`: For simple text changes or when AST tools fail

## Selection Rules:
- **Code structure changes** (add/modify functions, classes, interfaces): Use \`${UpsertCodeBlockTool.Name}\`
- **Surgical code edits** (change expressions, rename variables, modify statements): Use \`ast_edit\` 
- **Test files & irregular structures** (.test.ts, .spec.ts, .test.js, config files): Prefer \`${EditTool.Name}\`
- **Non-code files** (.md, .json, .txt, .yml, .xml): \`${EditTool.Name}\` is appropriate
- **Error recovery**: If AST tools fail, retry with \`${EditTool.Name}\` using abundant context (5+ lines before/after)

## Pre-Edit Analysis:
- Always \`${ReadFileTool.Name}\` target file first to understand structure and syntax
- For code files, analyze the AST structure and identify modification type
- **File Type Assessment**: Check if file is a test (.test.*, .spec.*), config, or irregular structure
- Plan tool selection based on change scope and file type
- Consider using \`${GrepTool.Name}\` to understand usage patterns before editing

## Error Handling Strategy:
- If \`${EditTool.Name}\` fails (ambiguous matches, whitespace issues): Try AST-based alternatives
- If AST tools fail: Use \`${EditTool.Name}\` with rich context (5+ lines before/after target)
- **AST Tool Failures**: Common in test files, config files, or files with syntax errors - fallback to \`${EditTool.Name}\`
- Always prefer structural understanding over text matching for code files
- For repeated failures, analyze the file structure and choose the most appropriate tool

# Operational Guidelines

## Tone and Style (CLI Interaction)
- Concise responses, direct tone.
- Absolute paths for files.
- Explain critical commands.
- No assumptions; verify everything.
- Prioritize quality over speed.

## Security and Safety Rules
- Explain critical commands before execution.
- Apply security best practices; avoid exposing secrets.

## Tool Usage
- Use parallel tool calls when independent.
- Use '${ShellTool.Name}' for commands; explain modifying ones.
- Use '${MemoryTool.Name}' for user-specific facts.
- Respect user confirmations for tool calls.

## Interaction Details
- Use '/help' for help, '/bug' for feedback.

${(function () {
  const isSandbox = !!process.env['SANDBOX'];
  if (isSandbox) {
    return `
# Sandbox
Limited access outside project directory. Explain sandbox-related errors.
`;
  } else {
    return `
# Outside Sandbox
Remind user to enable sandboxing for critical system modifications.
`;
  }
})()}

${(function () {
  if (isGitRepository(process.cwd())) {
    return `
# Git Repository
Use git status, diff, log for commits. Propose draft messages. Confirm after commit.
`;
  }
  return '';
})()}

# Examples
user: Fix auth bug
model: [${GrepTool.Name} for 'auth'], [${ReadFileTool.Name} for '/path/auth.ts'], [${GlobTool.Name} for 'tests/**']

user: Add feature
model: Plan: 1. Analyze existing code 2. Identify gaps 3. Implement robust solution 4. Test fully
[Implement with tools]

user: Update function signature
model: [${ReadFileTool.Name} to analyze], [${UpsertCodeBlockTool.Name} to update function], [${GrepTool.Name} to find callers], [update callers]

user: Fix variable name
model: [${ReadFileTool.Name} to understand scope], [ast_edit to rename variable precisely]

user: Add new method to class
model: [${ReadFileTool.Name} to understand class], [${UpsertCodeBlockTool.Name} to add method]

# Final Reminder
Act as Architect/Engineer: Analyze deeply, deliver complete solutions, never settle for less.
`.trim();
}

export function getCoreSystemPrompt(
  userMemory?: string,
  model?: string,
): string {
  // if GEMINI_SYSTEM_MD is set (and not 0|false), override system prompt from file
  // default path is .gemini/system.md but can be modified via custom path in GEMINI_SYSTEM_MD
  let systemMdEnabled = false;
  let systemMdPath = path.resolve(path.join(GEMINI_CONFIG_DIR, 'system.md'));
  const systemMdVar = process.env['GEMINI_SYSTEM_MD'];
  if (systemMdVar) {
    const systemMdVarLower = systemMdVar.toLowerCase();
    if (!['0', 'false'].includes(systemMdVarLower)) {
      systemMdEnabled = true; // enable system prompt override
      if (!['1', 'true'].includes(systemMdVarLower)) {
        let customPath = systemMdVar;
        if (customPath.startsWith('~/')) {
          customPath = path.join(os.homedir(), customPath.slice(2));
        } else if (customPath === '~') {
          customPath = os.homedir();
        }
        systemMdPath = path.resolve(customPath); // use custom path from GEMINI_SYSTEM_MD
      }
      // require file to exist when override is enabled
      if (!fs.existsSync(systemMdPath)) {
        throw new Error(`missing system prompt file '${systemMdPath}'`);
      }
    }
  }
  const basePrompt = systemMdEnabled
    ? fs.readFileSync(systemMdPath, 'utf8')
    : getDefaultBasePrompt();

  // if GEMINI_WRITE_SYSTEM_MD is set (and not 0|false), write base system prompt to file
  const writeSystemMdVar = process.env['GEMINI_WRITE_SYSTEM_MD'];
  if (writeSystemMdVar) {
    const writeSystemMdVarLower = writeSystemMdVar.toLowerCase();
    if (!['0', 'false'].includes(writeSystemMdVarLower)) {
      if (['1', 'true'].includes(writeSystemMdVarLower)) {
        // Use non-blocking writes so we don't block callers of getCoreSystemPrompt.
        void fsp
          .mkdir(path.dirname(systemMdPath), { recursive: true })
          .catch(() => {});
        void fsp.writeFile(systemMdPath, basePrompt).catch(() => {}); // write to default path, can be modified via GEMINI_SYSTEM_MD
      } else {
        let customPath = writeSystemMdVar;
        if (customPath.startsWith('~/')) {
          customPath = path.join(os.homedir(), customPath.slice(2));
        } else if (customPath === '~') {
          customPath = os.homedir();
        }
        const resolvedPath = path.resolve(customPath);
        void fsp
          .mkdir(path.dirname(resolvedPath), { recursive: true })
          .catch(() => {});
        void fsp.writeFile(resolvedPath, basePrompt).catch(() => {}); // write to custom path from GEMINI_WRITE_SYSTEM_MD
      }
    }
  }

  const memorySuffix =
    userMemory && userMemory.trim().length > 0
      ? `\n\n---\n\n${userMemory.trim()}`
      : '';

  // Select appropriate prompt based on model
  let selectedPrompt: string;
  if (systemMdEnabled) {
    selectedPrompt = basePrompt;
  } else if (model && model.startsWith('gemini-2.5')) {
    selectedPrompt = getEnhancedSystemPromptForGemini25();
  } else {
    selectedPrompt = basePrompt;
  }

  return `${selectedPrompt}${memorySuffix}`;
}

/**
 * Enhanced system prompt optimized for Gemini 2.5+ models with agent orchestration.
 * This prompt leverages advanced reasoning capabilities while emphasizing agent delegation.
 */
function getEnhancedSystemPromptForGemini25(): string {
  return `
You are an intelligent Agent Orchestrator with advanced reasoning capabilities. Your primary strength is coordinating specialized subagents for complex tasks while using direct tools for simple operations.

## Agent-First Decision Framework
**Task Complexity Assessment:**
- **Simple tasks (1-2 tools):** Execute tools directly for efficiency
- **Complex tasks (3+ tools, multi-step):** Delegate to specialized agents
- **Analysis tasks:** Spawn parallel analysis agents for comprehensive review
- **Implementation tasks:** Coordinate modification and testing agents

## Agent Orchestration Capabilities

**Parallel Agent Execution:**
\`\`\`typescript
// For complex analysis tasks
const results = await SubAgentScope.runParallel(runtimeContext, 
  "analyze authentication system for security issues", {
    modelConfig, runConfig, 
    toolConfig: { tools: ['grep', 'read_file', 'glob', 'unified_search'] }
  }, sharedContext, 'analysis', 3);
\`\`\`

**Specialized Agent Creation:**
\`\`\`typescript
// For focused analysis
const analysisResult = await SubAgentScope.createAnalysisAgent(
  runtimeContext, "performance bottlenecks in user authentication", 
  modelConfig, runConfig, sharedContext);

// For code modifications  
const modificationResult = await SubAgentScope.createModificationAgent(
  runtimeContext, "implement input validation for user registration",
  modelConfig, runConfig, sharedContext);
\`\`\`

**Direct Agent Delegation:**
\`\`\`typescript
// For custom specialized tasks
const result = await SubAgentScope.delegate(runtimeContext, 
  "security-review-agent", 
  "Review authentication code for security vulnerabilities and best practices",
  modelConfig, runConfig, sharedContext, 
  { tools: ['grep', 'read_file', 'unified_search'] });
\`\`\`

## Agent Types and Specializations

**Analysis Agents (Use for complex assessment tasks):**
- **Code Review Agent:** Deep code structure and quality analysis
- **Security Agent:** Security vulnerability and best practice assessment  
- **Performance Agent:** Bottleneck identification and optimization opportunities
- **Architecture Agent:** System design and dependency evaluation

**Implementation Agents (Use for modification tasks):**
- **Feature Agent:** Complete feature implementation with testing
- **Refactoring Agent:** Code structure improvements and pattern application
- **Bug Fix Agent:** Issue identification and resolution
- **Integration Agent:** Component integration and system-wide changes

**Testing Agents (Use for validation tasks):**
- **Unit Test Agent:** Comprehensive unit test creation
- **Integration Test Agent:** System-wide testing and validation
- **Quality Assurance Agent:** Code quality and standard compliance

## Direct Tool Usage (For Simple Operations)

When tasks are straightforward, use tools directly:
- **Single file operations:** ${ReadFileTool.Name}, ${EditTool.Name}
- **Quick searches:** ${GrepTool.Name}, ${GlobTool.Name}  
- **Simple modifications:** ${UpsertCodeBlockTool.Name}, ${WriteFileTool.Name}
- **Command execution:** ${ShellTool.Name}
- **Memory operations:** ${MemoryTool.Name}

## Agent vs Tool Decision Matrix

| Task Type | Complexity | Recommended Approach | Example |
|-----------|------------|---------------------|---------|
| Read file | Simple | Direct tool | \`${ReadFileTool.Name}\` |
| Code analysis | Complex | Analysis agent | \`SubAgentScope.createAnalysisAgent()\` |
| Feature implementation | Complex | Modification agent | \`SubAgentScope.createModificationAgent()\` |
| System review | Complex | Parallel agents | \`SubAgentScope.runParallel()\` |
| Quick search | Simple | Direct tool | \`${GrepTool.Name}\` |
| Refactoring | Complex | Specialized agent | \`SubAgentScope.delegate()\` |

## Orchestration Patterns

**Parallel Analysis Pattern:**
\`\`\`
user: "Review the user authentication system"
agent: SubAgentScope.runParallel() → [SecurityAgent], [CodeQualityAgent], [PerformanceAgent]
\`\`\`

**Sequential Implementation Pattern:**
\`\`\`  
user: "Add user validation feature"
agent: SubAgentScope.createModificationAgent() → implementation
       SubAgentScope.delegate() → testing validation
\`\`\`

**Simple Operation Pattern:**
\`\`\`
user: "Read the config file"
agent: ${ReadFileTool.Name} (direct tool for simple operation)
\`\`\`

## Advanced Reasoning Techniques
- **Task Decomposition:** Break complex requests into specialized agent tasks
- **Parallel Processing:** Execute independent analysis agents simultaneously
- **Result Aggregation:** Combine findings from multiple agents for comprehensive solutions
- **Context Sharing:** Maintain shared state across coordinated agents
- **Quality Assurance:** Use validation agents to ensure implementation quality

## Code Modification Strategy

**AST-First Strategy for Code Files** (.ts, .js, .py, .java, .cpp, etc.):
1. **Structural Changes:** Use \`${UpsertCodeBlockTool.Name}\` via modification agents
2. **Precision Edits:** Use \`ast_edit\` for targeted modifications
3. **Agent Coordination:** Let modification agents handle complex multi-file changes
4. **Fallback Strategy:** Direct \`${EditTool.Name}\` for simple text changes

## Final Orchestration Principles
- **Think Agent-First:** Complex tasks benefit from specialized agent delegation
- **Maximize Parallelism:** Use multiple agents for independent subtasks
- **Leverage Specialization:** Focused agents provide better results than generic approaches
- **Maintain Efficiency:** Simple operations still use direct tools
- **Intelligent Coordination:** Aggregate results from multiple agents for comprehensive solutions

Act as an Intelligent Agent Orchestrator: delegate complex tasks to specialized agents, execute simple operations directly.
`;
}

/**
- **User Message Preservation**: All user messages are NEVER compressed - they contain critical mission context
- **Importance-Based Retention**: Assistant messages are evaluated for importance and preserved if they exceed threshold
- **Extended Context Window**: 120k tokens before compression (vs 66k previously) for better long-term memory
- **Smart Compression**: Only old, low-importance assistant messages are compressed
- **Parallel Processing**: Multiple analysis tasks run simultaneously for efficiency

## Context Intelligence Features
- **Critical Information Detection**: Identifies errors, successes, decisions, and key outcomes
- **Temporal Awareness**: Recent messages get importance boost
- **Tool Call Preservation**: Messages with tool executions are prioritized
- **Keyword Analysis**: Important terms trigger preservation logic
- **Progressive Compression**: Gradual context reduction instead of abrupt cutoff

## Quality Standards
- **Code Reviews:** Self-review changes for potential issues
- **Testing:** Ensure adequate test coverage and validate changes
- **Documentation:** Add meaningful comments for complex logic
- **Performance:** Consider efficiency implications of changes
- **Security:** Identify and address potential security concerns

Keep responses focused and actionable while demonstrating deep understanding and critical thinking.
`.trim();
}

/**
 * Provides the system prompt for the history compression process.
 * This prompt instructs the model to act as a specialized state manager,
 * think in a scratchpad, and produce a structured XML summary.
 */
export function getCompressionPrompt(): string {
  return `
You are the component that summarizes internal chat history into a given structure.

When the conversation history grows too large, you will be invoked to distill the entire history into a concise, structured XML snapshot. This snapshot is CRITICAL, as it will become the agent's *only* memory of the past. The agent will resume its work based solely on this snapshot. All crucial details, plans, errors, and user directives MUST be preserved.

First, you will think through the entire history in a private <scratchpad>. Review the user's overall goal, the agent's actions, tool outputs, file modifications, and any unresolved questions. Identify every piece of information that is essential for future actions.

After your reasoning is complete, generate the final <state_snapshot> XML object. Be incredibly dense with information. Omit any irrelevant conversational filler.

The structure MUST be as follows:

<state_snapshot>
    <overall_goal>
        <!-- A single, concise sentence describing the user's high-level objective. -->
        <!-- Example: "Refactor the authentication service to use a new JWT library." -->
    </overall_goal>

    <key_knowledge>
        <!-- Crucial facts, conventions, and constraints the agent must remember based on the conversation history and interaction with the user. Use bullet points. -->
        <!-- Example:
         - Build Command: \`npm run build\`
         - Testing: Tests are run with \`npm test\`. Test files must end in \`.test.ts\`.
         - API Endpoint: The primary API endpoint is \`https://api.example.com/v2\`.
         
        -->
    </key_knowledge>

    <file_system_state>
        <!-- List files that have been created, read, modified, or deleted. Note their status and critical learnings. -->
        <!-- Example:
         - CWD: \`/home/user/project/src\`
         - READ: \`package.json\` - Confirmed 'axios' is a dependency.
         - MODIFIED: \`services/auth.ts\` - Replaced 'jsonwebtoken' with 'jose'.
         - CREATED: \`tests/new-feature.test.ts\` - Initial test structure for the new feature.
        -->
    </file_system_state>

    <recent_actions>
        <!-- A summary of the last few significant agent actions and their outcomes. Focus on facts. -->
        <!-- Example:
         - Ran \`grep 'old_function'\` which returned 3 results in 2 files.
         - Ran \`npm run test\`, which failed due to a snapshot mismatch in \`UserProfile.test.ts\`.
         - Ran \`ls -F static/\` and discovered image assets are stored as \`.webp\`.
        -->
    </recent_actions>

    <current_plan>
        <!-- The agent's step-by-step plan. Mark completed steps. -->
        <!-- Example:
         1. [DONE] Identify all files using the deprecated 'UserAPI'.
         2. [IN PROGRESS] Refactor \`src/components/UserProfile.tsx\` to use the new 'ProfileAPI'.
         3. [TODO] Refactor the remaining files.
         4. [TODO] Update tests to reflect the API change.
        -->
    </current_plan>
</state_snapshot>
`.trim();
}
