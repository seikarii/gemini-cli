# Agent Orchestration Guide

## Overview

The enhanced gemini-cli system now supports intelligent agent orchestration for complex tasks. Instead of relying solely on direct tool execution, the system can coordinate specialized subagents to handle complex, multi-step operations more efficiently.

## Core Philosophy

**Agent-First Thinking:** For complex tasks, consider agent delegation before direct tool execution. This approach leverages specialized agents that can focus on specific aspects of a problem while coordinating their efforts for comprehensive solutions.

## When to Use Agents vs Direct Tools

### Direct Tools (Simple Operations)
Use direct tools for straightforward, single-step operations:
- Reading a single file: `read_file`
- Quick text searches: `grep`
- Simple file modifications: `edit`
- Command execution: `shell`

### Agent Orchestration (Complex Operations)
Use agent coordination for multi-step, complex tasks:
- System-wide analysis requiring multiple tools
- Feature implementation spanning multiple files
- Code reviews requiring different perspectives
- Performance optimization across components

## Agent Types and Capabilities

### Analysis Agents
- **Code Analysis Agent:** Deep code structure and quality analysis
- **Security Agent:** Security vulnerability assessment
- **Performance Agent:** Bottleneck identification
- **Architecture Agent:** System design evaluation

### Implementation Agents  
- **Modification Agent:** Code changes with proper testing
- **Refactoring Agent:** Structure improvements
- **Feature Agent:** Complete feature implementation
- **Integration Agent:** System-wide changes

### Testing Agents
- **Unit Test Agent:** Comprehensive unit test creation
- **Integration Test Agent:** System-wide testing
- **Quality Assurance Agent:** Code quality compliance

## API Usage Examples

### Parallel Agent Execution
```typescript
// For complex analysis tasks
const results = await SubAgentScope.runParallel(
  runtimeContext,
  "analyze authentication system for security issues",
  {
    modelConfig,
    runConfig, 
    toolConfig: { tools: ['grep', 'read_file', 'glob', 'unified_search'] }
  },
  sharedContext,
  'analysis',
  3
);
```

### Specialized Agent Creation
```typescript
// For focused analysis
const analysisResult = await SubAgentScope.createAnalysisAgent(
  runtimeContext,
  "performance bottlenecks in user authentication", 
  modelConfig,
  runConfig,
  sharedContext
);

// For code modifications  
const modificationResult = await SubAgentScope.createModificationAgent(
  runtimeContext,
  "implement input validation for user registration",
  modelConfig,
  runConfig,
  sharedContext
);
```

### Direct Agent Delegation
```typescript
// For custom specialized tasks
const result = await SubAgentScope.delegate(
  runtimeContext,
  "security-review-agent", 
  "Review authentication code for security vulnerabilities",
  modelConfig,
  runConfig,
  sharedContext,
  { tools: ['grep', 'read_file', 'unified_search'] }
);
```

## Orchestration Patterns

### Parallel Analysis Pattern
For comprehensive system review:
```
user: "Review the user authentication system"
system: SubAgentScope.runParallel() → [SecurityAgent], [CodeQualityAgent], [PerformanceAgent]
```

### Sequential Implementation Pattern
For feature development:
```
user: "Add user validation feature"
system: SubAgentScope.createModificationAgent() → implementation
        SubAgentScope.delegate() → testing validation
```

### Hybrid Pattern
Combining direct tools with agent coordination:
```
user: "Optimize database queries"
system: Direct: grep for query patterns
        Agent: SubAgentScope.createAnalysisAgent() for performance review
        Agent: SubAgentScope.createModificationAgent() for optimization
```

## Decision Matrix

| Task Type | Complexity | Tools Needed | Recommended Approach |
|-----------|------------|--------------|---------------------|
| Read config | Simple | 1 | Direct tool: `read_file` |
| Code analysis | Complex | 3+ | Analysis agent |
| Feature implementation | Complex | 5+ | Modification agent + Test agent |
| System review | Complex | Multiple | Parallel agents |
| Quick search | Simple | 1 | Direct tool: `grep` |
| Refactoring | Complex | 3+ | Specialized agent |

## Benefits of Agent Orchestration

### Efficiency
- **Parallel Processing:** Multiple agents work simultaneously
- **Specialization:** Focused agents provide better results
- **Resource Management:** Controlled concurrency prevents overload

### Quality
- **Comprehensive Analysis:** Multiple perspectives on complex problems
- **Consistent Patterns:** Specialized agents follow established patterns
- **Error Recovery:** Agent failures can fallback to direct tools

### Maintainability
- **Modular Approach:** Clear separation of concerns
- **Reusable Patterns:** Agent templates for common tasks
- **Scalable Architecture:** Easy to add new agent types

## Integration with Existing Systems

### Sequential Thinking
The agent orchestration system integrates with the existing sequential thinking service to:
- Guide delegation decisions
- Plan complex multi-agent workflows
- Coordinate between different cognitive modes

### Concurrency Control
Uses existing infrastructure:
- **Semaphore:** Controls concurrent agent execution
- **FileOperationPool:** Manages file system operations
- **Resource Management:** Prevents system overload

### Prompt Enhancement
The enhanced prompts now include:
- Agent delegation decision trees
- Orchestration pattern examples
- Integration with direct tool usage
- Context sharing strategies

## Best Practices

### Task Assessment
1. **Evaluate Complexity:** Count required tools and steps
2. **Identify Parallelization:** Look for independent subtasks
3. **Consider Specialization:** Match agents to specific expertise
4. **Plan Coordination:** Design result aggregation strategy

### Agent Selection
1. **Match Expertise:** Choose agents suited to task domain
2. **Consider Resources:** Balance concurrent agents with system capacity
3. **Plan Integration:** Ensure agents can share context effectively
4. **Design Fallbacks:** Have direct tool alternatives ready

### Result Integration
1. **Aggregate Findings:** Combine results from multiple agents
2. **Resolve Conflicts:** Handle disagreements between agents
3. **Validate Quality:** Ensure comprehensive coverage
4. **Document Decisions:** Record rationale for future reference

## Migration Guide

### From Direct Tools to Agents
- **Identify Complex Tasks:** Look for multi-tool workflows
- **Group Related Operations:** Combine related tool calls into agent tasks
- **Add Parallelization:** Split independent operations across agents
- **Maintain Efficiency:** Keep simple operations as direct tools

### Gradual Adoption
1. Start with analysis tasks (low risk, high benefit)
2. Move to implementation tasks (moderate risk, high benefit)
3. Add specialized patterns (custom agents for specific domains)
4. Optimize coordination (fine-tune agent interactions)

This agent orchestration system transforms the gemini-cli from a tool-focused assistant to an intelligent coordinator that can handle complex, multi-faceted tasks through specialized agent collaboration while maintaining efficiency for simple operations.
