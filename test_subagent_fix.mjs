#!/usr/bin/env node

/**
 * Test script for subagent authentication fix
 */

import { Config } from './packages/core/dist/src/config/config.js';
import { SubAgentScope } from './packages/core/dist/src/core/subagent.js';

async function testSubagentAuthentication() {
  try {
    console.log('🚀 Testing subagent authentication fix...');
    
    // Create config
    const config = new Config();
    await config.initialize({
      authConfigPath: './',
      workspacePath: './',
      model: 'gemini-2.5-flash',
      maxFileSize: 1000000,
    });

    console.log('✅ Config initialized');
    
    // Initialize main client
    const geminiClient = config.getGeminiClient();
    const contentGeneratorConfig = config.getContentGeneratorConfig();
    await geminiClient.initialize(contentGeneratorConfig);
    
    console.log('✅ Main client initialized with authentication');
    
    // Get the authenticated content generator
    const contentGenerator = geminiClient.getContentGenerator();
    console.log('✅ ContentGenerator obtained from main client');
    
    // Create a subagent with the shared content generator
    const subagent = await SubAgentScope.create(
      'test-subagent',
      config,
      { 
        systemPrompt: 'You are a test subagent. Simply respond with "Hello from subagent!" and emit a test result.',
      },
      { 
        model: 'gemini-2.5-flash', 
        temp: 0.7, 
        top_p: 1 
      },
      { 
        max_time_minutes: 1, 
        max_turns: 1 
      },
      {
        tools: []
      },
      {
        outputs: {
          result: 'Test result from subagent'
        }
      },
      contentGenerator  // Pass the authenticated content generator
    );
    
    console.log('✅ Subagent created with shared authentication');
    
    // Test the subagent execution 
    console.log('🤖 Running subagent test...');
    
    // Create a simple context state
    const { ContextState } = await import('./packages/core/dist/src/core/subagent.js');
    const contextState = new ContextState();
    
    await subagent.runNonInteractive(contextState);
    
    console.log('✅ Subagent executed successfully!');
    console.log('📊 Results:', {
      terminateReason: subagent.output.terminate_reason,
      emittedVars: subagent.output.emitted_vars
    });
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Stack:', error.stack);
    return false;
  }
  
  return true;
}

// Run the test
testSubagentAuthentication()
  .then(success => {
    if (success) {
      console.log('\n🎉 SUBAGENT AUTHENTICATION FIX SUCCESS!');
      console.log('✅ Subagents can now reuse main system authentication');
    } else {
      console.log('\n💥 SUBAGENT AUTHENTICATION FIX FAILED');
      process.exit(1);
    }
  })
  .catch(error => {
    console.error('\n💥 UNEXPECTED ERROR:', error);
    process.exit(1);
  });
