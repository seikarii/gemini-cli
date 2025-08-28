#!/usr/bin/env node

/**
 * Simple test script to verify async conversions are working correctly
 */

import {
  getCoreSystemPrompt,
  getCoreSystemPromptAsync,
} from './packages/core/src/core/prompts.js';
import fs from 'node:fs';
import path from 'node:path';

console.log('🔍 Testing async conversion functionality...\n');

async function testAsyncConversions() {
  try {
    // Test 1: Compare sync vs async versions
    console.log('📊 Test 1: Comparing sync vs async prompts...');
    const syncResult = getCoreSystemPrompt();
    const asyncResult = await getCoreSystemPromptAsync();

    if (syncResult === asyncResult) {
      console.log('✅ Sync and async versions produce identical results');
    } else {
      console.log('❌ Sync and async versions produce different results');
      console.log('Sync length:', syncResult.length);
      console.log('Async length:', asyncResult.length);
    }

    // Test 2: Check performance difference (async should be non-blocking)
    console.log('\n⏱️  Test 2: Performance comparison...');

    const syncStart = Date.now();
    for (let i = 0; i < 100; i++) {
      getCoreSystemPrompt();
    }
    const syncTime = Date.now() - syncStart;

    const asyncStart = Date.now();
    const promises = [];
    for (let i = 0; i < 100; i++) {
      promises.push(getCoreSystemPromptAsync());
    }
    await Promise.all(promises);
    const asyncTime = Date.now() - asyncStart;

    console.log(`Sync version (100 calls): ${syncTime}ms`);
    console.log(`Async version (100 calls): ${asyncTime}ms`);

    if (asyncTime <= syncTime * 1.5) {
      console.log('✅ Async version performance is acceptable');
    } else {
      console.log('⚠️  Async version is significantly slower');
    }

    // Test 3: Verify with user memory
    console.log('\n🧠 Test 3: Testing with user memory...');
    const userMemory = 'Remember to use TypeScript best practices';
    const syncWithMemory = getCoreSystemPrompt(userMemory);
    const asyncWithMemory = await getCoreSystemPromptAsync(userMemory);

    if (syncWithMemory === asyncWithMemory) {
      console.log('✅ Memory handling works correctly in both versions');
    } else {
      console.log('❌ Memory handling differs between versions');
    }

    console.log('\n🎉 All async conversion tests completed!');
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

// Run the tests
testAsyncConversions();
