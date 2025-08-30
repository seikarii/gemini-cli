/**
 * @fileoverview Security Integration Demonstration
 * Shows how the new security features work and how to activate them
 */

import {
  initializeSecureFileProcessing,
  secureProcessReadContent,
  isSecureProcessingEnabled,
  getSecurityStats,
} from '../utils/secureFileUtils.js';

/**
 * Demonstration of security features
 */
async function demonstrateSecurity(): Promise<void> {
  console.log('🔒 Gemini CLI Security Integration Demonstration\n');

  // Step 1: Initialize security with default settings
  console.log('1. Initializing security features...');
  initializeSecureFileProcessing({
    enabled: true,
    enableRateLimit: true,
    enableInputValidation: true,
    enableContentSanitization: true,
    maxContentLength: 50000,
  });

  console.log(`   ✅ Security enabled: ${isSecureProcessingEnabled()}\n`);

  // Step 2: Test normal content processing
  console.log('2. Processing normal content...');
  const normalContent = 'Hello, world! This is normal content.';
  const result1 = await secureProcessReadContent(normalContent, '/test/normal.txt');
  
  if (result1.success) {
    console.log(`   ✅ Normal content processed successfully`);
    console.log(`   📄 Content length: ${result1.content?.length} characters`);
    console.log(`   ⚠️  Warnings: ${result1.warnings.length}\n`);
  } else {
    console.log(`   ❌ Failed: ${result1.error}\n`);
  }

  // Step 3: Test problematic content (escape sequences)
  console.log('3. Processing content with escape sequences...');
  const problematicContent = 'console.log(\\"Hello World\\"); alert(\\"XSS\\");';
  const result2 = await secureProcessReadContent(problematicContent, '/test/problematic.js');
  
  if (result2.success) {
    console.log(`   ✅ Problematic content sanitized successfully`);
    console.log(`   📄 Original: ${problematicContent}`);
    console.log(`   📄 Sanitized: ${result2.content}`);
    console.log(`   ⚠️  Warnings: ${result2.warnings.join(', ')}\n`);
  } else {
    console.log(`   ❌ Failed: ${result2.error}\n`);
  }

  // Step 4: Test content with null bytes and control characters
  console.log('4. Processing content with null bytes...');
  const nullByteContent = 'Text with\\0null bytes and\\x1Fcontrol chars';
  const result3 = await secureProcessReadContent(nullByteContent, '/test/nullbytes.txt');
  
  if (result3.success) {
    console.log(`   ✅ Null byte content sanitized successfully`);
    console.log(`   📄 Original: ${nullByteContent}`);
    console.log(`   📄 Sanitized: ${result3.content}`);
    console.log(`   ⚠️  Warnings: ${result3.warnings.join(', ')}\n`);
  } else {
    console.log(`   ❌ Failed: ${result3.error}\n`);
  }

  // Step 5: Test very large content
  console.log('5. Processing very large content...');
  const largeContent = 'x'.repeat(60000); // 60KB content
  const result4 = await secureProcessReadContent(largeContent, '/test/large.txt');
  
  if (result4.success) {
    console.log(`   ✅ Large content processed successfully`);
    console.log(`   📄 Original length: ${largeContent.length} characters`);
    console.log(`   📄 Processed length: ${result4.content?.length} characters`);
    console.log(`   ⚠️  Warnings: ${result4.warnings.join(', ')}\n`);
  } else {
    console.log(`   ❌ Failed: ${result4.error}\n`);
  }

  // Step 6: Test rate limiting (multiple rapid requests)
  console.log('6. Testing rate limiting...');
  const promises = [];
  for (let i = 0; i < 5; i++) {
    promises.push(secureProcessReadContent(`Content ${i}`, `/test/file${i}.txt`));
  }
  
  const results = await Promise.all(promises);
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  
  console.log(`   📊 Rapid requests: ${successful} successful, ${failed} rate-limited\n`);

  // Step 7: Show statistics
  console.log('7. Security statistics:');
  const stats = getSecurityStats();
  console.log(`   📊 Security enabled: ${stats.enabled}`);
  if (stats.stats) {
    console.log(`   📊 Rate limiter stats available: ${!!stats.stats.rateLimiterStats}`);
  }
  console.log();

  console.log('🎉 Security demonstration completed!\n');
  
  // Usage instructions
  console.log('💡 How to use in your application:');
  console.log('');
  console.log('1. Initialize security (do this once at startup):');
  console.log('   initializeSecureFileProcessing({');
  console.log('     enabled: true,');
  console.log('     enableRateLimit: true,');
  console.log('     enableInputValidation: true,');
  console.log('     enableContentSanitization: true');
  console.log('   });');
  console.log('');
  console.log('2. Process file content before sending to API:');
  console.log('   const result = await secureProcessReadContent(fileContent, filePath);');
  console.log('   if (result.success) {');
  console.log('     // Use result.content instead of original content');
  console.log('   } else {');
  console.log('     // Handle security error: result.error');
  console.log('   }');
  console.log('');
  console.log('3. Environment variables for configuration:');
  console.log('   GEMINI_SECURE_FILE_PROCESSING=true|false');
  console.log('   GEMINI_ENABLE_RATE_LIMIT=true|false');
  console.log('   GEMINI_ENABLE_INPUT_VALIDATION=true|false');
  console.log('   GEMINI_ENABLE_CONTENT_SANITIZATION=true|false');
  console.log('   GEMINI_MAX_CONTENT_LENGTH=100000');
  console.log('');
}

// Run demonstration if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateSecurity().catch(console.error);
}

export { demonstrateSecurity };
