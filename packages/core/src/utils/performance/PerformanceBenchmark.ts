/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { performance } from 'perf_hooks';
import {
  OptimizedFileOperations,
  Semaphore,
  FileOperationPool,
  BufferPool,
  EnhancedLruCache,
  LruCache
} from './index.js';
import { loadServerHierarchicalMemory } from '../memoryDiscovery.js';
import { FileDiscoveryService } from '../../services/fileDiscoveryService.js';
import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Benchmark results interface
 */
interface BenchmarkResult {
  testName: string;
  operation: string;
  iterations: number;
  totalTime: number;
  averageTime: number;
  operationsPerSecond: number;
  memoryUsage?: number;
  cacheHitRate?: number;
}

/**
 * Performance benchmarking suite for optimized file operations
 */
export class PerformanceBenchmark {
  private results: BenchmarkResult[] = [];
  private optimizedFileOps: OptimizedFileOperations;
  private filePool: FileOperationPool;
  private bufferPool: BufferPool;

  constructor() {
    this.optimizedFileOps = OptimizedFileOperations.getInstance({
      enableCache: true,
      enablePooling: true,
      enableBufferPool: true,
    });
    this.filePool = new FileOperationPool(10, 1000);
    this.bufferPool = new BufferPool();
  }

  /**
   * Run comprehensive performance benchmarks
   */
  async runBenchmarks(): Promise<BenchmarkResult[]> {
    console.log('🚀 Starting Performance Benchmarks...\n');

    // Cache performance tests
    await this.benchmarkCachePerformance();
    
    // File operation tests
    await this.benchmarkFileOperations();
    
    // Memory discovery tests
    await this.benchmarkMemoryDiscovery();
    
    // Pool performance tests
    await this.benchmarkPoolPerformance();

    return this.results;
  }

  /**
   * Benchmark cache performance: enhanced vs legacy
   */
  private async benchmarkCachePerformance(): Promise<void> {
    console.log('📊 Benchmarking Cache Performance...');

    const iterations = 10000;
    const testData = Array.from({ length: 100 }, (_, i) => ({
      key: `key-${i}`,
      value: `${'x'.repeat(1000 + i * 100)}`, // Variable sized data
    }));

    // Legacy LRU Cache
    const legacyCache = new LruCache<string, string>(1000);
    const legacyStartTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      const data = testData[i % testData.length];
      legacyCache.set(data.key, data.value);
      legacyCache.get(data.key);
    }
    
    const legacyTime = performance.now() - legacyStartTime;

    // Enhanced LRU Cache
    const enhancedCache = new EnhancedLruCache<string, string>(1000, {
      enableCompression: true,
      enableTTL: false,
      trackMemory: true,
    });
    
    const enhancedStartTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      const data = testData[i % testData.length];
      enhancedCache.set(data.key, data.value);
      enhancedCache.get(data.key);
    }
    
    const enhancedTime = performance.now() - enhancedStartTime;
    const enhancedStats = enhancedCache.getStats();

    this.results.push({
      testName: 'Cache Performance',
      operation: 'Legacy LRU Cache',
      iterations,
      totalTime: legacyTime,
      averageTime: legacyTime / iterations,
      operationsPerSecond: iterations / (legacyTime / 1000),
    });

    this.results.push({
      testName: 'Cache Performance',
      operation: 'Enhanced LRU Cache',
      iterations,
      totalTime: enhancedTime,
      averageTime: enhancedTime / iterations,
      operationsPerSecond: iterations / (enhancedTime / 1000),
      memoryUsage: enhancedStats.memoryUsage,
      cacheHitRate: enhancedStats.hitRate,
    });

    console.log(`  ✅ Legacy Cache: ${legacyTime.toFixed(2)}ms (${(iterations / (legacyTime / 1000)).toFixed(0)} ops/sec)`);
    console.log(`  ✅ Enhanced Cache: ${enhancedTime.toFixed(2)}ms (${(iterations / (enhancedTime / 1000)).toFixed(0)} ops/sec)`);
    console.log(`  📈 Improvement: ${((legacyTime - enhancedTime) / legacyTime * 100).toFixed(1)}%\n`);
  }

  /**
   * Benchmark file operations
   */
  private async benchmarkFileOperations(): Promise<void> {
    console.log('📁 Benchmarking File Operations...');

    const testFiles = await this.createTestFiles();
    const iterations = testFiles.length;

    // Standard fs operations
    const standardStartTime = performance.now();
    
    for (const filePath of testFiles) {
      try {
        await fs.readFile(filePath, 'utf-8');
      } catch {
        // Ignore errors for benchmark
      }
    }
    
    const standardTime = performance.now() - standardStartTime;

    // Optimized file operations
    const optimizedStartTime = performance.now();
    
    for (const filePath of testFiles) {
      await this.optimizedFileOps.safeReadFile(filePath);
    }
    
    const optimizedTime = performance.now() - optimizedStartTime;
    const stats = this.optimizedFileOps.getStats();

    this.results.push({
      testName: 'File Operations',
      operation: 'Standard fs.readFile',
      iterations,
      totalTime: standardTime,
      averageTime: standardTime / iterations,
      operationsPerSecond: iterations / (standardTime / 1000),
    });

    this.results.push({
      testName: 'File Operations',
      operation: 'Optimized File Operations',
      iterations,
      totalTime: optimizedTime,
      averageTime: optimizedTime / iterations,
      operationsPerSecond: iterations / (optimizedTime / 1000),
      cacheHitRate: stats.cache.hitRate,
    });

    console.log(`  ✅ Standard: ${standardTime.toFixed(2)}ms (${(iterations / (standardTime / 1000)).toFixed(0)} ops/sec)`);
    console.log(`  ✅ Optimized: ${optimizedTime.toFixed(2)}ms (${(iterations / (optimizedTime / 1000)).toFixed(0)} ops/sec)`);
    console.log(`  📈 Improvement: ${((standardTime - optimizedTime) / standardTime * 100).toFixed(1)}%\n`);

    // Cleanup test files
    await this.cleanupTestFiles(testFiles);
  }

  /**
   * Benchmark memory discovery operations
   */
  private async benchmarkMemoryDiscovery(): Promise<void> {
    console.log('🧠 Benchmarking Memory Discovery...');

    const testDirectory = process.cwd();
    const iterations = 5; // Fewer iterations for heavy operation

    const startTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      try {
        await loadServerHierarchicalMemory(
          testDirectory,
          [testDirectory],
          false, // debugMode
          {} as FileDiscoveryService, // fileService placeholder
          [], // extensionContextFilePaths
          'tree',
          undefined, // fileFilteringOptions
          50, // maxDirs
        );
      } catch {
        // Ignore errors for benchmark
      }
    }
    
    const totalTime = performance.now() - startTime;

    this.results.push({
      testName: 'Memory Discovery',
      operation: 'Hierarchical Memory Loading',
      iterations,
      totalTime,
      averageTime: totalTime / iterations,
      operationsPerSecond: iterations / (totalTime / 1000),
    });

    console.log(`  ✅ Memory Discovery: ${totalTime.toFixed(2)}ms (${(totalTime / iterations).toFixed(2)}ms avg)\n`);
  }

  /**
   * Benchmark pool performance
   */
  private async benchmarkPoolPerformance(): Promise<void> {
    console.log('🏊 Benchmarking Pool Performance...');

    const iterations = 1000;

    // FileOperationPool test
    const poolStartTime = performance.now();
    
    const promises = Array.from({ length: iterations }, (_, i) => 
      this.filePool.execute(`test-${i % 10}`, async () => {
        await new Promise(resolve => setTimeout(resolve, 1));
        return `result-${i}`;
      })
    );
    
    await Promise.all(promises);
    const poolTime = performance.now() - poolStartTime;
    const poolStats = this.filePool.getStats();

    // BufferPool test
    const bufferStartTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      const buffer = this.bufferPool.acquire(1024);
      buffer.write('test data');
      this.bufferPool.release(buffer);
    }
    
    const bufferTime = performance.now() - bufferStartTime;
    const bufferStats = this.bufferPool.getStats();

    this.results.push({
      testName: 'Pool Performance',
      operation: 'File Operation Pool',
      iterations,
      totalTime: poolTime,
      averageTime: poolTime / iterations,
      operationsPerSecond: iterations / (poolTime / 1000),
    });

    this.results.push({
      testName: 'Pool Performance',
      operation: 'Buffer Pool',
      iterations,
      totalTime: bufferTime,
      averageTime: bufferTime / iterations,
      operationsPerSecond: iterations / (bufferTime / 1000),
    });

    console.log(`  ✅ File Pool: ${poolTime.toFixed(2)}ms (${poolStats.totalOperations} ops)`);
    console.log(`  ✅ Buffer Pool: ${bufferTime.toFixed(2)}ms (${bufferStats.reuseRate.toFixed(1)}% reuse rate)\n`);
  }

  /**
   * Create test files for benchmarking
   */
  private async createTestFiles(): Promise<string[]> {
    const testDir = path.join(process.cwd(), '.benchmark-tmp');
    await fs.mkdir(testDir, { recursive: true });

    const files: string[] = [];
    
    for (let i = 0; i < 50; i++) {
      const filePath = path.join(testDir, `test-file-${i}.txt`);
      const content = `Test content for file ${i}\n`.repeat(100);
      await fs.writeFile(filePath, content);
      files.push(filePath);
    }

    return files;
  }

  /**
   * Cleanup test files
   */
  private async cleanupTestFiles(files: string[]): Promise<void> {
    if (files.length > 0) {
      const testDir = path.dirname(files[0]);
      await fs.rm(testDir, { recursive: true, force: true });
    }
  }

  /**
   * Print detailed benchmark results
   */
  printResults(): void {
    console.log('📊 Benchmark Results Summary:');
    console.log('=' .repeat(80));

    const grouped = this.results.reduce((acc, result) => {
      if (!acc[result.testName]) acc[result.testName] = [];
      acc[result.testName].push(result);
      return acc;
    }, {} as Record<string, BenchmarkResult[]>);

    for (const [testName, results] of Object.entries(grouped)) {
      console.log(`\n🔬 ${testName}:`);
      
      for (const result of results) {
        console.log(`  📈 ${result.operation}:`);
        console.log(`     Time: ${result.totalTime.toFixed(2)}ms`);
        console.log(`     Avg: ${result.averageTime.toFixed(4)}ms`);
        console.log(`     Ops/sec: ${result.operationsPerSecond.toFixed(0)}`);
        
        if (result.memoryUsage) {
          console.log(`     Memory: ${(result.memoryUsage / 1024).toFixed(2)}KB`);
        }
        
        if (result.cacheHitRate) {
          console.log(`     Cache Hit Rate: ${result.cacheHitRate.toFixed(1)}%`);
        }
      }
    }

    console.log('\n' + '=' .repeat(80));
  }
}

/**
 * Run benchmarks from CLI
 */
export async function runPerformanceBenchmarks(): Promise<void> {
  const benchmark = new PerformanceBenchmark();
  await benchmark.runBenchmarks();
  benchmark.printResults();
}
