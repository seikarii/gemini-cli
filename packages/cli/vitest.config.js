/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/// <reference types="vitest" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
export default defineConfig({
    test: {
        include: ['**/*.{test,spec}.?(c|m)[jt]s?(x)', 'config.test.ts'],
        exclude: ['**/node_modules/**', '**/dist/**', '**/cypress/**'],
        environment: 'jsdom',
        globals: true,
        reporters: ['default', 'junit'],
        silent: true,
        outputFile: {
            junit: 'junit.xml',
        },
        setupFiles: ['./test-setup.ts'],
        coverage: {
            enabled: true,
            provider: 'v8',
            reportsDirectory: './coverage',
            include: ['src/**/*'],
            reporter: [
                ['text', { file: 'full-text-summary.txt' }],
                'html',
                'json',
                'lcov',
                'cobertura',
                ['json-summary', { outputFile: 'coverage-summary.json' }],
            ],
        },
    },
    plugins: [react()],
});
//# sourceMappingURL=vitest.config.js.map