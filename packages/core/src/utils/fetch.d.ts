/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export declare class FetchError extends Error {
    code?: string;
    constructor(message: string, code?: string);
}
export declare function isPrivateIp(url: string): boolean;
export declare function fetchWithTimeout(url: string, timeout: number): Promise<Response>;
