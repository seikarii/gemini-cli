/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { UserTierId } from './types.js';
import { OAuth2Client } from 'google-auth-library';
import { AuthType } from '../core/contentGenerator.js';
export declare class ProjectIdRequiredError extends Error {
    constructor();
}
export declare class ProjectAccessError extends Error {
    constructor(projectId: string, details?: string);
}
export declare class LicenseMismatchError extends Error {
    constructor(expected: string, actual: string);
}
export interface UserData {
    projectId: string;
    userTier: UserTierId;
}
/**
 *
 * @param client OAuth2 client
 * @param authType the authentication type being used
 * @returns the user's actual project id and tier
 */
export declare function setupUser(client: OAuth2Client, authType: AuthType): Promise<UserData>;
