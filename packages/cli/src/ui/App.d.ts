/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { LoadedSettings } from '../config/settings.js';
import { type Config } from '@google/gemini-cli-core';
interface AppProps {
    config: Config;
    settings: LoadedSettings;
    startupWarnings?: string[];
    version: string;
    agent: unknown;
}
export declare const AppWrapper: (props: AppProps) => import("react/jsx-runtime").JSX.Element;
export declare const MemoizedApp: import("react").MemoExoticComponent<({ agent, config, settings, startupWarnings, version }: AppProps) => import("react/jsx-runtime").JSX.Element>;
export default MemoizedApp;
