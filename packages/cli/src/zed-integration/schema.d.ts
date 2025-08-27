/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { z } from 'zod';
export declare const AGENT_METHODS: {
    authenticate: string;
    initialize: string;
    session_cancel: string;
    session_load: string;
    session_new: string;
    session_prompt: string;
};
export declare const CLIENT_METHODS: {
    fs_read_text_file: string;
    fs_write_text_file: string;
    session_request_permission: string;
    session_update: string;
};
export declare const PROTOCOL_VERSION = 1;
export type WriteTextFileRequest = z.infer<typeof writeTextFileRequestSchema>;
export type ReadTextFileRequest = z.infer<typeof readTextFileRequestSchema>;
export type PermissionOptionKind = z.infer<typeof permissionOptionKindSchema>;
export type Role = z.infer<typeof roleSchema>;
export type TextResourceContents = z.infer<typeof textResourceContentsSchema>;
export type BlobResourceContents = z.infer<typeof blobResourceContentsSchema>;
export type ToolKind = z.infer<typeof toolKindSchema>;
export type ToolCallStatus = z.infer<typeof toolCallStatusSchema>;
export type WriteTextFileResponse = z.infer<typeof writeTextFileResponseSchema>;
export type ReadTextFileResponse = z.infer<typeof readTextFileResponseSchema>;
export type RequestPermissionOutcome = z.infer<typeof requestPermissionOutcomeSchema>;
export type CancelNotification = z.infer<typeof cancelNotificationSchema>;
export type AuthenticateRequest = z.infer<typeof authenticateRequestSchema>;
export type AuthenticateResponse = z.infer<typeof authenticateResponseSchema>;
export type NewSessionResponse = z.infer<typeof newSessionResponseSchema>;
export type LoadSessionResponse = z.infer<typeof loadSessionResponseSchema>;
export type StopReason = z.infer<typeof stopReasonSchema>;
export type PromptResponse = z.infer<typeof promptResponseSchema>;
export type ToolCallLocation = z.infer<typeof toolCallLocationSchema>;
export type PlanEntry = z.infer<typeof planEntrySchema>;
export type PermissionOption = z.infer<typeof permissionOptionSchema>;
export type Annotations = z.infer<typeof annotationsSchema>;
export type RequestPermissionResponse = z.infer<typeof requestPermissionResponseSchema>;
export type FileSystemCapability = z.infer<typeof fileSystemCapabilitySchema>;
export type EnvVariable = z.infer<typeof envVariableSchema>;
export type McpServer = z.infer<typeof mcpServerSchema>;
export type AgentCapabilities = z.infer<typeof agentCapabilitiesSchema>;
export type AuthMethod = z.infer<typeof authMethodSchema>;
export type PromptCapabilities = z.infer<typeof promptCapabilitiesSchema>;
export type ClientResponse = z.infer<typeof clientResponseSchema>;
export type ClientNotification = z.infer<typeof clientNotificationSchema>;
export type EmbeddedResourceResource = z.infer<typeof embeddedResourceResourceSchema>;
export type NewSessionRequest = z.infer<typeof newSessionRequestSchema>;
export type LoadSessionRequest = z.infer<typeof loadSessionRequestSchema>;
export type InitializeResponse = z.infer<typeof initializeResponseSchema>;
export type ContentBlock = z.infer<typeof contentBlockSchema>;
export type ToolCallContent = z.infer<typeof toolCallContentSchema>;
export type ToolCall = z.infer<typeof toolCallSchema>;
export type ClientCapabilities = z.infer<typeof clientCapabilitiesSchema>;
export type PromptRequest = z.infer<typeof promptRequestSchema>;
export type SessionUpdate = z.infer<typeof sessionUpdateSchema>;
export type AgentResponse = z.infer<typeof agentResponseSchema>;
export type RequestPermissionRequest = z.infer<typeof requestPermissionRequestSchema>;
export type InitializeRequest = z.infer<typeof initializeRequestSchema>;
export type SessionNotification = z.infer<typeof sessionNotificationSchema>;
export type ClientRequest = z.infer<typeof clientRequestSchema>;
export type AgentRequest = z.infer<typeof agentRequestSchema>;
export type AgentNotification = z.infer<typeof agentNotificationSchema>;
export declare const writeTextFileRequestSchema: z.ZodObject<{
    content: z.ZodString;
    path: z.ZodString;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    path?: string;
    content?: string;
    sessionId?: string;
}, {
    path?: string;
    content?: string;
    sessionId?: string;
}>;
export declare const readTextFileRequestSchema: z.ZodObject<{
    limit: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    path: z.ZodString;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    path?: string;
    line?: number;
    sessionId?: string;
    limit?: number;
}, {
    path?: string;
    line?: number;
    sessionId?: string;
    limit?: number;
}>;
export declare const permissionOptionKindSchema: z.ZodUnion<[z.ZodLiteral<"allow_once">, z.ZodLiteral<"allow_always">, z.ZodLiteral<"reject_once">, z.ZodLiteral<"reject_always">]>;
export declare const roleSchema: z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>;
export declare const textResourceContentsSchema: z.ZodObject<{
    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    text: z.ZodString;
    uri: z.ZodString;
}, "strip", z.ZodTypeAny, {
    text?: string;
    mimeType?: string;
    uri?: string;
}, {
    text?: string;
    mimeType?: string;
    uri?: string;
}>;
export declare const blobResourceContentsSchema: z.ZodObject<{
    blob: z.ZodString;
    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    uri: z.ZodString;
}, "strip", z.ZodTypeAny, {
    blob?: string;
    mimeType?: string;
    uri?: string;
}, {
    blob?: string;
    mimeType?: string;
    uri?: string;
}>;
export declare const toolKindSchema: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
export declare const toolCallStatusSchema: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
export declare const writeTextFileResponseSchema: z.ZodNull;
export declare const readTextFileResponseSchema: z.ZodObject<{
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content?: string;
}, {
    content?: string;
}>;
export declare const requestPermissionOutcomeSchema: z.ZodUnion<[z.ZodObject<{
    outcome: z.ZodLiteral<"cancelled">;
}, "strip", z.ZodTypeAny, {
    outcome?: "cancelled";
}, {
    outcome?: "cancelled";
}>, z.ZodObject<{
    optionId: z.ZodString;
    outcome: z.ZodLiteral<"selected">;
}, "strip", z.ZodTypeAny, {
    outcome?: "selected";
    optionId?: string;
}, {
    outcome?: "selected";
    optionId?: string;
}>]>;
export declare const cancelNotificationSchema: z.ZodObject<{
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
}, {
    sessionId?: string;
}>;
export declare const authenticateRequestSchema: z.ZodObject<{
    methodId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    methodId?: string;
}, {
    methodId?: string;
}>;
export declare const authenticateResponseSchema: z.ZodNull;
export declare const newSessionResponseSchema: z.ZodObject<{
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
}, {
    sessionId?: string;
}>;
export declare const loadSessionResponseSchema: z.ZodNull;
export declare const stopReasonSchema: z.ZodUnion<[z.ZodLiteral<"end_turn">, z.ZodLiteral<"max_tokens">, z.ZodLiteral<"refusal">, z.ZodLiteral<"cancelled">]>;
export declare const promptResponseSchema: z.ZodObject<{
    stopReason: z.ZodUnion<[z.ZodLiteral<"end_turn">, z.ZodLiteral<"max_tokens">, z.ZodLiteral<"refusal">, z.ZodLiteral<"cancelled">]>;
}, "strip", z.ZodTypeAny, {
    stopReason?: "cancelled" | "end_turn" | "max_tokens" | "refusal";
}, {
    stopReason?: "cancelled" | "end_turn" | "max_tokens" | "refusal";
}>;
export declare const toolCallLocationSchema: z.ZodObject<{
    line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    path: z.ZodString;
}, "strip", z.ZodTypeAny, {
    path?: string;
    line?: number;
}, {
    path?: string;
    line?: number;
}>;
export declare const planEntrySchema: z.ZodObject<{
    content: z.ZodString;
    priority: z.ZodUnion<[z.ZodLiteral<"high">, z.ZodLiteral<"medium">, z.ZodLiteral<"low">]>;
    status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">]>;
}, "strip", z.ZodTypeAny, {
    status?: "in_progress" | "completed" | "pending";
    content?: string;
    priority?: "medium" | "high" | "low";
}, {
    status?: "in_progress" | "completed" | "pending";
    content?: string;
    priority?: "medium" | "high" | "low";
}>;
export declare const permissionOptionSchema: z.ZodObject<{
    kind: z.ZodUnion<[z.ZodLiteral<"allow_once">, z.ZodLiteral<"allow_always">, z.ZodLiteral<"reject_once">, z.ZodLiteral<"reject_always">]>;
    name: z.ZodString;
    optionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    name?: string;
    kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
    optionId?: string;
}, {
    name?: string;
    kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
    optionId?: string;
}>;
export declare const annotationsSchema: z.ZodObject<{
    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    audience?: ("user" | "assistant")[];
    priority?: number;
    lastModified?: string;
}, {
    audience?: ("user" | "assistant")[];
    priority?: number;
    lastModified?: string;
}>;
export declare const requestPermissionResponseSchema: z.ZodObject<{
    outcome: z.ZodUnion<[z.ZodObject<{
        outcome: z.ZodLiteral<"cancelled">;
    }, "strip", z.ZodTypeAny, {
        outcome?: "cancelled";
    }, {
        outcome?: "cancelled";
    }>, z.ZodObject<{
        optionId: z.ZodString;
        outcome: z.ZodLiteral<"selected">;
    }, "strip", z.ZodTypeAny, {
        outcome?: "selected";
        optionId?: string;
    }, {
        outcome?: "selected";
        optionId?: string;
    }>]>;
}, "strip", z.ZodTypeAny, {
    outcome?: {
        outcome?: "cancelled";
    } | {
        outcome?: "selected";
        optionId?: string;
    };
}, {
    outcome?: {
        outcome?: "cancelled";
    } | {
        outcome?: "selected";
        optionId?: string;
    };
}>;
export declare const fileSystemCapabilitySchema: z.ZodObject<{
    readTextFile: z.ZodBoolean;
    writeTextFile: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    readTextFile?: boolean;
    writeTextFile?: boolean;
}, {
    readTextFile?: boolean;
    writeTextFile?: boolean;
}>;
export declare const envVariableSchema: z.ZodObject<{
    name: z.ZodString;
    value: z.ZodString;
}, "strip", z.ZodTypeAny, {
    value?: string;
    name?: string;
}, {
    value?: string;
    name?: string;
}>;
export declare const mcpServerSchema: z.ZodObject<{
    args: z.ZodArray<z.ZodString, "many">;
    command: z.ZodString;
    env: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        value: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        value?: string;
        name?: string;
    }, {
        value?: string;
        name?: string;
    }>, "many">;
    name: z.ZodString;
}, "strip", z.ZodTypeAny, {
    name?: string;
    env?: {
        value?: string;
        name?: string;
    }[];
    command?: string;
    args?: string[];
}, {
    name?: string;
    env?: {
        value?: string;
        name?: string;
    }[];
    command?: string;
    args?: string[];
}>;
export declare const promptCapabilitiesSchema: z.ZodObject<{
    audio: z.ZodOptional<z.ZodBoolean>;
    embeddedContext: z.ZodOptional<z.ZodBoolean>;
    image: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    image?: boolean;
    audio?: boolean;
    embeddedContext?: boolean;
}, {
    image?: boolean;
    audio?: boolean;
    embeddedContext?: boolean;
}>;
export declare const agentCapabilitiesSchema: z.ZodObject<{
    loadSession: z.ZodOptional<z.ZodBoolean>;
    promptCapabilities: z.ZodOptional<z.ZodObject<{
        audio: z.ZodOptional<z.ZodBoolean>;
        embeddedContext: z.ZodOptional<z.ZodBoolean>;
        image: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        image?: boolean;
        audio?: boolean;
        embeddedContext?: boolean;
    }, {
        image?: boolean;
        audio?: boolean;
        embeddedContext?: boolean;
    }>>;
}, "strip", z.ZodTypeAny, {
    loadSession?: boolean;
    promptCapabilities?: {
        image?: boolean;
        audio?: boolean;
        embeddedContext?: boolean;
    };
}, {
    loadSession?: boolean;
    promptCapabilities?: {
        image?: boolean;
        audio?: boolean;
        embeddedContext?: boolean;
    };
}>;
export declare const authMethodSchema: z.ZodObject<{
    description: z.ZodNullable<z.ZodString>;
    id: z.ZodString;
    name: z.ZodString;
}, "strip", z.ZodTypeAny, {
    id?: string;
    name?: string;
    description?: string;
}, {
    id?: string;
    name?: string;
    description?: string;
}>;
export declare const clientResponseSchema: z.ZodUnion<[z.ZodNull, z.ZodObject<{
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content?: string;
}, {
    content?: string;
}>, z.ZodObject<{
    outcome: z.ZodUnion<[z.ZodObject<{
        outcome: z.ZodLiteral<"cancelled">;
    }, "strip", z.ZodTypeAny, {
        outcome?: "cancelled";
    }, {
        outcome?: "cancelled";
    }>, z.ZodObject<{
        optionId: z.ZodString;
        outcome: z.ZodLiteral<"selected">;
    }, "strip", z.ZodTypeAny, {
        outcome?: "selected";
        optionId?: string;
    }, {
        outcome?: "selected";
        optionId?: string;
    }>]>;
}, "strip", z.ZodTypeAny, {
    outcome?: {
        outcome?: "cancelled";
    } | {
        outcome?: "selected";
        optionId?: string;
    };
}, {
    outcome?: {
        outcome?: "cancelled";
    } | {
        outcome?: "selected";
        optionId?: string;
    };
}>]>;
export declare const clientNotificationSchema: z.ZodObject<{
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
}, {
    sessionId?: string;
}>;
export declare const embeddedResourceResourceSchema: z.ZodUnion<[z.ZodObject<{
    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    text: z.ZodString;
    uri: z.ZodString;
}, "strip", z.ZodTypeAny, {
    text?: string;
    mimeType?: string;
    uri?: string;
}, {
    text?: string;
    mimeType?: string;
    uri?: string;
}>, z.ZodObject<{
    blob: z.ZodString;
    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    uri: z.ZodString;
}, "strip", z.ZodTypeAny, {
    blob?: string;
    mimeType?: string;
    uri?: string;
}, {
    blob?: string;
    mimeType?: string;
    uri?: string;
}>]>;
export declare const newSessionRequestSchema: z.ZodObject<{
    cwd: z.ZodString;
    mcpServers: z.ZodArray<z.ZodObject<{
        args: z.ZodArray<z.ZodString, "many">;
        command: z.ZodString;
        env: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value?: string;
            name?: string;
        }, {
            value?: string;
            name?: string;
        }>, "many">;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}, {
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}>;
export declare const loadSessionRequestSchema: z.ZodObject<{
    cwd: z.ZodString;
    mcpServers: z.ZodArray<z.ZodObject<{
        args: z.ZodArray<z.ZodString, "many">;
        command: z.ZodString;
        env: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value?: string;
            name?: string;
        }, {
            value?: string;
            name?: string;
        }>, "many">;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }>, "many">;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}, {
    sessionId?: string;
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}>;
export declare const initializeResponseSchema: z.ZodObject<{
    agentCapabilities: z.ZodObject<{
        loadSession: z.ZodOptional<z.ZodBoolean>;
        promptCapabilities: z.ZodOptional<z.ZodObject<{
            audio: z.ZodOptional<z.ZodBoolean>;
            embeddedContext: z.ZodOptional<z.ZodBoolean>;
            image: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        }, {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        }>>;
    }, "strip", z.ZodTypeAny, {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    }, {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    }>;
    authMethods: z.ZodArray<z.ZodObject<{
        description: z.ZodNullable<z.ZodString>;
        id: z.ZodString;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id?: string;
        name?: string;
        description?: string;
    }, {
        id?: string;
        name?: string;
        description?: string;
    }>, "many">;
    protocolVersion: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    protocolVersion?: number;
    agentCapabilities?: {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    };
    authMethods?: {
        id?: string;
        name?: string;
        description?: string;
    }[];
}, {
    protocolVersion?: number;
    agentCapabilities?: {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    };
    authMethods?: {
        id?: string;
        name?: string;
        description?: string;
    }[];
}>;
export declare const contentBlockSchema: z.ZodUnion<[z.ZodObject<{
    annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
        audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
        lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }>>>;
    text: z.ZodString;
    type: z.ZodLiteral<"text">;
}, "strip", z.ZodTypeAny, {
    type?: "text";
    text?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}, {
    type?: "text";
    text?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}>, z.ZodObject<{
    annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
        audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
        lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }>>>;
    data: z.ZodString;
    mimeType: z.ZodString;
    type: z.ZodLiteral<"image">;
}, "strip", z.ZodTypeAny, {
    type?: "image";
    data?: string;
    mimeType?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}, {
    type?: "image";
    data?: string;
    mimeType?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}>, z.ZodObject<{
    annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
        audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
        lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }>>>;
    data: z.ZodString;
    mimeType: z.ZodString;
    type: z.ZodLiteral<"audio">;
}, "strip", z.ZodTypeAny, {
    type?: "audio";
    data?: string;
    mimeType?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}, {
    type?: "audio";
    data?: string;
    mimeType?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}>, z.ZodObject<{
    annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
        audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
        lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }>>>;
    description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    name: z.ZodString;
    size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    type: z.ZodLiteral<"resource_link">;
    uri: z.ZodString;
}, "strip", z.ZodTypeAny, {
    type?: "resource_link";
    name?: string;
    title?: string;
    size?: number;
    description?: string;
    mimeType?: string;
    uri?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}, {
    type?: "resource_link";
    name?: string;
    title?: string;
    size?: number;
    description?: string;
    mimeType?: string;
    uri?: string;
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}>, z.ZodObject<{
    annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
        audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
        lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }, {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    }>>>;
    resource: z.ZodUnion<[z.ZodObject<{
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        text: z.ZodString;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        text?: string;
        mimeType?: string;
        uri?: string;
    }, {
        text?: string;
        mimeType?: string;
        uri?: string;
    }>, z.ZodObject<{
        blob: z.ZodString;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        blob?: string;
        mimeType?: string;
        uri?: string;
    }, {
        blob?: string;
        mimeType?: string;
        uri?: string;
    }>]>;
    type: z.ZodLiteral<"resource">;
}, "strip", z.ZodTypeAny, {
    type?: "resource";
    resource?: {
        text?: string;
        mimeType?: string;
        uri?: string;
    } | {
        blob?: string;
        mimeType?: string;
        uri?: string;
    };
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}, {
    type?: "resource";
    resource?: {
        text?: string;
        mimeType?: string;
        uri?: string;
    } | {
        blob?: string;
        mimeType?: string;
        uri?: string;
    };
    annotations?: {
        audience?: ("user" | "assistant")[];
        priority?: number;
        lastModified?: string;
    };
}>]>;
export declare const toolCallContentSchema: z.ZodUnion<[z.ZodObject<{
    content: z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>;
    type: z.ZodLiteral<"content">;
}, "strip", z.ZodTypeAny, {
    type?: "content";
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
}, {
    type?: "content";
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
}>, z.ZodObject<{
    newText: z.ZodString;
    oldText: z.ZodNullable<z.ZodString>;
    path: z.ZodString;
    type: z.ZodLiteral<"diff">;
}, "strip", z.ZodTypeAny, {
    path?: string;
    type?: "diff";
    newText?: string;
    oldText?: string;
}, {
    path?: string;
    type?: "diff";
    newText?: string;
    oldText?: string;
}>]>;
export declare const toolCallSchema: z.ZodObject<{
    content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        type: z.ZodLiteral<"content">;
    }, "strip", z.ZodTypeAny, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }>, z.ZodObject<{
        newText: z.ZodString;
        oldText: z.ZodNullable<z.ZodString>;
        path: z.ZodString;
        type: z.ZodLiteral<"diff">;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }>]>, "many">>;
    kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
    locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        path: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        line?: number;
    }, {
        path?: string;
        line?: number;
    }>, "many">>;
    rawInput: z.ZodOptional<z.ZodUnknown>;
    status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
    title: z.ZodString;
    toolCallId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
}, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
}>;
export declare const clientCapabilitiesSchema: z.ZodObject<{
    fs: z.ZodObject<{
        readTextFile: z.ZodBoolean;
        writeTextFile: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        readTextFile?: boolean;
        writeTextFile?: boolean;
    }, {
        readTextFile?: boolean;
        writeTextFile?: boolean;
    }>;
}, "strip", z.ZodTypeAny, {
    fs?: {
        readTextFile?: boolean;
        writeTextFile?: boolean;
    };
}, {
    fs?: {
        readTextFile?: boolean;
        writeTextFile?: boolean;
    };
}>;
export declare const promptRequestSchema: z.ZodObject<{
    prompt: z.ZodArray<z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>, "many">;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    prompt?: ({
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    })[];
    sessionId?: string;
}, {
    prompt?: ({
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    })[];
    sessionId?: string;
}>;
export declare const sessionUpdateSchema: z.ZodUnion<[z.ZodObject<{
    content: z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>;
    sessionUpdate: z.ZodLiteral<"user_message_chunk">;
}, "strip", z.ZodTypeAny, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "user_message_chunk";
}, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "user_message_chunk";
}>, z.ZodObject<{
    content: z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>;
    sessionUpdate: z.ZodLiteral<"agent_message_chunk">;
}, "strip", z.ZodTypeAny, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "agent_message_chunk";
}, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "agent_message_chunk";
}>, z.ZodObject<{
    content: z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>;
    sessionUpdate: z.ZodLiteral<"agent_thought_chunk">;
}, "strip", z.ZodTypeAny, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "agent_thought_chunk";
}, {
    content?: {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    };
    sessionUpdate?: "agent_thought_chunk";
}>, z.ZodObject<{
    content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        type: z.ZodLiteral<"content">;
    }, "strip", z.ZodTypeAny, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }>, z.ZodObject<{
        newText: z.ZodString;
        oldText: z.ZodNullable<z.ZodString>;
        path: z.ZodString;
        type: z.ZodLiteral<"diff">;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }>]>, "many">>;
    kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
    locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        path: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        line?: number;
    }, {
        path?: string;
        line?: number;
    }>, "many">>;
    rawInput: z.ZodOptional<z.ZodUnknown>;
    sessionUpdate: z.ZodLiteral<"tool_call">;
    status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
    title: z.ZodString;
    toolCallId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
    sessionUpdate?: "tool_call";
}, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
    sessionUpdate?: "tool_call";
}>, z.ZodObject<{
    content: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        type: z.ZodLiteral<"content">;
    }, "strip", z.ZodTypeAny, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }, {
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    }>, z.ZodObject<{
        newText: z.ZodString;
        oldText: z.ZodNullable<z.ZodString>;
        path: z.ZodString;
        type: z.ZodLiteral<"diff">;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }, {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    }>]>, "many">>>;
    kind: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>>>;
    locations: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodObject<{
        line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        path: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        path?: string;
        line?: number;
    }, {
        path?: string;
        line?: number;
    }>, "many">>>;
    rawInput: z.ZodOptional<z.ZodUnknown>;
    sessionUpdate: z.ZodLiteral<"tool_call_update">;
    status: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>>>;
    title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
    toolCallId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
    sessionUpdate?: "tool_call_update";
}, {
    status?: "failed" | "in_progress" | "completed" | "pending";
    content?: ({
        type?: "content";
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
    } | {
        path?: string;
        type?: "diff";
        newText?: string;
        oldText?: string;
    })[];
    title?: string;
    kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
    locations?: {
        path?: string;
        line?: number;
    }[];
    rawInput?: unknown;
    toolCallId?: string;
    sessionUpdate?: "tool_call_update";
}>, z.ZodObject<{
    entries: z.ZodArray<z.ZodObject<{
        content: z.ZodString;
        priority: z.ZodUnion<[z.ZodLiteral<"high">, z.ZodLiteral<"medium">, z.ZodLiteral<"low">]>;
        status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">]>;
    }, "strip", z.ZodTypeAny, {
        status?: "in_progress" | "completed" | "pending";
        content?: string;
        priority?: "medium" | "high" | "low";
    }, {
        status?: "in_progress" | "completed" | "pending";
        content?: string;
        priority?: "medium" | "high" | "low";
    }>, "many">;
    sessionUpdate: z.ZodLiteral<"plan">;
}, "strip", z.ZodTypeAny, {
    entries?: {
        status?: "in_progress" | "completed" | "pending";
        content?: string;
        priority?: "medium" | "high" | "low";
    }[];
    sessionUpdate?: "plan";
}, {
    entries?: {
        status?: "in_progress" | "completed" | "pending";
        content?: string;
        priority?: "medium" | "high" | "low";
    }[];
    sessionUpdate?: "plan";
}>]>;
export declare const agentResponseSchema: z.ZodUnion<[z.ZodObject<{
    agentCapabilities: z.ZodObject<{
        loadSession: z.ZodOptional<z.ZodBoolean>;
        promptCapabilities: z.ZodOptional<z.ZodObject<{
            audio: z.ZodOptional<z.ZodBoolean>;
            embeddedContext: z.ZodOptional<z.ZodBoolean>;
            image: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        }, {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        }>>;
    }, "strip", z.ZodTypeAny, {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    }, {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    }>;
    authMethods: z.ZodArray<z.ZodObject<{
        description: z.ZodNullable<z.ZodString>;
        id: z.ZodString;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id?: string;
        name?: string;
        description?: string;
    }, {
        id?: string;
        name?: string;
        description?: string;
    }>, "many">;
    protocolVersion: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    protocolVersion?: number;
    agentCapabilities?: {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    };
    authMethods?: {
        id?: string;
        name?: string;
        description?: string;
    }[];
}, {
    protocolVersion?: number;
    agentCapabilities?: {
        loadSession?: boolean;
        promptCapabilities?: {
            image?: boolean;
            audio?: boolean;
            embeddedContext?: boolean;
        };
    };
    authMethods?: {
        id?: string;
        name?: string;
        description?: string;
    }[];
}>, z.ZodNull, z.ZodObject<{
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
}, {
    sessionId?: string;
}>, z.ZodNull, z.ZodObject<{
    stopReason: z.ZodUnion<[z.ZodLiteral<"end_turn">, z.ZodLiteral<"max_tokens">, z.ZodLiteral<"refusal">, z.ZodLiteral<"cancelled">]>;
}, "strip", z.ZodTypeAny, {
    stopReason?: "cancelled" | "end_turn" | "max_tokens" | "refusal";
}, {
    stopReason?: "cancelled" | "end_turn" | "max_tokens" | "refusal";
}>]>;
export declare const requestPermissionRequestSchema: z.ZodObject<{
    options: z.ZodArray<z.ZodObject<{
        kind: z.ZodUnion<[z.ZodLiteral<"allow_once">, z.ZodLiteral<"allow_always">, z.ZodLiteral<"reject_once">, z.ZodLiteral<"reject_always">]>;
        name: z.ZodString;
        optionId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }, {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }>, "many">;
    sessionId: z.ZodString;
    toolCall: z.ZodObject<{
        content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>;
        kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
        locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
        title: z.ZodString;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    }>;
}, "strip", z.ZodTypeAny, {
    options?: {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }[];
    sessionId?: string;
    toolCall?: {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    };
}, {
    options?: {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }[];
    sessionId?: string;
    toolCall?: {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    };
}>;
export declare const initializeRequestSchema: z.ZodObject<{
    clientCapabilities: z.ZodObject<{
        fs: z.ZodObject<{
            readTextFile: z.ZodBoolean;
            writeTextFile: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        }, {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        }>;
    }, "strip", z.ZodTypeAny, {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    }, {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    }>;
    protocolVersion: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    protocolVersion?: number;
    clientCapabilities?: {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    };
}, {
    protocolVersion?: number;
    clientCapabilities?: {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    };
}>;
export declare const sessionNotificationSchema: z.ZodObject<{
    sessionId: z.ZodString;
    update: z.ZodUnion<[z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"user_message_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    }>, z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"agent_message_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    }>, z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"agent_thought_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    }>, z.ZodObject<{
        content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>;
        kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
        locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        sessionUpdate: z.ZodLiteral<"tool_call">;
        status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
        title: z.ZodString;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    }>, z.ZodObject<{
        content: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>>;
        kind: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>>>;
        locations: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        sessionUpdate: z.ZodLiteral<"tool_call_update">;
        status: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    }>, z.ZodObject<{
        entries: z.ZodArray<z.ZodObject<{
            content: z.ZodString;
            priority: z.ZodUnion<[z.ZodLiteral<"high">, z.ZodLiteral<"medium">, z.ZodLiteral<"low">]>;
            status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">]>;
        }, "strip", z.ZodTypeAny, {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }, {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }>, "many">;
        sessionUpdate: z.ZodLiteral<"plan">;
    }, "strip", z.ZodTypeAny, {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    }, {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    }>]>;
}, "strip", z.ZodTypeAny, {
    update?: {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    } | {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    };
    sessionId?: string;
}, {
    update?: {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    } | {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    };
    sessionId?: string;
}>;
export declare const clientRequestSchema: z.ZodUnion<[z.ZodObject<{
    content: z.ZodString;
    path: z.ZodString;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    path?: string;
    content?: string;
    sessionId?: string;
}, {
    path?: string;
    content?: string;
    sessionId?: string;
}>, z.ZodObject<{
    limit: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
    path: z.ZodString;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    path?: string;
    line?: number;
    sessionId?: string;
    limit?: number;
}, {
    path?: string;
    line?: number;
    sessionId?: string;
    limit?: number;
}>, z.ZodObject<{
    options: z.ZodArray<z.ZodObject<{
        kind: z.ZodUnion<[z.ZodLiteral<"allow_once">, z.ZodLiteral<"allow_always">, z.ZodLiteral<"reject_once">, z.ZodLiteral<"reject_always">]>;
        name: z.ZodString;
        optionId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }, {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }>, "many">;
    sessionId: z.ZodString;
    toolCall: z.ZodObject<{
        content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>;
        kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
        locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
        title: z.ZodString;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    }>;
}, "strip", z.ZodTypeAny, {
    options?: {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }[];
    sessionId?: string;
    toolCall?: {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    };
}, {
    options?: {
        name?: string;
        kind?: "allow_once" | "allow_always" | "reject_once" | "reject_always";
        optionId?: string;
    }[];
    sessionId?: string;
    toolCall?: {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
    };
}>]>;
export declare const agentRequestSchema: z.ZodUnion<[z.ZodObject<{
    clientCapabilities: z.ZodObject<{
        fs: z.ZodObject<{
            readTextFile: z.ZodBoolean;
            writeTextFile: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        }, {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        }>;
    }, "strip", z.ZodTypeAny, {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    }, {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    }>;
    protocolVersion: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    protocolVersion?: number;
    clientCapabilities?: {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    };
}, {
    protocolVersion?: number;
    clientCapabilities?: {
        fs?: {
            readTextFile?: boolean;
            writeTextFile?: boolean;
        };
    };
}>, z.ZodObject<{
    methodId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    methodId?: string;
}, {
    methodId?: string;
}>, z.ZodObject<{
    cwd: z.ZodString;
    mcpServers: z.ZodArray<z.ZodObject<{
        args: z.ZodArray<z.ZodString, "many">;
        command: z.ZodString;
        env: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value?: string;
            name?: string;
        }, {
            value?: string;
            name?: string;
        }>, "many">;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}, {
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}>, z.ZodObject<{
    cwd: z.ZodString;
    mcpServers: z.ZodArray<z.ZodObject<{
        args: z.ZodArray<z.ZodString, "many">;
        command: z.ZodString;
        env: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value?: string;
            name?: string;
        }, {
            value?: string;
            name?: string;
        }>, "many">;
        name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }, {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }>, "many">;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    sessionId?: string;
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}, {
    sessionId?: string;
    cwd?: string;
    mcpServers?: {
        name?: string;
        env?: {
            value?: string;
            name?: string;
        }[];
        command?: string;
        args?: string[];
    }[];
}>, z.ZodObject<{
    prompt: z.ZodArray<z.ZodUnion<[z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        text: z.ZodString;
        type: z.ZodLiteral<"text">;
    }, "strip", z.ZodTypeAny, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"image">;
    }, "strip", z.ZodTypeAny, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        data: z.ZodString;
        mimeType: z.ZodString;
        type: z.ZodLiteral<"audio">;
    }, "strip", z.ZodTypeAny, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        name: z.ZodString;
        size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        type: z.ZodLiteral<"resource_link">;
        uri: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>, z.ZodObject<{
        annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
            audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
            lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }, {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        }>>>;
        resource: z.ZodUnion<[z.ZodObject<{
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            text: z.ZodString;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }, {
            text?: string;
            mimeType?: string;
            uri?: string;
        }>, z.ZodObject<{
            blob: z.ZodString;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }, {
            blob?: string;
            mimeType?: string;
            uri?: string;
        }>]>;
        type: z.ZodLiteral<"resource">;
    }, "strip", z.ZodTypeAny, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }, {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    }>]>, "many">;
    sessionId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    prompt?: ({
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    })[];
    sessionId?: string;
}, {
    prompt?: ({
        type?: "text";
        text?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "image";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "audio";
        data?: string;
        mimeType?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource_link";
        name?: string;
        title?: string;
        size?: number;
        description?: string;
        mimeType?: string;
        uri?: string;
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    } | {
        type?: "resource";
        resource?: {
            text?: string;
            mimeType?: string;
            uri?: string;
        } | {
            blob?: string;
            mimeType?: string;
            uri?: string;
        };
        annotations?: {
            audience?: ("user" | "assistant")[];
            priority?: number;
            lastModified?: string;
        };
    })[];
    sessionId?: string;
}>]>;
export declare const agentNotificationSchema: z.ZodObject<{
    sessionId: z.ZodString;
    update: z.ZodUnion<[z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"user_message_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    }>, z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"agent_message_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    }>, z.ZodObject<{
        content: z.ZodUnion<[z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            text: z.ZodString;
            type: z.ZodLiteral<"text">;
        }, "strip", z.ZodTypeAny, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"image">;
        }, "strip", z.ZodTypeAny, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            data: z.ZodString;
            mimeType: z.ZodString;
            type: z.ZodLiteral<"audio">;
        }, "strip", z.ZodTypeAny, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            name: z.ZodString;
            size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
            type: z.ZodLiteral<"resource_link">;
            uri: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>, z.ZodObject<{
            annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            }, "strip", z.ZodTypeAny, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }, {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            }>>>;
            resource: z.ZodUnion<[z.ZodObject<{
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                text: z.ZodString;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }, {
                text?: string;
                mimeType?: string;
                uri?: string;
            }>, z.ZodObject<{
                blob: z.ZodString;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }, {
                blob?: string;
                mimeType?: string;
                uri?: string;
            }>]>;
            type: z.ZodLiteral<"resource">;
        }, "strip", z.ZodTypeAny, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }, {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        }>]>;
        sessionUpdate: z.ZodLiteral<"agent_thought_chunk">;
    }, "strip", z.ZodTypeAny, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    }, {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    }>, z.ZodObject<{
        content: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>;
        kind: z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>;
        locations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        sessionUpdate: z.ZodLiteral<"tool_call">;
        status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>;
        title: z.ZodString;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    }>, z.ZodObject<{
        content: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            content: z.ZodUnion<[z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                text: z.ZodString;
                type: z.ZodLiteral<"text">;
            }, "strip", z.ZodTypeAny, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"image">;
            }, "strip", z.ZodTypeAny, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                data: z.ZodString;
                mimeType: z.ZodString;
                type: z.ZodLiteral<"audio">;
            }, "strip", z.ZodTypeAny, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                description: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                name: z.ZodString;
                size: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                type: z.ZodLiteral<"resource_link">;
                uri: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>, z.ZodObject<{
                annotations: z.ZodNullable<z.ZodOptional<z.ZodObject<{
                    audience: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodLiteral<"assistant">, z.ZodLiteral<"user">]>, "many">>>;
                    lastModified: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    priority: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
                }, "strip", z.ZodTypeAny, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }, {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                }>>>;
                resource: z.ZodUnion<[z.ZodObject<{
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    text: z.ZodString;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                }>, z.ZodObject<{
                    blob: z.ZodString;
                    mimeType: z.ZodNullable<z.ZodOptional<z.ZodString>>;
                    uri: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }, {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                }>]>;
                type: z.ZodLiteral<"resource">;
            }, "strip", z.ZodTypeAny, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }, {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            }>]>;
            type: z.ZodLiteral<"content">;
        }, "strip", z.ZodTypeAny, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }, {
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        }>, z.ZodObject<{
            newText: z.ZodString;
            oldText: z.ZodNullable<z.ZodString>;
            path: z.ZodString;
            type: z.ZodLiteral<"diff">;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }, {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        }>]>, "many">>>;
        kind: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"read">, z.ZodLiteral<"edit">, z.ZodLiteral<"delete">, z.ZodLiteral<"move">, z.ZodLiteral<"search">, z.ZodLiteral<"execute">, z.ZodLiteral<"think">, z.ZodLiteral<"fetch">, z.ZodLiteral<"other">]>>>;
        locations: z.ZodNullable<z.ZodOptional<z.ZodArray<z.ZodObject<{
            line: z.ZodNullable<z.ZodOptional<z.ZodNumber>>;
            path: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            path?: string;
            line?: number;
        }, {
            path?: string;
            line?: number;
        }>, "many">>>;
        rawInput: z.ZodOptional<z.ZodUnknown>;
        sessionUpdate: z.ZodLiteral<"tool_call_update">;
        status: z.ZodNullable<z.ZodOptional<z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">, z.ZodLiteral<"failed">]>>>;
        title: z.ZodNullable<z.ZodOptional<z.ZodString>>;
        toolCallId: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    }, {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    }>, z.ZodObject<{
        entries: z.ZodArray<z.ZodObject<{
            content: z.ZodString;
            priority: z.ZodUnion<[z.ZodLiteral<"high">, z.ZodLiteral<"medium">, z.ZodLiteral<"low">]>;
            status: z.ZodUnion<[z.ZodLiteral<"pending">, z.ZodLiteral<"in_progress">, z.ZodLiteral<"completed">]>;
        }, "strip", z.ZodTypeAny, {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }, {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }>, "many">;
        sessionUpdate: z.ZodLiteral<"plan">;
    }, "strip", z.ZodTypeAny, {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    }, {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    }>]>;
}, "strip", z.ZodTypeAny, {
    update?: {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    } | {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    };
    sessionId?: string;
}, {
    update?: {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "user_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_message_chunk";
    } | {
        content?: {
            type?: "text";
            text?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "image";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "audio";
            data?: string;
            mimeType?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource_link";
            name?: string;
            title?: string;
            size?: number;
            description?: string;
            mimeType?: string;
            uri?: string;
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        } | {
            type?: "resource";
            resource?: {
                text?: string;
                mimeType?: string;
                uri?: string;
            } | {
                blob?: string;
                mimeType?: string;
                uri?: string;
            };
            annotations?: {
                audience?: ("user" | "assistant")[];
                priority?: number;
                lastModified?: string;
            };
        };
        sessionUpdate?: "agent_thought_chunk";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call";
    } | {
        status?: "failed" | "in_progress" | "completed" | "pending";
        content?: ({
            type?: "content";
            content?: {
                type?: "text";
                text?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "image";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "audio";
                data?: string;
                mimeType?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource_link";
                name?: string;
                title?: string;
                size?: number;
                description?: string;
                mimeType?: string;
                uri?: string;
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            } | {
                type?: "resource";
                resource?: {
                    text?: string;
                    mimeType?: string;
                    uri?: string;
                } | {
                    blob?: string;
                    mimeType?: string;
                    uri?: string;
                };
                annotations?: {
                    audience?: ("user" | "assistant")[];
                    priority?: number;
                    lastModified?: string;
                };
            };
        } | {
            path?: string;
            type?: "diff";
            newText?: string;
            oldText?: string;
        })[];
        title?: string;
        kind?: "edit" | "read" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "other";
        locations?: {
            path?: string;
            line?: number;
        }[];
        rawInput?: unknown;
        toolCallId?: string;
        sessionUpdate?: "tool_call_update";
    } | {
        entries?: {
            status?: "in_progress" | "completed" | "pending";
            content?: string;
            priority?: "medium" | "high" | "low";
        }[];
        sessionUpdate?: "plan";
    };
    sessionId?: string;
}>;
