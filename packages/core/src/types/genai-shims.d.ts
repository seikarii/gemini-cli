// Centralized minimal type shims for @google/genai used across packages.
// These are conservative, permissive declarations to allow the monorepo to type-check
// when the upstream `@google/genai` types may not be available. Replace with
// the official types from @google/genai when publishing or after adding it as a dependency.

declare module '@google/genai' {
  export type Part = { text?: string; name?: string } | string | any;
  export type PartListUnion = Part | Part[] | null;
  export type PartUnion = Part | PartListUnion;

  export type FunctionResponse = { name?: string; arguments?: any } | any;
  export type FunctionCall = { name?: string; args?: any } | any;

  export enum FinishReason {
    FINISH_REASON_UNSPECIFIED = 0,
    STOP = 1,
    MAX_TOKENS = 2,
    SAFETY = 3,
    RECITATION = 4,
    LANGUAGE = 5,
    BLOCKLIST = 6,
    PROHIBITED_CONTENT = 7,
    SPII = 8,
    OTHER = 9,
    MALFORMED_FUNCTION_CALL = 10,
    IMAGE_SAFETY = 11,
    UNEXPECTED_TOOL_CALL = 12,
  }

  export type Content = any;
  export type GenerateContentConfig = any;
  export type GenerateContentResponse = any;
  export type FunctionDeclaration = any;
  export type CallableTool = any;
  export type GroundingMetadata = any;
  export type GenerateContentResponseUsageMetadata = any;
}
