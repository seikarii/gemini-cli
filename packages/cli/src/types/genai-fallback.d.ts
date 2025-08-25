// Minimal local fallbacks for @google/genai types when upstream types aren't available during build.
// These are type-only declarations to keep the consumer compiling. Replace with real upstream types when available.

declare module '@google/genai' {
  // Keep types permissive to avoid leaking runtime coupling in this repo build.
  export type Part = { text?: string; name?: string; type?: string } | string | any;
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
}
