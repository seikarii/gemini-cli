// Re-export ambient @google/genai types as named type aliases so other packages
// can import them from the core package.
import type {
  Part,
  PartListUnion,
  PartUnion,
  FinishReason,
  FunctionCall,
  FunctionResponse,
} from '@google/genai';

export type { Part, PartListUnion, PartUnion, FinishReason, FunctionCall, FunctionResponse };
