Here are the changes I'd like you to make manually:

1.  **File:** `/media/seikarii/Nvme/gemini-cli/packages/core/src/utils/stringUtils.ts`
    *   **Change:** Add the `escapeBackticks` function.
    *   **Location:** At the end of the file, after the `escapeRegExp` function.

    ```typescript
    export function escapeBackticks(input: string): string {
      return input.replace(/```/g, "'''"); // Replace triple backticks with triple single quotes
    }
    ```

2.  **File:** `/media/seikarii/Nvme/gemini-cli/packages/core/src/tools/editCorrector.ts`
    *   **Change 1:** Import `escapeBackticks` from `../utils/stringUtils`.
    *   **Location:** With other imports at the top of the file.

    ```typescript
    import { escapeBackticks } from '../utils/stringUtils';
    ```

    *   **Change 2:** Modify the `correctNewStringEscaping` function to use `escapeBackticks`.
    *   **Location:** Inside the `correctNewStringEscaping` function, replace the existing `newString.replace(/```/g, "'''")` line.

    ```typescript
    // Old line to replace:
    // newString = newString.replace(/```/g, "'''");

    // New line:
    newString = escapeBackticks(newString);
    ```