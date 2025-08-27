/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolInvocation,
  ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { getErrorMessage } from '../utils/errors.js';
import { ApprovalMode, Config } from '../config/config.js';
import { getResponseText } from '../utils/generateContentResponseUtilities.js';
import { fetchWithTimeout, isPrivateIp } from '../utils/fetch.js';
import { convert } from 'html-to-text';
import { ProxyAgent, setGlobalDispatcher } from 'undici';

const URL_FETCH_TIMEOUT_MS = 10000;
const MAX_CONTENT_LENGTH = 100000;
const MIN_CONTENT_LENGTH = 100; // Minimum content length to be considered useful
const MAX_RETRIES = 2; // Maximum number of processing retries

// Helper function to extract URLs from a string
function extractUrls(text: string): string[] {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  return text.match(urlRegex) || [];
}

// Interfaces for grounding metadata (similar to web-search.ts)
interface GroundingChunkWeb {
  uri?: string;
  title?: string;
}

interface GroundingChunkItem {
  web?: GroundingChunkWeb;
}

interface GroundingSupportSegment {
  startIndex: number;
  endIndex: number;
  text?: string;
}

interface GroundingSupportItem {
  segment?: GroundingSupportSegment;
  groundingChunkIndices?: number[];
}

/**
 * Parameters for the WebFetch tool
 */
export interface WebFetchToolParams {
  /**
   * The prompt containing URL(s) (up to 20) and instructions for processing their content.
   */
  prompt: string;
}

class WebFetchToolInvocation extends BaseToolInvocation<
  WebFetchToolParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: WebFetchToolParams,
  ) {
    super(params);
  }

  private async executeFallback(signal: AbortSignal): Promise<ToolResult> {
    const urls = extractUrls(this.params.prompt);
    if (urls.length === 0) {
      return {
        llmContent: 'Error: No URLs found in the prompt for fallback processing.',
        returnDisplay: 'Error: No URLs found in the prompt for fallback processing.',
        error: {
          message: 'No URLs found in the prompt for fallback processing.',
          type: ToolErrorType.WEB_FETCH_FALLBACK_FAILED,
        },
      };
    }

    const results: string[] = [];
    let hasErrors = false;

    for (const url of urls) {
      try {
        let processedUrl = url;

        // Convert GitHub blob URL to raw URL
        if (processedUrl.includes('github.com') && processedUrl.includes('/blob/')) {
          processedUrl = processedUrl
            .replace('github.com', 'raw.githubusercontent.com')
            .replace('/blob/', '/');
        }

        const response = await fetchWithTimeout(processedUrl, URL_FETCH_TIMEOUT_MS);
        if (!response.ok) {
          throw new Error(
            `Request failed with status code ${response.status} ${response.statusText}`,
          );
        }

        const html = await response.text();

        // Enhanced HTML processing for better summarization
        const processedContent = this.processHtmlContent(html, processedUrl);

        // Validate processed content quality before sending to LLM
        if (!this.isContentValidForLLM(processedContent)) {
          throw new Error(`Unable to extract meaningful content from ${processedUrl}. The page may be dynamically loaded, require JavaScript, or have restricted access.`);
        }

        // Create a more focused and intelligent prompt for the LLM
        const llmPrompt = this.createIntelligentPrompt(processedContent, processedUrl, this.params.prompt);

        const geminiClient = this.config.getGeminiClient();
        const result = await geminiClient.generateContent(
          [{ role: 'user', parts: [{ text: llmPrompt }] }],
          {},
          signal,
        );
        const resultText = getResponseText(result) || '';

        results.push(`**${processedUrl}:**\n${resultText}`);
      } catch (error) {
        const errorMessage = `Error processing ${url}: ${error instanceof Error ? error.message : 'Unknown error'}`;
        results.push(`**${url}:**\nError: ${errorMessage}`);
        hasErrors = true;
      }
    }

    const combinedResults = results.join('\n\n---\n\n');
    const returnDisplay = hasErrors
      ? `Content processed with some errors using enhanced fallback fetch.`
      : `Content processed using enhanced fallback fetch.`;

    return {
      llmContent: combinedResults,
      returnDisplay,
    };
  }

  private processHtmlContent(html: string, url: string): string {
    // Enhanced HTML processing with multiple strategies and validation
    let processedText = '';
    let attempt = 0;
    const maxAttempts = MAX_RETRIES + 1;

    while (attempt < maxAttempts && !this.isContentValid(processedText)) {
      try {
        processedText = this.attemptHtmlProcessing(html, url, attempt);
        attempt++;
      } catch (error) {
        console.warn(`HTML processing attempt ${attempt + 1} failed:`, error);
        attempt++;
      }
    }

    // If all attempts failed or content is still invalid, provide fallback
    if (!this.isContentValid(processedText)) {
      processedText = this.createFallbackContent(html, url);
    }

    return processedText;
  }

  private attemptHtmlProcessing(html: string, url: string, attempt: number): string {
    const strategies = [
      // Strategy 1: Comprehensive content extraction
      () => this.extractWithComprehensiveStrategy(html, url),
      // Strategy 2: Main content focused
      () => this.extractWithMainContentStrategy(html, url),
      // Strategy 3: Basic cleanup
      () => this.extractWithBasicStrategy(html, url),
    ];

    const strategy = strategies[Math.min(attempt, strategies.length - 1)];
    return strategy();
  }

  private extractWithComprehensiveStrategy(html: string, url: string): string {
    const processedText = convert(html, {
      wordwrap: false,
      selectors: [
        { selector: 'script', format: 'skip' },
        { selector: 'style', format: 'skip' },
        { selector: 'nav', format: 'skip' },
        { selector: 'header', format: 'skip' },
        { selector: 'footer', format: 'skip' },
        { selector: 'aside', format: 'skip' },
        { selector: 'a', options: { ignoreHref: true } },
        { selector: 'img', format: 'skip' },
        { selector: 'form', format: 'skip' },
        { selector: 'input', format: 'skip' },
        { selector: 'button', format: 'skip' },
        { selector: 'noscript', format: 'skip' },
        { selector: 'iframe', format: 'skip' },
        { selector: 'object', format: 'skip' },
        { selector: 'embed', format: 'skip' },
        // Prioritize main content selectors
        { selector: 'main', format: 'block' },
        { selector: 'article', format: 'block' },
        { selector: 'section', format: 'block' },
        { selector: '[role="main"]', format: 'block' },
        { selector: '.content', format: 'block' },
        { selector: '.main-content', format: 'block' },
        { selector: '.post-content', format: 'block' },
        { selector: '.entry-content', format: 'block' },
        { selector: '#content', format: 'block' },
        { selector: '#main', format: 'block' },
        // Keep headings for structure
        { selector: 'h1', format: 'block' },
        { selector: 'h2', format: 'block' },
        { selector: 'h3', format: 'block' },
        { selector: 'h4', format: 'block' },
        { selector: 'h5', format: 'block' },
        { selector: 'h6', format: 'block' },
        // Keep paragraphs and lists
        { selector: 'p', format: 'block' },
        { selector: 'li', format: 'block' },
        { selector: 'ul', format: 'block' },
        { selector: 'ol', format: 'block' },
        // Keep tables
        { selector: 'table', format: 'block' },
        { selector: 'tr', format: 'block' },
        { selector: 'td', format: 'block' },
        { selector: 'th', format: 'block' },
      ],
      limits: {
        maxInputLength: 500000,
      },
    });

    return this.finalizeContent(processedText, html, url);
  }

  private extractWithMainContentStrategy(html: string, url: string): string {
    const processedText = convert(html, {
      wordwrap: false,
      selectors: [
        { selector: 'script', format: 'skip' },
        { selector: 'style', format: 'skip' },
        { selector: 'nav', format: 'skip' },
        { selector: 'header', format: 'skip' },
        { selector: 'footer', format: 'skip' },
        { selector: 'aside', format: 'skip' },
        // Focus only on main content areas
        { selector: 'main', format: 'block' },
        { selector: 'article', format: 'block' },
        { selector: '[role="main"]', format: 'block' },
        { selector: '.content', format: 'block' },
        { selector: '.main-content', format: 'block' },
        { selector: '.post-content', format: 'block' },
        { selector: '.entry-content', format: 'block' },
        { selector: '#content', format: 'block' },
        { selector: '#main', format: 'block' },
        // Keep essential elements
        { selector: 'h1', format: 'block' },
        { selector: 'h2', format: 'block' },
        { selector: 'h3', format: 'block' },
        { selector: 'p', format: 'block' },
        { selector: 'li', format: 'block' },
      ],
      limits: {
        maxInputLength: 500000,
      },
    });

    return this.finalizeContent(processedText, html, url);
  }

  private extractWithBasicStrategy(html: string, url: string): string {
    const processedText = convert(html, {
      wordwrap: false,
      selectors: [
        { selector: 'script', format: 'skip' },
        { selector: 'style', format: 'skip' },
        { selector: 'nav', format: 'skip' },
        { selector: 'header', format: 'skip' },
        { selector: 'footer', format: 'skip' },
        { selector: 'aside', format: 'skip' },
      ],
      limits: {
        maxInputLength: 500000,
      },
    });

    return this.finalizeContent(processedText, html, url);
  }

  private finalizeContent(processedText: string, html: string, url: string): string {
    // Clean up excessive whitespace and normalize
    let cleanedText = processedText
      .replace(/\n{3,}/g, '\n\n') // Replace 3+ newlines with 2
      .replace(/[ \t]+/g, ' ') // Replace multiple spaces/tabs with single space
      .trim();

    // Extract title if available
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    const title = titleMatch ? titleMatch[1].trim() : '';

    // Limit content length to prevent token overuse
    if (cleanedText.length > MAX_CONTENT_LENGTH) {
      cleanedText = cleanedText.substring(0, MAX_CONTENT_LENGTH - 100) +
        '\n\n[Content truncated due to length...]';
    }

    // Add metadata
    const metadata = `Source: ${url}${title ? `\nTitle: ${title}` : ''}\n\n`;

    return metadata + cleanedText;
  }

  private isContentValid(content: string): boolean {
    if (!content || content.trim().length < MIN_CONTENT_LENGTH) {
      return false;
    }

    // Check if content has meaningful text (not just metadata)
    const contentWithoutMetadata = content.split('\n\n').slice(1).join('\n\n');
    if (contentWithoutMetadata.trim().length < MIN_CONTENT_LENGTH) {
      return false;
    }

    // Check for excessive boilerplate or meaningless content
    const meaninglessPatterns = [
      /^(\s*error\s*:?\s*)+$/i,
      /^(\s*not found\s*:?\s*)+$/i,
      /^(\s*access denied\s*:?\s*)+$/i,
      /^(\s*loading\s*\.\.\.\s*)+$/i,
      /^(\s*please wait\s*:?\s*)+$/i,
    ];

    const meaningfulContent = contentWithoutMetadata.trim();
    for (const pattern of meaninglessPatterns) {
      if (pattern.test(meaningfulContent)) {
        return false;
      }
    }

    return true;
  }

  private createFallbackContent(html: string, url: string): string {
    // Extract basic information as fallback
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    const title = titleMatch ? titleMatch[1].trim() : 'No title available';

    const descriptionMatch = html.match(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']+)["'][^>]*>/i);
    const description = descriptionMatch ? descriptionMatch[1].trim() : '';

    let fallbackContent = `Source: ${url}\nTitle: ${title}\n\n`;

    if (description) {
      fallbackContent += `Description: ${description}\n\n`;
    }

    fallbackContent += `[Unable to extract full content from this webpage. The page may be dynamically loaded, require JavaScript, or have restricted access.]`;

    return fallbackContent;
  }

  private isContentValidForLLM(content: string): boolean {
    // First check basic validity
    if (!this.isContentValid(content)) {
      return false;
    }

    // Additional checks for LLM suitability
    const contentWithoutMetadata = content.split('\n\n').slice(1).join('\n\n').trim();

    // Check minimum useful content length for LLM processing
    if (contentWithoutMetadata.length < 200) {
      return false;
    }

    // Check for content that would confuse the LLM
    const confusingPatterns = [
      /^\[unable to extract/i,
      /^error:/i,
      /^not found/i,
      /^access denied/i,
      /^please try again/i,
      /^loading/i,
      /^waiting/i,
    ];

    for (const pattern of confusingPatterns) {
      if (pattern.test(contentWithoutMetadata)) {
        return false;
      }
    }

    // Check if content has enough substantive information
    const words = contentWithoutMetadata.split(/\s+/).filter(word => word.length > 3);
    if (words.length < 10) {
      return false;
    }

    return true;
  }

  private createIntelligentPrompt(content: string, url: string, userPrompt: string): string {
    // Extract key information from content
    const lines = content.split('\n');
    const title = lines.find(line => line.startsWith('Title:'))?.replace('Title:', '').trim() || '';
    const source = lines.find(line => line.startsWith('Source:'))?.replace('Source:', '').trim() || '';
    const mainContent = lines.slice(2).join('\n').trim(); // Skip metadata lines

    // Create a more intelligent prompt based on content analysis
    const contentSummary = this.analyzeContentForPrompt(mainContent);

    return `The user requested: "${userPrompt}"

I need to analyze the content from: ${source}${title ? ` (Title: ${title})` : ''}

Content Analysis:
${contentSummary}

Here is the extracted and cleaned content from the webpage:

---
${content}
---

Please provide a comprehensive and focused response based on the user's request. Focus on the most relevant information and provide actionable insights. If the content doesn't fully address the request, acknowledge this and provide the best analysis possible with the available information.`;
  }

  private analyzeContentForPrompt(content: string): string {
    const wordCount = content.split(/\s+/).length;
    const charCount = content.length;

    let analysis = `- Content length: ${charCount} characters, approximately ${wordCount} words
`;

    // Detect content type
    if (content.includes('function') || content.includes('const') || content.includes('var')) {
      analysis += `- Content type: Appears to contain code or technical documentation
`;
    } else if (content.match(/\b\d{4}-\d{2}-\d{2}\b/)) {
      analysis += `- Content type: Appears to contain dated information or articles
`;
    } else if (content.match(/\b(http|https):\/\//g)) {
      analysis += `- Content type: Appears to contain links or references
`;
    }

    // Check for structured content
    const headers = content.match(/^#{1,6}\s+.+$/gm);
    if (headers && headers.length > 0) {
      analysis += `- Structure: Contains ${headers.length} headings/sections
`;
    }

    // Check for lists
    const listItems = content.match(/^[-*+]\s+.+$/gm);
    if (listItems && listItems.length > 0) {
      analysis += `- Structure: Contains ${listItems.length} list items
`;
    }

    return analysis;
  }

  getDescription(): string {
    const displayPrompt =
      this.params.prompt.length > 100
        ? this.params.prompt.substring(0, 97) + '...'
        : this.params.prompt;
    return `Processing URLs and instructions from prompt: "${displayPrompt}"`;
  }

  override async shouldConfirmExecute(): Promise<
    ToolCallConfirmationDetails | false
  > {
    if (this.config.getApprovalMode() === ApprovalMode.AUTO_EDIT) {
      return false;
    }

    // Perform GitHub URL conversion here to differentiate between user-provided
    // URL and the actual URL to be fetched.
    const urls = extractUrls(this.params.prompt).map((url) => {
      if (url.includes('github.com') && url.includes('/blob/')) {
        return url
          .replace('github.com', 'raw.githubusercontent.com')
          .replace('/blob/', '/');
      }
      return url;
    });

    const confirmationDetails: ToolCallConfirmationDetails = {
      type: 'info',
      title: `Confirm Web Fetch`,
      prompt: this.params.prompt,
      urls,
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlways) {
          this.config.setApprovalMode(ApprovalMode.AUTO_EDIT);
        }
      },
    };
    return confirmationDetails;
  }

  async execute(signal: AbortSignal): Promise<ToolResult> {
    const userPrompt = this.params.prompt;
    const urls = extractUrls(userPrompt);
    const url = urls[0];
    const isPrivate = isPrivateIp(url);

    if (isPrivate) {
      return this.executeFallback(signal);
    }

    const geminiClient = this.config.getGeminiClient();

    try {
      const response = await geminiClient.generateContent(
        [{ role: 'user', parts: [{ text: userPrompt }] }],
        { tools: [{ urlContext: {} }] },
        signal, // Pass signal
      );

      console.debug(
        `[WebFetchTool] Full response for prompt "${userPrompt.substring(
          0,
          50,
        )}...":`,
        JSON.stringify(response, null, 2),
      );

      let responseText = getResponseText(response) || '';
      const urlContextMeta = response.candidates?.[0]?.urlContextMetadata;
      const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
      const sources = groundingMetadata?.groundingChunks as
        | GroundingChunkItem[]
        | undefined;
      const groundingSupports = groundingMetadata?.groundingSupports as
        | GroundingSupportItem[]
        | undefined;

      // Error Handling
      let processingError = false;

      if (
        urlContextMeta?.urlMetadata &&
        urlContextMeta.urlMetadata.length > 0
      ) {
        const allStatuses = urlContextMeta.urlMetadata.map(
          (m) => m.urlRetrievalStatus,
        );
        if (allStatuses.every((s) => s !== 'URL_RETRIEVAL_STATUS_SUCCESS')) {
          processingError = true;
        }
      } else if (!responseText.trim() && !sources?.length) {
        // No URL metadata and no content/sources
        processingError = true;
      }

      if (
        !processingError &&
        !responseText.trim() &&
        (!sources || sources.length === 0)
      ) {
        // Successfully retrieved some URL (or no specific error from urlContextMeta), but no usable text or grounding data.
        processingError = true;
      }

      if (processingError) {
        return this.executeFallback(signal);
      }

      const sourceListFormatted: string[] = [];
      if (sources && sources.length > 0) {
        sources.forEach((source: GroundingChunkItem, index: number) => {
          const title = source.web?.title || 'Untitled';
          const uri = source.web?.uri || 'Unknown URI'; // Fallback if URI is missing
          sourceListFormatted.push(`[${index + 1}] ${title} (${uri})`);
        });

        if (groundingSupports && groundingSupports.length > 0) {
          const insertions: Array<{ index: number; marker: string }> = [];
          groundingSupports.forEach((support: GroundingSupportItem) => {
            if (support.segment && support.groundingChunkIndices) {
              const citationMarker = support.groundingChunkIndices
                .map((chunkIndex: number) => `[${chunkIndex + 1}]`)
                .join('');
              insertions.push({
                index: support.segment.endIndex,
                marker: citationMarker,
              });
            }
          });

          insertions.sort((a, b) => b.index - a.index);
          const responseChars = responseText.split('');
          insertions.forEach((insertion) => {
            responseChars.splice(insertion.index, 0, insertion.marker);
          });
          responseText = responseChars.join('');
        }

        if (sourceListFormatted.length > 0) {
          responseText += `

Sources:
${sourceListFormatted.join('\n')}`;
        }
      }

      const llmContent = responseText;

      console.debug(
        `[WebFetchTool] Formatted tool response for prompt "${userPrompt}:\n\n":`,
        llmContent,
      );

      return {
        llmContent,
        returnDisplay: `Content processed from prompt.`,
      };
    } catch (error: unknown) {
      const errorMessage = `Error processing web content for prompt "${userPrompt.substring(
        0,
        50,
      )}...": ${getErrorMessage(error)}`;
      console.error(errorMessage, error);
      return {
        llmContent: `Error: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.WEB_FETCH_PROCESSING_ERROR,
        },
      };
    }
  }
}

/**
 * Implementation of the WebFetch tool logic
 */
export class WebFetchTool extends BaseDeclarativeTool<
  WebFetchToolParams,
  ToolResult
> {
  static readonly Name: string = 'web_fetch';

  constructor(private readonly config: Config) {
    super(
      WebFetchTool.Name,
      'WebFetch',
      "Processes content from URL(s), including local and private network addresses (e.g., localhost), embedded in a prompt. Include up to 20 URLs and instructions (e.g., summarize, extract specific data) directly in the 'prompt' parameter.",
      Kind.Fetch,
      {
        properties: {
          prompt: {
            description:
              'A comprehensive prompt that includes the URL(s) (up to 20) to fetch and specific instructions on how to process their content (e.g., "Summarize https://example.com/article and extract key points from https://another.com/data"). Must contain as least one URL starting with http:// or https://.',
            type: 'string',
          },
        },
        required: ['prompt'],
        type: 'object',
      },
    );
    const proxy = config.getProxy();
    if (proxy) {
      setGlobalDispatcher(new ProxyAgent(proxy as string));
    }
  }

  protected override validateToolParamValues(
    params: WebFetchToolParams,
  ): string | null {
    if (!params.prompt || params.prompt.trim() === '') {
      return "The 'prompt' parameter cannot be empty and must contain URL(s) and instructions.";
    }
    if (
      !params.prompt.includes('http://') &&
      !params.prompt.includes('https://')
    ) {
      return "The 'prompt' must contain at least one valid URL (starting with http:// or https://).";
    }
    return null;
  }

  protected createInvocation(
    params: WebFetchToolParams,
  ): ToolInvocation<WebFetchToolParams, ToolResult> {
    return new WebFetchToolInvocation(this.config, params);
  }
}
