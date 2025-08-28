/**
 * @fileoverview Lazy loading and pagination system for efficient data handling
 * with intelligent caching, prefetching, and memory-optimized loading patterns
 */

import { EventEmitter } from 'events';

/**
 * Pagination configuration options
 */
export interface IPaginationConfig {
  /** Default page size */
  pageSize: number;
  /** Maximum page size allowed */
  maxPageSize: number;
  /** Enable automatic prefetching of next pages */
  enablePrefetch: boolean;
  /** Number of pages to prefetch ahead */
  prefetchCount: number;
  /** Enable caching of loaded pages */
  enableCache: boolean;
  /** Maximum number of pages to keep in cache */
  maxCacheSize: number;
  /** Cache TTL in milliseconds */
  cacheTTL: number;
  /** Enable lazy loading for large datasets */
  enableLazyLoading: boolean;
  /** Threshold for triggering lazy loading */
  lazyLoadThreshold: number;
  /** Enable virtual scrolling optimization */
  enableVirtualScrolling: boolean;
  /** Virtual viewport size */
  virtualViewportSize: number;
}

/**
 * Page data structure
 */
export interface IPageData<T = unknown> {
  /** Page number (0-based) */
  pageNumber: number;
  /** Items in this page */
  items: T[];
  /** Total number of items across all pages */
  totalItems: number;
  /** Total number of pages */
  totalPages: number;
  /** Whether this is the first page */
  isFirst: boolean;
  /** Whether this is the last page */
  isLast: boolean;
  /** Timestamp when page was loaded */
  loadedAt: number;
  /** Size of this page */
  pageSize: number;
  /** Metadata for the page */
  metadata?: Record<string, unknown>;
}

/**
 * Lazy loading context for tracking state
 */
export interface ILazyLoadingContext<T = unknown> {
  /** Current page number */
  currentPage: number;
  /** Loaded pages cache */
  loadedPages: Map<number, IPageData<T>>;
  /** Pages currently being loaded */
  loadingPages: Set<number>;
  /** Total items count */
  totalItems: number;
  /** Loading state */
  isLoading: boolean;
  /** Error state */
  error: Error | null;
  /** Last access timestamp */
  lastAccess: number;
}

/**
 * Data loader function signature
 */
export type DataLoader<T> = (
  page: number,
  pageSize: number,
  filters?: Record<string, unknown>
) => Promise<{
  items: T[];
  totalItems: number;
  hasMore: boolean;
}>;

/**
 * Virtual scrolling item renderer
 */
export type VirtualItemRenderer<T> = (
  item: T,
  index: number,
  isVisible: boolean
) => {
  height: number;
  content: unknown;
};

/**
 * Lazy loading manager interface
 */
export interface ILazyLoadingManager<T = unknown> extends EventEmitter {
  /**
   * Initialize the lazy loading manager
   * @param dataLoader Function to load data
   * @param config Configuration options
   */
  initialize(dataLoader: DataLoader<T>, config?: Partial<IPaginationConfig>): void;

  /**
   * Load a specific page
   * @param pageNumber Page number to load (0-based)
   * @param force Force reload even if cached
   * @returns Promise resolving to page data
   */
  loadPage(pageNumber: number, force?: boolean): Promise<IPageData<T>>;

  /**
   * Load next page
   * @returns Promise resolving to page data
   */
  loadNextPage(): Promise<IPageData<T> | null>;

  /**
   * Load previous page
   * @returns Promise resolving to page data
   */
  loadPreviousPage(): Promise<IPageData<T> | null>;

  /**
   * Get items for a specific range (virtual scrolling)
   * @param startIndex Start index
   * @param endIndex End index
   * @returns Promise resolving to items in range
   */
  getItemsInRange(startIndex: number, endIndex: number): Promise<T[]>;

  /**
   * Prefetch upcoming pages
   * @param fromPage Starting page for prefetching
   * @param count Number of pages to prefetch
   */
  prefetchPages(fromPage: number, count?: number): Promise<void>;

  /**
   * Clear cache and reset state
   */
  reset(): void;

  /**
   * Get current loading context
   */
  getContext(): ILazyLoadingContext<T>;

  /**
   * Set filters for data loading
   * @param filters Filter parameters
   */
  setFilters(filters: Record<string, unknown>): void;

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    cachedPages: number;
    totalCacheSize: number;
    hitRate: number;
    missRate: number;
  };
}

/**
 * Virtual scrolling manager for efficient rendering of large lists
 */
export interface IVirtualScrollingManager<T = unknown> extends EventEmitter {
  /**
   * Initialize virtual scrolling
   * @param totalItems Total number of items
   * @param itemRenderer Item renderer function
   * @param config Configuration options
   */
  initialize(
    totalItems: number,
    itemRenderer: VirtualItemRenderer<T>,
    config?: Partial<IPaginationConfig>
  ): void;

  /**
   * Update viewport and calculate visible items
   * @param scrollTop Current scroll position
   * @param viewportHeight Height of the viewport
   * @returns Visible item range
   */
  updateViewport(scrollTop: number, viewportHeight: number): {
    startIndex: number;
    endIndex: number;
    visibleItems: Array<{ index: number; top: number; height: number }>;
  };

  /**
   * Get estimated total height
   * @returns Total estimated height
   */
  getTotalHeight(): number;

  /**
   * Update item height at specific index
   * @param index Item index
   * @param height Measured height
   */
  updateItemHeight(index: number, height: number): void;
}

/**
 * Advanced lazy loading manager implementation
 */
export class LazyLoadingManager<T = unknown> extends EventEmitter implements ILazyLoadingManager<T> {
  private config: IPaginationConfig;
  private context: ILazyLoadingContext<T>;
  private dataLoader: DataLoader<T> | null = null;
  private filters: Record<string, unknown> = {};
  private cacheStats = {
    hits: 0,
    misses: 0,
    totalRequests: 0
  };

  constructor() {
    super();
    this.config = {
      pageSize: 50,
      maxPageSize: 1000,
      enablePrefetch: true,
      prefetchCount: 2,
      enableCache: true,
      maxCacheSize: 10,
      cacheTTL: 300000, // 5 minutes
      enableLazyLoading: true,
      lazyLoadThreshold: 0.8,
      enableVirtualScrolling: false,
      virtualViewportSize: 10
    };

    this.context = this.createInitialContext();
  }

  /**
   * Initialize the lazy loading manager
   */
  initialize(dataLoader: DataLoader<T>, config?: Partial<IPaginationConfig>): void {
    this.dataLoader = dataLoader;
    this.config = { ...this.config, ...config };
    this.context = this.createInitialContext();
    this.emit('initialized');
  }

  /**
   * Load a specific page
   */
  async loadPage(pageNumber: number, force = false): Promise<IPageData<T>> {
    if (!this.dataLoader) {
      throw new Error('Data loader not initialized');
    }

    // Check cache first
    if (!force && this.config.enableCache && this.context.loadedPages.has(pageNumber)) {
      const cachedPage = this.context.loadedPages.get(pageNumber)!;
      
      // Check if cache is still valid
      if (Date.now() - cachedPage.loadedAt < this.config.cacheTTL) {
        this.cacheStats.hits++;
        this.cacheStats.totalRequests++;
        this.emit('page-loaded', cachedPage);
        return cachedPage;
      }
    }

    // Prevent duplicate loading
    if (this.context.loadingPages.has(pageNumber)) {
      return new Promise((resolve, reject) => {
        const onPageLoaded = (page: IPageData<T>) => {
          if (page.pageNumber === pageNumber) {
            this.removeListener('page-loaded', onPageLoaded);
            this.removeListener('page-error', onPageError);
            resolve(page);
          }
        };

        const onPageError = (error: Error, page: number) => {
          if (page === pageNumber) {
            this.removeListener('page-loaded', onPageLoaded);
            this.removeListener('page-error', onPageError);
            reject(error);
          }
        };

        this.on('page-loaded', onPageLoaded);
        this.on('page-error', onPageError);
      });
    }

    this.context.loadingPages.add(pageNumber);
    this.context.isLoading = true;
    this.cacheStats.misses++;
    this.cacheStats.totalRequests++;

    try {
      this.emit('page-loading', pageNumber);

      const result = await this.dataLoader(pageNumber, this.config.pageSize, this.filters);
      
      const pageData: IPageData<T> = {
        pageNumber,
        items: result.items,
        totalItems: result.totalItems,
        totalPages: Math.ceil(result.totalItems / this.config.pageSize),
        isFirst: pageNumber === 0,
        isLast: !result.hasMore,
        loadedAt: Date.now(),
        pageSize: result.items.length
      };

      // Update context
      this.context.totalItems = result.totalItems;
      this.context.currentPage = pageNumber;
      this.context.lastAccess = Date.now();
      this.context.error = null;

      // Cache the page
      if (this.config.enableCache) {
        this.cachePageData(pageData);
      }

      // Prefetch if enabled
      if (this.config.enablePrefetch && !pageData.isLast) {
        this.prefetchPages(pageNumber + 1, this.config.prefetchCount);
      }

      this.emit('page-loaded', pageData);
      return pageData;

    } catch (error) {
      this.context.error = error as Error;
      this.emit('page-error', error, pageNumber);
      throw error;
    } finally {
      this.context.loadingPages.delete(pageNumber);
      this.context.isLoading = this.context.loadingPages.size > 0;
    }
  }

  /**
   * Load next page
   */
  async loadNextPage(): Promise<IPageData<T> | null> {
    const nextPage = this.context.currentPage + 1;
    const totalPages = Math.ceil(this.context.totalItems / this.config.pageSize);
    
    if (nextPage >= totalPages) {
      return null;
    }

    return this.loadPage(nextPage);
  }

  /**
   * Load previous page
   */
  async loadPreviousPage(): Promise<IPageData<T> | null> {
    const prevPage = this.context.currentPage - 1;
    
    if (prevPage < 0) {
      return null;
    }

    return this.loadPage(prevPage);
  }

  /**
   * Get items for a specific range (virtual scrolling support)
   */
  async getItemsInRange(startIndex: number, endIndex: number): Promise<T[]> {
    const items: T[] = [];
    const startPage = Math.floor(startIndex / this.config.pageSize);
    const endPage = Math.floor(endIndex / this.config.pageSize);

    // Load all pages that contain items in the range
    for (let page = startPage; page <= endPage; page++) {
      try {
        const pageData = await this.loadPage(page);
        
        // Calculate which items from this page are in our range
        const pageStartIndex = page * this.config.pageSize;
        
        const rangeStart = Math.max(startIndex - pageStartIndex, 0);
        const rangeEnd = Math.min(endIndex - pageStartIndex, pageData.items.length - 1);
        
        if (rangeStart <= rangeEnd) {
          items.push(...pageData.items.slice(rangeStart, rangeEnd + 1));
        }
      } catch (error) {
        this.emit('range-load-error', error, { startIndex, endIndex, page });
      }
    }

    return items;
  }

  /**
   * Prefetch upcoming pages
   */
  async prefetchPages(fromPage: number, count = this.config.prefetchCount): Promise<void> {
    if (!this.config.enablePrefetch) {
      return;
    }

    const totalPages = Math.ceil(this.context.totalItems / this.config.pageSize);
    const endPage = Math.min(fromPage + count - 1, totalPages - 1);

    const prefetchPromises: Array<Promise<IPageData<T>>> = [];

    for (let page = fromPage; page <= endPage; page++) {
      if (!this.context.loadedPages.has(page) && !this.context.loadingPages.has(page)) {
        prefetchPromises.push(this.loadPage(page));
      }
    }

    try {
      await Promise.all(prefetchPromises);
      this.emit('prefetch-completed', { fromPage, count: prefetchPromises.length });
    } catch (error) {
      this.emit('prefetch-error', error);
    }
  }

  /**
   * Clear cache and reset state
   */
  reset(): void {
    this.context = this.createInitialContext();
    this.filters = {};
    this.cacheStats = { hits: 0, misses: 0, totalRequests: 0 };
    this.emit('reset');
  }

  /**
   * Get current loading context
   */
  getContext(): ILazyLoadingContext<T> {
    return { ...this.context };
  }

  /**
   * Set filters for data loading
   */
  setFilters(filters: Record<string, unknown>): void {
    this.filters = { ...filters };
    this.reset(); // Reset cache when filters change
    this.emit('filters-changed', filters);
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    cachedPages: number;
    totalCacheSize: number;
    hitRate: number;
    missRate: number;
  } {
    return {
      cachedPages: this.context.loadedPages.size,
      totalCacheSize: this.context.loadedPages.size,
      hitRate: this.cacheStats.totalRequests > 0 ? this.cacheStats.hits / this.cacheStats.totalRequests : 0,
      missRate: this.cacheStats.totalRequests > 0 ? this.cacheStats.misses / this.cacheStats.totalRequests : 0
    };
  }

  /**
   * Cache page data with LRU eviction
   */
  private cachePageData(pageData: IPageData<T>): void {
    // Remove oldest entries if cache is full
    if (this.context.loadedPages.size >= this.config.maxCacheSize) {
      const oldestKey = Array.from(this.context.loadedPages.keys())[0];
      this.context.loadedPages.delete(oldestKey);
    }

    this.context.loadedPages.set(pageData.pageNumber, pageData);
  }

  /**
   * Create initial loading context
   */
  private createInitialContext(): ILazyLoadingContext<T> {
    return {
      currentPage: 0,
      loadedPages: new Map(),
      loadingPages: new Set(),
      totalItems: 0,
      isLoading: false,
      error: null,
      lastAccess: Date.now()
    };
  }
}

/**
 * Virtual scrolling manager for efficient rendering
 */
export class VirtualScrollingManager<T = unknown> extends EventEmitter implements IVirtualScrollingManager<T> {
  private totalItems = 0;
  private itemRenderer: VirtualItemRenderer<T> | null = null;
  private config: IPaginationConfig;
  private itemHeights = new Map<number, number>();
  private averageItemHeight = 50; // Default estimate
  private measuredItemsCount = 0;

  constructor() {
    super();
    this.config = {
      pageSize: 50,
      maxPageSize: 1000,
      enablePrefetch: true,
      prefetchCount: 2,
      enableCache: true,
      maxCacheSize: 10,
      cacheTTL: 300000,
      enableLazyLoading: true,
      lazyLoadThreshold: 0.8,
      enableVirtualScrolling: true,
      virtualViewportSize: 10
    };
  }

  /**
   * Initialize virtual scrolling
   */
  initialize(
    totalItems: number,
    itemRenderer: VirtualItemRenderer<T>,
    config?: Partial<IPaginationConfig>
  ): void {
    this.totalItems = totalItems;
    this.itemRenderer = itemRenderer;
    this.config = { ...this.config, ...config };
    this.emit('initialized');
  }

  /**
   * Update viewport and calculate visible items
   */
  updateViewport(scrollTop: number, viewportHeight: number): {
    startIndex: number;
    endIndex: number;
    visibleItems: Array<{ index: number; top: number; height: number }>;
  } {
    if (!this.itemRenderer) {
      throw new Error('Item renderer not initialized');
    }

    // Calculate visible range
    const startIndex = this.findItemIndexByOffset(scrollTop);
    const endIndex = this.findItemIndexByOffset(scrollTop + viewportHeight);

    // Add buffer for smooth scrolling
    const bufferSize = Math.floor(this.config.virtualViewportSize / 2);
    const bufferedStart = Math.max(0, startIndex - bufferSize);
    const bufferedEnd = Math.min(this.totalItems - 1, endIndex + bufferSize);

    // Calculate visible items with positions
    const visibleItems: Array<{ index: number; top: number; height: number }> = [];
    let currentTop = this.getOffsetForIndex(bufferedStart);

    for (let i = bufferedStart; i <= bufferedEnd; i++) {
      const height = this.getItemHeight(i);
      
      visibleItems.push({
        index: i,
        top: currentTop,
        height
      });

      currentTop += height;
    }

    this.emit('viewport-updated', {
      startIndex: bufferedStart,
      endIndex: bufferedEnd,
      visibleCount: visibleItems.length
    });

    return {
      startIndex: bufferedStart,
      endIndex: bufferedEnd,
      visibleItems
    };
  }

  /**
   * Get estimated total height
   */
  getTotalHeight(): number {
    if (this.measuredItemsCount === 0) {
      return this.totalItems * this.averageItemHeight;
    }

    // Use measured heights for known items and average for unknown
    let totalHeight = 0;
    const measuredCount = this.itemHeights.size;
    const unmeasuredCount = this.totalItems - measuredCount;

    // Sum measured heights
    for (const height of this.itemHeights.values()) {
      totalHeight += height;
    }

    // Estimate unmeasured items
    totalHeight += unmeasuredCount * this.averageItemHeight;

    return totalHeight;
  }

  /**
   * Update item height at specific index
   */
  updateItemHeight(index: number, height: number): void {
    const oldHeight = this.itemHeights.get(index);
    this.itemHeights.set(index, height);

    // Update average height calculation
    if (oldHeight === undefined) {
      this.measuredItemsCount++;
    }

    // Recalculate average
    const totalMeasuredHeight = Array.from(this.itemHeights.values()).reduce((a, b) => a + b, 0);
    this.averageItemHeight = totalMeasuredHeight / this.measuredItemsCount;

    this.emit('item-height-updated', { index, height, averageHeight: this.averageItemHeight });
  }

  /**
   * Find item index by vertical offset
   */
  private findItemIndexByOffset(offset: number): number {
    let currentOffset = 0;
    
    for (let i = 0; i < this.totalItems; i++) {
      const itemHeight = this.getItemHeight(i);
      
      if (currentOffset + itemHeight > offset) {
        return i;
      }
      
      currentOffset += itemHeight;
    }

    return Math.max(0, this.totalItems - 1);
  }

  /**
   * Get vertical offset for item at index
   */
  private getOffsetForIndex(index: number): number {
    let offset = 0;
    
    for (let i = 0; i < index && i < this.totalItems; i++) {
      offset += this.getItemHeight(i);
    }

    return offset;
  }

  /**
   * Get height for item at index
   */
  private getItemHeight(index: number): number {
    return this.itemHeights.get(index) || this.averageItemHeight;
  }
}

/**
 * Create lazy loading manager
 */
export function createLazyLoadingManager<T = unknown>(): LazyLoadingManager<T> {
  return new LazyLoadingManager<T>();
}

/**
 * Create virtual scrolling manager
 */
export function createVirtualScrollingManager<T = unknown>(): VirtualScrollingManager<T> {
  return new VirtualScrollingManager<T>();
}

/**
 * Default pagination configurations
 */
export const PAGINATION_PRESETS = {
  /** Small datasets with frequent updates */
  SMALL_DATASET: {
    pageSize: 25,
    maxPageSize: 100,
    enablePrefetch: true,
    prefetchCount: 1,
    enableCache: true,
    maxCacheSize: 5,
    cacheTTL: 60000, // 1 minute
  },

  /** Large datasets with infrequent updates */
  LARGE_DATASET: {
    pageSize: 100,
    maxPageSize: 500,
    enablePrefetch: true,
    prefetchCount: 3,
    enableCache: true,
    maxCacheSize: 20,
    cacheTTL: 600000, // 10 minutes
  },

  /** Real-time data with frequent changes */
  REAL_TIME: {
    pageSize: 50,
    maxPageSize: 200,
    enablePrefetch: false,
    prefetchCount: 0,
    enableCache: false,
    maxCacheSize: 3,
    cacheTTL: 10000, // 10 seconds
  },

  /** Virtual scrolling optimized */
  VIRTUAL_SCROLLING: {
    pageSize: 200,
    maxPageSize: 1000,
    enablePrefetch: true,
    prefetchCount: 2,
    enableCache: true,
    maxCacheSize: 15,
    cacheTTL: 300000, // 5 minutes
    enableVirtualScrolling: true,
    virtualViewportSize: 20,
  }
} satisfies Record<string, Partial<IPaginationConfig>>;
