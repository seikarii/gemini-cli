/**
 * @fileoverview Encryption system for sensitive data protection
 * Provides AES-256-GCM encryption with key management and rotation
 */

import { createCipheriv, createDecipheriv, randomBytes, pbkdf2Sync, createHmac, Cipheriv, Decipheriv } from 'crypto';

// Extend crypto types for GCM mode
interface CipherGCM extends Cipheriv {
  getAuthTag(): Buffer;
}

interface DecipherGCM extends Decipheriv {
  setAuthTag(buffer: Buffer): void;
}

/**
 * Encryption configuration
 */
export interface IEncryptionConfig {
  /** Encryption algorithm */
  algorithm: string;
  /** Key derivation iterations */
  iterations: number;
  /** Salt length in bytes */
  saltLength: number;
  /** IV length in bytes */
  ivLength: number;
  /** Tag length in bytes for authenticated encryption */
  tagLength: number;
  /** Key rotation interval in milliseconds */
  keyRotationInterval: number;
  /** Enable automatic key rotation */
  autoRotation: boolean;
}

/**
 * Encrypted data structure
 */
export interface IEncryptedData {
  /** Base64 encoded encrypted data */
  data: string;
  /** Base64 encoded IV */
  iv: string;
  /** Base64 encoded salt */
  salt: string;
  /** Base64 encoded authentication tag */
  tag: string;
  /** Key version used for encryption */
  keyVersion: number;
  /** Timestamp of encryption */
  timestamp: string;
  /** Algorithm used */
  algorithm: string;
}

/**
 * Key information
 */
export interface IKeyInfo {
  /** Key version */
  version: number;
  /** Key creation timestamp */
  created: string;
  /** Key status */
  status: 'active' | 'rotating' | 'retired';
  /** Usage count */
  usageCount: number;
  /** Last used timestamp */
  lastUsed?: string;
}

/**
 * Encryption key manager
 */
class KeyManager {
  private keys = new Map<number, Buffer>();
  private currentVersion = 1;
  private keyInfo = new Map<number, IKeyInfo>();
  private masterKey: Buffer;

  constructor(masterPassword: string, salt?: Buffer) {
    const keySalt = salt || randomBytes(32);
    this.masterKey = pbkdf2Sync(masterPassword, keySalt, 100000, 32, 'sha256');
    
    // Generate initial key
    this.generateNewKey();
  }

  /**
   * Get current active key
   */
  getCurrentKey(): { key: Buffer; version: number } {
    const key = this.keys.get(this.currentVersion);
    if (!key) {
      throw new Error('No active encryption key available');
    }
    
    // Update usage stats
    const info = this.keyInfo.get(this.currentVersion);
    if (info) {
      info.usageCount++;
      info.lastUsed = new Date().toISOString();
    }
    
    return { key, version: this.currentVersion };
  }

  /**
   * Get key by version
   */
  getKey(version: number): Buffer | undefined {
    const key = this.keys.get(version);
    
    if (key) {
      // Update usage stats
      const info = this.keyInfo.get(version);
      if (info) {
        info.usageCount++;
        info.lastUsed = new Date().toISOString();
      }
    }
    
    return key;
  }

  /**
   * Generate new encryption key
   */
  generateNewKey(): number {
    const newVersion = this.currentVersion + 1;
    const keyMaterial = randomBytes(32);
    
    // Derive key from master key and key material
    const key = createHmac('sha256', this.masterKey)
      .update(keyMaterial)
      .update(Buffer.from(newVersion.toString()))
      .digest();
    
    this.keys.set(newVersion, key);
    this.keyInfo.set(newVersion, {
      version: newVersion,
      created: new Date().toISOString(),
      status: 'active',
      usageCount: 0
    });
    
    // Mark previous key as rotating
    if (this.currentVersion > 0) {
      const oldInfo = this.keyInfo.get(this.currentVersion);
      if (oldInfo) {
        oldInfo.status = 'rotating';
      }
    }
    
    this.currentVersion = newVersion;
    return newVersion;
  }

  /**
   * Retire old key
   */
  retireKey(version: number): void {
    const info = this.keyInfo.get(version);
    if (info && version !== this.currentVersion) {
      info.status = 'retired';
    }
  }

  /**
   * Get all key information
   */
  getKeyInfo(): IKeyInfo[] {
    return Array.from(this.keyInfo.values());
  }

  /**
   * Clean up retired keys
   */
  cleanupRetiredKeys(olderThanDays = 30): void {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);
    
    for (const [version, info] of this.keyInfo.entries()) {
      if (info.status === 'retired' && new Date(info.created) < cutoffDate) {
        this.keys.delete(version);
        this.keyInfo.delete(version);
      }
    }
  }
}

/**
 * Main encryption service
 */
export class EncryptionService {
  private config: IEncryptionConfig;
  private keyManager: KeyManager;
  private rotationTimer?: NodeJS.Timeout;

  constructor(masterPassword: string, config?: Partial<IEncryptionConfig>) {
    this.config = {
      algorithm: 'aes-256-gcm',
      iterations: 100000,
      saltLength: 32,
      ivLength: 16,
      tagLength: 16,
      keyRotationInterval: 24 * 60 * 60 * 1000, // 24 hours
      autoRotation: true,
      ...config
    };

    this.keyManager = new KeyManager(masterPassword);
    
    if (this.config.autoRotation) {
      this.startKeyRotation();
    }
  }

  /**
   * Encrypt data
   */
  encrypt(data: string | Buffer): IEncryptedData {
    const plaintext = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    const { key, version } = this.keyManager.getCurrentKey();
    
    // Generate random IV and salt
    const iv = randomBytes(this.config.ivLength);
    const salt = randomBytes(this.config.saltLength);
    
    // Create cipher
    const cipher = createCipheriv(this.config.algorithm, key, iv);
    
    // Encrypt data
    const encrypted = Buffer.concat([
      cipher.update(plaintext),
      cipher.final()
    ]);
    
    // Get authentication tag
    const tag = (cipher as unknown as CipherGCM).getAuthTag();
    
    return {
      data: encrypted.toString('base64'),
      iv: iv.toString('base64'),
      salt: salt.toString('base64'),
      tag: tag.toString('base64'),
      keyVersion: version,
      timestamp: new Date().toISOString(),
      algorithm: this.config.algorithm
    };
  }

  /**
   * Decrypt data
   */
  decrypt(encryptedData: IEncryptedData): Buffer {
    const key = this.keyManager.getKey(encryptedData.keyVersion);
    if (!key) {
      throw new Error(`Encryption key version ${encryptedData.keyVersion} not found`);
    }
    
    // Parse encrypted data
    const data = Buffer.from(encryptedData.data, 'base64');
    const iv = Buffer.from(encryptedData.iv, 'base64');
    const tag = Buffer.from(encryptedData.tag, 'base64');
    
    // Create decipher
    const decipher = createDecipheriv(encryptedData.algorithm, key, iv);
    (decipher as unknown as DecipherGCM).setAuthTag(tag);
    
    // Decrypt data
    try {
      const decrypted = Buffer.concat([
        decipher.update(data),
        decipher.final()
      ]);
      
      return decrypted;
    } catch (error) {
      throw new Error(`Decryption failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Encrypt string and return as string
   */
  encryptString(plaintext: string): string {
    const encrypted = this.encrypt(plaintext);
    return JSON.stringify(encrypted);
  }

  /**
   * Decrypt string from encrypted string
   */
  decryptString(encryptedString: string): string {
    try {
      const encryptedData: IEncryptedData = JSON.parse(encryptedString);
      const decrypted = this.decrypt(encryptedData);
      return decrypted.toString('utf8');
    } catch (error) {
      throw new Error(`Failed to decrypt string: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Encrypt object
   */
  encryptObject<T>(obj: T): IEncryptedData {
    const json = JSON.stringify(obj);
    return this.encrypt(json);
  }

  /**
   * Decrypt object
   */
  decryptObject<T>(encryptedData: IEncryptedData): T {
    const decrypted = this.decrypt(encryptedData);
    const json = decrypted.toString('utf8');
    
    try {
      return JSON.parse(json) as T;
    } catch (error) {
      throw new Error(`Failed to parse decrypted JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Re-encrypt data with current key (for key rotation)
   */
  reencrypt(encryptedData: IEncryptedData): IEncryptedData {
    const decrypted = this.decrypt(encryptedData);
    return this.encrypt(decrypted);
  }

  /**
   * Rotate encryption key
   */
  rotateKey(): number {
    const newVersion = this.keyManager.generateNewKey();
    
    // Retire old keys after grace period
    setTimeout(() => {
      const keyInfo = this.keyManager.getKeyInfo();
      for (const info of keyInfo) {
        if (info.version < newVersion && info.status === 'rotating') {
          this.keyManager.retireKey(info.version);
        }
      }
    }, 60000); // 1 minute grace period
    
    return newVersion;
  }

  /**
   * Get key information
   */
  getKeyInfo(): IKeyInfo[] {
    return this.keyManager.getKeyInfo();
  }

  /**
   * Hash data using HMAC
   */
  hmac(data: string | Buffer, key?: Buffer): string {
    const dataBuffer = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    const hmacKey = key || this.keyManager.getCurrentKey().key;
    
    return createHmac('sha256', hmacKey).update(dataBuffer).digest('hex');
  }

  /**
   * Verify HMAC
   */
  verifyHmac(data: string | Buffer, signature: string, key?: Buffer): boolean {
    const expectedSignature = this.hmac(data, key);
    return this.constantTimeCompare(signature, expectedSignature);
  }

  /**
   * Generate secure random bytes
   */
  static generateRandomBytes(length: number): Buffer {
    return randomBytes(length);
  }

  /**
   * Generate secure random string
   */
  static generateRandomString(length: number): string {
    const bytes = randomBytes(Math.ceil(length * 3 / 4));
    return bytes.toString('base64').slice(0, length);
  }

  /**
   * Derive key from password
   */
  static deriveKey(password: string, salt: Buffer, iterations = 100000): Buffer {
    return pbkdf2Sync(password, salt, iterations, 32, 'sha256');
  }

  /**
   * Shutdown encryption service
   */
  shutdown(): void {
    if (this.rotationTimer) {
      clearInterval(this.rotationTimer);
    }
    
    // Clean up retired keys
    this.keyManager.cleanupRetiredKeys();
  }

  /**
   * Start automatic key rotation
   */
  private startKeyRotation(): void {
    this.rotationTimer = setInterval(() => {
      this.rotateKey();
    }, this.config.keyRotationInterval);
  }

  /**
   * Constant time string comparison to prevent timing attacks
   */
  private constantTimeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) {
      return false;
    }
    
    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }
    
    return result === 0;
  }
}

/**
 * Field-level encryption decorator
 */
export function Encrypted(encryptionService: EncryptionService) {
  return function (target: Record<string, unknown>, propertyKey: string): void {
    const privateKey = `_${propertyKey}`;
    
    Object.defineProperty(target, propertyKey, {
      get(this: Record<string, unknown>): string | undefined {
        const encryptedValue = this[privateKey] as string;
        if (!encryptedValue) return undefined;
        
        try {
          return encryptionService.decryptString(encryptedValue);
        } catch {
          return undefined;
        }
      },
      
      set(this: Record<string, unknown>, value: string): void {
        if (value === undefined || value === null) {
          this[privateKey] = undefined;
          return;
        }
        
        this[privateKey] = encryptionService.encryptString(value);
      },
      
      configurable: true,
      enumerable: true
    });
  };
}

/**
 * Create encryption service with default configuration
 */
export function createEncryptionService(masterPassword: string, config?: Partial<IEncryptionConfig>): EncryptionService {
  return new EncryptionService(masterPassword, config);
}

/**
 * Encryption utilities
 */
export class EncryptionUtils {
  /**
   * Generate cryptographically secure password
   */
  static generatePassword(length = 32, includeSymbols = true): string {
    const lowercase = 'abcdefghijklmnopqrstuvwxyz';
    const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const numbers = '0123456789';
    const symbols = '!@#$%^&*()_+-=[]{}|;:,.<>?';
    
    let charset = lowercase + uppercase + numbers;
    if (includeSymbols) {
      charset += symbols;
    }
    
    let password = '';
    const bytes = randomBytes(length);
    
    for (let i = 0; i < length; i++) {
      password += charset[bytes[i] % charset.length];
    }
    
    return password;
  }

  /**
   * Hash password with salt
   */
  static hashPassword(password: string, salt?: Buffer): { hash: string; salt: string } {
    const passwordSalt = salt || randomBytes(32);
    const hash = pbkdf2Sync(password, passwordSalt, 100000, 64, 'sha256');
    
    return {
      hash: hash.toString('hex'),
      salt: passwordSalt.toString('hex')
    };
  }

  /**
   * Verify password against hash
   */
  static verifyPassword(password: string, hash: string, salt: string): boolean {
    const saltBuffer = Buffer.from(salt, 'hex');
    const hashBuffer = Buffer.from(hash, 'hex');
    const computed = pbkdf2Sync(password, saltBuffer, 100000, 64, 'sha256');
    
    return hashBuffer.equals(computed);
  }

  /**
   * Generate salt
   */
  static generateSalt(length = 32): Buffer {
    return randomBytes(length);
  }
}
