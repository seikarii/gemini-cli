/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import crypto from 'crypto';
import { Project, SourceFile, Node, ClassDeclaration, SyntaxKind } from 'ts-morph';
import { findNodes } from './finder.js';
import { ModificationSpec, ModificationOperation } from './models.js';

type ModifierResult = {
  success: boolean;
  output?: string;
  error?: string;
  modifiedText?: string;
  backupId?: string;
};

export class ASTModifier {
  private backups = new Map<string, string>();

  constructor(
    private readonly projectOptions: { useInMemoryFileSystem?: boolean } = {
      useInMemoryFileSystem: true,
    },
  ) {}

  private createProject() {
    return new Project({
      useInMemoryFileSystem: !!this.projectOptions.useInMemoryFileSystem,
    });
  }

  private createBackup(sourceText: string) {
    const id = crypto
      .createHash('md5')
      .update(sourceText + Date.now().toString())
      .digest('hex')
      .slice(0, 12);
    this.backups.set(id, sourceText);
    return id;
  }

  private restoreBackup(id: string): string | undefined {
    return this.backups.get(id);
  }

  /**
   * Apply a list of modifications to a source text. Returns modified text on success,
   * or restores the backup and returns an error on failure.
   */
  async applyModifications(
    sourceText: string,
    modifications: ModificationSpec[],
    opts?: { filePath?: string; format?: boolean },
  ): Promise<ModifierResult> {
    const project = this.createProject();
    const filePath = opts?.filePath ?? '/virtual-file.ts';
    const sourceFile = project.createSourceFile(filePath, sourceText, {
      overwrite: true,
    });

    const backupId = this.createBackup(sourceText);

  try {
      // Group modifications by target nodes if possible, otherwise apply sequentially
      for (const mod of modifications) {
        const targetQuery = (mod as any).targetQuery as any;
        const targets = findNodes(sourceFile, targetQuery);

        // If target nodes found, apply to each. If none, try to apply at file-level (e.g., add_import/add_class)
        if (targets && targets.length > 0) {
          for (const node of targets) {
            await this.applyModificationToNode(sourceFile, node, mod);
          }
        } else {
          // file-level ops
          await this.applyFileLevelModification(sourceFile, mod);
        }
      }

      // Save / format optionally
      let modifiedText = sourceFile.getFullText();
      if (opts?.format) {
        try {
          // try prettier if available (dynamic import to avoid require)
          const prettier = await import('prettier');
          const cfg = await (prettier as any).resolveConfig(filePath).catch(() => ({}));
          modifiedText = (prettier as any).format(modifiedText, {
            ...(cfg || {}),
            filepath: filePath,
          });
        } catch {
          // ignore formatting errors, return raw modified text
        }
      }

      return {
        success: true,
        output: 'Modifications applied',
        modifiedText,
        backupId,
      };
    } catch (_e: unknown) {
      const restored = this.restoreBackup(backupId);
      return {
        success: false,
        error: String(_e),
        output: 'Restored from backup',
        modifiedText: restored,
        backupId,
      };
    }
  }

  private async applyFileLevelModification(
    sourceFile: SourceFile,
    mod: ModificationSpec,
  ): Promise<void> {
    switch (mod.operation) {
      case ModificationOperation.ADD_IMPORT:
        if (!mod.newCode) throw new Error('add_import requires newCode');
        // Insert import at top
        sourceFile.insertText(0, mod.newCode.trim() + '\n');
        break;
      case ModificationOperation.ADD_CLASS: {
        if (!mod.newCode) throw new Error('add_class requires newCode');
        // Try to find sensible insertion point: after last import or at top
        const imports = sourceFile.getImportDeclarations();
        if (imports.length > 0) {
          const lastImp = imports[imports.length - 1];
          sourceFile.insertText(lastImp.getEnd(), '\n' + mod.newCode);
        } else {
          sourceFile.insertStatements(0, mod.newCode);
        }
        break;
      }
      default:
        // unsupported at file-level; nothing to do
        break;
    }
  }

  private async applyModificationToNode(
    sourceFile: SourceFile,
    node: Node,
    mod: ModificationSpec,
  ): Promise<void> {
    switch (mod.operation) {
      case ModificationOperation.REPLACE:
        if (!mod.newCode) throw new Error('replace requires newCode');
        node.replaceWithText(mod.newCode);
        break;

      case ModificationOperation.INSERT_BEFORE:
      case ModificationOperation.INSERT_AFTER:
        if (!mod.newCode) throw new Error('insert requires newCode');
        await this.insertRelative(
          node,
          mod.newCode,
          mod.operation === ModificationOperation.INSERT_BEFORE,
        );
        break;

      case ModificationOperation.DELETE:
        (node as any).remove();
        break;

      case ModificationOperation.MODIFY_ATTRIBUTE:
        if (!mod.attribute)
          throw new Error('modify_attribute requires attribute name');
        // Try to set property if exists
        try {
          // accessing internal ts-morph API; safe for now
          (node as any)[mod.attribute] = mod.value;
        } catch (_e) {
          // intentionally ignore errors from internal access
          void _e;
        }
        break;

      case ModificationOperation.WRAP:
        if (!mod.wrapperTemplate)
          throw new Error('wrap requires wrapperTemplate');
        await this.wrapNode(node, mod.wrapperTemplate);
        break;

      case ModificationOperation.EXTRACT:
        // create a new function at top-level with name=extractName and replace node with call
        if (!mod.extractName) throw new Error('extract requires extractName');
        await this.extractNodeAsFunction(sourceFile, node, mod.extractName);
        break;

      case ModificationOperation.REFACTOR:
        if (!mod.attribute)
          throw new Error('refactor requires attribute & value');
        try {
          // set attribute on node if possible
          // internal mutation of ts-morph node
          (node as any)[mod.attribute] = mod.value;
        } catch (_e) {
          void _e;
        }
        break;

      case ModificationOperation.RENAME_SYMBOL_SCOPED:
        if (!mod.attribute || typeof mod.value !== 'string')
          throw new Error(
            'rename_symbol_scoped requires attribute(oldName) and value(newName)',
          );
        await this.renameSymbolScoped(
          node,
          String(mod.attribute),
          String(mod.value),
        );
        break;

      case ModificationOperation.ADD_TO_CLASS_BASES:
        if (!mod.newCode)
          throw new Error('add_to_class_bases requires newCode');
        if (Node.isClassDeclaration(node)) {
          const cls = node as ClassDeclaration;
          const existing = cls.getExtends();
          if (existing) {
            // append to heritageClause by replacing with new text
            const baseText = cls
              .getHeritageClauses()
              .map((h: Node) => h.getText())
              .join(' ');
            cls.replaceWithText(
              cls.getText().replace(baseText, `${baseText}, ${mod.newCode}`),
            );
          } else {
            // add an extends clause
            cls.insertText(
              cls.getStart() + cls.getText().indexOf('{'),
              ` extends ${mod.newCode} `,
            );
          }
        }
        break;
      case ModificationOperation.UPDATE_METHOD_SIGNATURE:
        if (!mod.newCode)
          throw new Error('update_method_signature requires newCode');
        await this.updateMethodSignature(node, mod.newCode);
        break;

      case ModificationOperation.INSERT_STATEMENT_INTO_FUNCTION:
        if (!mod.newCode)
          throw new Error('insert_statement_into_function requires newCode');
        await this.insertStatementIntoFunction(
          node,
          mod.newCode,
          (mod.metadata?.['insert_at_end'] ?? true) as boolean,
        );
        break;

      case ModificationOperation.REPLACE_EXPRESSION:
        if (!mod.newCode)
          throw new Error('replace_expression requires newCode');
        node.replaceWithText(mod.newCode);
        break;

      default:
        // If unknown, try best-effort replaceWithText
        if (mod.newCode) node.replaceWithText(mod.newCode);
        break;
    }
  }

  private async insertRelative(node: Node, code: string, before: boolean) {
    const parent = node.getParent();
    if (!parent) {
      // fallback to replaceWithText surrounding insertion
      const text = before
        ? `${code}\n${node.getText()}`
        : `${node.getText()}\n${code}`;
      node.replaceWithText(text);
      return;
    }

    // If parent is a block (list of statements), use insertStatements
    const block = parent.getFirstChildByKind(SyntaxKind.Block) ?? parent;
    if (Node.isSourceFile(block) || Node.isBlock(block)) {
      // find index of node among block statements
      const statements = (block as any).getStatements ? (block as any).getStatements() : [];
      const idx = statements.findIndex((s: Node) => s.getStart() === node.getStart());
      if (idx >= 0) {
        if (before) (block as any).insertStatements(idx, code);
        else (block as any).insertStatements(idx + 1, code);
        return;
      }
    }

    // fallback: text replacement
    const text = before
      ? `${code}\n${node.getText()}`
      : `${node.getText()}\n${code}`;
    node.replaceWithText(text);
  }

  private async wrapNode(node: Node, wrapperTemplate: string) {
    // wrapperTemplate expected to contain a placeholder like "{node}" or "/*NODE*/"
    const raw = wrapperTemplate
      .replace('{node}', node.getText())
      .replace('/*NODE*/', node.getText());
    node.replaceWithText(raw);
  }

  private async extractNodeAsFunction(
    sourceFile: SourceFile,
    node: Node,
    funcName: string,
  ) {
    // Try to create a function at top of file that returns or contains the node
    const nodeText = node.getText();
    const funcText = `function ${funcName}() { return (${nodeText}); }`;
    // Insert at top, after imports if any
    const imports = sourceFile.getImportDeclarations();
    if (imports.length > 0) {
      sourceFile.insertText(
        imports[imports.length - 1].getEnd(),
        '\n' + funcText,
      );
    } else {
      sourceFile.insertStatements(0, funcText);
    }
    // Replace original with a call
    node.replaceWithText(`${funcName}()`);
  }

  private async renameSymbolScoped(
    scopeNode: Node,
    oldName: string,
    newName: string,
  ) {
    // Walk descendants and replace identifier text when it matches oldName
  scopeNode.forEachDescendant((n: Node) => {
      try {
        // look for identifiers (VariableDeclaration, Identifier nodes)
        if (
          (n.getKind && n.getKind() === SyntaxKind.Identifier) ||
          (n.getText && n.getText() === oldName)
        ) {
          if (n.getText() === oldName) {
            n.replaceWithText(newName);
          }
        } else {
          // sometimes ts-morph nodes (parameters, variable names) expose getName
          if (Node.isRenameable(n) && typeof (n as any).getName === 'function' && (n as any).getName() === oldName) {
            // rename is optional on some nodes
            try {
              (n as any).rename(newName);
            } catch {
              // ignore rename failures
            }
          }
        }
      } catch {
        // ignore problematic nodes
      }
    });
  }

  private async updateMethodSignature(node: Node, newSignature: string) {
    if (
      Node.isFunctionDeclaration(node) ||
      Node.isMethodDeclaration(node) ||
      Node.isArrowFunction(node) ||
      Node.isFunctionExpression(node)
    ) {
      try {
  // attempt to set parameters by replacing the signature portion
              const bodyText = (node.getFirstChildByKind(SyntaxKind.Block) as any)?.getText?.() ?? '{}';
              const name = ((node as any).getName && (node as any).getName()) ?? '<anonymous>';
  const replaceText = `${name}(${newSignature}) ${bodyText}`;
        // replace whole node with new signature + body (safer than trying to mutate params)
        node.replaceWithText(replaceText);
      } catch {
        // fallback: naive text replace
        node.replaceWithText(
          node.getText().replace(/\([^)]*\)/, `(${newSignature})`),
        );
      }
    } else {
      // not a function -> ignore
    }
  }

  private async insertStatementIntoFunction(
    node: Node,
    stmtCode: string,
    atEnd = true,
  ) {
    // If node is a function or method, insert into its body
    const block = node.getFirstChildByKind(SyntaxKind.Block) as any;
    if (block) {
      const stmts = block.getStatements ? block.getStatements() : [];
      if (atEnd) block.insertStatements(stmts.length, stmtCode);
      else block.insertStatements(0, stmtCode);
      return;
    }

    // fallback: replace node text by injecting before/after
    const text = atEnd
      ? `${node.getText()}\n${stmtCode}`
      : `${stmtCode}\n${node.getText()}`;
    node.replaceWithText(text);
  }
}
