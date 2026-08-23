import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';

import { type ClientEvent, WasmLspClient } from './wasm-client';

interface InitializeResult {
  capabilities?: {
    semanticTokensProvider?: {
      legend?: {
        tokenTypes?: string[];
        tokenModifiers?: string[];
      };
    };
  };
}

interface SemanticTokensResult {
  data?: number[];
}

interface PendingTokens {
  resolve(value: vscode.SemanticTokens | null): void;
  reject(error: Error): void;
}

class ReussirExtension implements vscode.Disposable {
  private readonly output = vscode.window.createOutputChannel('Reussir LSP', { log: true });
  private readonly disposables: vscode.Disposable[] = [this.output];
  private readonly openDocuments = new Map<string, number>();
  private readonly pendingTokens = new Map<number, PendingTokens>();
  private process: ChildProcessWithoutNullStreams | undefined;
  private client: WasmLspClient | undefined;
  private initializeWaiter:
    | { id: number; resolve(result: InitializeResult): void; reject(error: Error): void }
    | undefined;
  private shutdownWaiter: { id: number; resolve(): void } | undefined;
  private stopping = false;

  constructor(private readonly context: vscode.ExtensionContext) {}

  async start(): Promise<void> {
    const wasmUri = vscode.Uri.joinPath(this.context.extensionUri, 'dist', 'reussir_vscode_wasm.wasm');
    this.client = await WasmLspClient.instantiate(await vscode.workspace.fs.readFile(wasmUri));

    const command = resolveServerCommand(this.context);
    const cwd = vscode.workspace.workspaceFolders?.find(folder => folder.uri.scheme === 'file')?.uri.fsPath;
    this.output.info(`Starting ${command}`);
    this.process = spawn(command, [], {
      cwd,
      env: process.env,
      windowsHide: true,
      stdio: 'pipe'
    });
    this.process.stdout.on('data', (chunk: Buffer) => this.acceptServerBytes(chunk));
    this.process.stderr.on('data', (chunk: Buffer) => this.output.append(chunk.toString('utf8')));
    this.process.on('error', error => this.failAll(error));
    this.process.on('exit', (code, signal) => {
      if (!this.stopping) {
        this.failAll(new Error(`reussir-lsp exited unexpectedly (code ${code}, signal ${signal})`));
      }
    });

    const rootUri = vscode.workspace.workspaceFolders?.[0]?.uri.toString();
    const initialize = this.client.initialize(rootUri, process.pid);
    const result = await new Promise<InitializeResult>((resolve, reject) => {
      this.initializeWaiter = { id: initialize.id, resolve, reject };
      this.write(initialize.bytes);
    });
    this.write(this.client.initialized());

    const legend = result.capabilities?.semanticTokensProvider?.legend;
    if (!legend?.tokenTypes || !legend.tokenModifiers) {
      throw new Error('reussir-lsp did not advertise a semantic-token legend');
    }

    for (const document of vscode.workspace.textDocuments) {
      if (document.languageId === 'reussir') {
        this.open(document);
      }
    }
    this.disposables.push(
      vscode.workspace.onDidOpenTextDocument(document => {
        if (document.languageId === 'reussir') {
          this.open(document);
        }
      }),
      vscode.workspace.onDidChangeTextDocument(event => {
        if (event.document.languageId === 'reussir') {
          this.change(event.document);
        }
      }),
      vscode.workspace.onDidCloseTextDocument(document => {
        if (document.languageId === 'reussir') {
          this.close(document);
        }
      }),
      vscode.languages.registerDocumentSemanticTokensProvider(
        { language: 'reussir' },
        {
          provideDocumentSemanticTokens: (document, cancellation) =>
            this.provideSemanticTokens(document, cancellation)
        },
        new vscode.SemanticTokensLegend(legend.tokenTypes, legend.tokenModifiers)
      )
    );
  }

  async stop(): Promise<void> {
    if (this.stopping) {
      return;
    }
    this.stopping = true;
    for (const disposable of this.disposables.splice(0)) {
      disposable.dispose();
    }

    const client = this.client;
    const child = this.process;
    if (client && child && child.exitCode === null) {
      const shutdown = client.shutdown();
      const response = new Promise<void>(resolve => {
        this.shutdownWaiter = { id: shutdown.id, resolve };
      });
      this.write(shutdown.bytes);
      await Promise.race([response, new Promise<void>(resolve => setTimeout(resolve, 1_000))]);
      if (child.exitCode === null) {
        this.write(client.exit());
        child.stdin.end();
      }
      await Promise.race([
        new Promise<void>(resolve => child.once('exit', () => resolve())),
        new Promise<void>(resolve => setTimeout(resolve, 1_000))
      ]);
      if (child.exitCode === null) {
        child.kill();
      }
    }
    this.process = undefined;
    this.client = undefined;
  }

  dispose(): void {
    void this.stop();
  }

  private open(document: vscode.TextDocument): void {
    const uri = document.uri.toString();
    if (this.openDocuments.has(uri)) {
      return;
    }
    this.openDocuments.set(uri, document.version);
    this.write(this.requiredClient().didOpen(uri, document.version, document.getText()));
  }

  private change(document: vscode.TextDocument): void {
    const uri = document.uri.toString();
    if (!this.openDocuments.has(uri)) {
      this.open(document);
      return;
    }
    this.openDocuments.set(uri, document.version);
    this.write(this.requiredClient().didChange(uri, document.version, document.getText()));
  }

  private close(document: vscode.TextDocument): void {
    const uri = document.uri.toString();
    if (this.openDocuments.delete(uri)) {
      this.write(this.requiredClient().didClose(uri));
    }
  }

  private provideSemanticTokens(
    document: vscode.TextDocument,
    cancellation: vscode.CancellationToken
  ): Promise<vscode.SemanticTokens | null> {
    this.open(document);
    const request = this.requiredClient().semanticTokens(document.uri.toString());
    return new Promise((resolve, reject) => {
      this.pendingTokens.set(request.id, { resolve, reject });
      const subscription = cancellation.onCancellationRequested(() => {
        if (this.pendingTokens.delete(request.id)) {
          this.write(this.requiredClient().cancel(request.id));
          resolve(null);
        }
      });
      this.pendingTokens.set(request.id, {
        resolve: value => {
          subscription.dispose();
          resolve(value);
        },
        reject: error => {
          subscription.dispose();
          reject(error);
        }
      });
      this.write(request.bytes);
    });
  }

  private acceptServerBytes(chunk: Uint8Array): void {
    let events: ClientEvent[];
    try {
      events = this.requiredClient().feed(chunk);
    } catch (error) {
      this.failAll(asError(error));
      return;
    }
    for (const event of events) {
      switch (event.kind) {
        case 'initialized': {
          const waiter = this.initializeWaiter;
          if (waiter?.id === event.id) {
            this.initializeWaiter = undefined;
            waiter.resolve(event.result as InitializeResult);
          }
          break;
        }
        case 'semanticTokens': {
          const pending = this.pendingTokens.get(event.id);
          if (pending) {
            this.pendingTokens.delete(event.id);
            const result = event.result as SemanticTokensResult | null;
            pending.resolve(result?.data ? new vscode.SemanticTokens(Uint32Array.from(result.data)) : null);
          }
          break;
        }
        case 'shutdown':
          if (this.shutdownWaiter?.id === event.id) {
            this.shutdownWaiter.resolve();
            this.shutdownWaiter = undefined;
          }
          break;
        case 'error': {
          const error = new Error(`LSP ${event.code}: ${event.message}`);
          if (event.id === this.initializeWaiter?.id) {
            this.initializeWaiter.reject(error);
            this.initializeWaiter = undefined;
          } else if (event.id !== null) {
            this.pendingTokens.get(event.id)?.reject(error);
            this.pendingTokens.delete(event.id);
          } else {
            this.output.error(error.message);
          }
          break;
        }
        case 'notification':
          this.output.debug(`Server notification ${event.method}`);
          break;
        case 'unknownResponse':
          this.output.warn(`Ignoring response for unknown request ${event.id}`);
          break;
        case 'protocolError':
          this.failAll(new Error(event.message));
          break;
      }
    }
  }

  private write(bytes: Uint8Array): void {
    const child = this.process;
    if (!child || child.exitCode !== null || !child.stdin.writable) {
      throw new Error('reussir-lsp is not running');
    }
    child.stdin.write(bytes);
  }

  private requiredClient(): WasmLspClient {
    if (!this.client) {
      throw new Error('Reussir WASM client is not initialized');
    }
    return this.client;
  }

  private failAll(error: Error): void {
    this.output.error(error.message);
    this.initializeWaiter?.reject(error);
    this.initializeWaiter = undefined;
    for (const pending of this.pendingTokens.values()) {
      pending.reject(error);
    }
    this.pendingTokens.clear();
    if (!this.stopping) {
      void vscode.window.showErrorMessage(`Reussir language server: ${error.message}`);
    }
  }
}

function resolveServerCommand(context: vscode.ExtensionContext): string {
  const configured = vscode.workspace.getConfiguration('reussir').get<string>('server.path')?.trim();
  if (configured) {
    return configured;
  }
  if (process.env.REUSSIR_LSP_PATH) {
    return process.env.REUSSIR_LSP_PATH;
  }
  const executable = process.platform === 'win32' ? 'reussir-lsp.exe' : 'reussir-lsp';
  const development = path.resolve(context.extensionPath, '..', '..', 'build', 'bin', executable);
  return fs.existsSync(development) ? development : executable;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

let extension: ReussirExtension | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  extension = new ReussirExtension(context);
  context.subscriptions.push(extension);
  await extension.start();
}

export async function deactivate(): Promise<void> {
  await extension?.stop();
  extension = undefined;
}
