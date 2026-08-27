import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';

import { type ClientEvent, WasmClient, feedEvents } from './wasm-client';

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
  private readonly sessionDisposables: vscode.Disposable[] = [];
  private readonly openDocuments = new Map<string, number>();
  private readonly pendingTokens = new Map<number, PendingTokens>();
  private process: ChildProcessWithoutNullStreams | undefined;
  private client: WasmClient | undefined;
  private initializeWaiter:
    | { id: number; resolve(result: InitializeResult): void; reject(error: Error): void }
    | undefined;
  private shutdownWaiter: { id: number; resolve(): void } | undefined;
  private stopping = false;

  constructor(private readonly context: vscode.ExtensionContext) {}

  async start(): Promise<void> {
    // The wasm-bindgen module loads and instantiates its wasm when first
    // required; constructing the client allocates a fresh codec.
    this.client = new WasmClient();

    const command = resolveServerCommand(this.context);
    const cwd = vscode.workspace.workspaceFolders?.find(folder => folder.uri.scheme === 'file')?.uri.fsPath;
    this.output.info(`Starting ${command}`);
    const child = spawn(command, [], {
      cwd,
      env: process.env,
      windowsHide: true,
      stdio: 'pipe'
    });
    this.process = child;
    child.stdout.on('data', (chunk: Buffer) => this.acceptServerBytes(chunk));
    child.stderr.on('data', (chunk: Buffer) => this.output.append(chunk.toString('utf8')));
    // Pipe writes fail asynchronously when the server dies mid-write; without
    // this listener the EPIPE becomes an uncaught extension-host exception.
    child.stdin.on('error', error => this.output.error(`reussir-lsp stdin: ${error.message}`));
    child.on('error', error => {
      if (this.process === child) {
        this.failSession(error);
      }
    });
    child.on('exit', (code, signal) => {
      if (!this.stopping && this.process === child) {
        this.failSession(new Error(`reussir-lsp exited unexpectedly (code ${code}, signal ${signal})`));
      }
    });

    const rootUri = vscode.workspace.workspaceFolders?.[0]?.uri.toString();
    const initialize = this.client.initialize(rootUri, process.pid);
    const result = await new Promise<InitializeResult>((resolve, reject) => {
      this.initializeWaiter = { id: initialize.id, resolve, reject };
      this.write(initialize.frame);
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
    this.sessionDisposables.push(
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

    const client = this.client;
    const child = this.process;
    if (client && child && serverAlive(child)) {
      try {
        const shutdown = client.shutdown();
        const response = new Promise<void>(resolve => {
          this.shutdownWaiter = { id: shutdown.id, resolve };
        });
        this.write(shutdown.frame);
        await Promise.race([response, delay(1_000)]);
        if (serverAlive(child)) {
          this.write(client.exit());
          child.stdin.end();
        }
        await Promise.race([
          new Promise<void>(resolve => child.once('exit', () => resolve())),
          delay(1_000)
        ]);
      } catch (error) {
        this.output.error(`shutdown handshake failed: ${asError(error).message}`);
      }
    }
    this.teardownSession(new Error('the Reussir extension is shutting down'));
    this.output.dispose();
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
      const subscription = cancellation.onCancellationRequested(() => {
        if (this.pendingTokens.delete(request.id)) {
          subscription.dispose();
          try {
            this.write(this.requiredClient().cancel(request.id));
          } catch (error) {
            this.output.debug(`could not cancel request ${request.id}: ${asError(error).message}`);
          }
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
      try {
        this.write(request.frame);
      } catch (error) {
        if (this.pendingTokens.delete(request.id)) {
          subscription.dispose();
          reject(asError(error));
        }
      }
    });
  }

  private acceptServerBytes(chunk: Uint8Array): void {
    let events: ClientEvent[];
    try {
      events = feedEvents(this.requiredClient(), chunk);
    } catch (error) {
      this.failSession(asError(error));
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
          this.failSession(new Error(event.message));
          break;
      }
    }
  }

  private write(bytes: Uint8Array): void {
    const child = this.process;
    if (!child || !serverAlive(child)) {
      throw new Error('reussir-lsp is not running');
    }
    child.stdin.write(bytes);
  }

  private requiredClient(): WasmClient {
    if (!this.client) {
      throw new Error('Reussir WASM client is not initialized');
    }
    return this.client;
  }

  // Fatal-session handler: rejects in-flight work, tears the session down so
  // dead listeners stop firing, and offers a restart.
  private failSession(error: Error): void {
    this.output.error(error.message);
    this.teardownSession(error);
    if (this.stopping) {
      return;
    }
    void vscode.window
      .showErrorMessage(`Reussir language server: ${error.message}`, 'Restart Server')
      .then(choice => {
        if (choice === 'Restart Server' && !this.stopping) {
          this.start().catch(startError => {
            void vscode.window.showErrorMessage(
              `Reussir language server failed to restart: ${asError(startError).message}`
            );
          });
        }
      });
  }

  private teardownSession(error: Error): void {
    for (const disposable of this.sessionDisposables.splice(0)) {
      disposable.dispose();
    }
    this.openDocuments.clear();
    this.initializeWaiter?.reject(error);
    this.initializeWaiter = undefined;
    this.shutdownWaiter = undefined;
    for (const pending of this.pendingTokens.values()) {
      pending.reject(error);
    }
    this.pendingTokens.clear();
    const child = this.process;
    this.process = undefined;
    // Release the codec's wasm-side memory eagerly; a restart constructs a
    // fresh client rather than reusing this one.
    this.client?.free();
    this.client = undefined;
    if (child && serverAlive(child)) {
      child.kill();
    }
  }
}

function serverAlive(child: ChildProcessWithoutNullStreams): boolean {
  // exitCode stays null forever for a signal-killed process; signalCode is
  // what records that death.
  return child.exitCode === null && child.signalCode === null && child.stdin.writable;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
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
