import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { type ClientEvent, WasmLspClient } from '../src/wasm-client';

async function main(): Promise<void> {
  const extensionRoot = process.cwd();
  const repositoryRoot = path.resolve(extensionRoot, '../..');
  const executable = process.platform === 'win32' ? 'reussir-lsp.exe' : 'reussir-lsp';
  const server = process.env.REUSSIR_LSP_PATH || path.join(repositoryRoot, 'target', 'debug', executable);
  const wasm = await WasmLspClient.instantiate(
    await readFile(path.join(extensionRoot, 'dist', 'reussir_vscode_wasm.wasm'))
  );
  const child = spawn(server, [], { stdio: 'pipe', windowsHide: true });

  const waiting = new Map<number, (event: ClientEvent) => void>();
  let failure: Error | undefined;
  child.stderr.on('data', chunk => process.stderr.write(chunk));
  child.on('error', error => {
    failure = error;
    for (const resolve of waiting.values()) {
      resolve({ kind: 'protocolError', message: error.message });
    }
  });
  child.stdout.on('data', chunk => {
    for (const event of wasm.feed(chunk)) {
      if ('id' in event && typeof event.id === 'number') {
        waiting.get(event.id)?.(event);
        waiting.delete(event.id);
      }
    }
  });

  function write(bytes: Uint8Array): void {
    child.stdin.write(bytes);
  }

  function response(id: number): Promise<ClientEvent> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        waiting.delete(id);
        reject(new Error(`timed out waiting for LSP response ${id}`));
      }, 5_000);
      waiting.set(id, event => {
        clearTimeout(timer);
        resolve(event);
      });
    });
  }

  try {
    const initialize = wasm.initialize(`file://${repositoryRoot}`, process.pid);
    const initialized = response(initialize.id);
    write(initialize.bytes);
    const initializeEvent = await initialized;
    if (initializeEvent.kind !== 'initialized') {
      throw new Error(`initialization failed: ${JSON.stringify(initializeEvent)}`);
    }
    write(wasm.initialized());

    const uri = 'file:///reussir-vscode-smoke.rr';
    write(wasm.didOpen(uri, 1, 'pub fn highlighted(value: i64) -> i64 { value }'));
    const tokens = wasm.semanticTokens(uri);
    const tokenResponse = response(tokens.id);
    write(tokens.bytes);
    const tokenEvent = await tokenResponse;
    if (tokenEvent.kind !== 'semanticTokens') {
      throw new Error(`semantic token request failed: ${JSON.stringify(tokenEvent)}`);
    }
    const data = (tokenEvent.result as { data?: number[] } | null)?.data;
    if (!data || data.length === 0 || data.length % 5 !== 0) {
      throw new Error(`invalid semantic token payload: ${JSON.stringify(tokenEvent.result)}`);
    }
    write(wasm.didClose(uri));

    const shutdown = wasm.shutdown();
    const shutdownResponse = response(shutdown.id);
    write(shutdown.bytes);
    const shutdownEvent = await shutdownResponse;
    if (shutdownEvent.kind !== 'shutdown') {
      throw new Error(`shutdown failed: ${JSON.stringify(shutdownEvent)}`);
    }
    write(wasm.exit());
    child.stdin.end();
    console.log(`Reussir VS Code WASM protocol smoke test passed (${data.length / 5} tokens)`);
  } finally {
    if (child.exitCode === null) {
      child.kill();
    }
  }

  if (failure) {
    throw failure;
  }
}

void main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
