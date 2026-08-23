const ABI_VERSION = 1;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

interface WasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  reussir_vscode_abi_version(): number;
  reussir_vscode_alloc(length: number): number;
  reussir_vscode_dealloc(pointer: number, length: number): void;
  reussir_vscode_output_pointer(): number;
  reussir_vscode_output_length(): number;
  reussir_vscode_reset(): void;
  reussir_vscode_initialize(rootPointer: number, rootLength: number, processId: number): number;
  reussir_vscode_initialized(): void;
  reussir_vscode_did_open(
    uriPointer: number,
    uriLength: number,
    version: number,
    textPointer: number,
    textLength: number
  ): void;
  reussir_vscode_did_change(
    uriPointer: number,
    uriLength: number,
    version: number,
    textPointer: number,
    textLength: number
  ): void;
  reussir_vscode_did_close(uriPointer: number, uriLength: number): void;
  reussir_vscode_semantic_tokens(uriPointer: number, uriLength: number): number;
  reussir_vscode_cancel(id: number): void;
  reussir_vscode_shutdown(): number;
  reussir_vscode_exit(): void;
  reussir_vscode_feed(pointer: number, length: number): void;
}

export type ClientEvent =
  | { kind: 'initialized'; id: number; result: unknown }
  | { kind: 'semanticTokens'; id: number; result: unknown }
  | { kind: 'shutdown'; id: number }
  | { kind: 'error'; id: number | null; code: number; message: string }
  | { kind: 'notification'; method: string; params: unknown }
  | { kind: 'unknownResponse'; id: number; result: unknown }
  | { kind: 'protocolError'; message: string };

interface Allocation {
  pointer: number;
  length: number;
}

export class WasmLspClient {
  private constructor(private readonly exports: WasmExports) {
    const version = exports.reussir_vscode_abi_version();
    if (version !== ABI_VERSION) {
      throw new Error(`unsupported Reussir VS Code WASM ABI ${version}; expected ${ABI_VERSION}`);
    }
    exports.reussir_vscode_reset();
  }

  static async instantiate(bytes: Uint8Array): Promise<WasmLspClient> {
    const owned = Uint8Array.from(bytes);
    const module = await WebAssembly.compile(owned.buffer);
    const imports = WebAssembly.Module.imports(module);
    if (imports.length !== 0) {
      throw new Error(`Reussir VS Code WASM unexpectedly imports ${imports.length} host symbols`);
    }
    const instance = await WebAssembly.instantiate(module, {});
    return new WasmLspClient(instance.exports as WasmExports);
  }

  initialize(rootUri: string | undefined, processId: number | undefined): { id: number; bytes: Uint8Array } {
    return this.withStrings([rootUri ?? ''], ([root]) => {
      const id = this.exports.reussir_vscode_initialize(
        root.pointer,
        root.length,
        processId ?? -1
      );
      return { id, bytes: this.output() };
    });
  }

  initialized(): Uint8Array {
    this.exports.reussir_vscode_initialized();
    return this.output();
  }

  didOpen(uri: string, version: number, text: string): Uint8Array {
    return this.withStrings([uri, text], ([encodedUri, encodedText]) => {
      this.exports.reussir_vscode_did_open(
        encodedUri.pointer,
        encodedUri.length,
        version,
        encodedText.pointer,
        encodedText.length
      );
      return this.output();
    });
  }

  didChange(uri: string, version: number, text: string): Uint8Array {
    return this.withStrings([uri, text], ([encodedUri, encodedText]) => {
      this.exports.reussir_vscode_did_change(
        encodedUri.pointer,
        encodedUri.length,
        version,
        encodedText.pointer,
        encodedText.length
      );
      return this.output();
    });
  }

  didClose(uri: string): Uint8Array {
    return this.withStrings([uri], ([encodedUri]) => {
      this.exports.reussir_vscode_did_close(encodedUri.pointer, encodedUri.length);
      return this.output();
    });
  }

  semanticTokens(uri: string): { id: number; bytes: Uint8Array } {
    return this.withStrings([uri], ([encodedUri]) => {
      const id = this.exports.reussir_vscode_semantic_tokens(
        encodedUri.pointer,
        encodedUri.length
      );
      return { id, bytes: this.output() };
    });
  }

  cancel(id: number): Uint8Array {
    this.exports.reussir_vscode_cancel(id);
    return this.output();
  }

  shutdown(): { id: number; bytes: Uint8Array } {
    const id = this.exports.reussir_vscode_shutdown();
    return { id, bytes: this.output() };
  }

  exit(): Uint8Array {
    this.exports.reussir_vscode_exit();
    return this.output();
  }

  feed(chunk: Uint8Array): ClientEvent[] {
    return this.withBytes(chunk, allocation => {
      this.exports.reussir_vscode_feed(allocation.pointer, allocation.length);
      const output = this.output();
      return output.length === 0 ? [] : JSON.parse(decoder.decode(output)) as ClientEvent[];
    });
  }

  private output(): Uint8Array {
    const pointer = this.exports.reussir_vscode_output_pointer();
    const length = this.exports.reussir_vscode_output_length();
    return new Uint8Array(this.exports.memory.buffer, pointer, length).slice();
  }

  private withStrings<T>(values: string[], operation: (allocations: Allocation[]) => T): T {
    const encoded = values.map(value => encoder.encode(value));
    const allocations = encoded.map(bytes => this.allocate(bytes));
    try {
      return operation(allocations);
    } finally {
      for (const allocation of allocations) {
        this.exports.reussir_vscode_dealloc(allocation.pointer, allocation.length);
      }
    }
  }

  private withBytes<T>(bytes: Uint8Array, operation: (allocation: Allocation) => T): T {
    const allocation = this.allocate(bytes);
    try {
      return operation(allocation);
    } finally {
      this.exports.reussir_vscode_dealloc(allocation.pointer, allocation.length);
    }
  }

  private allocate(bytes: Uint8Array): Allocation {
    if (bytes.length === 0) {
      return { pointer: 0, length: 0 };
    }
    const pointer = this.exports.reussir_vscode_alloc(bytes.length);
    new Uint8Array(this.exports.memory.buffer, pointer, bytes.length).set(bytes);
    return { pointer, length: bytes.length };
  }
}
