// Typed surface over the wasm-bindgen-generated bindings. The generated
// module (dist/wasm/, produced by `npm run build:wasm`) owns instantiation,
// memory management, and string/buffer marshaling; this file only narrows
// the untyped `feed` result to the ClientEvent union the Rust codec emits
// (crates/reussir-vscode-wasm/src/lib.rs).
import { WasmClient, Request } from '../dist/wasm/reussir_vscode_wasm';

export type ClientEvent =
  | { kind: 'initialized'; id: number; result: unknown }
  | { kind: 'semanticTokens'; id: number; result: unknown }
  | { kind: 'shutdown'; id: number }
  | { kind: 'error'; id: number | null; code: number; message: string }
  | { kind: 'notification'; method: string; params: unknown }
  | { kind: 'unknownResponse'; id: number; result: unknown }
  | { kind: 'protocolError'; message: string };

export function feedEvents(client: WasmClient, chunk: Uint8Array): ClientEvent[] {
  return client.feed(chunk) as ClientEvent[];
}

export { WasmClient, Request };
