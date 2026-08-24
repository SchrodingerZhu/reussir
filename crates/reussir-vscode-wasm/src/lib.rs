//! Stateful LSP framing for the Rust/Wasm portion of the VS Code extension.
//!
//! VS Code and native-process access stay in the small TypeScript host. This
//! crate owns request identifiers, synchronization messages, response routing,
//! and streaming `Content-Length` framing. It deliberately knows nothing about
//! Reussir syntax; the native `reussir-lsp` remains the only semantic engine.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestKind {
    Initialize,
    SemanticTokens,
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ClientEvent {
    Initialized {
        id: i32,
        result: Value,
    },
    SemanticTokens {
        id: i32,
        result: Value,
    },
    Shutdown {
        id: i32,
    },
    Error {
        id: Option<i32>,
        code: i64,
        message: String,
    },
    Notification {
        method: String,
        params: Value,
    },
    UnknownResponse {
        id: i32,
        result: Value,
    },
    ProtocolError {
        message: String,
    },
}

#[derive(Debug)]
pub struct ClientCodec {
    next_id: i32,
    pending: HashMap<i32, RequestKind>,
}

impl Default for ClientCodec {
    fn default() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
        }
    }
}

impl ClientCodec {
    pub fn initialize(
        &mut self,
        root_uri: Option<&str>,
        process_id: Option<u32>,
    ) -> (i32, Vec<u8>) {
        let params = json!({
            "processId": process_id,
            "clientInfo": {
                "name": "reussir-vscode",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "rootUri": root_uri,
            "capabilities": {
                "general": {
                    "positionEncodings": ["utf-16"],
                },
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": false,
                        "didSave": false,
                    },
                    "semanticTokens": {
                        "dynamicRegistration": false,
                        "requests": {
                            "range": false,
                            "full": true,
                        },
                        "tokenTypes": [
                            "namespace", "type", "class", "enum", "interface", "struct",
                            "typeParameter", "parameter", "variable", "property", "enumMember",
                            "event", "function", "method", "macro", "keyword", "modifier",
                            "label", "comment", "string", "number", "regexp", "operator",
                            "decorator"
                        ],
                        "tokenModifiers": [
                            "declaration", "definition", "readonly", "static", "deprecated",
                            "abstract", "async", "modification", "documentation", "defaultLibrary"
                        ],
                        "formats": ["relative"],
                        "overlappingTokenSupport": false,
                        "multilineTokenSupport": false
                    }
                }
            },
            "workspaceFolders": Value::Null,
        });
        self.request("initialize", params, RequestKind::Initialize)
    }

    pub fn initialized(&self) -> Vec<u8> {
        notification("initialized", json!({}))
    }

    pub fn did_open(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
        notification(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": "reussir",
                    "version": version,
                    "text": text,
                }
            }),
        )
    }

    pub fn did_change(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
        notification(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [{ "text": text }],
            }),
        )
    }

    pub fn did_close(&self, uri: &str) -> Vec<u8> {
        notification(
            "textDocument/didClose",
            json!({ "textDocument": { "uri": uri } }),
        )
    }

    pub fn semantic_tokens(&mut self, uri: &str) -> (i32, Vec<u8>) {
        self.request(
            "textDocument/semanticTokens/full",
            json!({ "textDocument": { "uri": uri } }),
            RequestKind::SemanticTokens,
        )
    }

    pub fn cancel(&self, id: i32) -> Vec<u8> {
        notification("$/cancelRequest", json!({ "id": id }))
    }

    pub fn shutdown(&mut self) -> (i32, Vec<u8>) {
        self.request("shutdown", Value::Null, RequestKind::Shutdown)
    }

    pub fn exit(&self) -> Vec<u8> {
        notification("exit", Value::Null)
    }

    fn request(&mut self, method: &str, params: Value, kind: RequestKind) -> (i32, Vec<u8>) {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).unwrap_or(1);
        self.pending.insert(id, kind);
        (
            id,
            frame(&json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })),
        )
    }
}

fn notification(method: &str, params: Value) -> Vec<u8> {
    frame(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
}

fn frame(value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).expect("JSON-RPC values are serializable");
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    let mut output = Vec::with_capacity(header.len() + body.len());
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(&body);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(frame: &[u8]) -> Value {
        let split = frame.windows(4).position(|window| window == b"\r\n\r\n").unwrap() + 4;
        serde_json::from_slice(&frame[split..]).unwrap()
    }

    #[test]
    fn full_document_notifications_have_the_expected_shape() {
        let client = ClientCodec::default();
        let open = body(&client.did_open("file:///demo.rr", 3, "fn main() {}"));
        assert_eq!(open["method"], "textDocument/didOpen");
        assert_eq!(open["params"]["textDocument"]["version"], 3);

        let change = body(&client.did_change("file:///demo.rr", 4, "fn changed() {}"));
        assert_eq!(change["method"], "textDocument/didChange");
        assert!(change["params"]["contentChanges"][0].get("range").is_none());
    }

    #[test]
    fn request_ids_are_unique_and_monotonic() {
        let mut client = ClientCodec::default();
        let (first, _) = client.initialize(None, None);
        let (second, _) = client.semantic_tokens("file:///demo.rr");
        let (third, _) = client.shutdown();
        assert!(first < second && second < third);
    }
}
