//! Stateful LSP framing for the Rust/Wasm portion of the VS Code extension.
//!
//! VS Code and native-process access stay in the small TypeScript host. This
//! crate owns request identifiers, synchronization messages, response routing,
//! and streaming `Content-Length` framing. It deliberately knows nothing about
//! Reussir syntax; the native `reussir-lsp` remains the only semantic engine.

use std::collections::HashMap;

use lsp_types::{
    CancelParams, ClientCapabilities, ClientInfo, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, GeneralClientCapabilities,
    InitializeParams, InitializedParams, NumberOrString, PositionEncodingKind,
    SemanticTokenModifier, SemanticTokenType, SemanticTokensClientCapabilities,
    SemanticTokensClientCapabilitiesRequests, SemanticTokensFullOptions, SemanticTokensParams,
    TextDocumentClientCapabilities, TextDocumentContentChangeEvent, TextDocumentIdentifier,
    TextDocumentItem, TextDocumentSyncClientCapabilities, TokenFormat, Url,
    VersionedTextDocumentIdentifier,
};
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
        #[allow(deprecated)]
        let params = InitializeParams {
            process_id,
            root_uri: root_uri.map(parse_uri),
            client_info: Some(ClientInfo {
                name: "reussir-vscode".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
            capabilities: client_capabilities(),
            ..InitializeParams::default()
        };
        self.request("initialize", to_value(params), RequestKind::Initialize)
    }

    pub fn initialized(&self) -> Vec<u8> {
        notification("initialized", to_value(InitializedParams {}))
    }

    pub fn did_open(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
        notification(
            "textDocument/didOpen",
            to_value(DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: parse_uri(uri),
                    language_id: "reussir".into(),
                    version,
                    text: text.into(),
                },
            }),
        )
    }

    pub fn did_change(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
        notification(
            "textDocument/didChange",
            to_value(DidChangeTextDocumentParams {
                text_document: VersionedTextDocumentIdentifier {
                    uri: parse_uri(uri),
                    version,
                },
                content_changes: vec![TextDocumentContentChangeEvent {
                    range: None,
                    range_length: None,
                    text: text.into(),
                }],
            }),
        )
    }

    pub fn did_close(&self, uri: &str) -> Vec<u8> {
        notification(
            "textDocument/didClose",
            to_value(DidCloseTextDocumentParams {
                text_document: TextDocumentIdentifier {
                    uri: parse_uri(uri),
                },
            }),
        )
    }

    pub fn semantic_tokens(&mut self, uri: &str) -> (i32, Vec<u8>) {
        self.request(
            "textDocument/semanticTokens/full",
            to_value(SemanticTokensParams {
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                text_document: TextDocumentIdentifier {
                    uri: parse_uri(uri),
                },
            }),
            RequestKind::SemanticTokens,
        )
    }

    pub fn cancel(&self, id: i32) -> Vec<u8> {
        notification(
            "$/cancelRequest",
            to_value(CancelParams {
                id: NumberOrString::Number(id),
            }),
        )
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

fn client_capabilities() -> ClientCapabilities {
    ClientCapabilities {
        general: Some(GeneralClientCapabilities {
            position_encodings: Some(vec![PositionEncodingKind::UTF16]),
            ..GeneralClientCapabilities::default()
        }),
        text_document: Some(TextDocumentClientCapabilities {
            synchronization: Some(TextDocumentSyncClientCapabilities {
                dynamic_registration: Some(false),
                did_save: Some(false),
                ..TextDocumentSyncClientCapabilities::default()
            }),
            semantic_tokens: Some(SemanticTokensClientCapabilities {
                dynamic_registration: Some(false),
                requests: SemanticTokensClientCapabilitiesRequests {
                    range: Some(false),
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                },
                token_types: vec![
                    SemanticTokenType::NAMESPACE,
                    SemanticTokenType::TYPE,
                    SemanticTokenType::CLASS,
                    SemanticTokenType::ENUM,
                    SemanticTokenType::INTERFACE,
                    SemanticTokenType::STRUCT,
                    SemanticTokenType::TYPE_PARAMETER,
                    SemanticTokenType::PARAMETER,
                    SemanticTokenType::VARIABLE,
                    SemanticTokenType::PROPERTY,
                    SemanticTokenType::ENUM_MEMBER,
                    SemanticTokenType::EVENT,
                    SemanticTokenType::FUNCTION,
                    SemanticTokenType::METHOD,
                    SemanticTokenType::MACRO,
                    SemanticTokenType::KEYWORD,
                    SemanticTokenType::MODIFIER,
                    SemanticTokenType::new("label"),
                    SemanticTokenType::COMMENT,
                    SemanticTokenType::STRING,
                    SemanticTokenType::NUMBER,
                    SemanticTokenType::REGEXP,
                    SemanticTokenType::OPERATOR,
                    SemanticTokenType::DECORATOR,
                ],
                token_modifiers: vec![
                    SemanticTokenModifier::DECLARATION,
                    SemanticTokenModifier::DEFINITION,
                    SemanticTokenModifier::READONLY,
                    SemanticTokenModifier::STATIC,
                    SemanticTokenModifier::DEPRECATED,
                    SemanticTokenModifier::ABSTRACT,
                    SemanticTokenModifier::ASYNC,
                    SemanticTokenModifier::MODIFICATION,
                    SemanticTokenModifier::DOCUMENTATION,
                    SemanticTokenModifier::DEFAULT_LIBRARY,
                ],
                formats: vec![TokenFormat::RELATIVE],
                overlapping_token_support: Some(false),
                multiline_token_support: Some(false),
                ..SemanticTokensClientCapabilities::default()
            }),
            ..TextDocumentClientCapabilities::default()
        }),
        ..ClientCapabilities::default()
    }
}

/// The TypeScript host constructs every URI it passes in; a URI that does not
/// parse is a host bug, not a recoverable protocol state.
fn parse_uri(uri: &str) -> Url {
    Url::parse(uri).expect("the host must pass valid URIs")
}

fn to_value(params: impl Serialize) -> Value {
    serde_json::to_value(params).expect("LSP params are serializable")
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
        let split = frame
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4;
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
    fn initialize_advertises_utf16_and_full_only_semantic_tokens() {
        let mut client = ClientCodec::default();
        let (_, frame) = client.initialize(Some("file:///workspace"), Some(7));
        let init = body(&frame);
        assert_eq!(init["method"], "initialize");
        assert_eq!(init["params"]["processId"], 7);
        assert_eq!(init["params"]["rootUri"], "file:///workspace");
        let capabilities = &init["params"]["capabilities"];
        assert_eq!(capabilities["general"]["positionEncodings"][0], "utf-16");
        let semantic = &capabilities["textDocument"]["semanticTokens"];
        assert_eq!(semantic["requests"]["full"], true);
        assert_eq!(semantic["requests"]["range"], false);
        assert_eq!(semantic["formats"], json!(["relative"]));
        let token_types = semantic["tokenTypes"].as_array().unwrap();
        assert!(token_types.iter().any(|value| value == "label"));
        assert!(token_types.iter().any(|value| value == "decorator"));
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
