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
    input: Vec<u8>,
}

impl Default for ClientCodec {
    fn default() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
            input: Vec::new(),
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

    pub fn feed(&mut self, chunk: &[u8]) -> Vec<ClientEvent> {
        self.input.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(header_end) = find_bytes(&self.input, b"\r\n\r\n") {
            let header = &self.input[..header_end];
            let Some(content_length) = parse_content_length(header) else {
                events.push(ClientEvent::ProtocolError {
                    message: "LSP message is missing a valid Content-Length header".into(),
                });
                self.input.clear();
                break;
            };
            let body_start = header_end + 4;
            let Some(body_end) = body_start.checked_add(content_length) else {
                events.push(ClientEvent::ProtocolError {
                    message: "LSP Content-Length overflowed the address space".into(),
                });
                self.input.clear();
                break;
            };
            if self.input.len() < body_end {
                break;
            }

            let body = self.input[body_start..body_end].to_vec();
            self.input.drain(..body_end);
            match serde_json::from_slice::<Value>(&body) {
                Ok(message) => self.route_message(message, &mut events),
                Err(error) => events.push(ClientEvent::ProtocolError {
                    message: format!("invalid JSON-RPC payload: {error}"),
                }),
            }
        }
        events
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

    fn route_message(&mut self, message: Value, events: &mut Vec<ClientEvent>) {
        if let Some(method) = message.get("method").and_then(Value::as_str) {
            events.push(ClientEvent::Notification {
                method: method.to_owned(),
                params: message.get("params").cloned().unwrap_or(Value::Null),
            });
            return;
        }

        let Some(id) = message
            .get("id")
            .and_then(Value::as_i64)
            .and_then(|id| i32::try_from(id).ok())
        else {
            events.push(ClientEvent::ProtocolError {
                message: "JSON-RPC response has no supported numeric id".into(),
            });
            return;
        };

        if let Some(error) = message.get("error") {
            self.pending.remove(&id);
            events.push(ClientEvent::Error {
                id: Some(id),
                code: error.get("code").and_then(Value::as_i64).unwrap_or(-32603),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown LSP error")
                    .to_owned(),
            });
            return;
        }

        let result = message.get("result").cloned().unwrap_or(Value::Null);
        match self.pending.remove(&id) {
            Some(RequestKind::Initialize) => events.push(ClientEvent::Initialized { id, result }),
            Some(RequestKind::SemanticTokens) => {
                events.push(ClientEvent::SemanticTokens { id, result })
            }
            Some(RequestKind::Shutdown) => events.push(ClientEvent::Shutdown { id }),
            None => events.push(ClientEvent::UnknownResponse { id, result }),
        }
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

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_content_length(header: &[u8]) -> Option<usize> {
    std::str::from_utf8(header)
        .ok()?
        .split("\r\n")
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())
                .flatten()
        })
}

#[cfg(target_arch = "wasm32")]
mod wasm_binding {
    use wasm_bindgen::prelude::*;

    use super::ClientCodec;

    /// A request's id together with the framed bytes to write to the server.
    #[wasm_bindgen(getter_with_clone)]
    pub struct Request {
        pub id: i32,
        pub frame: Vec<u8>,
    }

    /// The stateful LSP client codec; the host constructs one per server
    /// process. wasm-bindgen generates the JS class and marshals every
    /// string and byte buffer across the boundary.
    #[wasm_bindgen]
    #[derive(Default)]
    pub struct WasmClient {
        codec: ClientCodec,
    }

    #[wasm_bindgen]
    impl WasmClient {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self::default()
        }

        pub fn initialize(&mut self, root_uri: Option<String>, process_id: Option<u32>) -> Request {
            let (id, frame) = self.codec.initialize(root_uri.as_deref(), process_id);
            Request { id, frame }
        }

        pub fn initialized(&self) -> Vec<u8> {
            self.codec.initialized()
        }

        #[wasm_bindgen(js_name = didOpen)]
        pub fn did_open(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
            self.codec.did_open(uri, version, text)
        }

        #[wasm_bindgen(js_name = didChange)]
        pub fn did_change(&self, uri: &str, version: i32, text: &str) -> Vec<u8> {
            self.codec.did_change(uri, version, text)
        }

        #[wasm_bindgen(js_name = didClose)]
        pub fn did_close(&self, uri: &str) -> Vec<u8> {
            self.codec.did_close(uri)
        }

        #[wasm_bindgen(js_name = semanticTokens)]
        pub fn semantic_tokens(&mut self, uri: &str) -> Request {
            let (id, frame) = self.codec.semantic_tokens(uri);
            Request { id, frame }
        }

        pub fn cancel(&self, id: i32) -> Vec<u8> {
            self.codec.cancel(id)
        }

        pub fn shutdown(&mut self) -> Request {
            let (id, frame) = self.codec.shutdown();
            Request { id, frame }
        }

        pub fn exit(&self) -> Vec<u8> {
            self.codec.exit()
        }

        /// Feed bytes read from the server's stdout; returns the decoded
        /// `ClientEvent`s as a JS array. The json-compatible serializer keeps
        /// nested payloads as plain objects rather than ES2015 Maps.
        pub fn feed(&mut self, chunk: &[u8]) -> JsValue {
            use serde::Serialize;

            let events = self.codec.feed(chunk);
            events
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .expect("client events are serializable")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(frame: &[u8]) -> Value {
        let split = find_bytes(frame, b"\r\n\r\n").unwrap() + 4;
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

    #[test]
    fn fragmented_responses_are_buffered_and_routed() {
        let mut client = ClientCodec::default();
        let (id, _) = client.semantic_tokens("file:///demo.rr");
        let response = frame(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "data": [0, 0, 2, 13, 0] }
        }));
        let split = response.len() / 2;
        assert!(client.feed(&response[..split]).is_empty());
        assert_eq!(
            client.feed(&response[split..]),
            vec![ClientEvent::SemanticTokens {
                id,
                result: json!({ "data": [0, 0, 2, 13, 0] }),
            }]
        );
    }

    #[test]
    fn several_messages_in_one_chunk_preserve_order() {
        let mut client = ClientCodec::default();
        let (initialize_id, _) = client.initialize(None, None);
        let (tokens_id, _) = client.semantic_tokens("file:///demo.rr");
        let mut responses = frame(&json!({
            "jsonrpc": "2.0",
            "id": initialize_id,
            "result": { "capabilities": {} }
        }));
        responses.extend(frame(&json!({
            "jsonrpc": "2.0",
            "id": tokens_id,
            "result": null
        })));
        assert_eq!(
            client.feed(&responses),
            vec![
                ClientEvent::Initialized {
                    id: initialize_id,
                    result: json!({ "capabilities": {} }),
                },
                ClientEvent::SemanticTokens {
                    id: tokens_id,
                    result: Value::Null,
                },
            ]
        );
    }

    #[test]
    fn server_errors_clear_pending_requests() {
        let mut client = ClientCodec::default();
        let (id, _) = client.shutdown();
        let error = frame(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32603, "message": "broken" }
        }));
        assert_eq!(
            client.feed(&error),
            vec![ClientEvent::Error {
                id: Some(id),
                code: -32603,
                message: "broken".into(),
            }]
        );
    }

    #[test]
    fn malformed_headers_fail_soft() {
        let mut client = ClientCodec::default();
        assert!(matches!(
            client.feed(b"X-Test: no-length\r\n\r\n{}"),
            events if matches!(events.as_slice(), [ClientEvent::ProtocolError { .. }])
        ));
    }
}
