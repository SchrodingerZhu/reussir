//! async-lsp routing and whole-document synchronization.

use std::ops::ControlFlow;
use std::sync::Arc;

use async_lsp::lsp_types::{
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    InitializeParams, InitializeResult, PositionEncodingKind, SemanticTokens,
    SemanticTokensFullOptions, SemanticTokensOptions, SemanticTokensParams, SemanticTokensResult,
    SemanticTokensServerCapabilities, ServerCapabilities, ServerInfo, TextDocumentSyncCapability,
    TextDocumentSyncKind, TextDocumentSyncOptions, Url, WorkDoneProgressOptions, notification,
    request,
};
use async_lsp::router::Router;
use dashmap::DashMap;

use crate::semantic::{legend_token_modifiers, legend_token_types, semantic_tokens};

#[derive(Clone, Debug)]
struct Document {
    version: i32,
    text: Arc<str>,
}

#[derive(Default)]
pub(crate) struct ServerState {
    documents: Arc<DashMap<Url, Document>>,
}

impl ServerState {
    pub(crate) fn router() -> Router<Self> {
        let mut router = Router::new(Self::default());
        router
            .request::<request::Initialize, _>(|_, params| async move { Ok(initialize(params)) })
            .request::<request::Shutdown, _>(|_, ()| async move { Ok(()) })
            .request::<request::SemanticTokensFullRequest, _>(|state, params| {
                let documents = Arc::clone(&state.documents);
                async move {
                    let Some(text) = snapshot(&documents, &params) else {
                        return Ok(None);
                    };
                    Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
                        result_id: None,
                        data: semantic_tokens(&text),
                    })))
                }
            })
            .notification::<notification::Initialized>(|_, _| ControlFlow::Continue(()))
            .notification::<notification::DidOpenTextDocument>(|state, params| {
                state.did_open(params);
                ControlFlow::Continue(())
            })
            .notification::<notification::DidChangeTextDocument>(|state, params| {
                state.did_change(params);
                ControlFlow::Continue(())
            })
            .notification::<notification::DidCloseTextDocument>(|state, params| {
                state.did_close(params);
                ControlFlow::Continue(())
            })
            .notification::<notification::Exit>(|_, ()| ControlFlow::Continue(()));
        router
    }

    fn did_open(&mut self, params: DidOpenTextDocumentParams) {
        let document = params.text_document;
        self.documents.insert(
            document.uri,
            Document {
                version: document.version,
                text: Arc::from(document.text),
            },
        );
    }

    fn did_change(&mut self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let Some(mut document) = self.documents.get_mut(&uri) else {
            tracing::warn!(%uri, version, "ignoring change for unopened document");
            return;
        };
        if version <= document.version {
            tracing::warn!(%uri, version, current = document.version, "ignoring stale document change");
            return;
        }
        let mut changes = params.content_changes.into_iter();
        let Some(change) = changes.next() else {
            tracing::warn!(%uri, version, "ignoring empty full-document change");
            return;
        };
        if changes.next().is_some() || change.range.is_some() || change.range_length.is_some() {
            tracing::warn!(%uri, version, "ignoring change that is not one whole-document replacement");
            return;
        }
        document.version = version;
        document.text = Arc::from(change.text);
    }

    fn did_close(&mut self, params: DidCloseTextDocumentParams) {
        self.documents.remove(&params.text_document.uri);
    }
}

fn snapshot(documents: &DashMap<Url, Document>, params: &SemanticTokensParams) -> Option<Arc<str>> {
    documents
        .get(&params.text_document.uri)
        .map(|document| Arc::clone(&document.text))
}

pub(crate) fn initialize(_params: InitializeParams) -> InitializeResult {
    InitializeResult {
        capabilities: ServerCapabilities {
            position_encoding: Some(PositionEncodingKind::UTF16),
            text_document_sync: Some(TextDocumentSyncCapability::Options(
                TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::FULL),
                    ..TextDocumentSyncOptions::default()
                },
            )),
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                    work_done_progress_options: WorkDoneProgressOptions {
                        work_done_progress: None,
                    },
                    legend: async_lsp::lsp_types::SemanticTokensLegend {
                        token_types: legend_token_types(),
                        token_modifiers: legend_token_modifiers(),
                    },
                    range: None,
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                }),
            ),
            ..ServerCapabilities::default()
        },
        server_info: Some(ServerInfo {
            name: "reussir-lsp".into(),
            version: Some(env!("CARGO_PKG_VERSION").into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use async_lsp::lsp_types::{
        ClientCapabilities, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
        DidOpenTextDocumentParams, InitializeParams, Position, Range, SemanticTokensParams,
        TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem,
        VersionedTextDocumentIdentifier, WorkDoneProgressParams,
    };

    use super::*;

    fn uri() -> Url {
        Url::parse("file:///test.rr").unwrap()
    }

    fn semantic_params() -> SemanticTokensParams {
        SemanticTokensParams {
            text_document: TextDocumentIdentifier { uri: uri() },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        }
    }

    #[test]
    fn capabilities_are_full_only() {
        let result = initialize(InitializeParams {
            capabilities: ClientCapabilities::default(),
            ..InitializeParams::default()
        });
        assert_eq!(
            result.capabilities.position_encoding,
            Some(PositionEncodingKind::UTF16)
        );
        assert!(matches!(
            result.capabilities.text_document_sync,
            Some(TextDocumentSyncCapability::Options(
                TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::FULL),
                    ..
                }
            ))
        ));
        let Some(SemanticTokensServerCapabilities::SemanticTokensOptions(options)) =
            result.capabilities.semantic_tokens_provider
        else {
            panic!("semantic tokens must use static options")
        };
        assert!(options.range.is_none());
        assert!(matches!(
            options.full,
            Some(SemanticTokensFullOptions::Bool(true))
        ));
        assert_eq!(options.legend.token_types, legend_token_types());
    }

    #[test]
    fn store_accepts_only_newer_whole_document_replacements() {
        let mut state = ServerState::default();
        state.did_open(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: uri(),
                language_id: "reussir".into(),
                version: 1,
                text: "fn old() {}".into(),
            },
        });
        let in_flight_snapshot = snapshot(&state.documents, &semantic_params()).unwrap();
        state.did_change(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: uri(),
                version: 2,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: "fn new() {}".into(),
            }],
        });
        assert_eq!(
            &*snapshot(&state.documents, &semantic_params()).unwrap(),
            "fn new() {}"
        );
        assert_eq!(&*in_flight_snapshot, "fn old() {}");

        state.did_change(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: uri(),
                version: 1,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: "fn stale() {}".into(),
            }],
        });
        assert_eq!(
            &*snapshot(&state.documents, &semantic_params()).unwrap(),
            "fn new() {}"
        );

        state.did_change(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: uri(),
                version: 3,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: Some(Range::new(Position::new(0, 0), Position::new(0, 2))),
                range_length: Some(2),
                text: "fn".into(),
            }],
        });
        assert_eq!(
            &*snapshot(&state.documents, &semantic_params()).unwrap(),
            "fn new() {}"
        );

        state.did_close(DidCloseTextDocumentParams {
            text_document: TextDocumentIdentifier { uri: uri() },
        });
        assert!(snapshot(&state.documents, &semantic_params()).is_none());
    }
}
