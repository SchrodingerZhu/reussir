//! Translates the flat event stream into a lossless `cstree` green tree,
//! re-attaching trivia (whitespace and comments) between the meaningful
//! tokens.

use cstree::build::GreenNodeBuilder;
use cstree::green::GreenNode;
use cstree::interning::Interner;

use super::Event;
use crate::kind::SyntaxKind;
use crate::lexer::Token;

pub(crate) struct Sink<'s, 'i, I> {
    source: &'s str,
    /// All tokens, trivia included.
    tokens: &'s [Token],
    cursor: usize,
    events: Vec<Event>,
    builder: GreenNodeBuilder<'i, 'i, SyntaxKind, I>,
    /// Number of currently open nodes. Trivia may only be attached while a
    /// node is open, and the root must stay the single top-level element.
    depth: usize,
}

impl<'s, 'i, I> Sink<'s, 'i, I>
where
    I: Interner<cstree::interning::TokenKey>,
{
    pub(crate) fn new(
        source: &'s str,
        tokens: &'s [Token],
        events: Vec<Event>,
        interner: &'i mut I,
    ) -> Self {
        Sink {
            source,
            tokens,
            cursor: 0,
            events,
            builder: GreenNodeBuilder::with_interner(interner),
            depth: 0,
        }
    }

    pub(crate) fn finish(mut self) -> GreenNode {
        for idx in 0..self.events.len() {
            match std::mem::replace(&mut self.events[idx], Event::Token) {
                Event::Start { kind: None, .. } => {}
                Event::Start {
                    kind: Some(kind),
                    forward_parent,
                } => {
                    // Resolve the forward-parent chain: parents created by
                    // `precede` appear later in the stream but must open
                    // before this node.
                    let mut kinds = vec![kind];
                    let mut fp = forward_parent;
                    while let Some(parent_idx) = fp {
                        let parent_idx = parent_idx as usize;
                        match std::mem::replace(&mut self.events[parent_idx], Event::Token) {
                            Event::Start {
                                kind,
                                forward_parent,
                            } => {
                                if let Some(kind) = kind {
                                    kinds.push(kind);
                                }
                                fp = forward_parent;
                            }
                            _ => unreachable!("forward parent must be a start event"),
                        }
                        // Mark the parent event as consumed (tombstone).
                        self.events[parent_idx] = Event::Start {
                            kind: None,
                            forward_parent: None,
                        };
                    }
                    for kind in kinds.into_iter().rev() {
                        // Outermost first: each `precede` parent wraps
                        // everything started after it.
                        self.attach_trivia();
                        self.builder.start_node(kind);
                        self.depth += 1;
                    }
                }
                Event::Finish => {
                    if self.depth == 1 {
                        // The root is about to close: pull in any trailing
                        // trivia so the tree stays lossless.
                        self.attach_trivia();
                    }
                    self.builder.finish_node();
                    self.depth -= 1;
                }
                Event::Token => {
                    self.attach_trivia();
                    self.token();
                }
            }
        }
        let (green, _cache) = self.builder.finish();
        green
    }

    /// Trivia (and lexer error tokens, which the parser never consumes)
    /// attaches to the innermost open node; before a node opens this is the
    /// enclosing node, so leading whitespace does not become part of the
    /// new node. Nothing can be attached before the root is open.
    fn attach_trivia(&mut self) {
        if self.depth == 0 {
            return;
        }
        while let Some(token) = self.tokens.get(self.cursor) {
            if !token.kind.is_trivia() && token.kind != SyntaxKind::ErrorToken {
                break;
            }
            self.token();
        }
    }

    fn token(&mut self) {
        let token = self.tokens[self.cursor];
        self.builder.token(token.kind, token.text(self.source));
        self.cursor += 1;
    }
}
