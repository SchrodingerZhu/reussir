//! The recursive-descent parser core.
//!
//! The parser is event driven: grammar code emits a flat stream of
//! [`Event`]s instead of building the tree directly. This buys two things
//! that a tree-building parser cannot easily provide:
//!
//! * cheap speculation — ambiguities like `a < b` (comparison) versus
//!   `a<b>(c)` (instantiation) are resolved by trial parsing followed by
//!   truncating the event buffer;
//! * structural error recovery — error nodes and skipped tokens are
//!   ordinary events, so a best-effort tree is always produced.
//!
//! Grammar rules live in [`grammar`] (statements, types, patterns) and
//! [`expr`] (Pratt expression parsing) as methods on [`Parser`]. Deeply
//! recursive rules are guarded with [`stacker::maybe_grow`], so pathological
//! inputs spill to a heap-allocated stack instead of overflowing.

pub(crate) mod expr;
pub(crate) mod grammar;
pub(crate) mod sink;

use crate::diagnostics::ParseError;
use crate::kind::SyntaxKind;
use crate::lexer::Token;

/// Red zone / growth parameters for [`stacker::maybe_grow`].
pub(crate) const STACK_RED_ZONE: usize = 128 * 1024;
pub(crate) const STACK_GROW: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum Event {
    /// `kind == None` is a tombstone: an abandoned or forward-parented
    /// start.
    Start {
        kind: Option<SyntaxKind>,
        forward_parent: Option<u32>,
    },
    Finish,
    /// Consume the next non-trivia token.
    Token,
}

pub(crate) struct Parser<'s> {
    source: &'s str,
    /// Non-trivia tokens only; the sink re-interleaves trivia.
    tokens: Vec<Token>,
    pos: usize,
    pub(crate) events: Vec<Event>,
    pub(crate) errors: Vec<ParseError>,
}

/// A snapshot of the parser state, for speculative parsing.
pub(crate) struct Snapshot {
    events: usize,
    pos: usize,
    errors: usize,
}

impl<'s> Parser<'s> {
    pub(crate) fn new(source: &'s str, tokens: Vec<Token>) -> Self {
        Parser {
            source,
            tokens,
            pos: 0,
            events: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    pub(crate) fn nth(&self, n: usize) -> SyntaxKind {
        self.tokens
            .get(self.pos + n)
            .map_or(SyntaxKind::Eof, |t| t.kind)
    }

    pub(crate) fn current_text(&self) -> &'s str {
        self.nth_text(0)
    }

    pub(crate) fn nth_text(&self, n: usize) -> &'s str {
        self.tokens
            .get(self.pos + n)
            .map_or("", |t| t.text(self.source))
    }

    /// The byte span of the current token (an empty span at the end of the
    /// last token for EOF).
    pub(crate) fn current_span(&self) -> (u32, u32) {
        match self.tokens.get(self.pos) {
            Some(t) => t.range,
            None => {
                let end = self.tokens.last().map_or(0, |t| t.range.1);
                (end, end)
            }
        }
    }

    pub(crate) fn at(&self, kind: SyntaxKind) -> bool {
        self.current() == kind
    }

    /// An adjacent `<<`/`>>` pair glued into a shift operator, or `None`.
    /// The lexer keeps angle tokens single so the type grammar can close
    /// nested generics one level at a time; a shift operator therefore
    /// exists only as two angle tokens with nothing between them, detected
    /// where an infix operator is expected. Returns the angle kind.
    pub(crate) fn glued_shift(&self) -> Option<SyntaxKind> {
        let kind = self.current();
        if !matches!(kind, SyntaxKind::LAngle | SyntaxKind::RAngle) || self.nth(1) != kind {
            return None;
        }
        let a = self.tokens.get(self.pos)?;
        let b = self.tokens.get(self.pos + 1)?;
        (a.range.1 == b.range.0).then_some(kind)
    }

    pub(crate) fn at_eof(&self) -> bool {
        self.at(SyntaxKind::Eof)
    }

    pub(crate) fn at_ident_like(&self) -> bool {
        self.current().is_ident_like()
    }

    /// Is the current token an identifier with exactly the given text
    /// (a contextual keyword such as `trampoline` or `shared`)?
    pub(crate) fn at_ctx_kw(&self, text: &str) -> bool {
        self.at_ident_like() && self.current_text() == text
    }

    pub(crate) fn bump(&mut self) {
        assert!(!self.at_eof(), "bump at EOF");
        self.pos += 1;
        self.events.push(Event::Token);
    }

    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            self.error(format!(
                "expected {}, found {}",
                kind.describe(),
                self.current().describe()
            ));
            false
        }
    }

    /// Expect any identifier-like token (see [`SyntaxKind::is_ident_like`]).
    pub(crate) fn expect_ident(&mut self, what: &str) -> bool {
        if self.at_ident_like() {
            self.bump();
            true
        } else {
            self.error(format!(
                "expected {what}, found {}",
                self.current().describe()
            ));
            false
        }
    }

    /// Record an error at the current token.
    pub(crate) fn error(&mut self, message: impl Into<String>) {
        let span = self.current_span();
        self.errors.push(ParseError {
            span,
            message: message.into(),
        });
    }

    pub(crate) fn start(&mut self) -> Marker {
        let pos = self.events.len() as u32;
        self.events.push(Event::Start {
            kind: None,
            forward_parent: None,
        });
        Marker { pos }
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            events: self.events.len(),
            pos: self.pos,
            errors: self.errors.len(),
        }
    }

    /// Did any error occur since the snapshot was taken?
    pub(crate) fn errored_since(&self, snapshot: &Snapshot) -> bool {
        self.errors.len() > snapshot.errors
    }

    pub(crate) fn rollback(&mut self, snapshot: Snapshot) {
        self.events.truncate(snapshot.events);
        self.pos = snapshot.pos;
        self.errors.truncate(snapshot.errors);
    }

    pub(crate) fn finish(self) -> (Vec<Event>, Vec<ParseError>) {
        (self.events, self.errors)
    }
}

pub(crate) struct Marker {
    pos: u32,
}

impl Marker {
    pub(crate) fn complete(self, p: &mut Parser<'_>, kind: SyntaxKind) -> CompletedMarker {
        match &mut p.events[self.pos as usize] {
            Event::Start { kind: slot, .. } => *slot = Some(kind),
            _ => unreachable!("marker does not point at a start event"),
        }
        p.events.push(Event::Finish);
        CompletedMarker { pos: self.pos }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CompletedMarker {
    pos: u32,
}

impl CompletedMarker {
    /// Create a new marker that will wrap this completed node, used to
    /// build left-recursive structures (binary operators, call chains).
    pub(crate) fn precede(self, p: &mut Parser<'_>) -> Marker {
        let new = p.start();
        match &mut p.events[self.pos as usize] {
            Event::Start { forward_parent, .. } => *forward_parent = Some(new.pos),
            _ => unreachable!("completed marker does not point at a start event"),
        }
        new
    }
}
