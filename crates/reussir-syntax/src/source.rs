//! Multi-file source management: a [`SourceCache`] owns every source file of a
//! compilation (or REPL session), keyed by [`FileId`].
//!
//! Spans stay *file-local* byte offsets (`ResolvedNode::text_range` of that
//! file's parse); a `FileId` identifies which file a span indexes. Items never
//! straddle files, so the compiler carries one `FileId` per item/diagnostic
//! rather than widening every span.
//!
//! The cache implements [`ariadne::Cache`], so diagnostics render straight from
//! it (per-file caret reports), and it doubles as the line index the code
//! generator resolves `FileLineColLoc` debug locations against — one authority
//! for "what does offset N of file F mean".

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ariadne::Source;

/// Identifies a file in a [`SourceCache`]. Allocated densely in insertion
/// order; the first file added is [`FileId::ROOT`] — the crate root (or the
/// single input of a one-file compile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(u32);

impl FileId {
    /// The first file added to a cache: the crate root / primary input.
    pub const ROOT: FileId = FileId(0);

    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The id of the file at position `index` (files are dense, in insertion
    /// order) — for reconstructing references from a serialized file table.
    pub fn from_index(index: u32) -> FileId {
        FileId(index)
    }
}

struct SourceFile {
    /// Display name: the path as typed for an on-disk file, or a synthetic
    /// name like `<stdin>` / `<repl>` for a virtual one.
    name: String,
    /// The on-disk path, when the file is real (feeds DWARF file attributes).
    path: Option<PathBuf>,
    /// The text plus `ariadne`'s line index over it.
    source: Source<String>,
    /// `false` for a placeholder whose content could not be obtained (a
    /// virtual or missing file referenced by a re-ingested IR dump): spans
    /// into it are structurally valid but cannot resolve to source text, so
    /// diagnostics degrade to plain lines and debug locations to `unknown`.
    available: bool,
    /// `byte_to_char[b]` = number of chars strictly before byte `b`; built on
    /// first use (only diagnostics need it — `ariadne` indexes by char).
    byte_to_char: OnceLock<Vec<u32>>,
}

/// All source files of one compilation, in insertion order.
#[derive(Default)]
pub struct SourceCache {
    files: Vec<SourceFile>,
}

impl SourceCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cache holding a single virtual file — for rendering diagnostics
    /// against an ad-hoc snippet (tests, REPL parse errors).
    pub fn single(name: impl Into<String>, source: impl Into<String>) -> Self {
        let mut cache = Self::new();
        cache.add_virtual(name, source);
        cache
    }

    /// Register an on-disk file's contents. The display name is the path as
    /// given (not canonicalized), matching how the user referred to it.
    pub fn add_file(&mut self, path: impl Into<PathBuf>, source: impl Into<String>) -> FileId {
        let path = path.into();
        let name = path.display().to_string();
        self.push(SourceFile {
            name,
            path: Some(path),
            source: Source::from(source.into()),
            available: true,
            byte_to_char: OnceLock::new(),
        })
    }

    /// Register in-memory content with a synthetic display name (`<stdin>`,
    /// `<repl>`, …).
    pub fn add_virtual(&mut self, name: impl Into<String>, source: impl Into<String>) -> FileId {
        self.push(SourceFile {
            name: name.into(),
            path: None,
            source: Source::from(source.into()),
            available: true,
            byte_to_char: OnceLock::new(),
        })
    }

    /// Register a name-only placeholder for a file whose content cannot be
    /// obtained (a virtual or since-deleted file referenced by an IR dump).
    /// Spans into it stay attributed to the name but render without source
    /// context and resolve to no debug location.
    pub fn add_unavailable(&mut self, name: impl Into<String>) -> FileId {
        self.push(SourceFile {
            name: name.into(),
            path: None,
            source: Source::from(String::new()),
            available: false,
            byte_to_char: OnceLock::new(),
        })
    }

    /// Whether the file's content is present (see
    /// [`add_unavailable`](Self::add_unavailable)).
    pub fn is_available(&self, id: FileId) -> bool {
        self.file(id).available
    }

    fn push(&mut self, file: SourceFile) -> FileId {
        let id = FileId(u32::try_from(self.files.len()).expect("more than u32::MAX source files"));
        self.files.push(file);
        id
    }

    fn file(&self, id: FileId) -> &SourceFile {
        &self.files[id.index()]
    }

    /// The file's text.
    pub fn source(&self, id: FileId) -> &str {
        self.file(id).source.text()
    }

    /// The file's display name.
    pub fn name(&self, id: FileId) -> &str {
        &self.file(id).name
    }

    /// The on-disk path, if the file is real.
    pub fn path(&self, id: FileId) -> Option<&Path> {
        self.file(id).path.as_deref()
    }

    /// The file name component for the DWARF `DIFile` basename. A virtual file
    /// uses its display name.
    pub fn basename(&self, id: FileId) -> Cow<'_, str> {
        match self.path(id) {
            Some(p) => p
                .file_name()
                .map_or(Cow::Borrowed(""), |n| n.to_string_lossy()),
            None => Cow::Borrowed(self.name(id)),
        }
    }

    /// The parent directory for the DWARF `DIFile` directory (empty for a
    /// virtual file).
    pub fn directory(&self, id: FileId) -> Cow<'_, str> {
        match self.path(id) {
            Some(p) => p
                .parent()
                .map_or(Cow::Borrowed(""), |d| d.to_string_lossy()),
            None => Cow::Borrowed(""),
        }
    }

    /// The 1-based `(line, column)` of byte `offset`, for `FileLineColLoc`
    /// debug locations (`ariadne` reports both zero-indexed).
    pub fn line_col(&self, id: FileId, offset: usize) -> (usize, usize) {
        match self.file(id).source.get_byte_line(offset) {
            Some((_, line, col)) => (line + 1, col + 1),
            None => (1, 1),
        }
    }

    /// `byte_to_char[b]` = chars strictly before byte `b`, built on first use.
    fn char_map(&self, id: FileId) -> &[u32] {
        let file = self.file(id);
        file.byte_to_char.get_or_init(|| {
            let text = file.source.text();
            let mut map = vec![0u32; text.len() + 1];
            let mut chars = 0u32;
            for (idx, c) in text.char_indices() {
                map[idx..idx + c.len_utf8()].fill(chars);
                chars += 1;
            }
            map[text.len()] = chars;
            map
        })
    }

    /// Convert a byte-offset span into the char-offset span `ariadne` indexes
    /// by. Out-of-range offsets clamp to the end of the file.
    pub fn char_span(&self, id: FileId, span: (u32, u32)) -> (u32, u32) {
        let map = self.char_map(id);
        let of = |byte: u32| map[(byte as usize).min(map.len() - 1)];
        (of(span.0), of(span.1))
    }

    /// The file's length in characters (Unicode scalars).
    pub fn char_len(&self, id: FileId) -> u32 {
        *self.char_map(id).last().expect("map covers offset 0")
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// All file ids, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = FileId> {
        (0..self.files.len() as u32).map(FileId)
    }
}

/// A standalone byte→char offset table over one source string, for consumers
/// that need character offsets without a whole [`SourceCache`] (e.g. the
/// surface-verify JSON emitter, whose wire format records char offsets).
pub struct CharMap {
    /// `byte_to_char[b]` is the number of chars strictly before byte `b`.
    byte_to_char: Vec<u32>,
}

impl CharMap {
    pub fn new(source: &str) -> Self {
        let mut byte_to_char = vec![0u32; source.len() + 1];
        let mut chars = 0u32;
        for (idx, c) in source.char_indices() {
            byte_to_char[idx..idx + c.len_utf8()].fill(chars);
            chars += 1;
        }
        byte_to_char[source.len()] = chars;
        Self { byte_to_char }
    }

    pub fn char_offset(&self, byte: u32) -> u32 {
        self.byte_to_char[byte as usize]
    }

    pub fn char_span(&self, span: (u32, u32)) -> (u32, u32) {
        (self.char_offset(span.0), self.char_offset(span.1))
    }
}

/// Feed `ariadne` reports straight from the cache: `fetch` hands out the
/// per-file line index, `display` the file's name. Implemented on the shared
/// reference so one cache serves many reports without cloning sources.
impl ariadne::Cache<FileId> for &SourceCache {
    type Storage = String;

    fn fetch(&mut self, id: &FileId) -> Result<&Source<String>, impl fmt::Debug> {
        self.files
            .get(id.index())
            .map(|f| &f.source)
            .ok_or_else(|| format!("no file with id {id:?} in the source cache"))
    }

    fn display<'a>(&self, id: &'a FileId) -> Option<impl fmt::Display + 'a> {
        Some(self.files.get(id.index())?.name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_dense_and_root_is_first() {
        let mut cache = SourceCache::new();
        let a = cache.add_file("dir/a.rr", "fn a() {}\n");
        let b = cache.add_virtual("<repl>", "1 + 1\n");
        assert_eq!(a, FileId::ROOT);
        assert_eq!(b.index(), 1);
        assert_eq!(cache.name(a), "dir/a.rr");
        assert_eq!(cache.name(b), "<repl>");
        assert_eq!(cache.source(b), "1 + 1\n");
        assert_eq!(cache.ids().collect::<Vec<_>>(), vec![a, b]);
    }

    #[test]
    fn dwarf_components_split_real_paths_and_pass_virtual_names_through() {
        let mut cache = SourceCache::new();
        let real = cache.add_file("pkg/src/lib.rr", "");
        let virt = cache.add_virtual("<stdin>", "");
        assert_eq!(cache.basename(real), "lib.rr");
        assert_eq!(cache.directory(real), "pkg/src");
        assert_eq!(cache.basename(virt), "<stdin>");
        assert_eq!(cache.directory(virt), "");
    }

    #[test]
    fn line_col_is_one_based_per_file() {
        let mut cache = SourceCache::new();
        let a = cache.add_virtual("a", "one\ntwo\n");
        let b = cache.add_virtual("b", "x\n");
        assert_eq!(cache.line_col(a, 0), (1, 1));
        assert_eq!(cache.line_col(a, 4), (2, 1));
        assert_eq!(cache.line_col(a, 6), (2, 3));
        // Files are independent coordinate systems.
        assert_eq!(cache.line_col(b, 0), (1, 1));
    }

    #[test]
    fn char_span_counts_multibyte_chars_once() {
        let mut cache = SourceCache::new();
        // 'é' is two bytes; the byte span after it lands two chars in.
        let f = cache.add_virtual("u", "é=1");
        assert_eq!(cache.char_span(f, (0, 2)), (0, 1));
        assert_eq!(cache.char_span(f, (2, 3)), (1, 2));
        // Out-of-range clamps instead of panicking.
        assert_eq!(cache.char_span(f, (100, 200)), (3, 3));
    }
}
