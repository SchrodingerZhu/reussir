use std::cell::RefCell;

use crate::types::Type;
use crate::{Location, Path};
use frizbee::{IncrementalMatcher, Options};
use rustc_hash::FxHashMapRand;
use ustr::Ustr;
#[derive(Clone, Debug)]
pub struct Param {
    pub name: Option<Ustr>,
    pub location: Location,
    pub ty: Type,
}
#[derive(Clone, Debug)]
pub struct FunctionProto {
    pub path: Path,
    pub location: Location,
    pub name_location: Location,
    pub type_args: Option<Box<[Ustr]>>,
    pub params: Box<[Param]>,
    pub return_type: Type,
    pub has_region: bool,
    pub is_public: bool,
    pub is_opaque: bool,
}

struct FuzzyIndices {
    paths: Box<[Path]>,
    matcher: RefCell<IncrementalMatcher>,
}

pub struct FunctionDatabase {
    functions: FxHashMapRand<Path, FunctionProto>,
    fuzzy_indices: Option<FuzzyIndices>,
}

impl std::fmt::Debug for FunctionDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.functions)
    }
}

impl Default for FunctionDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionDatabase {
    pub fn new() -> Self {
        Self {
            functions: FxHashMapRand::default(),
            fuzzy_indices: None,
        }
    }

    pub fn build_fuzzy_indices(&mut self) {
        let mut names = Vec::new();
        let mut paths = Vec::new();
        for (path, proto) in &self.functions {
            names.push(proto.path.consolidate());
            paths.push(path.clone());
        }
        let matcher = IncrementalMatcher::new(&names);
        self.fuzzy_indices = Some(FuzzyIndices {
            paths: paths.into_boxed_slice(),
            matcher: RefCell::new(matcher),
        });
    }

    pub fn fuzzy_search(&self, query: &Path) -> Vec<&FunctionProto> {
        let string_search = query.consolidate();
        let Some(indices) = self.fuzzy_indices.as_ref() else {
            return Vec::new();
        };
        let options = Options {
            prefilter: true,
            max_typos: None,
            sort: true,
            min_score: string_search.len() as u16 * 6,
        };
        let result = indices
            .matcher
            .borrow_mut()
            .match_needle(&string_search, options);
        result
            .into_iter()
            .filter_map(|m| indices.paths.get(m.index_in_haystack as usize))
            .filter_map(|path| self.get(path))
            .collect()
    }

    pub fn get(&self, path: &Path) -> Option<&FunctionProto> {
        self.functions.get(path)
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.functions.contains_key(path)
    }

    pub fn add_function(&mut self, proto: FunctionProto) {
        self.functions.insert(proto.path.clone(), proto);
    }
}
