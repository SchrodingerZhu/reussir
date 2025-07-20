use rustc_hash::FxHashMapRand;
use std::alloc::Layout;
use std::fmt::Debug;
use strum::{Display, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};
use ustr::Ustr;

use crate::{Location, Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoStaticStr, Display, EnumIter, EnumString)]
pub enum Primitive {
    #[strum(serialize = "bool")]
    Bool,
    #[strum(serialize = "i8")]
    I8,
    #[strum(serialize = "i16")]
    I16,
    #[strum(serialize = "i32")]
    I32,
    #[strum(serialize = "i64")]
    I64,
    #[strum(serialize = "i128")]
    I128,
    #[strum(serialize = "u8")]
    U8,
    #[strum(serialize = "u16")]
    U16,
    #[strum(serialize = "u32")]
    U32,
    #[strum(serialize = "u64")]
    U64,
    #[strum(serialize = "u128")]
    U128,
    #[strum(serialize = "bf16")]
    BF16,
    #[strum(serialize = "f16")]
    F16,
    #[strum(serialize = "f32")]
    F32,
    #[strum(serialize = "f64")]
    F64,
    #[strum(serialize = "f128")]
    F128,
    #[strum(serialize = "char")]
    Char,
    #[strum(serialize = "str")]
    Str,
    #[strum(serialize = "unit")]
    Unit,
    #[strum(serialize = "never")]
    Never,
    #[strum(serialize = "region")]
    Region,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capacity {
    /// Data is stored as a plain value
    Value,
    /// Data is behind a rigid reference. That is, the data is frozen and cannot be mutated.
    Rigid,
    /// Data is behind a mutable reference. A flex reference must be inside a region. It cannot be materialized.
    Flex,
    /// An assignable field for flex references. A field can be mutated if the object it resides is behind a flex reference.
    Field,
    /// Normal RC-managed data or primitive types. It is immutable but can be reused via compiler analysis.
    Default,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub capacity: Capacity,
    pub expr: TypeExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeExpr {
    Var(usize),
    App(Path, Box<[Type]>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub parameters: Box<[Type]>,
    pub return_type: Box<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpaqueType {
    pub path: Path,
    pub location: Option<Location>,
    pub layout: Layout,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConcreteType {
    Primitive(Primitive),
    Record(Record),
    Array(Array),
    Function(FunctionType),
    Closure(FunctionType),
    Opaque(OpaqueType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Array {
    pub element_type: Box<Type>,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    pub path: Path,
    pub location: Option<Location>,
    pub num_args: usize,
    pub kind: RecordKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variant {
    pub name: String,
    pub fields: Box<[Compound]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Compound {
    Tuple(Box<[Type]>),
    Struct(Box<[(Ustr, Type)]>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordKind {
    Enum(Box<[Variant]>),
    Compound(Box<[Compound]>),
}

pub struct TypeDatabase {
    types: FxHashMapRand<Path, ConcreteType>,
}

impl Debug for TypeDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.types.iter()).finish()
    }
}
impl Default for TypeDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeDatabase {
    pub fn new() -> Self {
        TypeDatabase {
            types: FxHashMapRand::default(),
        }
    }

    pub fn new_with_primitives() -> Self {
        let mut db = Self::new();
        for primitive in Primitive::iter() {
            let path = Path::new(ustr::Ustr::from(primitive.into()), []);
            let concrete_type = ConcreteType::Primitive(primitive);
            db.types.insert(path, concrete_type);
        }
        db
    }

    pub fn get(&self, path: &Path) -> Option<&ConcreteType> {
        self.types.get(path)
    }

    pub fn add_compound_record(
        &mut self,
        path: Path,
        location: Option<Location>,
        num_args: usize,
        fields: impl IntoIterator<Item = Compound>,
    ) {
        let concrete_type = ConcreteType::Record(Record {
            path: path.clone(),
            location,
            num_args,
            kind: RecordKind::Compound(fields.into_iter().collect()),
        });
        self.types.insert(path, concrete_type);
    }

    pub fn add_enum_record(
        &mut self,
        path: Path,
        location: Option<Location>,
        num_args: usize,
        variants: impl IntoIterator<Item = Variant>,
    ) {
        let concrete_type = ConcreteType::Record(Record {
            path: path.clone(),
            location,
            num_args,
            kind: RecordKind::Enum(variants.into_iter().collect()),
        });
        self.types.insert(path, concrete_type);
    }

    pub fn get_location(&self, path: &Path) -> Option<Location> {
        self.types.get(path).and_then(|ct| match ct {
            ConcreteType::Record(record) => record.location,
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_path;

    #[test]
    fn test_type_database() {
        let db = TypeDatabase::new_with_primitives();
        println!("Type database: {:#?}", db);
        assert!(db.get(&type_path!("f128")).is_some());
        assert!(db.get(&type_path!("u64")).is_some());
        assert!(db.get(&type_path!("non_existent_type")).is_none());
    }
}
