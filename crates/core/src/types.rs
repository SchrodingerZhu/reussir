use rustc_hash::FxHashMapRand;
use std::alloc::Layout;
use std::fmt::Debug;
use strum::{Display, EnumIter, IntoEnumIterator, IntoStaticStr};
use ustr::Ustr;

use crate::{Location, Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoStaticStr, Display, EnumIter)]
pub enum Primitive {
    #[strum(to_string = "bool")]
    Bool,
    #[strum(to_string = "i8")]
    I8,
    #[strum(to_string = "i16")]
    I16,
    #[strum(to_string = "i32")]
    I32,
    #[strum(to_string = "i64")]
    I64,
    #[strum(to_string = "i128")]
    I128,
    #[strum(to_string = "u8")]
    U8,
    #[strum(to_string = "u16")]
    U16,
    #[strum(to_string = "u32")]
    U32,
    #[strum(to_string = "u64")]
    U64,
    #[strum(to_string = "u128")]
    U128,
    #[strum(to_string = "f16")]
    F16,
    #[strum(to_string = "f32")]
    F32,
    #[strum(to_string = "f64")]
    F64,
    #[strum(to_string = "f128")]
    F128,
    #[strum(to_string = "char")]
    Char,
    #[strum(to_string = "str")]
    Str,
    #[strum(to_string = "unit")]
    Unit,
    #[strum(to_string = "never")]
    Never,
    #[strum(to_string = "region")]
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
    /// Normal RC-managed data. It is immutable but can be reused via compiler analysis.
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub capacity: Capacity,
    pub expr: TypeExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeBruijnIdx(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeExpr {
    Var(DeBruijnIdx),
    App(Path, Box<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub parameters: Box<[Type]>,
    pub return_type: Box<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpaqueType(pub Layout);

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
        f.debug_map()
            .entries(self.types.iter().map(|(k, v)| (k, v)))
            .finish()
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
    use crate::type_path;
    use super::*;

    #[test]
    fn test_type_database() {
        let db = TypeDatabase::new_with_primitives();
        println!("Type database: {:#?}", db);
        assert!(db.get(&type_path!("f128")).is_some());
        assert!(db.get(&type_path!("u64")).is_some());
        assert!(
            db.get(&type_path!("non_existent_type"))
                .is_none()
        );
    }
}
