use rustc_hash::FxHashMapRand;
use std::alloc::Layout;
use std::fmt::Debug;
use strum::{Display, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};
use ustr::Ustr;

use crate::{CGContext, Location, Path};

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
    #[strum(serialize = "usize")]
    Usize,
    #[strum(serialize = "isize")]
    Isize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
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
pub enum Type {
    Atom {
        capability: Capability,
        expr: TypeExpr,
    },
    Arrow(Box<[Self]>, Box<Self>),
    Meta(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeExpr {
    pub path: Path,
    pub args: Option<Box<[Type]>>,
}

impl TypeExpr {
    pub fn as_primitive(&self) -> Option<Primitive> {
        if !self.path.prefix().is_empty() || !self.args.is_none() {
            return None;
        }
        Primitive::iter().find(|p| Into::<&str>::into(*p) == self.path.basename())
    }
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
    pub is_public: bool,
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
    pub type_args: Option<Box<[Ustr]>>,
    pub kind: RecordKind,
    pub is_public: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variant {
    pub location: Option<Location>,
    pub name: Ustr,
    pub body: Box<Compound>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Compound {
    Tuple(Option<Box<[Type]>>),
    Struct(Box<[(Ustr, Type)]>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordKind {
    Enum(Box<[Variant]>),
    Compound(Box<Compound>),
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

impl Type {
    pub fn is_float(&self) -> bool {
        match self {
            Type::Atom {
                capability: Capability::Default,
                expr: TypeExpr { path, args: None },
            } if path.prefix().is_empty() => [
                Primitive::BF16,
                Primitive::F16,
                Primitive::F32,
                Primitive::F64,
                Primitive::F128,
            ]
            .iter()
            .any(|&p| Into::<&'static str>::into(p) == path.basename()),
            _ => false,
        }
    }
    pub fn is_never(&self) -> bool {
        matches!(
            self,
            Type::Atom {
                capability: Capability::Default,
                expr: TypeExpr { path, args: None }
            } if path.basename() == Into::<&'static str>::into(Primitive::Never)
        )
    }
    pub fn is_integer(&self) -> bool {
        match self {
            Type::Atom {
                capability: Capability::Default,
                expr: TypeExpr { path, args: None },
            } if path.prefix().is_empty() => [
                Primitive::I8,
                Primitive::I16,
                Primitive::I32,
                Primitive::I64,
                Primitive::I128,
                Primitive::U8,
                Primitive::U16,
                Primitive::U32,
                Primitive::U64,
                Primitive::U128,
            ]
            .iter()
            .any(|&p| Into::<&'static str>::into(p) == path.basename()),
            _ => false,
        }
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
        is_public: bool,
        location: Option<Location>,
        type_args: Option<Box<[Ustr]>>,
        compound: Compound,
    ) {
        let concrete_type = ConcreteType::Record(Record {
            path: path.clone(),
            location,
            type_args,
            kind: RecordKind::Compound(Box::new(compound)),
            is_public,
        });
        self.types.insert(path, concrete_type);
    }

    pub fn add_enum_record(
        &mut self,
        path: Path,
        is_public: bool,
        location: Option<Location>,
        type_args: Option<Box<[Ustr]>>,
        variants: impl IntoIterator<Item = Variant>,
    ) {
        let concrete_type = ConcreteType::Record(Record {
            path: path.clone(),
            location,
            type_args,
            kind: RecordKind::Enum(variants.into_iter().collect()),
            is_public,
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

impl Primitive {
    pub fn codegen(&self, ctx: &mut CGContext) -> std::fmt::Result {
        match self {
            Primitive::Bool => write!(ctx.output, "i1"),
            Primitive::I8 => write!(ctx.output, "i8"),
            Primitive::I16 => write!(ctx.output, "i16"),
            Primitive::I32 => write!(ctx.output, "i32"),
            Primitive::I64 => write!(ctx.output, "i64"),
            Primitive::I128 => write!(ctx.output, "i128"),
            Primitive::U8 => write!(ctx.output, "u8"),
            Primitive::U16 => write!(ctx.output, "u16"),
            Primitive::U32 => write!(ctx.output, "u32"),
            Primitive::U64 => write!(ctx.output, "u64"),
            Primitive::U128 => write!(ctx.output, "u128"),
            Primitive::BF16 => write!(ctx.output, "bf16"),
            Primitive::F16 => write!(ctx.output, "f16"),
            Primitive::F32 => write!(ctx.output, "f32"),
            Primitive::F64 => write!(ctx.output, "f64"),
            Primitive::F128 => write!(ctx.output, "f128"),
            Primitive::Char => write!(ctx.output, "!reussir.char"),
            Primitive::Str => write!(ctx.output, "!reussir.str"),
            Primitive::Unit => write!(ctx.output, "none"),
            Primitive::Never => write!(ctx.output, "<todo>"), // TODO: Adjust for actual never type
            Primitive::Region => write!(ctx.output, "<todo>"), // TODO: Adjust for actual region
            Primitive::Usize => write!(ctx.output, "index"),
            Primitive::Isize => write!(ctx.output, "<todo>"),
        }
    }
}

impl Type {
    pub fn codegen(&self, ctx: &mut CGContext) -> std::fmt::Result {
        match self {
            Type::Atom {
                capability: Capability::Default,
                expr,
            } => {
                if let Some(primitive) = expr.as_primitive() {
                    primitive.codegen(ctx)
                } else {
                    todo!("Handle non-primitive atom types")
                }
            }
            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path;

    #[test]
    fn test_type_database() {
        let db = TypeDatabase::new_with_primitives();
        println!("Type database: {:#?}", db);
        assert!(db.get(&path!("f128")).is_some());
        assert!(db.get(&path!("u64")).is_some());
        assert!(db.get(&path!("non_existent_type")).is_none());
    }
}
