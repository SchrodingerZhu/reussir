use std::alloc::Layout;

use ustr::Ustr;

use crate::Path;


pub enum Primitive {
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F16,
    F32,
    F64,
    F128,
    Char,
    Str,
    Unit,
    Never,
}

pub struct FunctionType {
    pub parameters: Box<[Type]>,
    pub return_type: Box<Type>,
}

pub struct OpaqueType(pub Layout);

pub enum Type {
    Primitive(Primitive),
    Record(Record),
    Array(Array),
    Function(FunctionType),
    Closure(FunctionType),
    Opaque(OpaqueType),
}

pub struct Array {
    pub element_type: Box<Type>,
    pub length: usize,
}

pub struct Record {
    pub path: Path,
    pub record_type: RecordType,
}


pub struct Variant {
    pub name: String,
    pub fields: Box<[Compound]>,
}

pub enum Compound {
    Tuple(Box<[Type]>),
    Struct(Box<[(Ustr, Type)]>)
}

pub enum RecordType {
    Enum(Box<[Variant]>),
    Compound(Box<[Compound]>),
}