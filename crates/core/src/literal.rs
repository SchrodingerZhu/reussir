use rustc_apfloat::ieee::{BFloat, Double, Half, Quad, Single};

#[derive(Clone, PartialEq, Eq, Hash)]

pub enum IntegerLiteral {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
}

impl std::fmt::Debug for IntegerLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegerLiteral::I8(val) => write!(f, "{val}i8"),
            IntegerLiteral::I16(val) => write!(f, "{val}i16"),
            IntegerLiteral::I32(val) => write!(f, "{val}i32"),
            IntegerLiteral::I64(val) => write!(f, "{val}i64"),
            IntegerLiteral::I128(val) => write!(f, "{val}i128"),
            IntegerLiteral::U8(val) => write!(f, "{val}u8"),
            IntegerLiteral::U16(val) => write!(f, "{val}u16"),
            IntegerLiteral::U32(val) => write!(f, "{val}u32"),
            IntegerLiteral::U64(val) => write!(f, "{val}u64"),
            IntegerLiteral::U128(val) => write!(f, "{val}u128"),
        }
    }
}

impl std::fmt::Display for IntegerLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegerLiteral::I8(val) => write!(f, "{val}"),
            IntegerLiteral::I16(val) => write!(f, "{val}"),
            IntegerLiteral::I32(val) => write!(f, "{val}"),
            IntegerLiteral::I64(val) => write!(f, "{val}"),
            IntegerLiteral::I128(val) => write!(f, "{val}"),
            IntegerLiteral::U8(val) => write!(f, "{val}"),
            IntegerLiteral::U16(val) => write!(f, "{val}"),
            IntegerLiteral::U32(val) => write!(f, "{val}"),
            IntegerLiteral::U64(val) => write!(f, "{val}"),
            IntegerLiteral::U128(val) => write!(f, "{val}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum FloatLiteral {
    BF16(BFloat),
    F16(Half),
    F32(Single),
    F64(Double),
    F128(Quad),
}

impl std::fmt::Debug for FloatLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatLiteral::BF16(val) => write!(f, "{val}bf16"),
            FloatLiteral::F16(val) => write!(f, "{val}f16"),
            FloatLiteral::F32(val) => write!(f, "{val}f32"),
            FloatLiteral::F64(val) => write!(f, "{val}f64"),
            FloatLiteral::F128(val) => write!(f, "{val}f128"),
        }
    }
}

impl std::fmt::Display for FloatLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatLiteral::BF16(val) => write!(f, "{val}"),
            FloatLiteral::F16(val) => write!(f, "{val}"),
            FloatLiteral::F32(val) => write!(f, "{val}"),
            FloatLiteral::F64(val) => write!(f, "{val}"),
            FloatLiteral::F128(val) => write!(f, "{val}"),
        }
    }
}
