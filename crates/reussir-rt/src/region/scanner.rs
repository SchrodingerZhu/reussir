use std::ptr::NonNull;

use crate::region::Header;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    /// Load and yield current position as a header pointer,
    /// then advance the instr pointer by 1.
    Field,
    /// Load current position as a variant index of the given byte width
    /// (1, 2, or 4 — the record's minimal tag width), then advance the
    /// instr pointer by (index + 1).
    Variant(u32),
    /// Scanner ended.
    End,
    /// Advance the cursor by the given number of bytes.
    Advance(u32),
    /// Advance the instruction pointer by a given amount
    Jump(u32),
}

const END_VALUE: i32 = 0;
const VARIANT8_VALUE: i32 = -1;
const FIELD_VALUE: i32 = -2;
const VARIANT16_VALUE: i32 = -3;
const VARIANT32_VALUE: i32 = -4;
const JUMP_BASE: i32 = -5;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PackedInstr(i32);

impl std::fmt::Debug for PackedInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let instr: Instruction = (*self).into();
        write!(f, "{instr:?}")
    }
}

impl From<PackedInstr> for Instruction {
    fn from(value: PackedInstr) -> Self {
        match value.0 {
            END_VALUE => Instruction::End,
            VARIANT8_VALUE => Instruction::Variant(1),
            FIELD_VALUE => Instruction::Field,
            VARIANT16_VALUE => Instruction::Variant(2),
            VARIANT32_VALUE => Instruction::Variant(4),
            val if val > 0 => Instruction::Advance(val as u32),
            val => Instruction::Jump((val - JUMP_BASE).unsigned_abs()),
        }
    }
}

impl From<Instruction> for PackedInstr {
    fn from(value: Instruction) -> Self {
        value.pack()
    }
}

impl Instruction {
    pub const fn pack(self) -> PackedInstr {
        match self {
            Instruction::End => PackedInstr(END_VALUE),
            Instruction::Variant(width) => match width {
                1 => PackedInstr(VARIANT8_VALUE),
                2 => PackedInstr(VARIANT16_VALUE),
                4 => PackedInstr(VARIANT32_VALUE),
                _ => panic!("unsupported variant tag width"),
            },
            Instruction::Field => PackedInstr(FIELD_VALUE),
            Instruction::Advance(bytes) => {
                debug_assert!(bytes > 0);
                PackedInstr(bytes as i32)
            }
            Instruction::Jump(instrs) => {
                debug_assert!(instrs > 0);
                PackedInstr(JUMP_BASE - (instrs as i32))
            }
        }
    }
}

pub struct Scanner {
    instr: NonNull<PackedInstr>,
    cursor: NonNull<u8>,
}

impl Scanner {
    fn next(&mut self) -> Option<NonNull<Header>> {
        unsafe {
            loop {
                let instr: Instruction = self.instr.read().into();
                match instr {
                    Instruction::End => return None,
                    Instruction::Field => {
                        let header_ptr = self.cursor.cast::<*mut Header>();
                        self.instr = self.instr.add(1);
                        if let Some(child) = NonNull::new(header_ptr.read()) {
                            return Some(child);
                        }
                    }
                    Instruction::Variant(width) => {
                        let index = match width {
                            1 => self.cursor.cast::<u8>().read() as usize,
                            2 => self.cursor.cast::<u16>().read() as usize,
                            _ => self.cursor.cast::<u32>().read() as usize,
                        };
                        self.instr = self.instr.add(index + 1);
                    }
                    Instruction::Advance(bytes) => {
                        self.cursor = self.cursor.byte_add(bytes as usize);
                        self.instr = self.instr.add(1);
                    }
                    Instruction::Jump(instrs) => {
                        self.instr = self.instr.add(instrs as usize);
                    }
                }
            }
        }
    }
}

impl Iterator for Scanner {
    type Item = NonNull<Header>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl Scanner {
    pub unsafe fn new(object: NonNull<Header>) -> Option<Self> {
        let vtable = unsafe { &*object.as_ref().vtable };
        let instr = NonNull::new(vtable.scan_instrs as *mut PackedInstr)?;
        let cursor = unsafe { Header::get_object_ptr(object) };
        Some(Self { instr, cursor })
    }
}
