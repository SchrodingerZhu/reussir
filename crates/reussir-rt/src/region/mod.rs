use std::{alloc::Layout, num::NonZeroUsize, ptr::NonNull};

use num_enum::TryFromPrimitive;
use smallvec::SmallVec;

use crate::region::{
    rusty::Region,
    scanner::{PackedInstr, Scanner},
};
pub mod rusty;
mod scanner;

const STATUS_BITS: usize = 2;
const STATUS_MASK: usize = (1 << STATUS_BITS) - 1;
const HIGHEST_BIT: usize = isize::MIN as usize;
const MAX_VALUE: usize = usize::MAX >> (STATUS_BITS + 1);
const MIN_RC_VALUE: usize = 1 << STATUS_BITS | (PackedStatusTag::Rc as usize) | HIGHEST_BIT;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct PackedStatus(*mut Header);

impl PackedStatus {
    pub unsafe fn increase_unchecked(&mut self) {
        debug_assert!({
            let status: Status = (*self).into();
            matches!(status, Status::Rc(_) | Status::Rank(_))
        });
        self.0 = self.0.map_addr(|addr| addr + (1 << STATUS_BITS));
    }
    pub unsafe fn saturating_decrease_unchecked(&mut self) -> bool {
        debug_assert!({
            let status: Status = (*self).into();
            matches!(status, Status::Rc(_) | Status::Rank(_))
        });
        if self.0.addr() <= MIN_RC_VALUE {
            return true;
        }
        self.0 = self.0.map_addr(|addr| addr - (1 << STATUS_BITS));
        false
    }
}

impl PackedStatus {
    pub fn from_scalar(scalar: usize) -> Self {
        Self(std::ptr::without_provenance_mut(scalar))
    }
}

#[derive(TryFromPrimitive)]
#[repr(u8)]
enum PackedStatusTag {
    Unmarked = 0b00,
    Disposing = 0b01,
    Rank = 0b10,
    Rc = 0b11,
}

#[derive(Debug, Copy, Clone)]
enum Status {
    Unmarked,
    Rank(NonZeroUsize),
    Rc(NonZeroUsize),
    Disposing,
    Parent(NonNull<Header>),
}

impl Status {
    pub fn rank1() -> Self {
        Status::Rank(unsafe { NonZeroUsize::new_unchecked(1) })
    }
    pub fn rc1() -> Self {
        Status::Rc(unsafe { NonZeroUsize::new_unchecked(1) })
    }
}

impl From<Status> for PackedStatus {
    fn from(status: Status) -> Self {
        match status {
            Status::Parent(ptr) => Self(ptr.as_ptr()),
            Status::Unmarked => Self::from_scalar(PackedStatusTag::Unmarked as usize | HIGHEST_BIT),
            Status::Rank(non_zero) => {
                debug_assert!(non_zero.get() <= MAX_VALUE, "Rank value too large to pack");
                Self::from_scalar(
                    (non_zero.get() << STATUS_BITS) | PackedStatusTag::Rank as usize | HIGHEST_BIT,
                )
            }
            Status::Rc(non_zero) => {
                debug_assert!(non_zero.get() <= MAX_VALUE, "RC value too large to pack");
                Self::from_scalar(
                    (non_zero.get() << STATUS_BITS) | PackedStatusTag::Rc as usize | HIGHEST_BIT,
                )
            }
            Status::Disposing => {
                Self::from_scalar(PackedStatusTag::Disposing as usize | HIGHEST_BIT)
            }
        }
    }
}

impl From<PackedStatus> for Status {
    fn from(packed: PackedStatus) -> Self {
        if packed.0.addr() as isize > 0 {
            let ptr = unsafe { NonNull::new_unchecked(packed.0) };
            return Status::Parent(ptr);
        }
        let tag = unsafe {
            PackedStatusTag::try_from_primitive((packed.0.addr() & STATUS_MASK) as u8)
                .unwrap_unchecked()
        };
        match tag {
            PackedStatusTag::Unmarked => Status::Unmarked,
            PackedStatusTag::Rank => {
                let rank = (packed.0.addr() & !HIGHEST_BIT) >> STATUS_BITS;
                Status::Rank(unsafe { NonZeroUsize::new_unchecked(rank) })
            }
            PackedStatusTag::Rc => {
                let rc = (packed.0.addr() & !HIGHEST_BIT) >> STATUS_BITS;
                Status::Rc(unsafe { NonZeroUsize::new_unchecked(rc) })
            }
            PackedStatusTag::Disposing => Status::Disposing,
        }
    }
}

#[repr(C)]
pub struct VTable {
    pub drop: Option<unsafe extern "C" fn(*mut u8)>,
    pub scan_instrs: *const PackedInstr,
    pub size: usize,
    pub alignment: usize,
}

impl VTable {
    pub fn extended_layout(&self) -> (Layout, usize) {
        let header_layout = Layout::new::<Header>();
        unsafe {
            let object_layout = Layout::from_size_align_unchecked(self.size, self.alignment);
            let (layout, offset) = header_layout.extend(object_layout).unwrap_unchecked();
            (layout.pad_to_align(), offset)
        }
    }
}

#[repr(C)]
pub struct Header {
    status: PackedStatus,
    next: *mut Self,
    vtable: *mut VTable,
}

impl Header {
    pub unsafe fn children(this: NonNull<Self>) -> impl Iterator<Item = NonNull<Header>> {
        unsafe { Scanner::new(this).into_iter().flatten() }
    }

    pub unsafe fn get_object_ptr(this: NonNull<Self>) -> NonNull<u8> {
        unsafe {
            let (_, offset) = (&*this.as_ref().vtable).extended_layout();
            this.byte_add(offset).cast()
        }
    }

    pub unsafe fn get_total_layout(this: NonNull<Self>) -> Layout {
        let (layout, _) = unsafe { (&*this.as_ref().vtable).extended_layout() };
        layout
    }

    pub unsafe fn drop(this: NonNull<Self>) {
        let object_ptr = unsafe { Self::get_object_ptr(this) };
        let table = unsafe { &*this.as_ref().vtable };
        if let Some(drop_fn) = table.drop {
            unsafe { drop_fn(object_ptr.as_ptr()) };
        }
    }
    pub unsafe fn deallocate(this: NonNull<Self>) {
        let layout = unsafe { Self::get_total_layout(this) };
        unsafe { std::alloc::dealloc(this.as_ptr() as *mut u8, layout) };
    }

    unsafe fn find(mut cursor: NonNull<Self>) -> NonNull<Self> {
        while let Status::Parent(parent) = unsafe { cursor.as_ref().status }.into() {
            let parent_status = unsafe { parent.as_ref().status };
            if let Status::Parent(..) = parent_status.into() {
                unsafe { cursor.as_mut().status = parent_status };
            }
            cursor = parent;
        }
        cursor
    }

    unsafe fn rank(this: NonNull<Self>) -> NonZeroUsize {
        let status = unsafe { this.as_ref().status };
        if let Status::Rank(rank) = status.into() {
            rank
        } else {
            unsafe { std::hint::unreachable_unchecked() }
        }
    }

    unsafe fn union(mut r1: NonNull<Self>, mut r2: NonNull<Self>) -> bool {
        r1 = unsafe { Self::find(r1) };
        r2 = unsafe { Self::find(r2) };
        if r1 == r2 {
            return false;
        }
        let rank1 = unsafe { Self::rank(r1) };
        let rank2 = unsafe { Self::rank(r2) };
        if rank1 > rank2 {
            std::mem::swap(&mut r1, &mut r2);
        } else if rank1 == rank2 {
            unsafe {
                r1.as_mut().status.increase_unchecked();
            };
        }
        unsafe { r2.as_mut().status = Status::Parent(r1).into() };
        true
    }

    pub unsafe fn dispose(this: NonNull<Self>) {
        type Stack = SmallVec<[NonNull<Header>; 8]>;
        unsafe fn add_stack(stack: &mut Stack, mut this: NonNull<Header>) {
            stack.push(this);
            unsafe { this.as_mut().status = Status::Disposing.into() }
        }
        let mut dfs = Stack::new();
        let mut scc = Stack::new();
        let mut free_list = Stack::new();
        unsafe { add_stack(&mut dfs, Header::find(this)) };
        while let Some(node) = dfs.pop() {
            scc.push(node);
            while let Some(node) = scc.pop() {
                free_list.push(node);
                for child in unsafe { Header::children(node) } {
                    let mut child_root = unsafe { Header::find(child) };
                    match unsafe { child_root.as_ref().status.into() } {
                        Status::Disposing => {
                            if child != child_root {
                                unsafe { add_stack(&mut scc, child) };
                            }
                        }
                        Status::Rc(_) => {
                            if unsafe { child_root.as_mut().status.saturating_decrease_unchecked() }
                            {
                                unsafe { add_stack(&mut dfs, child_root) };
                            }
                        }
                        _ => unsafe { std::hint::unreachable_unchecked() },
                    }
                }
            }
        }
        for elem in free_list {
            unsafe { Header::drop(elem) };
            unsafe { Header::deallocate(elem) };
        }
    }
    pub unsafe fn acquire(this: NonNull<Self>) {
        let mut root = unsafe { Header::find(this) };
        unsafe { root.as_mut().status.increase_unchecked() };
    }
    pub unsafe fn release(this: NonNull<Self>) {
        let mut root = unsafe { Header::find(this) };
        if unsafe { root.as_mut().status.saturating_decrease_unchecked() } {
            unsafe { Header::dispose(this) };
        }
    }
    pub unsafe fn freeze(this: NonNull<Self>) {
        type Pending = SmallVec<[NonNull<Header>; 16]>;
        const RED_ZONE_SIZE: usize = 4096;
        const BUMP_SIZE: usize = 32768;
        let mut pending = Pending::new();
        fn freeze_recursive(mut this: NonNull<Header>, pending: &mut Pending) {
            stacker::maybe_grow(RED_ZONE_SIZE, BUMP_SIZE, move || {
                let status: Status = unsafe { Header::find(this).as_ref().status }.into();
                match status {
                    Status::Unmarked => {
                        unsafe {
                            this.as_mut().status = Status::rank1().into();
                        }
                        pending.push(this);
                        unsafe {
                            for child in Header::children(this) {
                                freeze_recursive(child, pending);
                            }
                        }
                        if let Some(peek) = pending.last().copied()
                            && core::ptr::eq(peek.as_ptr(), this.as_ptr())
                        {
                            pending.pop();
                            unsafe { Header::find(this).as_mut().status = Status::rc1().into() };
                        }
                    }
                    Status::Rc(_) => unsafe {
                        Header::find(this).as_mut().status.increase_unchecked()
                    },
                    Status::Rank(_) => {
                        while pending
                            .last()
                            .copied()
                            .map(|p| unsafe { Header::union(this, p) })
                            .unwrap_or(false)
                        {
                            pending.pop();
                        }
                    }
                    _ => (),
                }
            })
        }
        freeze_recursive(this, &mut pending);
    }
}

impl Default for Header {
    fn default() -> Self {
        Self {
            status: Status::Unmarked.into(),
            next: std::ptr::null_mut(),
            vtable: std::ptr::null_mut(),
        }
    }
}

/*
  addRuntimeFunction(body, "__reussir_freeze_flex_object", {llvmPtrType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_cleanup_region", {llvmPtrType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_acquire_rigid_object", {llvmPtrType}, {});
  addRuntimeFunction(body, "__reussir_release_rigid_object", {llvmPtrType}, {});
*/

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_freeze_flex_object(ptr: *mut u8) -> *mut u8 {
    unsafe {
        match NonNull::new(ptr) {
            Some(ptr) => {
                Header::freeze(ptr.cast());
                ptr.as_ptr()
            }
            None => crate::panic!("Invalid pointer passed to __reussir_freeze_flex_object"),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_cleanup_region(ptr: *mut u8) {
    unsafe {
        match NonNull::new(ptr) {
            Some(ptr) => {
                Region::clean(ptr.cast().as_ref());
            }
            None => crate::panic!("Invalid pointer passed to __reussir_cleanup_region"),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_acquire_rigid_object(ptr: *mut u8) {
    unsafe {
        match NonNull::new(ptr) {
            Some(ptr) => {
                Header::acquire(ptr.cast());
            }
            None => crate::panic!("Invalid pointer passed to __reussir_acquire_rigid_object"),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_release_rigid_object(ptr: *mut u8) {
    unsafe {
        match NonNull::new(ptr) {
            Some(ptr) => {
                Header::release(ptr.cast());
            }
            None => crate::panic!("Invalid pointer passed to __reussir_release_rigid_object"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_unmarked() {
        let original = Status::Unmarked;
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();

        match unpacked {
            Status::Unmarked => (),
            _ => panic!("Expected Unmarked, got {:?}", unpacked),
        }
    }

    #[test]
    fn test_pack_unpack_disposing() {
        let original = Status::Disposing;
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();

        match unpacked {
            Status::Disposing => (),
            _ => panic!("Expected Disposing, got {:?}", unpacked),
        }
    }

    #[test]
    fn test_pack_unpack_rank() {
        // Test various rank values
        let test_values = [1, 2, 42, 100, 1000, MAX_VALUE];

        for &value in &test_values {
            let rank = NonZeroUsize::new(value).unwrap();
            let original = Status::Rank(rank);
            let packed: PackedStatus = original.into();
            let unpacked: Status = packed.into();

            match unpacked {
                Status::Rank(unpacked_rank) => {
                    assert_eq!(
                        rank.get(),
                        unpacked_rank.get(),
                        "Rank value mismatch: original {}, unpacked {}",
                        rank.get(),
                        unpacked_rank.get()
                    );
                }
                _ => panic!("Expected Rank({}), got {:?}", value, unpacked),
            }
        }
    }

    #[test]
    fn test_pack_unpack_rc() {
        // Test various reference count values
        let test_values = [1, 2, 42, 100, 1000, MAX_VALUE];

        for &value in &test_values {
            let rc = NonZeroUsize::new(value).unwrap();
            let original = Status::Rc(rc);
            let packed: PackedStatus = original.into();
            let unpacked: Status = packed.into();

            match unpacked {
                Status::Rc(unpacked_rc) => {
                    assert_eq!(
                        rc.get(),
                        unpacked_rc.get(),
                        "RC value mismatch: original {}, unpacked {}",
                        rc.get(),
                        unpacked_rc.get()
                    );
                }
                _ => panic!("Expected Rc({}), got {:?}", value, unpacked),
            }
        }
    }

    #[test]
    fn test_pack_unpack_parent() {
        // Create a dummy header on the stack for testing
        let header = Header::default();
        let ptr = NonNull::from(&header);
        let original = Status::Parent(ptr);
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();

        match unpacked {
            Status::Parent(unpacked_ptr) => {
                assert_eq!(
                    ptr.as_ptr(),
                    unpacked_ptr.as_ptr(),
                    "Parent pointer mismatch"
                );
            }
            _ => panic!("Expected Parent, got {:?}", unpacked),
        }
    }

    #[test]
    fn test_pack_unpack_round_trip_all_variants() {
        let header = Header::default();
        let ptr = NonNull::from(&header);

        // Test each variant individually to avoid ownership issues

        // Test Unmarked
        let original = Status::Unmarked;
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();
        assert!(
            matches!(unpacked, Status::Unmarked),
            "Unmarked variant failed"
        );

        // Test Disposing
        let original = Status::Disposing;
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();
        assert!(
            matches!(unpacked, Status::Disposing),
            "Disposing variant failed"
        );

        // Test Rank
        let original = Status::Rank(NonZeroUsize::new(42).unwrap());
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();
        match unpacked {
            Status::Rank(val) => assert_eq!(val.get(), 42, "Rank value mismatch"),
            _ => panic!("Expected Rank variant"),
        }

        // Test Rc
        let original = Status::Rc(NonZeroUsize::new(100).unwrap());
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();
        match unpacked {
            Status::Rc(val) => assert_eq!(val.get(), 100, "Rc value mismatch"),
            _ => panic!("Expected Rc variant"),
        }

        // Test Parent
        let original = Status::Parent(ptr);
        let packed: PackedStatus = original.into();
        let unpacked: Status = packed.into();
        match unpacked {
            Status::Parent(unp_ptr) => {
                assert_eq!(ptr.as_ptr(), unp_ptr.as_ptr(), "Parent pointer mismatch")
            }
            _ => panic!("Expected Parent variant"),
        }
    }

    #[test]
    fn freeze_singleton() {
        let empty_table = VTable {
            drop: None,
            scan_instrs: std::ptr::null(),
            size: std::mem::size_of::<Header>(),
            alignment: std::mem::align_of::<Header>(),
        };
        let mut header = Header {
            status: Status::Unmarked.into(),
            next: std::ptr::null_mut(),
            vtable: &empty_table as *const VTable as *mut VTable,
        };
        let ptr = NonNull::from(&mut header);
        unsafe { Header::freeze(ptr) };
        let status: Status = header.status.into();
        match status {
            Status::Rc(rc) => assert_eq!(rc.get(), 1, "Expected RC to be 1"),
            _ => panic!("Expected Rc status, got {:?}", status),
        }
    }
}
