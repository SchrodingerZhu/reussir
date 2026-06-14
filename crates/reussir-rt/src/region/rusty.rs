use std::{cell::Cell, mem::ManuallyDrop, ptr::NonNull};

use crate::region::{
    Header, Status, VTable,
    scanner::{Instruction, PackedInstr},
};

// For rust side interoperability
#[repr(C)]
pub struct RegionalRcBox<T> {
    header: Header,
    data: ManuallyDrop<T>,
}

pub unsafe trait RegionalObjectTrait: Sized {
    const SCAN_INSTRS: &'static [PackedInstr];
    unsafe extern "C" fn drop(ptr: *mut u8) {
        if let Some(ptr) = NonNull::new(ptr) {
            let mut obj = ptr.cast::<ManuallyDrop<Self>>();
            unsafe { ManuallyDrop::drop(obj.as_mut()) };
        }
    }
    const VTABLE: VTable = VTable {
        drop: Some(Self::drop),
        scan_instrs: Self::SCAN_INSTRS.as_ptr(),
        size: std::mem::size_of::<Self>(),
        alignment: std::mem::align_of::<Self>(),
    };
}

#[repr(transparent)]
pub struct RegionalRc<T>(NonNull<RegionalRcBox<T>>);

impl<T> Copy for RegionalRc<T> {}

impl<T> Clone for RegionalRc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> RegionalRc<T> {
    pub unsafe fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut self.0.as_mut().data }
    }
    pub unsafe fn deref(&self) -> &T {
        unsafe { &self.0.as_ref().data }
    }
    fn header(&self) -> NonNull<Header> {
        self.0.cast()
    }
    pub unsafe fn freeze(self) -> RigidRc<T> {
        unsafe { Header::freeze(self.header()) };
        RigidRc(self)
    }
    pub unsafe fn read_field(&self) -> RigidRc<T> {
        unsafe { Header::acquire(self.header()) };
        RigidRc(*self)
    }
}

#[repr(transparent)]
pub struct RigidRc<T>(RegionalRc<T>);

impl<T> RigidRc<T> {
    pub unsafe fn deref(&self) -> &T {
        unsafe { self.0.deref() }
    }
}

impl<T> Clone for RigidRc<T> {
    fn clone(&self) -> Self {
        unsafe {
            Header::acquire(self.0.header());
        }
        RigidRc(self.0)
    }
}

impl<T> Drop for RigidRc<T> {
    fn drop(&mut self) {
        unsafe {
            Header::release(self.0.header());
        }
    }
}

#[repr(C)]
pub struct Region {
    head: Cell<Option<NonNull<Header>>>,
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Region {
    pub fn new() -> Self {
        Self {
            head: Cell::new(None),
        }
    }
    pub fn create<T: RegionalObjectTrait>(&self, value: T) -> RegionalRc<T> {
        let rc_box = Box::new(RegionalRcBox {
            header: Header {
                status: Status::Unmarked.into(),
                next: self
                    .head
                    .get()
                    .map(|h| h.as_ptr())
                    .unwrap_or(std::ptr::null_mut()),
                vtable: &T::VTABLE as *const VTable as *mut VTable,
            },
            data: ManuallyDrop::new(value),
        });
        let ptr = NonNull::from(Box::leak(rc_box));
        self.head.set(Some(ptr.cast()));
        RegionalRc(ptr)
    }
    pub unsafe fn clean(&self) {
        while let Some(head) = self.head.get() {
            self.head.set(NonNull::new(unsafe { head.as_ref().next }));
            if let Status::Unmarked = unsafe { head.as_ref().status.into() } {
                unsafe {
                    Header::drop(head);
                    Header::deallocate(head);
                }
            }
        }
    }
}

macro_rules! impl_regional_object_traits_for_primitive {
    ($($ty:ty),*) => {
        $(
            unsafe impl RegionalObjectTrait for $ty {
                const SCAN_INSTRS: &'static [PackedInstr] = &[Instruction::End.pack()];
                unsafe extern "C" fn drop(_ptr: *mut u8) {
                }
            }
        )*
    };
}

impl_regional_object_traits_for_primitive!(
    bool, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize, isize
);

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn test_region() {
        let region = Region::new();
        unsafe {
            region.clean();
        }
    }

    struct ListNode {
        prev: Option<RegionalRc<Self>>,
        next: Option<RegionalRc<Self>>,
        data: usize,
    }

    unsafe impl RegionalObjectTrait for ListNode {
        const SCAN_INSTRS: &'static [PackedInstr] = &[
            Instruction::Field.pack(), // prev
            Instruction::Advance(offset_of!(ListNode, next) as u32).pack(),
            Instruction::Field.pack(), // next
            Instruction::End.pack(),
        ];
    }

    #[test]
    fn test_list_node() {
        let region = Region::new();
        let _node = region.create(ListNode {
            prev: None,
            next: None,
            data: 0,
        });
        unsafe {
            region.clean();
        }
    }
    #[test]
    fn test_list_0_1_2() {
        let region = Region::new();
        let mut node1 = region.create(ListNode {
            prev: None,
            next: None,
            data: 0,
        });
        let mut node2 = region.create(ListNode {
            prev: None,
            next: None,
            data: 1,
        });
        let mut node3 = region.create(ListNode {
            prev: None,
            next: None,
            data: 2,
        });
        unsafe {
            node3.deref_mut().next = Some(node1);
            node2.deref_mut().next = Some(node3);
            node1.deref_mut().next = Some(node2);
            node1.deref_mut().prev = Some(node3);
            node2.deref_mut().prev = Some(node1);
            node3.deref_mut().prev = Some(node2);
        }

        let rigid_node1 = unsafe { node1.freeze() };
        let rigid_node2 = unsafe { node2.freeze() };
        let rigid_node3 = unsafe { node3.freeze() };
        unsafe {
            region.clean();
            assert_eq!(rigid_node1.deref().data, 0);
            assert_eq!(rigid_node1.deref().prev.unwrap().deref().data, 2);
            assert_eq!(rigid_node1.deref().next.unwrap().deref().data, 1);
            assert_eq!(rigid_node2.deref().data, 1);
            assert_eq!(rigid_node2.deref().prev.unwrap().deref().data, 0);
            assert_eq!(rigid_node2.deref().next.unwrap().deref().data, 2);
            assert_eq!(rigid_node3.deref().data, 2);
            assert_eq!(rigid_node3.deref().prev.unwrap().deref().data, 1);
            assert_eq!(rigid_node3.deref().next.unwrap().deref().data, 0);
            let _node2_clone = rigid_node2.clone();
            let cloned = rigid_node3.deref().next.unwrap().read_field();
            assert_eq!(cloned.deref().data, 0);
        }
    }

    #[repr(C)]
    struct Variants {
        tag: usize,
        storage: VariantsStorage,
    }

    #[repr(C)]
    union VariantsStorage {
        foo: ManuallyDrop<Foo>,
        bar: ManuallyDrop<Bar>,
        baz: ManuallyDrop<Baz>,
    }
    #[repr(C)]
    struct Foo(Option<RegionalRc<Variants>>);
    #[repr(C)]
    struct Bar(
        Option<RegionalRc<Variants>>,
        i128,
        Option<RegionalRc<Variants>>,
    );
    #[repr(C)]
    struct Baz;

    unsafe impl RegionalObjectTrait for Variants {
        const SCAN_INSTRS: &'static [PackedInstr] = &[
            Instruction::Variant.pack(),
            Instruction::Jump(3).pack(), // Goto Foo
            Instruction::Jump(5).pack(), // Goto Bar
            Instruction::End.pack(),     // Goto Baz, but no fields to scan
            // Foo
            Instruction::Advance(offset_of!(Variants, storage) as u32).pack(),
            Instruction::Field.pack(),
            Instruction::End.pack(),
            // Bar
            Instruction::Advance(offset_of!(Variants, storage) as u32).pack(),
            Instruction::Field.pack(),
            Instruction::Advance(offset_of!(Bar, 2) as u32).pack(),
            Instruction::Field.pack(),
            Instruction::End.pack(),
        ];
    }

    #[test]
    fn test_complex_foo_bar_baz_variants() {
        let region = Region::new();

        // Create a complex interconnected structure using the existing Foo, Bar, Baz variants
        // Structure: Create a circular reference pattern and tree-like structure
        //
        //     root_foo
        //        |
        //        v
        //      bar1 (value: 100)
        //      /              \
        //     v                v
        //   foo1             bar2 (value: 200)
        //    |               /            \
        //    v              v              v
        //   bar3         foo2            baz (shared)
        //  (300)          |
        //   |             v
        //   v           foo3 -> back to bar1 (circular)
        //  baz (shared)

        // Create the deepest nodes first
        let baz_leaf = region.create(Variants {
            tag: 2, // Baz variant
            storage: VariantsStorage {
                baz: ManuallyDrop::new(Baz),
            },
        });

        // Create bar3 that points to baz_leaf
        let bar3 = region.create(Variants {
            tag: 1, // Bar variant
            storage: VariantsStorage {
                bar: ManuallyDrop::new(Bar(Some(baz_leaf), 300, None)),
            },
        });

        // Create foo1 that points to bar3
        let foo1 = region.create(Variants {
            tag: 0, // Foo variant
            storage: VariantsStorage {
                foo: ManuallyDrop::new(Foo(Some(bar3))),
            },
        });

        // Create foo3 (will be connected to bar1 later for circular reference)
        let mut foo3 = region.create(Variants {
            tag: 0, // Foo variant
            storage: VariantsStorage {
                foo: ManuallyDrop::new(Foo(None)), // Will be updated later
            },
        });

        // Create foo2 that points to foo3
        let foo2 = region.create(Variants {
            tag: 0, // Foo variant
            storage: VariantsStorage {
                foo: ManuallyDrop::new(Foo(Some(foo3))),
            },
        });

        // Create bar2 that points to foo2 and baz_leaf2
        let bar2 = region.create(Variants {
            tag: 1, // Bar variant
            storage: VariantsStorage {
                bar: ManuallyDrop::new(Bar(Some(foo2), 200, Some(baz_leaf))),
            },
        });

        // Create bar1 (main bar)
        let bar1 = region.create(Variants {
            tag: 1, // Bar variant
            storage: VariantsStorage {
                bar: ManuallyDrop::new(Bar(Some(foo1), 100, Some(bar2))),
            },
        });

        // Create circular reference: foo3 points back to bar1
        unsafe {
            *foo3.deref_mut() = Variants {
                tag: 0,
                storage: VariantsStorage {
                    foo: ManuallyDrop::new(Foo(Some(bar1))),
                },
            };
        }

        // Create root foo that points to bar1
        let root_foo = region.create(Variants {
            tag: 0, // Foo variant
            storage: VariantsStorage {
                foo: ManuallyDrop::new(Foo(Some(bar1))),
            },
        });

        // Verify the structure before freezing
        unsafe {
            // Check root_foo -> bar1
            assert_eq!(root_foo.deref().tag, 0); // Foo
            let root_inner = &root_foo.deref().storage.foo.0.as_ref().unwrap();
            assert_eq!(root_inner.deref().tag, 1); // Bar
            let bar1_data = &root_inner.deref().storage.bar;
            assert_eq!(bar1_data.1, 100); // bar1 value

            // Check bar1 -> foo1 (left branch)
            let foo1_ref = bar1_data.0.as_ref().unwrap();
            assert_eq!(foo1_ref.deref().tag, 0); // Foo

            // Check foo1 -> bar3
            let bar3_ref = &foo1_ref.deref().storage.foo.0.as_ref().unwrap();
            assert_eq!(bar3_ref.deref().tag, 1); // Bar
            let bar3_data = &bar3_ref.deref().storage.bar;
            assert_eq!(bar3_data.1, 300); // bar3 value

            // Check bar3 -> baz_leaf
            let baz_from_bar3 = bar3_data.0.as_ref().unwrap();
            assert_eq!(baz_from_bar3.deref().tag, 2); // Baz

            // Check bar1 -> bar2 (right branch)
            let bar2_ref = bar1_data.2.as_ref().unwrap();
            assert_eq!(bar2_ref.deref().tag, 1); // Bar
            let bar2_data = &bar2_ref.deref().storage.bar;
            assert_eq!(bar2_data.1, 200); // bar2 value

            // Check bar2 -> foo2
            let foo2_ref = bar2_data.0.as_ref().unwrap();
            assert_eq!(foo2_ref.deref().tag, 0); // Foo

            // Check foo2 -> foo3
            let foo3_ref = &foo2_ref.deref().storage.foo.0.as_ref().unwrap();
            assert_eq!(foo3_ref.deref().tag, 0); // Foo

            // Check circular reference: foo3 -> bar1
            let bar1_circular = &foo3_ref.deref().storage.foo.0.as_ref().unwrap();
            assert_eq!(bar1_circular.deref().tag, 1); // Bar
            let bar1_circular_data = &bar1_circular.deref().storage.bar;
            assert_eq!(bar1_circular_data.1, 100); // Should be the same bar1

            // Check bar2 -> baz_leaf2
            let baz2_ref = bar2_data.2.as_ref().unwrap();
            assert_eq!(baz2_ref.deref().tag, 2); // Baz
        }

        // Freeze the root - this should handle the entire connected structure
        let frozen_root = unsafe { root_foo.freeze() };
        let frozen_bar1 = unsafe { bar1.freeze() };

        // Clean up unmarked objects
        unsafe {
            region.clean();
        }

        // Verify the frozen structure works correctly
        unsafe {
            // Navigate through the complex structure
            let root_data = frozen_root.deref();
            assert_eq!(root_data.tag, 0); // Foo

            let bar1_from_root = &root_data.storage.foo.0.as_ref().unwrap();
            assert_eq!(bar1_from_root.deref().tag, 1); // Bar
            let bar1_frozen_data = &bar1_from_root.deref().storage.bar;
            assert_eq!(bar1_frozen_data.1, 100);

            // Test left path: root -> bar1 -> foo1 -> bar3 -> baz
            let foo1_frozen = bar1_frozen_data.0.as_ref().unwrap();
            let bar3_frozen = &foo1_frozen.deref().storage.foo.0.as_ref().unwrap();
            let bar3_frozen_data = &bar3_frozen.deref().storage.bar;
            assert_eq!(bar3_frozen_data.1, 300);
            let baz_frozen = bar3_frozen_data.0.as_ref().unwrap();
            assert_eq!(baz_frozen.deref().tag, 2); // Baz

            // Test right path: root -> bar1 -> bar2 -> foo2 -> foo3 -> bar1 (circular)
            let bar2_frozen = bar1_frozen_data.2.as_ref().unwrap();
            let bar2_frozen_data = &bar2_frozen.deref().storage.bar;
            assert_eq!(bar2_frozen_data.1, 200);

            let foo2_frozen = bar2_frozen_data.0.as_ref().unwrap();
            let foo3_frozen = &foo2_frozen.deref().storage.foo.0.as_ref().unwrap();
            let bar1_circular_frozen = &foo3_frozen.deref().storage.foo.0.as_ref().unwrap();
            let bar1_circular_data = &*bar1_circular_frozen.deref().storage.bar;
            assert_eq!(bar1_circular_data.1, 100); // Circular reference preserved
        }

        // Test cloning and read_field operations
        let cloned_root = frozen_root.clone();
        let cloned_bar1 = frozen_bar1.clone();

        unsafe {
            // Verify cloned references work
            assert_eq!(cloned_root.deref().tag, 0);
            assert_eq!(cloned_bar1.deref().tag, 1);
            let cloned_bar1_data = &*cloned_bar1.deref().storage.bar;
            assert_eq!(cloned_bar1_data.1, 100);

            // Test read_field
            let root_data = frozen_root.deref();
            let bar1_via_read = root_data.storage.foo.0.as_ref().unwrap().read_field();
            assert_eq!(bar1_via_read.deref().tag, 1);
            let bar1_read_data = &*bar1_via_read.deref().storage.bar;
            assert_eq!(bar1_read_data.1, 100);
        }
    }
}
