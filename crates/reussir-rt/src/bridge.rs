use std::cmp::Ordering;

/// Supplies equality for a compiler-emitted PolyFFI wrapper.
pub trait PartialEqBridge {
    fn eq_bridge(&self, other: &Self) -> bool;
}

/// Marks a bridged equality implementation as reflexive.
pub trait EqBridge: PartialEqBridge {}

/// Supplies partial ordering for a compiler-emitted PolyFFI wrapper.
pub trait PartialOrdBridge: PartialEqBridge {
    fn partial_cmp_bridge(&self, other: &Self) -> Option<Ordering>;
}

/// Supplies total ordering for a compiler-emitted PolyFFI wrapper.
pub trait OrdBridge: EqBridge + PartialOrdBridge {
    fn cmp_bridge(&self, other: &Self) -> Ordering;
}

/// Adds Rust comparison traits to a compiler-emitted PolyFFI wrapper.
///
/// The inner type owns the Reussir-specific behavior. This transparent newtype
/// only adapts that behavior to Rust's standard traits and does not add runtime
/// state to values crossing the FFI boundary.
#[repr(transparent)]
pub struct Bridge<T>(T);

impl<T> Bridge<T> {
    pub const fn new(inner: T) -> Self {
        Self(inner)
    }

    pub const fn as_inner(&self) -> &T {
        &self.0
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Clone> Clone for Bridge<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: PartialEqBridge> PartialEq for Bridge<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_bridge(&other.0)
    }
}

impl<T: EqBridge> Eq for Bridge<T> {}

impl<T: PartialOrdBridge> PartialOrd for Bridge<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp_bridge(&other.0)
    }
}

impl<T: OrdBridge> Ord for Bridge<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_bridge(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, mem};

    use super::*;

    struct CountingValue<'a> {
        value: i32,
        calls: &'a Cell<usize>,
    }

    impl PartialEqBridge for CountingValue<'_> {
        fn eq_bridge(&self, other: &Self) -> bool {
            self.calls.set(self.calls.get() + 1);
            self.value == other.value
        }
    }

    impl EqBridge for CountingValue<'_> {}

    impl PartialOrdBridge for CountingValue<'_> {
        fn partial_cmp_bridge(&self, other: &Self) -> Option<Ordering> {
            self.calls.set(self.calls.get() + 1);
            Some(self.value.cmp(&other.value))
        }
    }

    impl OrdBridge for CountingValue<'_> {
        fn cmp_bridge(&self, other: &Self) -> Ordering {
            self.calls.set(self.calls.get() + 1);
            self.value.cmp(&other.value)
        }
    }

    struct PartialValue(f32);

    impl PartialEqBridge for PartialValue {
        fn eq_bridge(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl PartialOrdBridge for PartialValue {
        fn partial_cmp_bridge(&self, other: &Self) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    #[test]
    fn bridge_is_transparent_and_dispatches_once() {
        assert_eq!(mem::size_of::<Bridge<usize>>(), mem::size_of::<usize>());
        assert_eq!(mem::align_of::<Bridge<usize>>(), mem::align_of::<usize>());

        let calls = Cell::new(0);
        let lhs = Bridge::new(CountingValue {
            value: 4,
            calls: &calls,
        });
        let rhs = Bridge::new(CountingValue {
            value: 7,
            calls: &calls,
        });

        assert!(lhs != rhs);
        assert_eq!(calls.get(), 1);
        calls.set(0);
        assert_eq!(lhs.cmp(&rhs), Ordering::Less);
        assert_eq!(calls.get(), 1);

        fn requires_eq<T: Eq>(_: &T) {}
        requires_eq(&lhs);
    }

    #[test]
    fn partial_order_preserves_all_results() {
        let one = Bridge::new(PartialValue(1.0));
        let two = Bridge::new(PartialValue(2.0));
        let nan = Bridge::new(PartialValue(f32::NAN));

        assert_eq!(one.partial_cmp(&two), Some(Ordering::Less));
        assert_eq!(two.partial_cmp(&two), Some(Ordering::Equal));
        assert_eq!(two.partial_cmp(&one), Some(Ordering::Greater));
        assert_eq!(one.partial_cmp(&nan), None);
    }

    struct Tracked<'a> {
        clones: &'a Cell<usize>,
        drops: &'a Cell<usize>,
    }

    impl Clone for Tracked<'_> {
        fn clone(&self) -> Self {
            self.clones.set(self.clones.get() + 1);
            Self {
                clones: self.clones,
                drops: self.drops,
            }
        }
    }

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn clone_and_drop_delegate_to_the_inner_value() {
        let clones = Cell::new(0);
        let drops = Cell::new(0);
        let original = Bridge::new(Tracked {
            clones: &clones,
            drops: &drops,
        });

        let cloned = original.clone();
        assert_eq!(clones.get(), 1);
        drop(cloned);
        assert_eq!(drops.get(), 1);
        drop(original);
        assert_eq!(drops.get(), 2);
    }
}
