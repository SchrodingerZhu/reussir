use std::cell::Cell;

use bumpalo::Bump;
use thiserror::Error;

enum PathState<'a, T> {
    Internal(&'a Node<'a, T>),
    Top(Option<&'a T>),
}

impl<'a, T> Clone for PathState<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for PathState<'a, T> {}

pub struct Node<'a, T> {
    tree_parent: Cell<Option<&'a Self>>,
    tree_left: Cell<Option<&'a Self>>,
    tree_right: Cell<Option<&'a Self>>,
    tree_reversed: Cell<bool>,
    state: Cell<PathState<'a, T>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

impl Direction {
    fn opposite(self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Error {
    #[error("Conflict detected")]
    Conflict,

    #[error("Already connected")]
    AlreadyConnected,
}

impl<'a, T> Node<'a, T> {
    fn get_child(&self, direction: Direction) -> Option<&Self> {
        match direction {
            Direction::Left => self.tree_left.get(),
            Direction::Right => self.tree_right.get(),
        }
    }
    fn set_child(&self, direction: Direction, child: Option<&'a Self>) {
        match direction {
            Direction::Left => self.tree_left.set(child),
            Direction::Right => self.tree_right.set(child),
        }
    }
    fn get_direction(&self) -> Direction {
        if self
            .tree_parent
            .get()
            .and_then(|parent| {
                parent
                    .tree_right
                    .get()
                    .map(|child| core::ptr::eq(child, self))
            })
            .unwrap_or(false)
        {
            Direction::Right
        } else {
            Direction::Left
        }
    }
    pub fn new(bump: &Bump) -> &Self {
        bump.alloc(Node {
            tree_parent: Cell::new(None),
            tree_left: Cell::new(None),
            tree_right: Cell::new(None),
            tree_reversed: Cell::new(false),
            state: Cell::new(PathState::Top(None)),
        })
    }
    fn push_down(&self) {
        if self.tree_reversed.get() {
            let tmp = self.tree_left.get();
            self.tree_left.set(self.tree_right.get());
            self.tree_right.set(tmp);
            if let Some(left) = self.tree_left.get() {
                left.tree_reversed.set(!left.tree_reversed.get());
            }
            if let Some(right) = self.tree_right.get() {
                right.tree_reversed.set(!right.tree_reversed.get());
            }
            self.tree_reversed.set(false);
        }
    }
    fn swap_path_state(&self, other: &Self) {
        self.state.swap(&other.state);
    }
    fn rotate(&'a self) {
        // Node cannot be rotated if it has no parent.
        let Some(parent) = self.tree_parent.get() else {
            return;
        };

        // First, check the `reversed` flags for all touched nodes. Push down the flags if needed.
        if let Some(grandparent) = parent.tree_parent.get() {
            grandparent.push_down();
        }
        parent.push_down();
        self.push_down();

        // Secondly, during the process of going up, the lower node also need to hand over the pointer to upper-level
        // paths so that only the root for each auxiliary tree can have non-null upper pointers.
        self.swap_path_state(parent);

        // Prepare to swap pointers.
        let node_dir = self.get_direction();
        let node_dir_inv = node_dir.opposite();
        let parent_dir = parent.get_direction();

        // update parent pointers
        if let Some(grandparent) = parent.tree_parent.get() {
            grandparent.set_child(parent_dir, Some(self));
        }
        self.tree_parent.set(parent.tree_parent.get());

        // update child pointers
        let target_child = self.get_child(node_dir_inv);
        parent.set_child(node_dir, target_child);
        if let Some(target_child) = target_child {
            target_child.tree_parent.set(Some(parent));
        }

        // update current node pointers
        self.set_child(node_dir_inv, Some(parent));
        parent.tree_parent.set(Some(self));
    }

    fn splay(&'a self) {
        while let Some(parent) = self.tree_parent.get() {
            if let Some(grandparent) = parent.tree_parent.get() {
                // Push down before check direction.
                grandparent.push_down();
                parent.push_down();
                if grandparent.get_direction() == parent.get_direction() {
                    // Zig-Zig case
                    parent.rotate();
                    self.rotate();
                } else {
                    // Zig-Zag case
                    self.rotate();
                }
            } else {
                // Zig case
                self.rotate();
            }
        }
    }
    // Separate current auxiliary tree into two smaller trees such that all nodes that are deeper than the current node
    // (those who come later during in-order traversal) are cut off from the auxiliary tree.
    // After the separation, current node is the root of its auxiliary tree.
    fn separate_deeper_nodes(&'a self) {
        self.splay();
        self.push_down();
        if let Some(right) = self.tree_right.get() {
            right.tree_parent.set(None);
            self.tree_right.set(None);
            right.state.set(PathState::Internal(self))
        }
    }

    // Merge current auxiliary tree with the upper-level one.
    // The merge process makes sure the current node is the "deepest" one in the merged auxiliary tree (by cutting off
    // irrelevant subtrees).
    // After the extension, current node is the root of the merged tree.
    // Return false if there is no upper level path.
    fn extend_upper_level_path(&'a self) -> bool {
        self.splay();
        if let PathState::Internal(upper) = self.state.get() {
            upper.separate_deeper_nodes();
            self.tree_parent.set(Some(upper));
            self.state.set(PathState::Top(None));
            upper.tree_right.set(Some(self));
            return true;
        }
        false
    }

    // Extend the auxiliary tree all the way to root.
    // After the extension, current node is the root of its auxiliary tree.
    fn extend_to_root(&'a self) {
        self.separate_deeper_nodes();
        while self.extend_upper_level_path() {}
    }

    // Lift the node to the root of its tree (not the auxiliary tree).
    // To do so, we first extend the auxiliary tree to root, which represents the path from root to the current node.
    // To set the current node as root, we reverse the order of the auxiliary tree such that previous
    // root (who has the least depth) now has the deepest depth and the current node (who has the deepest depth) now has
    // the lowest depth.
    fn lift_to_root(&'a self) {
        self.extend_to_root();
        self.splay();
        self.tree_reversed.set(!self.tree_reversed.get());
    }

    fn find_min(&'a self) -> &'a Self {
        let mut cursor = self;
        cursor.push_down();
        while let Some(left) = cursor.tree_left.get() {
            cursor = left;
            cursor.push_down();
        }
        cursor.splay();
        cursor
    }

    pub fn is_connected(&'a self, other: &'a Self) -> bool {
        if core::ptr::eq(self, other) {
            return true;
        }
        self.lift_to_root();
        other.extend_to_root();
        core::ptr::eq(other.find_min(), self)
    }

    pub fn connect_resolve<F>(
        &'a self,
        other: &'a Self,
        resolver: F,
    ) -> Result<UndoLog<'a, T>, Error>
    where
        F: FnOnce(&'a T, &'a T) -> Option<&'a T>,
    {
        if self.is_connected(other) {
            return Err(Error::AlreadyConnected);
        }
        other.lift_to_root();
        self.lift_to_root();
        let lhs_state = self.state.get();
        let rhs_state = other.state.get();
        match (lhs_state, rhs_state) {
            (PathState::Top(None), PathState::Top(Some(x))) => {
                self.state.set(PathState::Top(Some(x)));
            }
            (PathState::Top(Some(_)), PathState::Top(None)) => (),
            (PathState::Top(None), PathState::Top(None)) => (),
            (PathState::Top(Some(x)), PathState::Top(Some(y))) => {
                if let Some(resolved) = resolver(x, y) {
                    self.state.set(PathState::Top(Some(resolved)));
                } else {
                    return Err(Error::Conflict);
                }
            }
            _ => unreachable!("Unexpected path state combination"),
        }
        other.state.set(PathState::Internal(self));
        Ok(ConnectionGuard {
            lhs: self,
            rhs: other,
            lhs_state,
            rhs_state,
        })
        .map(|guard| UndoLog(UndoLogImpl::Connection(guard)))
    }

    pub fn connect(&'a self, other: &'a Self) -> Result<UndoLog<'a, T>, Error> {
        self.connect_resolve(other, |_, _| None)
    }

    pub fn attach_resolve<F>(&'a self, data: &'a T, resolver: F) -> Result<UndoLog<'a, T>, Error>
    where
        F: FnOnce(&'a T, &'a T) -> Option<&'a T>,
    {
        self.lift_to_root();
        let old_state = self.state.get();
        if let PathState::Top(Some(x)) = self.state.get() {
            if let Some(resolved) = resolver(x, data) {
                self.state.set(PathState::Top(Some(resolved)));
            } else {
                return Err(Error::Conflict);
            }
        } else {
            self.state.set(PathState::Top(Some(data)));
        }
        Ok(AttachGuard {
            node: self,
            state: old_state,
        })
        .map(|guard| UndoLog(UndoLogImpl::Attach(guard)))
    }

    pub fn attach_data(&'a self, data: &'a T) -> Result<UndoLog<'a, T>, Error> {
        self.attach_resolve(data, |_, _| None)
    }

    pub fn override_data(&'a self, data: &'a T) -> Result<UndoLog<'a, T>, Error> {
        self.attach_resolve(data, |_, _| Some(data))
    }

    pub fn get_data(&'a self) -> Option<&'a T> {
        self.lift_to_root();
        match self.state.get() {
            PathState::Top(Some(data)) => Some(data),
            PathState::Top(None) => None,
            PathState::Internal(_) => unreachable!("Cannot get data from internal path state"),
        }
    }
}

/// It is up to user to make sure that operations are undone in the reverse order of their application.
/// Otherwise, there is no memory error but the final state may not make sense.
pub struct UndoLog<'a, T>(UndoLogImpl<'a, T>);

impl<'a, T> UndoLog<'a, T> {
    pub fn undo(self) {
        match self.0 {
            UndoLogImpl::Attach(guard) => guard.undo(),
            UndoLogImpl::Connection(guard) => guard.undo(),
        }
    }
}

enum UndoLogImpl<'a, T> {
    Attach(AttachGuard<'a, T>),
    Connection(ConnectionGuard<'a, T>),
}

struct AttachGuard<'a, T> {
    node: &'a Node<'a, T>,
    state: PathState<'a, T>,
}

impl<'a, T> AttachGuard<'a, T> {
    fn undo(self) {
        self.node.lift_to_root();
        self.node.state.set(self.state);
    }
}

struct ConnectionGuard<'a, T> {
    lhs: &'a Node<'a, T>,
    rhs: &'a Node<'a, T>,
    lhs_state: PathState<'a, T>,
    rhs_state: PathState<'a, T>,
}

impl<T> ConnectionGuard<'_, T> {
    fn undo(self) {
        self.lhs.lift_to_root();
        self.rhs.extend_to_root();
        self.rhs.splay();
        self.rhs.push_down();
        debug_assert!(
            self.rhs
                .tree_left
                .get()
                .map(|x| core::ptr::eq(x, self.lhs))
                .unwrap_or(false)
        );
        // Reset the connection for rhs.
        self.rhs.tree_left.set(None);
        self.rhs.state.set(self.rhs_state);
        // Reset the connection for lhs.
        self.lhs.tree_parent.set(None);
        self.lhs.splay();
        self.lhs.state.set(self.lhs_state);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::Rng;

    use super::*;

    #[test]
    fn test_node_creation() {
        let bump = Bump::new();
        let node = Node::<i32>::new(&bump);
        assert!(node.get_child(Direction::Left).is_none());
        assert!(node.get_child(Direction::Right).is_none());
    }

    #[test]
    fn test_trivial_connection() {
        let bump = Bump::new();
        let a = Node::<i32>::new(&bump);
        let b = Node::<i32>::new(&bump);
        let c = Node::<i32>::new(&bump);
        let d = Node::<i32>::new(&bump);
        let e = Node::<i32>::new(&bump);

        let handles = [a, b, c, d, e];

        for i in handles.iter() {
            for j in handles.iter() {
                assert!(!i.is_connected(j) || core::ptr::eq(i, j));
            }
        }

        let _ab = a.connect(b).unwrap();
        let cd = c.connect(d).unwrap();

        assert!(a.is_connected(b));
        assert!(b.is_connected(a));
        assert!(c.is_connected(d));
        assert!(d.is_connected(c));

        for i in [a, b] {
            for j in [c, d, e] {
                assert!(!i.is_connected(j));
            }
        }

        for i in [c, d] {
            for j in [a, b, e] {
                assert!(!i.is_connected(j));
            }
        }

        for i in [a, b, c, d] {
            assert!(!i.is_connected(e));
        }

        let _eb = e.connect(b).unwrap();
        let _ad = a.connect(d).unwrap();

        for i in handles.iter() {
            for j in handles.iter() {
                assert!(i.is_connected(j));
            }
        }

        cd.undo();

        for i in [a, b, d, e] {
            for j in [a, b, d, e] {
                assert!(i.is_connected(j));
                assert!(!c.is_connected(i));
                assert!(!i.is_connected(c));
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_large_forests() {
        const LENGTH: usize = 1000;
        const STEP: usize = LENGTH / 10;
        let bump = Bump::new();
        let mut handles = Vec::new();
        let mut connections = std::collections::HashMap::new();
        for _ in 0..LENGTH {
            handles.push(Node::<()>::new(&bump));
        }
        for i in 1..LENGTH {
            connections.insert((i - 1, i), handles[i - 1].connect(handles[i]).unwrap());
        }
        for i in 0..LENGTH {
            for j in 0..LENGTH {
                assert!(handles[i].is_connected(handles[j]));
            }
        }
        for i in (STEP..LENGTH).step_by(STEP) {
            connections.remove(&(i - 1, i)).unwrap().undo();
        }
        for i in (0..LENGTH).step_by(STEP) {
            for j in i..(i + STEP) {
                for k in i..(i + STEP) {
                    assert!(handles[j].is_connected(handles[k]));
                }
            }
        }
        for i in (0..LENGTH).step_by(STEP) {
            for j in i..(i + STEP) {
                for k in 0..i {
                    assert!(!handles[j].is_connected(handles[k]));
                }
                for k in i + STEP..LENGTH {
                    assert!(!handles[j].is_connected(handles[k]));
                }
            }
        }

        let mut count = 0;

        for i in 0..(LENGTH / STEP - 1) {
            let a = handles[count + STEP + i];
            let b = handles[count + i];
            let handle = a.connect(b).unwrap();
            connections.insert((count + i, count + STEP + i), handle);
            count += STEP;
        }

        for i in 0..LENGTH {
            for j in 0..LENGTH {
                assert!(handles[i].is_connected(handles[j]));
            }
        }

        for i in (0..(LENGTH / 2 - STEP)).step_by(STEP) {
            connections
                .remove(&(i + (STEP / 2) - 1, i + (STEP / 2)))
                .unwrap()
                .undo();
            for j in (i + STEP / 2)..(i + STEP) {
                for k in (i + STEP / 2)..(i + STEP) {
                    assert!(handles[j].is_connected(handles[k]));
                }
            }
            for j in (i + STEP / 2)..(i + STEP) {
                for k in 0..(i + STEP / 2) {
                    assert!(!handles[j].is_connected(handles[k]));
                }
            }
            for j in (i + STEP / 2)..(i + STEP) {
                for k in (i + STEP)..LENGTH {
                    assert!(!handles[j].is_connected(handles[k]));
                }
            }
            let a = handles[i + (STEP / 2) - 1];
            let b = handles[i + (STEP / 2)];
            connections.insert((i + (STEP / 2) - 1, i + (STEP / 2)), a.connect(b).unwrap());
        }

        for i in 0..LENGTH {
            for j in 0..LENGTH {
                assert!(handles[i].is_connected(handles[j]));
            }
        }
    }

    #[test]
    fn test_random() {
        #[cfg(not(miri))]
        const LENGTH: usize = 200;

        #[cfg(miri)]
        const LENGTH: usize = 15;
        use rand::Rng;
        let bump = bumpalo::Bump::new();
        let mut rng = rand::rng();
        let mut handles = Vec::new();
        let mut connections = std::collections::HashMap::new();
        for _ in 0..LENGTH {
            handles.push(Node::<()>::new(&bump));
        }
        for _ in 0..10 * LENGTH {
            let i = rng.random_range(0..LENGTH - 1);
            let j = rng.random_range(0..(i + 1));
            if i == j {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = connections.entry((j, i)) {
                let a = handles[i];
                let b = handles[j];
                if let Ok(h) = a.connect(&b) {
                    e.insert(h);
                }
            } else {
                assert!(handles[i].is_connected(handles[j]));
                connections.remove(&(j, i)).unwrap().undo();
                assert!(!handles[j].is_connected(handles[i]));
            }
        }
        for i in 0..LENGTH {
            for j in i..LENGTH {
                if i == j || connections.contains_key(&(i, j)) {
                    assert!(handles[i].is_connected(handles[j]));
                }
            }
        }
        for i in 0..LENGTH {
            // symmetric
            assert!(handles[i].is_connected(handles[i]));
            for j in i..LENGTH {
                if handles[i].is_connected(handles[j]) {
                    // reflexive
                    assert!(handles[j].is_connected(handles[i]));
                    for k in 0..LENGTH {
                        if handles[j].is_connected(handles[k]) {
                            // transitive
                            assert!(handles[i].is_connected(handles[k]));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_trivial_attach() {
        let bump = Bump::new();
        let a = Node::<i32>::new(&bump);
        let b = Node::<i32>::new(&bump);
        let c = Node::<i32>::new(&bump);

        assert!(a.get_data().is_none());
        assert!(b.get_data().is_none());
        assert!(c.get_data().is_none());

        let _guard_a = a.attach_data(&42).unwrap();
        assert_eq!(a.get_data(), Some(&42));
        assert!(b.get_data().is_none());
        let _ab = a.connect(b).unwrap();
        assert_eq!(b.get_data(), Some(&42));
        assert!(c.get_data().is_none());
        let bc = b.connect(c).unwrap();
        assert_eq!(c.get_data(), Some(&42));
        bc.undo();
        assert_eq!(a.get_data(), Some(&42));
        assert_eq!(b.get_data(), Some(&42));
        assert!(c.get_data().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dfs_random() {
        const NUM_NODES: usize = 1000;
        const SAMPLE_CONNECT_PER_LEVEL: usize = 200;
        const SAMPLE_DATA_PER_LEVEL: usize = 200;
        const DFS_DEPTH: usize = 40;
        const ACTIONS_PER_LEVEL: usize = 40;
        enum Action {
            Attach(usize, i32),
            Connect(usize, usize),
        }

        impl Action {
            fn next() -> Self {
                let mut rng = rand::rng();
                let action = rng.random_range(0..2);
                match action {
                    0 => {
                        let a = rng.random_range(0..NUM_NODES);
                        let data = rng.random_range(0..1000);
                        Action::Attach(a, data)
                    }
                    _ => {
                        let a = rng.random_range(0..NUM_NODES);
                        let b = rng.random_range(0..NUM_NODES);
                        Action::Connect(a, b)
                    }
                }
            }
        }

        let bump = Bump::new();

        let mut nodes = Vec::with_capacity(NUM_NODES);
        for _ in 0..NUM_NODES {
            nodes.push(Node::<i32>::new(&bump));
        }
        fn go<'a>(nodes: &[&'a Node<'a, i32>], bump: &'a Bump, depth: usize) {
            if depth >= DFS_DEPTH {
                return;
            }
            let mut rng = rand::rng();
            let mut connection_samples = HashMap::new();
            for _ in 0..SAMPLE_CONNECT_PER_LEVEL {
                let a = rng.random_range(0..nodes.len());
                let b = rng.random_range(0..nodes.len());
                connection_samples.insert((a, b), nodes[a].is_connected(nodes[b]));
            }
            let mut data_samples = HashMap::new();
            for _ in 0..SAMPLE_DATA_PER_LEVEL {
                let a = rng.random_range(0..nodes.len());
                data_samples.insert(a, nodes[a].get_data().copied());
            }
            let mut logs = Vec::with_capacity(ACTIONS_PER_LEVEL);
            for _ in 0..ACTIONS_PER_LEVEL {
                let action = Action::next();
                match action {
                    Action::Attach(a, data) => {
                        let data = bump.alloc(data);
                        if let Ok(log) = nodes[a].attach_data(data) {
                            logs.push(log);
                        }
                    }
                    Action::Connect(a, b) => {
                        if let Ok(log) = nodes[a].connect(nodes[b]) {
                            logs.push(log);
                        }
                    }
                }
            }
            go(nodes, bump, depth + 1);
            for log in logs.into_iter().rev() {
                log.undo();
            }
            for (i, j) in connection_samples.keys() {
                assert_eq!(
                    nodes[*i].is_connected(nodes[*j]),
                    *connection_samples.get(&(*i, *j)).unwrap()
                );
            }
            for (i, data) in data_samples.iter() {
                if let Some(data) = data {
                    assert_eq!(nodes[*i].get_data(), Some(data));
                } else {
                    assert!(nodes[*i].get_data().is_none());
                }
            }
        }
        go(&nodes, &bump, 0);
    }
}
