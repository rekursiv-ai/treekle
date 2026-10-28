//! ``plain.immutable(tree)``: a tree of dicts, lists and leaves copied as
//! ``MappingProxyType`` and tuples, without recursing on the native stack.

use hashbrown::HashSet;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::py::*;

/// A node of ``freeze``'s copy: its source, how far it is read, its copies.
enum Open<'py> {
    List(Bound<'py, PyList>, usize, Vec<B<'py>>),
    Dict(Vec<(B<'py>, B<'py>)>, usize, Vec<B<'py>>),
}

impl<'py> Open<'py> {
    fn of(node: &B<'py>) -> Option<Self> {
        let kind = tp(node);
        if kind == t_list() {
            return Some(Self::List(
                unsafe { node.cast_unchecked::<PyList>() }.clone(),
                0,
                Vec::new(),
            ));
        }
        if kind == t_dict() {
            let entries = dict_entries(unsafe { node.cast_unchecked::<PyDict>() });
            if entries.iter().any(|(key, _)| !is_str(key)) {
                return None;
            }
            return Some(Self::Dict(entries, 0, Vec::new()));
        }
        None
    }

    fn next(&mut self) -> PyResult<Option<B<'py>>> {
        Ok(match self {
            Self::List(list, at, _) if *at < list.len() => {
                *at += 1;
                Some(list.get_item(*at - 1)?)
            }
            Self::Dict(entries, at, _) if *at < entries.len() => {
                *at += 1;
                Some(entries[*at - 1].1.clone())
            }
            _ => None,
        })
    }

    fn done(&mut self) -> &mut Vec<B<'py>> {
        match self {
            Self::List(_, _, done) | Self::Dict(_, _, done) => done,
        }
    }

    fn close(self, py: Python<'py>) -> PyResult<B<'py>> {
        match self {
            Self::List(_, _, done) => Ok(PyTuple::new(py, done)?.into_any()),
            Self::Dict(entries, _, done) => {
                let copied = PyDict::new(py);
                for ((key, _), value) in entries.into_iter().zip(done) {
                    copied.set_item(key, value)?;
                }
                unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyDictProxy_New(copied.as_ptr())) }
            }
        }
    }
}

/// ``plain.immutable(tree)`` for a tree of dicts, lists, and leaves; ``None``
/// for anything else, such as a cycle or a node a Python encoder made, which
/// the Python function then copies or rejects itself.
pub fn freeze<'py>(py: Python<'py>, tree: &B<'py>) -> PyResult<Option<B<'py>>> {
    let is_leaf =
        |value: &B<'py>| value.is_none() || is_str(value) || is_int(value) || is_float(value);
    if is_leaf(tree) {
        return Ok(Some(tree.clone()));
    }
    let Some(first) = Open::of(tree) else {
        return Ok(None);
    };
    let mut stack = vec![(tree.as_ptr() as usize, first)];
    let mut open: HashSet<usize> = HashSet::from([tree.as_ptr() as usize]);
    loop {
        let top = &mut stack
            .last_mut()
            .expect("the stack holds the root until it closes")
            .1;
        if let Some(item) = top.next()? {
            if is_leaf(&item) {
                top.done().push(item);
                continue;
            }
            let id = item.as_ptr() as usize;
            if open.contains(&id) {
                return Ok(None);
            }
            let Some(child) = Open::of(&item) else {
                return Ok(None);
            };
            open.insert(id);
            stack.push((id, child));
            continue;
        }
        let (id, closed) = stack.pop().expect("the stack holds the node being closed");
        open.remove(&id);
        let node = closed.close(py)?;
        match stack.last_mut() {
            Some((_, parent)) => parent.done().push(node),
            None => return Ok(Some(node)),
        }
    }
}
