//! Where the encoder writes: one builder for both outputs. Each open
//! container is either a tree node, a Python list or dict filled as its
//! parts arrive, or JSON text written in place. A tree node opened inside
//! text leaves a ``Hole`` there, filled by ``json`` once the walk is done.
//!
//! Parts arrive depth first: a part's whole subtree closes before the next
//! part starts, so a tree node takes each part as it closes, in order.

use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::json::{self, Style, Writer};
use crate::py::*;

/// Text deeper than this is built as a tree and left to ``json``, which
/// nests as deep and fails, if it fails, as it would on the whole tree.
const TEXT_DEPTH: usize = 96;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Root,
    List,
    Dict,
}

struct Open<'py> {
    shape: Shape,
    text: bool,
    /// Tree: the node, or for a root the value it holds once written.
    node: Option<B<'py>>,
    count: usize,
    /// A dict's key last written, whose value comes next.
    key: Option<B<'py>>,
    /// A tree node made whole when it opened, each part not yet written
    /// held by ``None``: a value replaces its slot.
    filled: bool,
}

/// One step of a hole's path: its index in a list or its key in a dict.
pub enum Step<'py> {
    Item(usize),
    Key(B<'py>),
}

pub struct Hole<'py> {
    offset: usize,
    path: Vec<Step<'py>>,
    tree: B<'py>,
}

/// Where the builder stood, to come back to when a reduce restarts. The
/// node being written then is put in its parent only when it closes, so the
/// parent holds nothing written since.
#[derive(Clone, Copy)]
pub struct Mark {
    opens: usize,
    count: usize,
    text: usize,
    holes: usize,
}

pub struct Sink<'py> {
    py: Python<'py>,
    opens: Vec<Open<'py>>,
    pub w: Writer,
    holes: Vec<Hole<'py>>,
    text_depth: usize,
}

impl<'py> Sink<'py> {
    pub fn new(py: Python<'py>, style: Option<Style>) -> Self {
        let text = style.is_some();
        let w = style.map_or_else(Writer::idle, Writer::new);
        let mut opens = SPARE.take(py).unwrap_or_default().bind();
        opens.push(Open {
            shape: Shape::Root,
            text,
            node: None,
            count: 0,
            key: None,
            filled: false,
        });
        Self {
            py,
            opens,
            w,
            holes: Vec::new(),
            text_depth: 0,
        }
    }

    /// Keep this sink's buffers for the next.
    fn give(&mut self) {
        let spare = Spare::keep(std::mem::take(&mut self.opens));
        let stale = SPARE.replace(self.py, Some(spare));
        drop(stale);
    }

    #[inline(always)]
    fn top(&mut self) -> &mut Open<'py> {
        self.opens.last_mut().expect("the root stays open")
    }

    #[inline(always)]
    pub fn text(&self) -> bool {
        self.opens.last().is_some_and(|open| open.text)
    }

    /// Before a list item or the root.
    #[inline]
    pub fn item(&mut self) {
        let at = self.opens.last().map_or(0, |open| open.count);
        self.item_at(at);
    }

    /// Before the list item at ``index``: every item before it is written.
    #[inline]
    pub fn item_at(&mut self, index: usize) {
        let top = self.top();
        top.count = index + 1;
        if top.text && top.shape == Shape::List && index > 0 {
            self.w.item_sep();
        }
    }

    /// Before a dict value: its key, an exact str.
    #[inline]
    pub fn key(&mut self, key: &B<'py>) {
        self.key_text(key, None);
    }

    /// ``key``, with its quoted text where it is already made.
    #[inline]
    pub fn key_text(&mut self, key: &B<'py>, text: Option<&[u8]>) {
        let top = self.opens.last_mut().expect("the root stays open");
        let first = top.count == 0;
        top.count += 1;
        top.key = Some(key.clone());
        if top.text {
            if !first {
                self.w.item_sep();
            }
            match text {
                Some(text) => self.w.raw(text),
                None => self.w.str(key),
            }
            self.w.key_sep();
        }
    }

    /// A leaf of any type the tree holds as it is.
    #[inline]
    pub fn leaf(&mut self, value: &B<'py>) -> PyResult<()> {
        if !self.text() {
            return self.put(value.clone());
        }
        let mark = self.w.buf.len();
        if !matches!(self.w.leaf(value), Ok(true)) {
            self.w.buf.truncate(mark);
            self.hole(value.clone());
        }
        Ok(())
    }

    /// Put ``value`` in the tree node on top, where its key or index says.
    #[inline]
    fn put(&mut self, value: B<'py>) -> PyResult<()> {
        let top = self.top();
        let Some(node) = &top.node else {
            top.node = Some(value);
            return Ok(());
        };
        let done = unsafe {
            match top.shape {
                Shape::List if top.filled => ffi::PyList_SetItem(
                    node.as_ptr(),
                    (top.count - 1) as ffi::Py_ssize_t,
                    value.into_ptr(),
                ),
                Shape::List => ffi::PyList_Append(node.as_ptr(), value.as_ptr()),
                Shape::Dict => {
                    let key = top.key.take().expect("a dict's value follows its key");
                    ffi::PyDict_SetItem(node.as_ptr(), key.as_ptr(), value.as_ptr())
                }
                Shape::Root => unreachable!("a root holds one value"),
            }
        };
        if done < 0 {
            return Err(PyErr::fetch(self.py));
        }
        Ok(())
    }

    /// Open a list, or a dict. A dict whose keys may repeat once escaped is
    /// opened as a tree, which keeps one entry per key as Python's dict does.
    #[inline]
    pub fn open(&mut self, dict: bool, tree: bool) -> PyResult<()> {
        let text = self.text() && !tree && self.text_depth < TEXT_DEPTH;
        let node = if text {
            self.text_depth += 1;
            self.w.buf.push(if dict { b'{' } else { b'[' });
            None
        } else {
            let made = unsafe {
                if dict {
                    ffi::PyDict_New()
                } else {
                    ffi::PyList_New(0)
                }
            };
            Some(unsafe { Bound::from_owned_ptr_or_err(self.py, made)? })
        };
        self.opens.push(Open {
            shape: if dict { Shape::Dict } else { Shape::List },
            text,
            node,
            count: 0,
            key: None,
            filled: false,
        });
        Ok(())
    }

    /// Whether ``open`` would make a tree node, which ``open_filled`` may
    /// make whole instead.
    #[inline]
    pub fn makes_tree(&self) -> bool {
        !self.text()
    }

    /// Open ``node``, a list or dict made whole already.
    #[inline]
    pub fn open_filled(&mut self, node: B<'py>, dict: bool) {
        self.opens.push(Open {
            shape: if dict { Shape::Dict } else { Shape::List },
            text: false,
            node: Some(node),
            count: 0,
            key: None,
            filled: true,
        });
    }

    pub fn close(&mut self) -> PyResult<()> {
        let open = self.opens.pop().expect("a container is open");
        if open.text {
            self.text_depth -= 1;
            self.w.buf.push(if open.shape == Shape::Dict {
                b'}'
            } else {
                b']'
            });
            return Ok(());
        }
        let node = open.node.expect("a tree node is made when it opens");
        if self.text() {
            self.hole(node);
            return Ok(());
        }
        self.put(node)
    }

    /// A node only ``json`` writes exactly, left where it goes in the text.
    fn hole(&mut self, tree: B<'py>) {
        let mut path = Vec::new();
        for open in &self.opens {
            match open.shape {
                Shape::Root => {}
                Shape::List => path.push(Step::Item(open.count - 1)),
                Shape::Dict => path.push(Step::Key(
                    open.key
                        .clone()
                        .expect("a text dict's value follows its key"),
                )),
            }
        }
        self.holes.push(Hole {
            offset: self.w.buf.len(),
            path,
            tree,
        });
    }

    pub fn mark(&self) -> Mark {
        Mark {
            opens: self.opens.len(),
            count: self.opens.last().map_or(0, |open| open.count),
            text: self.w.buf.len(),
            holes: self.holes.len(),
        }
    }

    /// Back to ``mark``: everything written since is dropped.
    pub fn rewind(&mut self, mark: Mark) {
        while self.opens.len() > mark.opens {
            if self.opens.pop().is_some_and(|open| open.text) {
                self.text_depth -= 1;
            }
        }
        if let Some(top) = self.opens.last_mut() {
            top.count = mark.count;
        }
        self.w.buf.truncate(mark.text);
        self.holes.truncate(mark.holes);
    }

    /// Open a tree of its own on top, for a walk nested in this one (a key,
    /// a set member's sort key); ``take_nested`` closes it.
    pub fn open_nested(&mut self) {
        self.opens.push(Open {
            shape: Shape::Root,
            text: false,
            node: None,
            count: 0,
            key: None,
            filled: false,
        });
    }

    pub fn take_nested(&mut self) -> B<'py> {
        let open = self.opens.pop().expect("the nested root is open");
        open.node.expect("a nested walk writes one value")
    }

    /// The tree the walk built.
    pub fn take_tree(mut self) -> B<'py> {
        let tree = self
            .opens
            .pop()
            .and_then(|root| root.node)
            .expect("the walk wrote its root");
        self.give();
        tree
    }

    /// The text the walk wrote, its holes filled as ``json.dumps`` of the
    /// whole tree would write them.
    pub fn take_text(mut self, options: [Option<&B<'py>>; 4]) -> PyResult<B<'py>> {
        self.give();
        let holes = std::mem::take(&mut self.holes);
        if !holes.is_empty() {
            fill(self.py, &mut self.w, holes, options)?;
        }
        self.w.finish(self.py)
    }
}

/// A finished sink's emptied buffer, kept for the next call's.
#[derive(Default)]
struct Spare {
    opens: Vec<Open<'static>>,
}

static SPARE: GilCell<Option<Spare>> = GilCell::new(None);

/// Past this many entries a buffer is dropped rather than kept.
const SPARE_LIMIT: usize = 1 << 16;

impl Spare {
    fn keep(mut opens: Vec<Open<'_>>) -> Self {
        opens.clear();
        if opens.capacity() > SPARE_LIMIT {
            opens = Vec::new();
        }
        // SAFETY: it is empty, so no value of the old lifetime is kept.
        Self {
            opens: unsafe { std::mem::transmute::<Vec<Open<'_>>, Vec<Open<'static>>>(opens) },
        }
    }

    fn bind<'py>(self) -> Vec<Open<'py>> {
        // SAFETY: it is empty, so no value of the old lifetime is used.
        unsafe { std::mem::transmute::<Vec<Open<'static>>, Vec<Open<'py>>>(self.opens) }
    }
}

/// Write each hole's text as ``json.dumps`` of the whole tree would write it,
/// or raise what that raises, first hole first.
///
/// Each hole's tree is dumped by ``json``'s own ``dumps`` inside containers
/// of the same kinds, on the same path, as in the whole tree, each holding
/// only what comes before it (``None`` for items before an index), so
/// ``json`` nests as deep and fails as it would there. The hole's text is
/// what falls between the path's openings and closings.
#[cold]
fn fill<'py>(
    py: Python<'py>,
    writer: &mut Writer,
    holes: Vec<Hole<'py>>,
    options: [Option<&B<'py>>; 4],
) -> PyResult<()> {
    let [ensure_ascii, separators, indent, sort_keys] = options;
    let dumps = json::own_dumps(py)?;
    let none = py.None().into_bound(py);
    let mut texts = Vec::with_capacity(holes.len());
    for hole in holes {
        let mut opening = Writer::new(writer.style.clone());
        let mut closing = Vec::with_capacity(hole.path.len());
        for step in &hole.path {
            match step {
                Step::Item(at) => {
                    opening.raw(b"[");
                    for _ in 0..*at {
                        opening.raw(b"null");
                        opening.item_sep();
                    }
                    closing.push(b']');
                }
                Step::Key(key) => {
                    opening.raw(b"{");
                    opening.str(key);
                    opening.key_sep();
                    closing.push(b'}');
                }
            }
        }
        closing.reverse();
        let mut tree = hole.tree;
        for step in hole.path.iter().rev() {
            tree = match step {
                Step::Item(at) => {
                    let list = PyList::new(py, std::iter::repeat_n(&none, *at))?;
                    list.append(tree)?;
                    list.into_any()
                }
                Step::Key(key) => {
                    let node = PyDict::new(py);
                    node.set_item(key, &tree)?;
                    node.into_any()
                }
            };
        }
        let text = json::call_dumps(&dumps, &tree, ensure_ascii, separators, indent, sort_keys)?;
        let bytes = text_bytes(&text)?;
        let Some(inner) = bytes
            .strip_prefix(opening.buf.as_slice())
            .and_then(|rest| rest.strip_suffix(closing.as_slice()))
        else {
            return Err(PyRuntimeError::new_err(
                "json.dumps wrote a list or dict otherwise than json writes it; was json's encoder replaced after treekle was imported?",
            ));
        };
        texts.push((hole.offset, inner.to_vec()));
    }
    writer.splice(texts);
    Ok(())
}

/// A str's text as the writer holds it: UTF-8, lone surrogates as their
/// three bytes.
fn text_bytes(text: &B<'_>) -> PyResult<Vec<u8>> {
    let bytes = unsafe {
        Bound::from_owned_ptr_or_err(
            text.py(),
            ffi::PyUnicode_AsEncodedString(
                text.as_ptr(),
                c"utf-8".as_ptr(),
                c"surrogatepass".as_ptr(),
            ),
        )?
    };
    Ok(bytes
        .cast_into::<pyo3::types::PyBytes>()?
        .as_bytes()
        .to_vec())
}
