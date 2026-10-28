//! Dict watchers that tell a native cache when the module state it copied
//! changes.
//!
//! An import path proved to name an object stays proved while neither
//! ``sys.modules`` nor the namespace it was read from changes; the decoder's
//! resolved imports stay resolved while its own globals stand. Each watched
//! dict advances a generation whenever it changes, and a fact is trusted only
//! while the generation it was read at is current. CPython calls the watcher
//! before every change to a watched dict (3.12+), including deallocation, so
//! no change goes unseen. Class facts need no watcher: a class's version tag
//! changes with it.

use std::ffi::c_int;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Callback = unsafe extern "C" fn(
    event: c_int,
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
    new_value: *mut ffi::PyObject,
) -> c_int;

unsafe extern "C" {
    fn PyDict_AddWatcher(callback: Callback) -> c_int;
    fn PyDict_Watch(watcher_id: c_int, dict: *mut ffi::PyObject) -> c_int;
}

const UNTRIED: i32 = -2;
const UNAVAILABLE: i32 = -1;

/// One set of watched dicts' generation. ``MODULES`` and ``DECODE`` share one
/// CPython watcher slot (``SLOT``), of the few an interpreter allows; one
/// callback tells their dicts apart.
pub struct Watch {
    generation: AtomicU64,
}

/// The CPython watcher every ``Watch`` uses, once registered.
static SLOT: AtomicI32 = AtomicI32::new(UNTRIED);

/// The ``sys.modules`` dict ``MODULES`` watches last, by address, or 0.
static MODULES_DICT: AtomicUsize = AtomicUsize::new(0);

impl Watch {
    const fn new() -> Self {
        Self {
            generation: AtomicU64::new(0),
        }
    }

    /// The current generation, or ``None`` when no watcher could be had, so
    /// nothing read from a watched dict may be trusted past its first use.
    #[inline]
    pub fn generation(&self) -> Option<u64> {
        if SLOT.load(Ordering::Relaxed) < 0 {
            return None;
        }
        Some(self.generation.load(Ordering::Relaxed))
    }

    /// Watch ``dict``; ``false`` when it cannot be watched. Watching a dict
    /// twice is harmless.
    pub fn watch(&self, dict: &Bound<'_, PyDict>) -> bool {
        let Some(id) = slot() else {
            return false;
        };
        // Noted before watching, so its first change is told apart.
        if std::ptr::eq(self, &MODULES) {
            MODULES_DICT.store(dict.as_ptr() as usize, Ordering::Relaxed);
        } else if std::ptr::eq(self, &DECODE) && !note_decode_dict(dict.as_ptr() as usize) {
            return false;
        }
        // Watching fails only for a non-dict or a bad id, and neither is passed.
        unsafe { PyDict_Watch(id, dict.as_ptr()) };
        true
    }
}

/// The watcher slot, registered on first use; ``None`` when none is left.
fn slot() -> Option<c_int> {
    let mut id = SLOT.load(Ordering::Relaxed);
    if id == UNTRIED {
        // CPython allows a few watchers per interpreter; when none is left,
        // callers read the dicts at each use.
        id = unsafe { PyDict_AddWatcher(changed) };
        if id < 0 {
            unsafe { ffi::PyErr_Clear() };
            id = UNAVAILABLE;
        }
        SLOT.store(id, Ordering::Relaxed);
    }
    (id >= 0).then_some(id)
}

/// ``PyDict_EVENT_DEALLOCATED`` (3.12 to 3.14).
const PY_DICT_EVENT_DEALLOCATED: c_int = 5;

/// The one callback for every watched dict. Every dict moves ``MODULES``,
/// which is never wrong: an extra change only reads a module again. A
/// ``DECODE`` dict moves ``DECODE`` too, and ``sys.modules`` ``IMPORTS``.
unsafe extern "C" fn changed(
    event: c_int,
    dict: *mut ffi::PyObject,
    _key: *mut ffi::PyObject,
    _new_value: *mut ffi::PyObject,
) -> c_int {
    MODULES.generation.fetch_add(1, Ordering::Relaxed);
    if event == PY_DICT_EVENT_DEALLOCATED {
        forget_namespace(dict as usize);
    }
    if is_decode_dict(dict as usize) {
        DECODE.generation.fetch_add(1, Ordering::Relaxed);
        if event == PY_DICT_EVENT_DEALLOCATED {
            forget_decode_dict(dict as usize);
        }
        return 0;
    }
    if dict as usize == MODULES_DICT.load(Ordering::Relaxed) {
        IMPORTS.fetch_add(1, Ordering::Relaxed);
        // A freed ``sys.modules`` forgets its address, so a dict made there
        // later is not taken for it.
        if event == PY_DICT_EVENT_DEALLOCATED {
            MODULES_DICT.store(0, Ordering::Relaxed);
        }
    }
    0
}

/// ``sys.modules`` and the module namespaces import paths were read from.
pub static MODULES: Watch = Watch::new();

/// Moved by changes to ``sys.modules`` alone, of every dict ``MODULES``
/// watches.
static IMPORTS: AtomicU64 = AtomicU64::new(0);

/// ``IMPORTS``' count, or ``None`` until ``sys.modules`` is watched.
pub fn imports() -> Option<u64> {
    (MODULES_DICT.load(Ordering::Relaxed) != 0 && MODULES.generation().is_some())
        .then(|| IMPORTS.load(Ordering::Relaxed))
}

/// The dicts ``from_plain`` copies import results from: the
/// ``treekle.deserialize`` globals and their ``_RESOLVED`` cache.
pub static DECODE: Watch = Watch::new();

/// Module namespaces a kept import path was read from, by address. Each is
/// watched, so rebinding a name in one moves ``MODULES``.
static NAMESPACES: std::sync::Mutex<Option<hashbrown::HashSet<usize>>> =
    std::sync::Mutex::new(None);

/// Past this many namespaces, a new one is not watched and its paths are
/// rechecked at each use instead.
const NAMESPACE_LIMIT: usize = 1 << 12;

/// Watch module namespace ``dict`` so its changes move ``MODULES``; ``false``
/// when it cannot be watched.
pub fn watch_namespace(dict: &Bound<'_, PyDict>) -> bool {
    let address = dict.as_ptr() as usize;
    let Ok(mut held) = NAMESPACES.lock() else {
        return false;
    };
    let held = held.get_or_insert_with(Default::default);
    if held.contains(&address) {
        return true;
    }
    if held.len() >= NAMESPACE_LIMIT {
        return false;
    }
    let Some(id) = slot() else {
        return false;
    };
    held.insert(address);
    // Watching fails only for a non-dict or a bad id, and neither is passed.
    unsafe { PyDict_Watch(id, dict.as_ptr()) };
    true
}

/// A freed namespace forgets its address, so a dict made there later is
/// watched before a path read from it is trusted.
fn forget_namespace(address: usize) {
    if let Ok(mut held) = NAMESPACES.lock()
        && let Some(held) = held.as_mut()
    {
        held.remove(&address);
    }
}

/// The dicts ``DECODE`` watches, by address; 0 marks a free entry.
static DECODE_DICTS: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

#[inline]
fn is_decode_dict(address: usize) -> bool {
    DECODE_DICTS
        .iter()
        .any(|each| each.load(Ordering::Relaxed) == address)
}

/// Note ``address`` as a ``DECODE`` dict; ``false`` when the table is full,
/// as its changes would then move only ``MODULES``.
fn note_decode_dict(address: usize) -> bool {
    is_decode_dict(address)
        || DECODE_DICTS.iter().any(|each| {
            each.compare_exchange(0, address, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        })
}

fn forget_decode_dict(address: usize) {
    for each in &DECODE_DICTS {
        let _ = each.compare_exchange(address, 0, Ordering::Relaxed, Ordering::Relaxed);
    }
}
