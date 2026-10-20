//! Dict watchers that tell a native cache when the Python state it copied
//! changes.
//!
//! A walk keeps facts it read from Python dicts (a module's globals, a cache
//! such as ``_ENCODER_CACHE``) across nodes and calls. Rereading them at every
//! node would cost a lookup or a Python call each time; instead each watched
//! dict advances a generation whenever it changes, and a fact is trusted only
//! while the generation it was read at is current. CPython calls the watcher
//! before every change to a watched dict (3.12+), including deletion by a
//! weak-reference callback and deallocation, so no change goes unseen.

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

/// One watcher's generations. ``ENCODE``, ``MODULES`` and ``DECODE`` share
/// one CPython watcher slot (``SLOT``), of the few an interpreter allows; one
/// callback tells their dicts apart and advances each set of generations.
pub struct Watch {
    generation: AtomicU64,
    /// Advanced by every change but an entry added for ``expected``.
    rewrites: AtomicU64,
    /// The address of the class a walk is asking Python to cache now, or 0.
    expected: AtomicUsize,
}

/// The CPython watcher both ``Watch``es use, once registered.
static SLOT: AtomicI32 = AtomicI32::new(UNTRIED);

/// The ``sys.modules`` dict ``MODULES`` watches last, by address, or 0.
static MODULES_DICT: AtomicUsize = AtomicUsize::new(0);

impl Watch {
    const fn new() -> Self {
        Self {
            generation: AtomicU64::new(0),
            rewrites: AtomicU64::new(0),
            expected: AtomicUsize::new(0),
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

    /// The generation of every change but the entry a walk's own call adds
    /// for ``expected``: that call is the walk asking Python for the very fact
    /// it then keeps, and no fact it kept before read that key. ``None`` as
    /// ``generation`` gives it.
    #[inline]
    pub fn rewrites(&self) -> Option<u64> {
        if SLOT.load(Ordering::Relaxed) < 0 {
            return None;
        }
        Some(self.rewrites.load(Ordering::Relaxed))
    }

    fn id(&self, _py: Python<'_>) -> Option<c_int> {
        let mut id = SLOT.load(Ordering::Relaxed);
        if id == UNTRIED {
            // CPython allows a few watchers per interpreter; when none is
            // left, callers fall back to reading the dicts at each use.
            id = unsafe { PyDict_AddWatcher(changed) };
            if id < 0 {
                unsafe { ffi::PyErr_Clear() };
                id = UNAVAILABLE;
            }
            SLOT.store(id, Ordering::Relaxed);
        }
        (id >= 0).then_some(id)
    }

    /// Watch ``dict``; ``false`` when it cannot be watched. Watching a dict
    /// twice is harmless.
    pub fn watch(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> bool {
        let Some(id) = self.id(py) else {
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

    /// ``watch``, and every key ``dict`` already holds recorded as planted,
    /// once per dict: a cache filled before it was watched is then known.
    pub fn watch_seeded(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> bool {
        if !self.watch(py, dict) {
            return false;
        }
        let first = SEEDED.lock().map_or(true, |mut seen| {
            seen.get_or_insert_with(Default::default)
                .insert(dict.as_ptr() as usize)
        });
        if first {
            let mut position: ffi::Py_ssize_t = 0;
            let mut key = std::ptr::null_mut();
            let mut value = std::ptr::null_mut();
            while unsafe { ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) }
                != 0
            {
                plant(key);
            }
        }
        true
    }

    /// Run ``f`` while entries Python adds for class ``kind`` leave
    /// ``rewrites`` where it is.
    pub fn expecting<T>(&self, kind: *mut ffi::PyObject, f: impl FnOnce() -> T) -> T {
        let before = self.expected.swap(kind as usize, Ordering::Relaxed);
        let result = f();
        self.expected.store(before, Ordering::Relaxed);
        result
    }

    #[inline]
    fn changed(&self, event: c_int, key: *mut ffi::PyObject) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        if event == PY_DICT_EVENT_ADDED {
            plant(key);
        }
        let expected = self.expected.load(Ordering::Relaxed);
        if !(event == PY_DICT_EVENT_ADDED && expected != 0 && named(key) == expected) {
            self.rewrites.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// ``PyDict_EVENT_ADDED``, first of ``PyDict_WatchEvent`` (3.12 to 3.14).
const PY_DICT_EVENT_ADDED: c_int = 0;

/// The addresses each key added to a watched dict names: the key itself,
/// the address an exact int holds (``_IMPORT_PATHS``' ``id`` keys), or a
/// weak reference's referent (a ``WeakKeyDictionary``'s keys), as bits of a
/// filter. Bits are only ever set, so an address whose bit is clear was
/// never a key of any watched cache; a set bit only makes a lookup happen
/// that was not needed.
static PLANTED: [AtomicU64; PLANTED_WORDS] = [const { AtomicU64::new(0) }; PLANTED_WORDS];
const PLANTED_WORDS: usize = 1 << 10;

#[inline]
fn planted_bit(address: usize) -> (usize, u64) {
    // Objects are 16-byte aligned; a multiplicative hash spreads the rest.
    let hash = ((address >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
    ((hash >> 6) as usize % PLANTED_WORDS, 1 << (hash & 63))
}

/// The dicts whose keys ``watch_seeded`` has read, by address. A dict freed
/// and another made at its address is read again only if its address is
/// new; a dict there unread would be missed, so a dict's deallocation drops
/// its address (``PyDict_EVENT_DEALLOCATED``).
static SEEDED: std::sync::Mutex<Option<std::collections::HashSet<usize>>> =
    std::sync::Mutex::new(None);

/// ``PyDict_EVENT_CLONED`` and ``PyDict_EVENT_DEALLOCATED`` (3.12 to 3.14).
const PY_DICT_EVENT_CLONED: c_int = 3;
const PY_DICT_EVENT_DEALLOCATED: c_int = 5;

/// The address a cache key names: what an exact int holds
/// (``_IMPORT_PATHS``' ``id`` keys), a live weak reference's referent (a
/// ``WeakKeyDictionary``'s keys), or else the key itself. Reading any of
/// these runs no Python code; an int too large for an address names none.
fn named(key: *mut ffi::PyObject) -> usize {
    unsafe {
        let kind = (*key).ob_type;
        if kind == &raw mut ffi::PyLong_Type {
            let value = ffi::PyLong_AsVoidPtr(key);
            if value.is_null() {
                ffi::PyErr_Clear();
            }
            return value as usize;
        }
        if kind == &raw mut ffi::_PyWeakref_RefType {
            let referent = referent(key);
            if !referent.is_null() {
                return referent as usize;
            }
        }
    }
    key as usize
}

/// A live weak reference's referent, borrowed; null once it is dead.
#[cfg(Py_3_13)]
unsafe fn referent(weak: *mut ffi::PyObject) -> *mut ffi::PyObject {
    unsafe {
        let mut referent = std::ptr::null_mut();
        if ffi::PyWeakref_GetRef(weak, &mut referent) != 1 {
            return std::ptr::null_mut();
        }
        ffi::Py_DECREF(referent);
        referent
    }
}

/// A live weak reference's referent, borrowed; null once it is dead.
#[cfg(not(Py_3_13))]
unsafe fn referent(weak: *mut ffi::PyObject) -> *mut ffi::PyObject {
    unsafe {
        let referent = ffi::PyWeakref_GetObject(weak);
        if referent == ffi::Py_None() {
            return std::ptr::null_mut();
        }
        referent
    }
}

fn plant(key: *mut ffi::PyObject) {
    let (word, bit) = planted_bit(named(key));
    PLANTED[word].fetch_or(bit, Ordering::Relaxed);
}

/// Whether a watched cache may hold a key naming ``address``.
#[inline]
pub fn planted(address: usize) -> bool {
    let (word, bit) = planted_bit(address);
    PLANTED[word].load(Ordering::Relaxed) & bit != 0
}

/// The one callback for every dict any ``Watch`` watches.
///
/// Every dict moves ``MODULES``, which is never wrong: an extra change only
/// reads a module again. A ``DECODE`` dict moves ``DECODE`` too, and
/// ``sys.modules`` nothing more: imports are common, and each would otherwise
/// drop every fact a walk kept. Every other dict is an ``ENCODE`` dict. A
/// ``sys.modules`` replaced since it was watched is told apart no longer, so
/// it moves ``ENCODE`` too.
unsafe extern "C" fn changed(
    event: c_int,
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
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
        // A freed ``sys.modules`` forgets its address, so a dict made there
        // later is not taken for it.
        if event == PY_DICT_EVENT_DEALLOCATED {
            MODULES_DICT.store(0, Ordering::Relaxed);
        }
        return 0;
    }
    ENCODE.changed(event, key);
    // A dict freed, or filled wholesale from another (``CLONED``), is read
    // again when next seeded.
    if (event == PY_DICT_EVENT_DEALLOCATED || event == PY_DICT_EVENT_CLONED)
        && let Ok(mut seen) = SEEDED.lock()
        && let Some(seen) = seen.as_mut()
    {
        seen.remove(&(dict as usize));
    }
    0
}

/// The dicts ``to_plain`` copies facts from: ``treekle.serialize`` and
/// ``treekle.reflect`` globals and the caches they hold.
pub static ENCODE: Watch = Watch::new();

/// ``sys.modules``, so a module found there by name stays found until it moves.
pub static MODULES: Watch = Watch::new();

/// The dicts ``from_plain`` copies import results from: the
/// ``treekle.deserialize`` globals and their ``_RESOLVED`` cache. Their
/// changes move ``DECODE`` alone, so resolving an import leaves the
/// encoder's kept facts standing.
pub static DECODE: Watch = Watch::new();

/// Module namespaces a kept import path was read from, by address, watched
/// so a rebound name moves ``MODULES``, as every watched dict's change does;
/// ``changed`` takes one for an ``ENCODE`` dict like any other.
static NAMESPACES: std::sync::Mutex<Option<std::collections::HashSet<usize>>> =
    std::sync::Mutex::new(None);

/// Past this many namespaces, a new one is not watched and its paths are
/// rechecked at each use instead.
const NAMESPACE_LIMIT: usize = 1 << 12;

/// Watch module namespace ``dict`` so its changes move ``MODULES``; ``false``
/// when it cannot be watched.
pub fn watch_namespace(py: Python<'_>, dict: &Bound<'_, PyDict>) -> bool {
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
    let Some(id) = MODULES.id(py) else {
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
/// as its changes would then move ``ENCODE`` instead.
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
