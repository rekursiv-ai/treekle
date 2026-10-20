//! Python object access the decoder and encoder share: the GIL cell, option
//! flags, recursion guards, type tests, str bytes and dict iteration.

use std::cell::UnsafeCell;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyString, PyTuple};

pub type B<'py> = Bound<'py, PyAny>;

/// A value only touched while the GIL is held. It is only ever swapped or
/// cloned, never borrowed, so Python code that runs mid-call and re-enters
/// (a hook calling ``to_plain``) can never alias a live reference into it.
pub struct GilCell<T>(UnsafeCell<T>);

// SAFETY: the module declares it needs the GIL, and every access takes a
// `Python` token, so no two threads reach the cell at once.
unsafe impl<T> Sync for GilCell<T> {}

impl<T> GilCell<T> {
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Put ``value`` in the cell and return what it held. The old value is
    /// dropped by the caller, after the swap, so a ``__del__`` it triggers
    /// finds the cell consistent.
    pub fn replace(&self, _py: Python<'_>, value: T) -> T {
        // SAFETY: the GIL is held and no reference into the cell outlives a
        // call, so nothing else reads or writes it during the swap.
        unsafe { std::mem::replace(&mut *self.0.get(), value) }
    }
}

impl<T: Default> GilCell<T> {
    /// Take the cell's value, leaving the default.
    pub fn take(&self, py: Python<'_>) -> T {
        self.replace(py, T::default())
    }
}

impl<T: Clone> GilCell<T> {
    /// A copy of the cell's value.
    pub fn cloned(&self, _py: Python<'_>) -> T {
        // SAFETY: as in `replace`; `T::clone` here is an `Rc` or `Option`
        // clone, which runs no Python code.
        unsafe { (*self.0.get()).clone() }
    }
}

/// A ``to_plain`` or ``from_plain`` option, read as Python reads it: by truth
/// value at each use.
pub enum Flag<'py> {
    Known(bool),
    Object(B<'py>),
}

impl<'py> Flag<'py> {
    pub fn new(value: Option<B<'py>>, default: bool) -> Self {
        match value {
            None => Self::Known(default),
            Some(value) if tp(&value) == t_bool() => {
                Self::Known(value.as_ptr() == unsafe { ffi::Py_True() })
            }
            Some(value) => Self::Object(value),
        }
    }

    #[inline]
    pub fn get(&self) -> PyResult<bool> {
        match self {
            Self::Known(value) => Ok(*value),
            Self::Object(value) => value.is_truthy(),
        }
    }

    /// The option as Python holds it.
    pub fn object(&self, py: Python<'py>) -> B<'py> {
        match self {
            Self::Known(value) => pyo3::types::PyBool::new(py, *value).to_owned().into_any(),
            Self::Object(value) => value.clone(),
        }
    }
}

// Before 3.14, `Py_EnterRecursiveCall` only counts calls against a fixed limit
// (`Py_C_RECURSION_LIMIT`, 10000 on x86-64 Linux), and a walk's frames can
// exhaust the thread's stack long before that: an 8 MB stack near 7000 levels,
// a 512 KiB thread stack by 400. So those versions also check the real stack,
// as 3.14 does.
#[cfg(not(Py_3_14))]
mod stack {
    use std::cell::Cell;

    /// Stack left unused below the deepest level, for the Python code a walk
    /// calls there (hooks, helpers, reprs), which 3.12 and 3.13 run without
    /// stack checks of their own. A quarter of a small stack at most.
    const MARGIN: usize = 128 * 1024;

    #[derive(Clone, Copy)]
    struct Bounds {
        base: usize,
        limit: usize,
        top: usize,
    }

    thread_local! {
        /// `None` until read; `Some(None)` where the platform does not say.
        static BOUNDS: Cell<Option<Option<Bounds>>> = const { Cell::new(None) };
        static DEPTH: Cell<i32> = const { Cell::new(0) };
    }

    /// The thread's usable stack as `(lowest, highest)` address.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn read() -> Option<(usize, usize)> {
        unsafe {
            let mut attr: libc::pthread_attr_t = std::mem::zeroed();
            if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
                return None;
            }
            let mut addr: *mut libc::c_void = std::ptr::null_mut();
            let mut size: libc::size_t = 0;
            let mut guard: libc::size_t = 0;
            let ok = libc::pthread_attr_getstack(&attr, &mut addr, &mut size) == 0
                && libc::pthread_attr_getguardsize(&attr, &mut guard) == 0;
            libc::pthread_attr_destroy(&mut attr);
            (ok && size > guard).then(|| (addr as usize + guard, addr as usize + size))
        }
    }

    #[cfg(target_vendor = "apple")]
    fn read() -> Option<(usize, usize)> {
        unsafe {
            let thread = libc::pthread_self();
            let top = libc::pthread_get_stackaddr_np(thread) as usize;
            let size = libc::pthread_get_stacksize_np(thread);
            (size > 0 && top > size).then(|| (top - size, top))
        }
    }

    #[cfg(windows)]
    fn read() -> Option<(usize, usize)> {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
        }
        let (mut low, mut high) = (0, 0);
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        (high > low).then_some((low, high))
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    )))]
    fn read() -> Option<(usize, usize)> {
        None
    }

    fn bounds() -> Option<Bounds> {
        if let Some(known) = BOUNDS.get() {
            return known;
        }
        let known = read().map(|(base, top)| Bounds {
            base,
            limit: base + MARGIN.min((top - base) / 4),
            top,
        });
        BOUNDS.set(Some(known));
        known
    }

    /// Whether one more level, the `depth`-th, would leave too little stack.
    /// Where the stack is unknown, or this code runs outside it (a user-space
    /// thread), `depth` is held to `sys.getrecursionlimit()` instead.
    #[inline]
    pub fn too_deep(depth: i32) -> bool {
        let marker = 0u8;
        let here = std::hint::black_box(&marker) as *const u8 as usize;
        match bounds() {
            Some(known) if here >= known.base && here < known.top => here < known.limit,
            _ => depth > unsafe { pyo3::ffi::Py_GetRecursionLimit() },
        }
    }

    pub fn depth() -> &'static std::thread::LocalKey<Cell<i32>> {
        &DEPTH
    }
}

/// One level of recursion, counted as CPython counts C recursion; before 3.14
/// the real stack is checked too, so a deep tree raises ``RecursionError``
/// instead of overflowing it.
pub struct Recursion;

impl Recursion {
    #[inline]
    pub fn enter(py: Python<'_>, context: &'static std::ffi::CStr) -> PyResult<Self> {
        #[cfg(not(Py_3_14))]
        let depth = stack::depth().get() + 1;
        #[cfg(not(Py_3_14))]
        if stack::too_deep(depth) {
            return Err(pyo3::exceptions::PyRecursionError::new_err(format!(
                "maximum recursion depth exceeded{}",
                context.to_string_lossy()
            )));
        }
        if unsafe { ffi::Py_EnterRecursiveCall(context.as_ptr()) } != 0 {
            return Err(PyErr::fetch(py));
        }
        #[cfg(not(Py_3_14))]
        stack::depth().set(depth);
        Ok(Self)
    }
}

impl Drop for Recursion {
    #[inline]
    fn drop(&mut self) {
        #[cfg(not(Py_3_14))]
        stack::depth().set(stack::depth().get() - 1);
        unsafe { ffi::Py_LeaveRecursiveCall() };
    }
}

// Perf: exp000 from_plain, paired: 1.09x over calling the exported Py_TYPE / Py*_Check.
// On 3.14 pyo3-ffi binds ``Py_TYPE`` to libpython's exported function, a call
// per check; CPython's own header reads the field inline, as this does. The
// crate refuses free-threaded builds, so the field is a plain pointer.
/// ``Py_TYPE(ptr)``.
#[inline(always)]
pub fn ob_type(ptr: *mut ffi::PyObject) -> *mut ffi::PyTypeObject {
    unsafe { (*ptr).ob_type }
}

// On 3.14 pyo3-ffi also binds ``PyUnicode_KIND``/``PyUnicode_DATA`` to
// exported calls. These read the 3.12-3.14 (GIL build) layout directly:
// ``state`` holds interned:2, kind:3, compact:1, ascii:1 from bit 0.
/// A compact str's ``(kind, data, length, ascii)``; ``None`` for a legacy str.
#[inline(always)]
pub unsafe fn str_view(op: *mut ffi::PyObject) -> Option<(u32, *const u8, usize, bool)> {
    unsafe {
        let ascii_obj = op.cast::<ffi::PyASCIIObject>();
        let state = (*ascii_obj).state;
        if state >> 5 & 1 == 0 {
            return None;
        }
        let len = (*ascii_obj).length as usize;
        let ascii = state >> 6 & 1 == 1;
        let data = if ascii {
            op.cast::<u8>()
                .add(std::mem::size_of::<ffi::PyASCIIObject>())
        } else {
            op.cast::<u8>()
                .add(std::mem::size_of::<ffi::PyCompactUnicodeObject>())
        };
        Some((state >> 2 & 7, data, len, ascii))
    }
}

/// An exact int's value when it is compact (one 30-bit digit at most), read
/// from the 3.12-3.14 ``lv_tag`` layout: sign in the low two bits, digit
/// count from bit 3.
#[inline(always)]
pub unsafe fn compact_int(op: *mut ffi::PyObject) -> Option<i64> {
    unsafe {
        let base = op.cast::<u8>().add(std::mem::size_of::<ffi::PyObject>());
        let tag = *base.cast::<usize>();
        if tag >= 2 << 3 {
            return None;
        }
        let digit = i64::from(*base.add(std::mem::size_of::<usize>()).cast::<u32>());
        Some((1 - (tag & 3) as i64) * digit)
    }
}

/// Whether ``raw``'s class has ``flag``, as the ``Py*_Check`` macros test.
#[inline(always)]
pub fn has_flag(raw: *mut ffi::PyObject, flag: std::ffi::c_ulong) -> bool {
    unsafe { (*ob_type(raw)).tp_flags & flag != 0 }
}

#[inline(always)]
pub fn tp(obj: &B<'_>) -> *mut ffi::PyTypeObject {
    ob_type(obj.as_ptr())
}

macro_rules! type_ptr {
    ($name:ident, $sym:ident) => {
        #[inline(always)]
        pub fn $name() -> *mut ffi::PyTypeObject {
            &raw mut ffi::$sym
        }
    };
}

type_ptr!(t_dict, PyDict_Type);
type_ptr!(t_list, PyList_Type);
type_ptr!(t_tuple, PyTuple_Type);
type_ptr!(t_str, PyUnicode_Type);
type_ptr!(t_int, PyLong_Type);
type_ptr!(t_float, PyFloat_Type);
type_ptr!(t_bool, PyBool_Type);
type_ptr!(t_proxy, PyDictProxy_Type);

#[inline(always)]
pub fn t_none() -> *mut ffi::PyTypeObject {
    ob_type(unsafe { ffi::Py_None() })
}

#[inline(always)]
pub fn is_str(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_UNICODE_SUBCLASS)
}

#[inline(always)]
pub fn is_int(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_LONG_SUBCLASS)
}

#[inline(always)]
pub fn is_float(obj: &B<'_>) -> bool {
    let kind = tp(obj);
    kind == t_float() || unsafe { ffi::PyType_IsSubtype(kind, t_float()) != 0 }
}

#[inline(always)]
pub fn is_tuple(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_TUPLE_SUBCLASS)
}

#[inline(always)]
pub fn is_list(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_LIST_SUBCLASS)
}

#[inline(always)]
pub fn is_dict(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_DICT_SUBCLASS)
}

#[inline(always)]
pub fn is_class(obj: &B<'_>) -> bool {
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_TYPE_SUBCLASS)
}

/// ``isinstance(obj, (bytes, bytearray))``.
#[inline(always)]
pub fn is_bytes_like(obj: &B<'_>) -> bool {
    let kind = tp(obj);
    has_flag(obj.as_ptr(), ffi::Py_TPFLAGS_BYTES_SUBCLASS)
        || kind == &raw mut ffi::PyByteArray_Type
        || unsafe { ffi::PyType_IsSubtype(kind, &raw mut ffi::PyByteArray_Type) != 0 }
}

#[inline(always)]
pub fn is_callable(obj: &B<'_>) -> bool {
    unsafe { ffi::PyCallable_Check(obj.as_ptr()) != 0 }
}

// A libpython slot function is recognized by the address CPython stores in a
// builtin type's slot, not by naming the function. pyo3-ffi links libpython's
// functions without ``dllimport``, so on Windows a name resolves to this
// module's import stub, which no slot ever holds. Type objects are linked as
// data, so their slots hold the real addresses on every platform.

/// ``PyObject_HashNotImplemented``, which marks a class unhashable: ``list``
/// has it, as every class with ``__hash__ = None`` does.
#[inline(always)]
pub fn hash_not_implemented() -> Option<usize> {
    unsafe { ffi::PyList_Type.tp_hash }.map(|f| f as usize)
}

/// ``PyObject_GenericGetAttr``, ``object``'s own attribute lookup.
#[inline(always)]
pub fn generic_getattr_slot() -> Option<usize> {
    unsafe { ffi::PyBaseObject_Type.tp_getattro }.map(|f| f as usize)
}

/// ``type(obj).__hash__ is None``: whether instances of ``obj``'s class opt
/// out of hashing.
#[inline]
pub fn is_unhashable(obj: &B<'_>) -> PyResult<bool> {
    let kind = tp(obj);
    // Under a plain ``type`` metaclass the attribute is the class's own
    // ``__hash__``, which CPython keeps in step with ``tp_hash``. Any other
    // metaclass may answer the lookup itself (a ``__getattribute__``, a
    // ``__hash__`` property) while the slot still hashes.
    if ob_type(kind.cast()) == &raw mut ffi::PyType_Type {
        return Ok(unsafe { (*kind).tp_hash }.map(|f| f as usize) == hash_not_implemented());
    }
    let py = obj.py();
    Ok(obj
        .get_type()
        .getattr(pyo3::intern!(py, "__hash__"))?
        .is_none())
}

/// ``PyUnicode_IS_COMPACT_ASCII``, which pyo3 does not expose on 3.14.
///
/// With the GIL (the only build this module compiles for), 3.12 to 3.14 lay
/// out ``PyASCIIObject.state`` as one ``unsigned int`` after ``length`` and
/// ``hash``: ``interned:2, kind:3, compact:1, ascii:1``, low bit first.
#[inline(always)]
unsafe fn compact_ascii(raw: *mut ffi::PyObject) -> bool {
    const COMPACT: u32 = 1 << 5;
    const ASCII: u32 = 1 << 6;
    let header = raw
        .cast::<u8>()
        .wrapping_add(std::mem::size_of::<ffi::PyObject>())
        .cast::<ffi::Py_ssize_t>();
    let state = unsafe { *header.add(2).cast::<u32>() };
    state & (COMPACT | ASCII) == COMPACT | ASCII
}

/// Whether the exact str at ``raw`` is compact ASCII.
///
/// # Safety
/// ``raw`` is a live exact ``str``.
#[inline]
pub unsafe fn compact_ascii_str(raw: *mut ffi::PyObject) -> bool {
    unsafe { compact_ascii(raw) }
}

/// The bytes of a compact ASCII str.
///
/// # Safety
/// ``raw`` is a live compact ASCII ``str``, held while the bytes are used.
#[inline]
pub unsafe fn ascii_bytes<'a>(raw: *mut ffi::PyObject) -> &'a [u8] {
    unsafe {
        let ascii = raw.cast::<ffi::PyASCIIObject>();
        std::slice::from_raw_parts(ascii.add(1).cast::<u8>(), (*ascii).length as usize)
    }
}

/// The UTF-8 bytes of a str, or ``None`` when it holds lone surrogates.
#[inline]
pub fn utf8<'a>(obj: &'a B<'_>) -> Option<&'a [u8]> {
    unsafe { utf8_ptr(obj.as_ptr()) }
}

// Perf: exp000 from_plain, lockstep: ~1.03x.
/// ``utf8`` for a borrowed str pointer, valid while the str lives.
#[inline]
pub unsafe fn utf8_ptr<'a>(obj: *mut ffi::PyObject) -> Option<&'a [u8]> {
    unsafe {
        // A compact ASCII str's characters are its UTF-8 form and follow the
        // header, as ``PyUnicode_AsUTF8AndSize`` (an exported call) would
        // answer. The state word follows ``hash``; on non-free-threaded
        // 3.12-3.14 its bits are interned:2, kind:3, compact:1, ascii:1.
        debug_assert!(ffi::PyUnicode_Check(obj) != 0);
        let ascii = obj.cast::<ffi::PyASCIIObject>();
        let state = *std::ptr::addr_of!((*ascii).hash).add(1).cast::<u32>();
        if state & (0b11 << 5) == 0b11 << 5 {
            let size = (*ascii).length as usize;
            return Some(std::slice::from_raw_parts(ascii.add(1).cast::<u8>(), size));
        }
        let mut size: ffi::Py_ssize_t = 0;
        let data = ffi::PyUnicode_AsUTF8AndSize(obj, &mut size);
        if data.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        Some(std::slice::from_raw_parts(data.cast::<u8>(), size as usize))
    }
}

/// ``isinstance(obj, str) and obj.startswith(prefixes)``. A subclass may
/// override ``startswith``, and a str with lone surrogates has no UTF-8 form,
/// so both ask Python.
#[inline]
pub fn str_starts_any(obj: &B<'_>, prefixes: &[&str]) -> PyResult<bool> {
    if !is_str(obj) {
        return Ok(false);
    }
    if tp(obj) == t_str()
        && let Some(text) = utf8(obj)
    {
        return Ok(prefixes
            .iter()
            .any(|prefix| text.starts_with(prefix.as_bytes())));
    }
    let py = obj.py();
    let argument = match prefixes {
        [prefix] => pystr(py, prefix),
        _ => PyTuple::new(py, prefixes)?.into_any(),
    };
    obj.call_method1(pyo3::intern!(py, "startswith"), (argument,))?
        .is_truthy()
}

/// ``isinstance(obj, str) and obj.startswith(prefix)``.
#[inline]
pub fn str_starts(obj: &B<'_>, prefix: &str) -> PyResult<bool> {
    str_starts_any(obj, &[prefix])
}

pub fn repr(obj: &B<'_>) -> PyResult<String> {
    Ok(obj.repr()?.to_string_lossy().into_owned())
}

pub fn text(obj: &B<'_>) -> PyResult<String> {
    Ok(obj.str()?.to_string_lossy().into_owned())
}

pub fn err_text(py: Python<'_>, err: &PyErr) -> PyResult<String> {
    text(err.value(py).as_any())
}

/// ``obj == other`` with Python's identity shortcut.
pub fn py_eq(obj: &B<'_>, other: &B<'_>) -> PyResult<bool> {
    if obj.is(other) {
        return Ok(true);
    }
    let result = unsafe { ffi::PyObject_RichCompareBool(obj.as_ptr(), other.as_ptr(), ffi::Py_EQ) };
    if result < 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    Ok(result == 1)
}

/// ``obj != other`` as a condition.
pub fn py_ne(obj: &B<'_>, other: &B<'_>) -> PyResult<bool> {
    let result = unsafe { ffi::PyObject_RichCompareBool(obj.as_ptr(), other.as_ptr(), ffi::Py_NE) };
    if result < 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    Ok(result == 1)
}

/// The items of an exact list or tuple, borrowed: valid only until Python
/// code next runs, which may resize the list.
///
/// # Safety
/// ``obj`` must be an exact list or tuple.
#[inline(always)]
pub unsafe fn borrowed_items<'a>(obj: &'a B<'_>) -> &'a [*mut ffi::PyObject] {
    unsafe {
        let raw = obj.as_ptr();
        if ffi::Py_TYPE(raw) == t_list() {
            let list = raw.cast::<ffi::PyListObject>();
            let size = ffi::Py_SIZE(raw) as usize;
            if size == 0 {
                return &[];
            }
            std::slice::from_raw_parts((*list).ob_item, size)
        } else {
            let tuple = raw.cast::<ffi::PyTupleObject>();
            std::slice::from_raw_parts((*tuple).ob_item.as_ptr(), ffi::Py_SIZE(raw) as usize)
        }
    }
}

/// Whether ``obj`` is an exact dict whose keys are all exact ``str``.
pub fn is_str_keyed_dict(obj: &B<'_>) -> bool {
    if tp(obj) != t_dict() {
        return false;
    }
    let mut position: ffi::Py_ssize_t = 0;
    let mut key = std::ptr::null_mut();
    let mut value = std::ptr::null_mut();
    unsafe {
        while ffi::PyDict_Next(obj.as_ptr(), &mut position, &mut key, &mut value) != 0 {
            if ob_type(key) != t_str() {
                return false;
            }
        }
    }
    true
}

/// Every entry of an exact dict, read at once. No Python code runs while it
/// reads, so nothing can change the dict mid-read.
pub fn dict_entries<'py>(dict: &Bound<'py, PyDict>) -> Vec<(B<'py>, B<'py>)> {
    let py = dict.py();
    let mut entries = Vec::with_capacity(dict.len());
    let mut position: ffi::Py_ssize_t = 0;
    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
    let mut value: *mut ffi::PyObject = std::ptr::null_mut();
    unsafe {
        while ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) != 0 {
            entries.push((
                Bound::from_borrowed_ptr(py, key),
                Bound::from_borrowed_ptr(py, value),
            ));
        }
    }
    entries
}

/// An exact dict's items, read live as ``dict.items()`` would be.
///
/// User code can run between two items (a hook, an ``__init__``) and change
/// the dict. ``PyDict_Next`` walks entries in the order CPython's item
/// iterator does (3.12 to 3.14), so the iterator's two checks, made here the
/// same way, raise the same errors at the same item: the size it started
/// with, and no more items than that.
pub struct DictItems<'py> {
    dict: Bound<'py, PyDict>,
    position: ffi::Py_ssize_t,
    /// The size when iteration began; -1 once a size change was reported,
    /// so it is reported again, as the iterator's is.
    used: ffi::Py_ssize_t,
    left: ffi::Py_ssize_t,
}

impl<'py> DictItems<'py> {
    pub fn new(dict: &Bound<'py, PyDict>) -> Self {
        let used = unsafe { (*dict.as_ptr().cast::<ffi::PyDictObject>()).ma_used };
        Self {
            dict: dict.clone(),
            position: 0,
            used,
            left: used,
        }
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<(B<'py>, B<'py>)>> {
        let dict = self.dict.as_ptr();
        // ``PyDict_Size`` is an exported call; the field read is its body.
        if unsafe { (*dict.cast::<ffi::PyDictObject>()).ma_used } != self.used {
            self.used = -1;
            return Err(PyRuntimeError::new_err(
                "dictionary changed size during iteration",
            ));
        }
        let mut key = std::ptr::null_mut();
        let mut value = std::ptr::null_mut();
        if unsafe { ffi::PyDict_Next(dict, &mut self.position, &mut key, &mut value) } == 0 {
            return Ok(None);
        }
        if self.left == 0 {
            return Err(PyRuntimeError::new_err(
                "dictionary keys changed during iteration",
            ));
        }
        self.left -= 1;
        let py = self.dict.py();
        // SAFETY: both are borrowed from the live dict under the GIL and
        // owned here before any other code can run.
        unsafe {
            Ok(Some((
                Bound::from_borrowed_ptr(py, key),
                Bound::from_borrowed_ptr(py, value),
            )))
        }
    }
}

/// ``key, value = pair``, raising what that unpacking raises. A tuple or
/// list is sized first and names its size when it is too long; anything
/// else is iterated once, three items at most.
fn unpack_pair<'py>(pair: B<'py>) -> PyResult<(B<'py>, B<'py>)> {
    let kind = tp(&pair);
    if kind == t_tuple() || kind == t_list() {
        let size = pair.len()?;
        if size == 2 {
            return Ok((pair.get_item(0)?, pair.get_item(1)?));
        }
        // CPython names the size of a too-long sequence from 3.14 on.
        let message = if size < 2 {
            format!("not enough values to unpack (expected 2, got {size})")
        } else if cfg!(Py_3_14) {
            format!("too many values to unpack (expected 2, got {size})")
        } else {
            "too many values to unpack (expected 2)".to_owned()
        };
        return Err(PyValueError::new_err(message));
    }
    let mut iter = pair.try_iter()?;
    let mut next = |got: usize| {
        iter.next().transpose()?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "not enough values to unpack (expected 2, got {got})"
            ))
        })
    };
    let (first, second) = (next(0)?, next(1)?);
    if iter.next().transpose()?.is_some() {
        return Err(PyValueError::new_err(
            "too many values to unpack (expected 2)",
        ));
    }
    Ok((first, second))
}

/// The items of any mapping: a dict directly, else through ``.items()``.
pub enum MapItems<'py> {
    Dict(DictItems<'py>),
    Iter(Bound<'py, PyIterator>),
}

impl<'py> MapItems<'py> {
    pub fn new(mapping: &B<'py>) -> PyResult<Self> {
        if tp(mapping) == t_dict() {
            return Ok(Self::Dict(DictItems::new(unsafe {
                mapping.cast_unchecked::<PyDict>()
            })));
        }
        let items = mapping.call_method0(pyo3::intern!(mapping.py(), "items"))?;
        Ok(Self::Iter(items.try_iter()?))
    }

    /// The entry count of a dict, for sizing what reads them; 0 otherwise.
    pub fn len_hint(&self) -> usize {
        match self {
            Self::Dict(items) => items.used.max(0) as usize,
            Self::Iter(_) => 0,
        }
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<(B<'py>, B<'py>)>> {
        match self {
            Self::Dict(items) => items.next_item(),
            Self::Iter(iter) => iter.next().transpose()?.map(unpack_pair).transpose(),
        }
    }
}

/// The items of any iterable, a list read live as its iterator would.
pub enum SeqItems<'py> {
    List(Bound<'py, PyList>, usize),
    Tuple(Bound<'py, PyTuple>, usize),
    Iter(Bound<'py, PyIterator>),
}

impl<'py> SeqItems<'py> {
    pub fn new(seq: &B<'py>) -> PyResult<Self> {
        let kind = tp(seq);
        if kind == t_list() {
            return Ok(Self::List(
                unsafe { seq.cast_unchecked::<PyList>() }.clone(),
                0,
            ));
        }
        if kind == t_tuple() {
            return Ok(Self::Tuple(
                unsafe { seq.cast_unchecked::<PyTuple>() }.clone(),
                0,
            ));
        }
        Ok(Self::Iter(seq.try_iter()?))
    }

    /// The item count where known, for sizing what reads them.
    pub fn len_hint(&self) -> usize {
        match self {
            Self::List(list, _) => list.len(),
            Self::Tuple(tuple, _) => tuple.len(),
            Self::Iter(_) => 0,
        }
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<B<'py>>> {
        match self {
            Self::List(list, index) => {
                if *index >= list.len() {
                    return Ok(None);
                }
                let item = list.get_item(*index)?;
                *index += 1;
                Ok(Some(item))
            }
            Self::Tuple(tuple, index) => {
                if *index >= tuple.len() {
                    return Ok(None);
                }
                let item = tuple.get_item(*index)?;
                *index += 1;
                Ok(Some(item))
            }
            Self::Iter(iter) => iter.next().transpose(),
        }
    }
}

pub fn pystr<'py>(py: Python<'py>, value: &str) -> B<'py> {
    PyString::new(py, value).into_any()
}

/// An optional keyword argument whose absence differs from an explicit ``None``.
pub struct Maybe<'py>(pub Option<B<'py>>);

impl<'a, 'py> FromPyObject<'a, 'py> for Maybe<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(Some(obj.to_owned())))
    }
}
