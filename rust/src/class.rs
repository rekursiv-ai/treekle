//! What is read from a class natively, with no Python code run: its stamp,
//! its attribute lookups and slots, and whether an import path still names it.

use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::py::*;
use crate::sync;

unsafe extern "C" {
    fn _PyType_Lookup(kind: *mut ffi::PyTypeObject, name: *mut ffi::PyObject)
    -> *mut ffi::PyObject;
}

// ------------------------------------------------------------------ stamps --

unsafe extern "C" {
    fn PyUnstable_Type_AssignVersionTag(kind: *mut ffi::PyTypeObject) -> std::ffi::c_int;
}

/// The class's version tag, assigned if it has none; 0 when out of tags.
#[inline]
pub fn version_tag(kind: *mut ffi::PyTypeObject) -> u32 {
    unsafe {
        if (*kind).tp_version_tag == 0 {
            PyUnstable_Type_AssignVersionTag(kind);
        }
        (*kind).tp_version_tag
    }
}

/// What a fact learned from a class assumed about it. CPython replaces a
/// class's version tag whenever the class or a base changes, and never reuses
/// one; but 3.13 and later rename a class (``__qualname__``) without a new tag,
/// so the name object is held and compared too. Immutable types never change.
pub struct Stamp {
    tag: u32,
    qualname: Option<Py<PyAny>>,
}

impl Stamp {
    /// The class's current state, or ``None`` if it cannot change.
    pub fn of(class: &B<'_>) -> Option<Self> {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_IMMUTABLETYPE) } != 0 {
            return None;
        }
        Some(Self::always(class))
    }

    /// The class's current state, immutable classes included. A fact kept
    /// across calls by class address needs this: version tags are never
    /// reused, so a new class that takes a dead one's address never matches.
    /// A class out of tags stays at 0, which never matches.
    pub fn always(class: &B<'_>) -> Self {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        let tag = version_tag(kind);
        let qualname = unsafe { heap_qualname(kind) };
        Self {
            tag,
            qualname: (!qualname.is_null())
                .then(|| unsafe { Py::from_borrowed_ptr(class.py(), qualname) }),
        }
    }

    /// Whether ``class`` is unchanged since this stamp was taken of it.
    #[inline]
    pub fn holds(&self, class: &B<'_>) -> bool {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        let qualname = self
            .qualname
            .as_ref()
            .map_or(std::ptr::null_mut(), Py::as_ptr);
        self.tag != 0
            && unsafe { (*kind).tp_version_tag == self.tag && heap_qualname(kind) == qualname }
    }
}

#[inline]
unsafe fn heap_qualname(kind: *mut ffi::PyTypeObject) -> *mut ffi::PyObject {
    unsafe {
        if ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
            return std::ptr::null_mut();
        }
        (*kind.cast::<ffi::PyHeapTypeObject>()).ht_qualname
    }
}

/// ``_PyType_Lookup``: the first of the MRO's dicts holding ``name``. Read
/// here dict by dict, as it reads them, without filling the type cache a
/// class seen once never reuses. Borrowed, as the class's dicts hold it.
#[inline]
pub unsafe fn type_lookup(
    kind: *mut ffi::PyTypeObject,
    name: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // An exact str's hash and ``==`` cannot fail, so the lookup runs no code;
    // CPython's type cache answers a repeated one.
    unsafe { _PyType_Lookup(kind, name) }
}

/// What ``getattr(class, name)`` does, where that is known with no code run.
pub enum ClassAttr {
    /// Gives the MRO's own entry, borrowed; with ``class``'s metaclass, which
    /// a caller keeping the answer must stamp too.
    Found(*mut ffi::PyObject, *mut ffi::PyTypeObject),
    /// Raises ``AttributeError``: neither the metaclass nor the MRO holds it.
    Missing,
    /// Only Python can say: the metaclass could answer first, or the entry
    /// could answer for itself (a class whose metaclass has ``__get__``, or
    /// any descriptor but a plain function, whose ``__get__`` on a class
    /// gives the function).
    Ask,
}

/// ``getattr(class, name)``, read as ``type``'s lookup reads it.
///
/// # Safety
/// ``class`` is live; ``name`` is an exact str.
pub unsafe fn class_attr(class: *mut ffi::PyObject, name: *mut ffi::PyObject) -> ClassAttr {
    unsafe {
        let Some(meta) = type_reads(class, name) else {
            return ClassAttr::Ask;
        };
        let entry = type_lookup(class.cast(), name);
        if entry.is_null() {
            return ClassAttr::Missing;
        }
        let of = ob_type(entry);
        if (*of).tp_descr_get.is_some() && of != &raw mut ffi::PyFunction_Type {
            return ClassAttr::Ask;
        }
        ClassAttr::Found(entry, meta)
    }
}

/// ``class``'s metaclass where ``getattr(class, name)`` answers from
/// ``class``'s MRO: ``type``, or a subclass (``ABCMeta``) keeping ``type``'s
/// lookup, whose own MRO holds no ``name`` to answer first. ``type`` itself
/// holds some (``__name__``, ``__bases__``), which answer before the class.
///
/// # Safety
/// ``class`` is live; ``name`` is an exact str.
pub unsafe fn type_reads(
    class: *mut ffi::PyObject,
    name: *mut ffi::PyObject,
) -> Option<*mut ffi::PyTypeObject> {
    unsafe {
        let meta = ob_type(class);
        let base = &raw mut ffi::PyType_Type;
        let reads = (meta == base
            || ffi::PyType_IsSubtype(meta, base) != 0
                && (*meta).tp_getattro.map(|f| f as usize)
                    == (*base).tp_getattro.map(|f| f as usize))
            && type_lookup(meta, name).is_null();
        reads.then_some(meta)
    }
}

/// ``namespace[key]`` as a borrowed reference, or null where it is absent.
/// Callers pass an exact dict and an exact str key, whose lookup cannot fail.
#[cfg(Py_3_13)]
unsafe fn dict_borrowed(
    namespace: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    unsafe {
        let mut found = std::ptr::null_mut();
        if ffi::PyDict_GetItemRef(namespace, key, &mut found) != 1 {
            return std::ptr::null_mut();
        }
        ffi::Py_DECREF(found);
        found
    }
}

/// ``namespace[key]`` as a borrowed reference, or null where it is absent.
/// Callers pass an exact dict and an exact str key, whose lookup cannot fail.
#[cfg(not(Py_3_13))]
unsafe fn dict_borrowed(
    namespace: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    unsafe { ffi::PyDict_GetItem(namespace, key) }
}

pub const PY_AUDIT_READ: std::ffi::c_int = 2;

/// The slot ``found`` describes for instances of ``kind``, if it is a member
/// descriptor that applies to them. A class can install another class's
/// member descriptor under a slot name; its offset is not an offset in this
/// instance, so that descriptor is left to ``getattr``, which enforces
/// ownership, rather than read as arbitrary memory.
///
/// # Safety
/// ``kind`` is a live class; ``found`` is null or a live object.
pub unsafe fn member_of(
    kind: *mut ffi::PyTypeObject,
    found: *mut ffi::PyObject,
) -> *mut ffi::PyMemberDef {
    unsafe {
        if found.is_null() || ob_type(found) != &raw mut ffi::PyMemberDescr_Type {
            return std::ptr::null_mut();
        }
        let descriptor = found.cast::<ffi::PyMemberDescrObject>();
        if ffi::PyType_IsSubtype(kind, (*descriptor).d_common.d_type) == 0 {
            return std::ptr::null_mut();
        }
        (*descriptor).d_member.cast::<ffi::PyMemberDef>()
    }
}

/// The offset of the object member ``kind``'s MRO finds first under
/// ``name``, when it has none of the ``refused`` flags.
fn member_offset(
    kind: *mut ffi::PyTypeObject,
    name: &B<'_>,
    refused: std::ffi::c_int,
) -> Option<isize> {
    if tp(name) != t_str() {
        return None;
    }
    unsafe {
        let member = member_of(kind, type_lookup(kind, name.as_ptr()));
        if member.is_null()
            || (*member).type_code != ffi::Py_T_OBJECT_EX
            || (*member).flags & refused != 0
        {
            return None;
        }
        Some((*member).offset)
    }
}

/// Where ``object.__setattr__`` stores ``name`` in ``kind``'s instances: a
/// writable plain object member.
pub fn store_offset(kind: *mut ffi::PyTypeObject, name: &B<'_>) -> Option<isize> {
    member_offset(kind, name, ffi::Py_READONLY)
}

/// Where ``object.__setattr__`` stores one attribute of an instance.
#[derive(Clone, Copy)]
pub enum Store {
    /// A writable plain object member at this offset.
    Slot(isize),
    /// The instance dict: no data descriptor on the class takes the name.
    Dict,
}

/// Where ``object.__setattr__`` stores each of ``names`` in ``kind``'s
/// instances, ``None`` for a name only the call itself can store; stamped
/// before it is read. Empty where the call is not
/// ``PyObject_GenericSetAttr``: it is wherever the first class in ``kind``'s
/// base chain that Python did not make sets attributes as ``object`` does.
pub fn stores(kind: &B<'_>, names: &[Box<[u8]>]) -> (Stamp, Vec<Option<Store>>) {
    let py = kind.py();
    let stamp = Stamp::always(kind);
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    let generic = unsafe {
        // ``object.__setattr__`` passes over Python classes' own
        // ``__setattr__``; the first static class it meets must set as
        // ``object`` does.
        let mut base = class;
        while !base.is_null() && ffi::PyType_HasFeature(base, ffi::Py_TPFLAGS_HEAPTYPE) != 0 {
            base = (*base).tp_base;
        }
        ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
            && !base.is_null()
            && (*base).tp_setattro.map(|f| f as usize)
                == ffi::PyBaseObject_Type.tp_setattro.map(|f| f as usize)
    };
    if !generic {
        return (stamp, Vec::new());
    }
    let dict = dict_storage(class);
    let stores = names
        .iter()
        .map(|name| {
            let text = std::str::from_utf8(name).ok()?;
            let name = PyString::intern(py, text);
            if let Some(offset) = store_offset(class, name.as_any()) {
                return Some(Store::Slot(offset));
            }
            let found = unsafe { type_lookup(class, name.as_ptr()) };
            let plain = found.is_null() || unsafe { (*ob_type(found)).tp_descr_set.is_none() };
            (dict && plain).then_some(Store::Dict)
        })
        .collect();
    (stamp, stores)
}

/// ``kind``'s own namespace, for a heap class straight under ``object``
/// with a plain ``type`` metaclass: then every attribute lookup on it reads
/// this dict, then ``object``'s.
pub fn own_namespace<'py>(kind: &B<'py>) -> Option<Bound<'py, PyDict>> {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    unsafe {
        if ob_type(kind.as_ptr()) != &raw mut ffi::PyType_Type
            || ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0
        {
            return None;
        }
        let mro = (*class).tp_mro;
        if ffi::PyTuple_GET_SIZE(mro) != 2
            || ffi::PyTuple_GET_ITEM(mro, 1) != (&raw mut ffi::PyBaseObject_Type).cast()
        {
            return None;
        }
        // A heap class's namespace is always an exact dict.
        Some(Bound::from_borrowed_ptr(kind.py(), (*class).tp_dict).cast_into_unchecked::<PyDict>())
    }
}

/// Whether ``kind``'s instances have room for an instance dict: a dict
/// slot, or one CPython manages.
#[inline]
pub fn dict_storage(kind: *mut ffi::PyTypeObject) -> bool {
    unsafe {
        (*kind).tp_dictoffset != 0
            || ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_MANAGED_DICT) != 0
    }
}

/// ``object.__new__``'s slot, where ``kind.__new__(kind)`` is a call of it:
/// a plain ``type`` metaclass finds ``object``'s own ``__new__``, and
/// ``kind`` inherits its slot.
pub fn object_new(kind: &B<'_>) -> Option<ffi::newfunc> {
    let py = kind.py();
    unsafe {
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let name = intern!(py, "__new__").as_ptr();
        let base = &raw mut ffi::PyBaseObject_Type;
        let new = (*base).tp_new?;
        let plain = ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
            && (*class).tp_new.map(|f| f as usize) == Some(new as usize)
            && !type_lookup(class, name).is_null()
            && type_lookup(class, name) == type_lookup(base, name);
        plain.then_some(new)
    }
}

// ----------------------------------------------------------- import paths --

/// What the path reader reads besides the class: ``sys.__dict__``, where
/// ``sys.modules`` is bound, and ``types.ModuleType``.
pub struct PathGlobals {
    pub sys_dict: Py<PyDict>,
    pub module_type: Py<PyAny>,
    /// The module a path was last read from: its name, the ``MODULES``
    /// generation then, and its watched namespace. Paths of one module come
    /// in runs, and each reads that module again.
    pub last: std::cell::RefCell<Option<ModuleRead>>,
}

/// A module's name, the ``MODULES`` generation it was read at, and its
/// namespace.
pub type ModuleRead = (Py<PyString>, u64, Py<PyDict>);

impl PathGlobals {
    /// ``sys.modules``, where it is an exact dict, read from ``sys``'s own
    /// dict by an interned key.
    pub fn modules<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        Ok(self
            .sys_dict
            .bind(py)
            .get_item(intern!(py, "modules"))?
            .and_then(|modules| modules.cast_into_exact::<PyDict>().ok()))
    }

    /// The namespace of the module ``sys.modules`` holds under ``name``,
    /// kept for the next path of the same module while ``sys.modules`` and
    /// the namespaces watched stand; ``None`` where Python must say.
    pub fn module_namespace<'py>(
        &self,
        name: &Bound<'py, PyString>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let py = name.py();
        if let Some(generation) = sync::MODULES.generation()
            && let Some((module, at, namespace)) = &*self.last.borrow()
            && *at == generation
            && module.as_ptr() == name.as_ptr()
        {
            return Ok(Some(namespace.bind(py).clone()));
        }
        let Some(modules) = self.modules(py)? else {
            return Ok(None);
        };
        // Watched before it is read, so a change from here on is seen.
        let watched = sync::MODULES.watch(&modules);
        let Some(module) = modules.get_item(name)? else {
            return Ok(None);
        };
        let Some(namespace) = self.namespace(&module) else {
            return Ok(None);
        };
        if watched
            && sync::watch_namespace(&namespace)
            && let Some(generation) = sync::MODULES.generation()
        {
            let stale = self.last.replace(Some((
                name.clone().unbind(),
                generation,
                namespace.clone().unbind(),
            )));
            drop(stale);
        }
        Ok(Some(namespace))
    }

    /// ``module``'s namespace where it is an exact ``types.ModuleType``,
    /// whose attributes ``module_entry`` then reads from it.
    pub fn namespace<'py>(&self, module: &B<'py>) -> Option<Bound<'py, PyDict>> {
        (module.get_type().as_ptr() == self.module_type.as_ptr()).then(|| unsafe {
            Bound::from_borrowed_ptr(module.py(), ffi::PyModule_GetDict(module.as_ptr()))
                .cast_into_unchecked::<PyDict>()
        })
    }

    /// ``getattr(module, name)`` read from the exact module's ``namespace``
    /// with no code run, borrowed: null where the dict lacks ``name``, which
    /// a module ``__getattr__`` may answer, or where ``ModuleType`` holds a
    /// data descriptor of that name (``__class__``, ``__dict__``), which
    /// answers before the dict.
    ///
    /// # Safety
    /// ``namespace`` is an exact module's dict; ``name`` is an exact str.
    pub unsafe fn module_entry(
        &self,
        namespace: *mut ffi::PyObject,
        name: *mut ffi::PyObject,
    ) -> *mut ffi::PyObject {
        unsafe {
            // Every data descriptor ``ModuleType`` and ``object`` hold is a
            // dunder, so no other name needs the lookup.
            if utf8_ptr(name).is_some_and(|text| text.starts_with(b"__")) {
                let found = type_lookup(self.module_type.as_ptr().cast(), name);
                if !found.is_null() && (*ob_type(found)).tp_descr_set.is_some() {
                    return std::ptr::null_mut();
                }
            }
            dict_borrowed(namespace, name)
        }
    }
}

// Perf: rows to_plain, perf stat instructions: +0.5% over checking the
// ``MODULES`` generation alone. A recheck is one atomic load and one stamp compare per class the
// path passes through; see ``holds``.
/// How to ask, with no Python code run, whether a kept import path still
/// names its object, as ``reflect.resolves_to`` would answer.
///
/// The path is its module's name, then attribute names. It names its object
/// while no longer prefix is a loaded module, and the module's entry for the
/// first name (``module_entry``), then each class's entry for the next
/// (``class_attr``), leads there. A dunder name after the first may be a
/// class's own descriptor, so it is left to Python.
///
/// Its answer depends on ``sys.modules``, the module's namespace, and each
/// class walked through to reach the last name. The first two are watched
/// dicts, so a change to either moves ``sync::MODULES``; a class's version
/// tag moves whenever it or a base gains, loses, or rebinds an attribute.
pub struct PathCheck {
    /// Each dotted prefix longer than the module's name.
    longer: Vec<Py<PyString>>,
    module: Py<PyString>,
    attrs: Vec<Py<PyString>>,
    /// The module's namespace, found while ``sync::MODULES`` stood at this
    /// generation: until ``sys.modules`` changes, the path's module part
    /// resolves to it again, so only its names are read.
    found: std::cell::RefCell<Option<(u64, Py<PyDict>)>>,
    /// The ``sync::MODULES`` generation the path last held at, with the
    /// stamp of each class the walk passed through: while the generation
    /// stands and every stamp holds, the path still holds.
    held: std::cell::RefCell<Option<(u64, Through)>>,
}

/// Each class a path's walk passed through, with its stamp at the time.
type Through = Vec<(Py<PyAny>, Stamp)>;

impl PathCheck {
    /// The check for ``path`` read in module ``module``, or ``None`` where
    /// only Python can tell.
    pub fn new(path: &B<'_>, module: &B<'_>) -> Option<Self> {
        let py = path.py();
        let text =
            std::str::from_utf8((tp(path) == t_str()).then(|| utf8(path)).flatten()?).ok()?;
        let module_text =
            std::str::from_utf8((tp(module) == t_str()).then(|| utf8(module)).flatten()?).ok()?;
        let rest = text.strip_prefix(module_text)?.strip_prefix('.')?;
        let names: Vec<&str> = rest.split('.').collect();
        // A dunder module attribute (``copyreg.__newobj__``) is read as
        // ``module_entry`` reads it; a dunder class attribute is not.
        if names.iter().any(|name| name.is_empty())
            || names.iter().skip(1).any(|name| name.starts_with("__"))
        {
            return None;
        }
        let mut longer = Vec::with_capacity(names.len());
        let mut prefix = module_text.to_owned();
        for name in &names {
            prefix.push('.');
            prefix.push_str(name);
            longer.push(PyString::new(py, &prefix).unbind());
        }
        Some(Self {
            longer,
            module: PyString::intern(py, module_text).unbind(),
            attrs: names
                .iter()
                .map(|name| PyString::intern(py, name).unbind())
                .collect(),
            found: std::cell::RefCell::new(None),
            held: std::cell::RefCell::new(None),
        })
    }

    /// The check for ``path`` read from its longest loaded prefix, as
    /// ``resolves_to`` reads it, or ``None`` where only Python can tell.
    pub fn loaded(g: &PathGlobals, path: &B<'_>) -> PyResult<Option<Self>> {
        let py = path.py();
        let Some(text) = (tp(path) == t_str()).then(|| utf8(path)).flatten() else {
            return Ok(None);
        };
        let Ok(text) = std::str::from_utf8(text) else {
            return Ok(None);
        };
        let Some(modules) = g.modules(py)? else {
            return Ok(None);
        };
        for (at, _) in text.rmatch_indices('.') {
            let module = PyString::new(py, &text[..at]);
            if modules.contains(&module)? {
                return Ok(Self::new(path, &module));
            }
        }
        Ok(None)
    }

    /// Whether the path names ``value`` now; ``None`` where Python must say.
    /// The check is only ever asked about the one object it was made for.
    #[inline]
    pub fn holds(&self, g: &PathGlobals, value: &B<'_>) -> PyResult<Option<bool>> {
        let py = value.py();
        if let Some(generation) = sync::MODULES.generation()
            && let Some((at, through)) = &*self.held.borrow()
            && *at == generation
            && through
                .iter()
                .all(|(class, stamp)| stamp.holds(class.bind(py)))
        {
            return Ok(Some(true));
        }
        self.held_now(g, value)
    }

    #[cold]
    fn held_now(&self, g: &PathGlobals, value: &B<'_>) -> PyResult<Option<bool>> {
        let mut through = Vec::with_capacity(self.attrs.len().saturating_sub(1));
        let verdict = self.walk(g, value, &mut through)?;
        // Kept only once each dict the walk read is watched (``sys.modules``
        // by ``namespace``, the module's namespace here) and each class it
        // passed through is stamped, its stamp taken before it was read.
        // A namespace is kept in ``found`` only once it is watched.
        let watched = self.found.borrow().is_some();
        let kept = if verdict == Some(true) && watched {
            sync::MODULES.generation().map(|now| (now, through))
        } else {
            None
        };
        let stale = self.held.replace(kept);
        drop(stale);
        Ok(verdict)
    }

    /// The walk ``holds`` trusts, each class it passes through recorded in
    /// ``through`` with the stamp taken before its entry was read; ``None``
    /// where a class cannot be stamped or only Python can say.
    fn walk(
        &self,
        g: &PathGlobals,
        value: &B<'_>,
        through: &mut Through,
    ) -> PyResult<Option<bool>> {
        let py = value.py();
        let namespace = match self.namespace(g, py)? {
            Some(namespace) => namespace,
            None => return Ok(None),
        };
        let (first, rest) = self.attrs.split_first().expect("a path names an attribute");
        let entry = unsafe { g.module_entry(namespace.as_ptr(), first.as_ptr()) };
        if entry.is_null() {
            return Ok(None);
        }
        let mut found = unsafe { Bound::from_borrowed_ptr(py, entry) };
        for name in rest {
            // A tag of 0 never holds, so a class out of tags is read each
            // time; stamped before its entry is read.
            let stamp = Stamp::always(&found);
            let ClassAttr::Found(entry, meta) =
                (unsafe { class_attr(found.as_ptr(), name.as_ptr()) })
            else {
                return Ok(None);
            };
            through.push((found.clone().unbind(), stamp));
            // A metaclass that gains ``name`` would answer before the class.
            if meta != &raw mut ffi::PyType_Type {
                let meta = unsafe { Bound::from_borrowed_ptr(py, meta.cast()) };
                let stamp = Stamp::always(&meta);
                through.push((meta.unbind(), stamp));
            }
            found = unsafe { Bound::from_borrowed_ptr(py, entry) };
        }
        Ok(Some(found.is(value)))
    }

    /// The namespace of the module the path is read from, kept while
    /// ``sys.modules`` is unchanged; ``None`` where Python must say.
    fn namespace<'py>(
        &self,
        g: &PathGlobals,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(generation) = sync::MODULES.generation()
            && let Some((at, namespace)) = &*self.found.borrow()
            && *at == generation
        {
            return Ok(Some(namespace.bind(py).clone()));
        }
        let Some(modules) = g.modules(py)? else {
            return Ok(None);
        };
        // Watched before it is read, so a change from here on is seen.
        let watched = sync::MODULES.watch(&modules);
        for prefix in &self.longer {
            if modules.contains(prefix.bind(py))? {
                return Ok(None);
            }
        }
        let Some(module) = modules.get_item(self.module.bind(py))? else {
            return Ok(None);
        };
        let Some(namespace) = g.namespace(&module) else {
            return Ok(None);
        };
        if watched
            && sync::watch_namespace(&namespace)
            && let Some(generation) = sync::MODULES.generation()
        {
            let stale = self
                .found
                .replace(Some((generation, namespace.clone().unbind())));
            drop(stale);
        }
        Ok(Some(namespace))
    }
}
