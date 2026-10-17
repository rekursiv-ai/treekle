//! ``to_plain``: Python values to plain trees, matching ``treekle.serialize``.
//!
//! The walk is Python's ``_encode``: each container is built at once with its
//! leaves in place, and every other part waits on an explicit stack until it
//! is popped and written into its node, in the order a depth-first walk
//! reaches it. A reduce pushes a checkpoint below its parts, so a
//! ``TypeError`` from any of them drops back to it and encodes the value by
//! protocol instead, as ``_recover`` does. Nothing recurses on the native
//! stack but the sub-walks that set members and escaped keys take for their
//! text, which Python nests too, so a tree of any depth encodes natively.
//!
//! Encoders are chosen as Python's ``_encoder_for`` chooses them, from
//! ``_ENCODERS`` and ``_ENCODER_CACHE``; one that is not the module's own runs
//! as Python against an ``_Encoding`` built from this walk's state. What a walk
//! learns about a class (its encoder, import path, slots, and whether it
//! reduces as an enum member) is kept across calls. Each fact is trusted only
//! while the class keeps its version tag and while ``watch::ENCODE``, which
//! watches every Python dict the fact was read from, stays at the generation
//! it was read at. So a module global, ``_ENCODER_CACHE``, ``_IMPORT_PATHS``,
//! or a ``reflect`` slot cache changed mid-walk is seen at the next node, as
//! Python sees it.

use std::ffi::c_void;
use std::rc::Rc;

use hashbrown::{HashMap, HashSet};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyString, PyTuple, PyWeakrefReference};

use crate::common::*;
use crate::watch;

// ---------------------------------------------------------------- globals --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Nat {
    Float,
    Complex,
    Bytes,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
    MappingProxy,
    Function,
    Type,
    Reduce,
    Path,
    Uuid,
    Datetime,
    Object,
    Hooked,
    Unreduced,
}

#[derive(Clone)]
enum Enc<'py> {
    Native(Nat),
    Foreign(B<'py>),
}

/// An encoder as a class's kept facts hold it.
enum Kept {
    Native(Nat),
    Foreign(Py<PyAny>),
}

impl Kept {
    fn bind<'py>(&self, py: Python<'py>) -> Enc<'py> {
        match self {
            Self::Native(nat) => Enc::Native(*nat),
            Self::Foreign(func) => Enc::Foreign(func.bind(py).clone()),
        }
    }
}

impl Enc<'_> {
    fn kept(&self) -> Kept {
        match self {
            Self::Native(nat) => Kept::Native(*nat),
            Self::Foreign(func) => Kept::Foreign(func.clone().unbind()),
        }
    }
}

pub struct EncGlobals {
    module: Py<PyDict>,
    reflect: Py<PyDict>,
    natives: Vec<(Py<PyAny>, Nat)>,
    encoders: Py<PyAny>,
    /// Each ``_ENCODERS`` class whose encoder is native, by address.
    exact: Vec<(usize, Nat)>,
    /// Whether every ``_ENCODERS`` entry is in ``exact``.
    encoders_all_native: bool,
    encoder_for: Py<PyAny>,
    import_path: Py<PyAny>,
    canonical_reduce: Py<PyAny>,
    attribute_names: Py<PyAny>,
    slot_names: Py<PyAny>,
    sorted_slots: Py<PyAny>,
    freeze: Py<PyAny>,
    pending: Py<PyAny>,
    weak_key_dict: Py<PyAny>,
    dialects: Py<PyAny>,
    default_dialect: Py<PyAny>,
    default_hooks: Py<PyAny>,
    module_type: Py<PyAny>,
    builtin_function_type: Py<PyAny>,
    zoneinfo: Py<PyAny>,
    mapping: Py<PyAny>,
    sequence: Py<PyAny>,
    abstract_set: Py<PyAny>,
    str_bytes: Py<PyAny>,
    dict_type: Py<PyAny>,
    b64encode: Py<PyAny>,
    json: Py<PyAny>,
    enum_type: Py<PyAny>,
    /// ``Enum.__reduce_ex__``, which returns ``(self.__class__, (self._value_,))``.
    enum_reduce_ex: Py<PyAny>,
    /// ``object.__dict__["__class__"]``.
    object_class: Py<PyAny>,
    pure_path: Py<PyAny>,
    partial: Py<PyAny>,
    /// ``_IN_KEY``, the linked place of the ``IN_KEY`` frame, so a Python walk
    /// called inside a key sees the same sentinel.
    in_key: Py<PyAny>,
}

/// Shared, so a call keeps the globals it started with even if a hook reloads
/// ``treekle.serialize`` and so replaces them.
static ENC: GilCell<Option<Rc<EncGlobals>>> = GilCell::new(None);

/// What walks have learned about classes, by address. A call owns the map
/// while it runs and puts it back after, so a nested ``to_plain`` from a hook
/// starts with an empty one rather than finding it borrowed.
static CLASSES: GilCell<Option<HashMap<usize, ClassInfo>>> = GilCell::new(None);

/// Past this many classes the map starts over; entries for dead classes are
/// only ever replaced, never looked for.
const CLASS_LIMIT: usize = 1 << 12;

fn item<'py>(module: &Bound<'py, PyDict>, name: &str) -> PyResult<B<'py>> {
    module
        .get_item(name)?
        .ok_or_else(|| PyAttributeError::new_err(format!("treekle.serialize has no {name}")))
}

fn interned_item<'py>(
    module: &Bound<'py, PyDict>,
    name: &Bound<'py, PyString>,
) -> PyResult<B<'py>> {
    module
        .get_item(name)?
        .ok_or_else(|| PyAttributeError::new_err(format!("treekle.serialize has no {name}")))
}

const NATIVES: [(&str, Nat); 18] = [
    ("_encode_float", Nat::Float),
    ("_encode_complex", Nat::Complex),
    ("_encode_bytes", Nat::Bytes),
    ("_encode_list", Nat::List),
    ("_encode_tuple", Nat::Tuple),
    ("_encode_set", Nat::Set),
    ("_encode_frozenset", Nat::FrozenSet),
    ("_encode_dict", Nat::Dict),
    ("_encode_mappingproxy", Nat::MappingProxy),
    ("_encode_function", Nat::Function),
    ("_encode_type", Nat::Type),
    ("_encode_reduce", Nat::Reduce),
    ("_encode_path", Nat::Path),
    ("_encode_uuid", Nat::Uuid),
    ("_encode_datetime", Nat::Datetime),
    ("_encode_object", Nat::Object),
    ("_encode_hooked", Nat::Hooked),
    ("_encode_unreduced", Nat::Unreduced),
];

fn native_of(natives: &[(Py<PyAny>, Nat)], func: &B<'_>) -> Option<Nat> {
    natives
        .iter()
        .find(|(each, _)| each.as_ptr() == func.as_ptr())
        .map(|(_, nat)| *nat)
}

pub fn setup(module: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = module.py();
    let get = |name: &str| item(module, name).map(Bound::unbind);
    let builtins = py.import("builtins")?;
    let natives = NATIVES
        .iter()
        .map(|(name, nat)| Ok((get(name)?, *nat)))
        .collect::<PyResult<Vec<_>>>()?;
    let encoders = item(module, "_ENCODERS")?;
    let mut exact = Vec::new();
    for pair in encoders.call_method0("items")?.try_iter()? {
        let pair = pair?;
        let (kind, func) = (pair.get_item(0)?, pair.get_item(1)?);
        if let Some(nat) = native_of(&natives, &func) {
            exact.push((kind.as_ptr() as usize, nat));
        }
    }
    let encoders_all_native = exact.len() == encoders.len()?;
    let python_to_plain = item(module, "to_plain")?;
    let kwdefaults = python_to_plain.getattr("__kwdefaults__")?;
    let dialect = item(module, "Dialect")?;
    let dialects = item(module, "get_args")?.call1((dialect.getattr("__value__")?,))?;
    let abc = py.import("collections.abc")?;
    // The package is wherever ``serialize`` lives: ``treekle`` once exported,
    // ``loop.treekle`` in the monorepo, which builds from this same source.
    let name = item(module, "__name__")?.extract::<String>()?;
    let package = name.rsplit_once('.').map_or("", |(package, _)| package);
    let reflect = py.import(format!("{package}.reflect").as_str())?;
    let plain = py.import(format!("{package}.plain").as_str())?;
    let enum_type = py.import("enum")?.getattr("Enum")?;
    let enum_reduce_ex = enum_type.getattr("__dict__")?.get_item("__reduce_ex__")?;
    let object_class = builtins
        .getattr("object")?
        .getattr("__dict__")?
        .get_item("__class__")?;
    let globals = EncGlobals {
        module: module.clone().unbind(),
        reflect: reflect.dict().unbind(),
        natives,
        encoders: encoders.unbind(),
        exact,
        encoders_all_native,
        encoder_for: get("_encoder_for")?,
        import_path: get("_import_path")?,
        canonical_reduce: get("_canonical_reduce")?,
        attribute_names: reflect.getattr("attribute_names")?.unbind(),
        slot_names: reflect.getattr("slot_names")?.unbind(),
        sorted_slots: reflect.getattr("_sorted_slots")?.unbind(),
        freeze: plain.getattr("immutable")?.unbind(),
        pending: get("PENDING")?,
        weak_key_dict: py.import("weakref")?.getattr("WeakKeyDictionary")?.unbind(),
        dialects: dialects.unbind(),
        default_dialect: kwdefaults.get_item("dialect")?.unbind(),
        default_hooks: kwdefaults.get_item("hooks")?.unbind(),
        module_type: get("ModuleType")?,
        builtin_function_type: get("BuiltinFunctionType")?,
        zoneinfo: get("ZoneInfo")?,
        mapping: get("Mapping")?,
        sequence: get("Sequence")?,
        abstract_set: abc.getattr("Set")?.unbind(),
        str_bytes: PyTuple::new(py, [builtins.getattr("str")?, builtins.getattr("bytes")?])?
            .into_any()
            .unbind(),
        dict_type: builtins.getattr("dict")?.unbind(),
        b64encode: get("base64")?.bind(py).getattr("b64encode")?.unbind(),
        json: get("json")?,
        enum_type: enum_type.unbind(),
        enum_reduce_ex: enum_reduce_ex.unbind(),
        object_class: object_class.unbind(),
        pure_path: py.import("pathlib")?.getattr("PurePath")?.unbind(),
        partial: py.import("functools")?.getattr("partial")?.unbind(),
        in_key: get("_IN_KEY")?,
    };
    // Kept facts name the old module's encoders, so they go with it.
    let stale = (ENC.replace(py, Some(Rc::new(globals))), CLASSES.take(py));
    drop(stale);
    Ok(())
}

// -------------------------------------------------------- python state --

unsafe extern "C" {
    fn _PyType_Lookup(kind: *mut ffi::PyTypeObject, name: *mut ffi::PyObject)
    -> *mut ffi::PyObject;
    fn PyObject_GenericGetDict(obj: *mut ffi::PyObject, context: *mut c_void)
    -> *mut ffi::PyObject;
    /// A dict sized for ``minused`` entries, so filling it never resizes.
    fn _PyDict_NewPresized(minused: ffi::Py_ssize_t) -> *mut ffi::PyObject;
}

/// What the walk reads from ``treekle.serialize`` and ``treekle.reflect``,
/// read again whenever ``watch::ENCODE`` moves on.
struct Env<'py> {
    encoders: B<'py>,
    encoders_own: bool,
    /// ``_ENCODERS``, when it is a watched dict other than the module's own.
    encoders_dict: Option<Bound<'py, PyDict>>,
    encoder_for: B<'py>,
    encoder_for_own: bool,
    /// ``_ENCODER_CACHE.data``, when the cache is a watched ``WeakKeyDictionary``.
    cache_data: Option<Bound<'py, PyDict>>,
    import_path: B<'py>,
    /// ``_IMPORT_PATHS``, while ``_import_path`` is the module's own.
    import_paths: Option<Bound<'py, PyDict>>,
    canonical_reduce: B<'py>,
    reduce_own: bool,
    verified_path: B<'py>,
    attribute_names: B<'py>,
    /// Whether ``attribute_names`` and the ``reflect`` functions it calls are
    /// the modules' own, so their caches can be read in their place.
    names_own: bool,
    slot_cache: Option<Bound<'py, PyDict>>,
    sorted_cache: Option<Bound<'py, PyDict>>,
    walk: B<'py>,
    encoding: B<'py>,
}

/// A ``WeakKeyDictionary``'s ``data``, watched along with the instance's own
/// attributes so a rebound ``data`` is seen too.
fn weak_data<'py>(g: &EncGlobals, cache: &B<'py>) -> Option<Bound<'py, PyDict>> {
    let py = cache.py();
    if !cache.get_type().is(g.weak_key_dict.bind(py)) {
        return None;
    }
    let attrs = unsafe {
        Bound::from_owned_ptr_or_opt(
            py,
            PyObject_GenericGetDict(cache.as_ptr(), std::ptr::null_mut()),
        )
    };
    let Some(attrs) = attrs else {
        unsafe { ffi::PyErr_Clear() };
        return None;
    };
    let attrs = attrs.cast_into_exact::<PyDict>().ok()?;
    if !watch::ENCODE.watch(py, &attrs) {
        return None;
    }
    let data = attrs.get_item("data").ok().flatten()?;
    let data = data.cast_into_exact::<PyDict>().ok()?;
    watch::ENCODE.watch(py, &data).then_some(data)
}

/// ``WeakKeyDictionary.get(kind)`` read from its ``data``.
fn weak_get<'py>(data: &Bound<'py, PyDict>, kind: &B<'py>) -> PyResult<Option<B<'py>>> {
    data.get_item(PyWeakrefReference::new(kind)?)
}

fn watched<'py>(value: B<'py>) -> Option<Bound<'py, PyDict>> {
    let dict = value.cast_into_exact::<PyDict>().ok()?;
    watch::ENCODE.watch(dict.py(), &dict).then_some(dict)
}

impl<'py> Env<'py> {
    fn read(py: Python<'py>, g: &EncGlobals) -> PyResult<Self> {
        let module = g.module.bind(py);
        let reflect = g.reflect.bind(py);
        watch::ENCODE.watch(py, module);
        watch::ENCODE.watch(py, reflect);
        let global = |name: &Bound<'py, PyString>| interned_item(module, name);
        let own_in_reflect = |name: &Bound<'py, PyString>, original: &Py<PyAny>| {
            Ok::<_, PyErr>(
                reflect
                    .get_item(name)?
                    .is_some_and(|found| found.is(original.bind(py))),
            )
        };
        let encoders = global(intern!(py, "_ENCODERS"))?;
        let encoders_own = encoders.is(g.encoders.bind(py));
        let encoders_dict = if encoders_own {
            None
        } else {
            watched(encoders.clone())
        };
        let encoder_for = global(intern!(py, "_encoder_for"))?;
        let cache_data = weak_data(g, &global(intern!(py, "_ENCODER_CACHE"))?);
        let import_path = global(intern!(py, "_import_path"))?;
        let import_paths = if import_path.is(g.import_path.bind(py)) {
            watched(global(intern!(py, "_IMPORT_PATHS"))?)
        } else {
            None
        };
        let canonical_reduce = global(intern!(py, "_canonical_reduce"))?;
        let attribute_names = global(intern!(py, "attribute_names"))?;
        let names_own = attribute_names.is(g.attribute_names.bind(py))
            && own_in_reflect(intern!(py, "slot_names"), &g.slot_names)?
            && own_in_reflect(intern!(py, "_sorted_slots"), &g.sorted_slots)?;
        let reflect_cache = |name: &Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyDict>>> {
            Ok(match reflect.get_item(name)? {
                Some(cache) if names_own => weak_data(g, &cache),
                _ => None,
            })
        };
        Ok(Self {
            encoder_for_own: encoder_for.is(g.encoder_for.bind(py)),
            encoders,
            encoders_own,
            encoders_dict,
            encoder_for,
            cache_data,
            import_path,
            import_paths,
            reduce_own: canonical_reduce.is(g.canonical_reduce.bind(py)),
            canonical_reduce,
            verified_path: global(intern!(py, "_verified_path"))?,
            attribute_names,
            names_own,
            slot_cache: reflect_cache(intern!(py, "_SLOT_NAMES"))?,
            sorted_cache: reflect_cache(intern!(py, "_SORTED_SLOT_NAMES"))?,
            walk: global(intern!(py, "_encode"))?,
            encoding: global(intern!(py, "_Encoding"))?,
        })
    }
}

// ----------------------------------------------------------- class facts --

/// What walks learned about one class, each fact read at ``generation``.
struct ClassInfo {
    stamp: Stamp,
    generation: u64,
    encoder: Option<Kept>,
    /// The class's own verified import path.
    path: Option<Py<PyAny>>,
    /// How its instances' attributes are named and read, unsorted then sorted.
    layouts: [Option<Rc<Layout>>; 2],
    /// Whether it reduces as ``Enum.__reduce_ex__`` does, read natively.
    enum_member: Option<bool>,
    /// The ``__dict__`` keys of the last instance read, by identity, and the
    /// order sorting gave them; instances of one class usually share both.
    dict_order: Option<DictOrder>,
}

/// The keys one instance's ``__dict__`` held, and the order they sort in.
struct DictOrder {
    keys: Vec<Py<PyAny>>,
    sorted: Vec<usize>,
}

/// A class's slots, as ``attribute_names`` names them, and where each lives.
struct Layout {
    /// Taken when the offsets were read, so a class changed while an
    /// instance is being read stops them being used mid-object.
    stamp: Stamp,
    names: Py<PyAny>,
    slots: Vec<Py<PyAny>>,
    /// Each slot's offset in the instance, where ``getattr`` would only read it.
    offsets: Vec<Option<isize>>,
}

const PY_AUDIT_READ: std::ffi::c_int = 2;

#[inline]
fn generic_getattr(kind: *mut ffi::PyTypeObject) -> bool {
    unsafe { (*kind).tp_getattro }.map(|f| f as usize) == generic_getattr_slot()
}

/// Where ``kind``'s instances hold slot ``name``, if generic attribute lookup
/// would read it straight from there: a plain object member, nothing audited.
fn slot_offset(kind: *mut ffi::PyTypeObject, name: &B<'_>) -> Option<isize> {
    if tp(name) != t_str() {
        return None;
    }
    unsafe {
        let found = _PyType_Lookup(kind, name.as_ptr());
        if found.is_null() || ob_type(found) != &raw mut ffi::PyMemberDescr_Type {
            return None;
        }
        let descriptor = found.cast::<ffi::PyMemberDescrObject>();
        // A class can install another class's member descriptor under a slot
        // name. Its offset is not an offset in this instance; let getattr
        // enforce descriptor ownership rather than reading arbitrary memory.
        if ffi::PyType_IsSubtype(kind, (*descriptor).d_common.d_type) == 0 {
            return None;
        }
        let member = (*descriptor).d_member.cast::<ffi::PyMemberDef>();
        if (*member).type_code != ffi::Py_T_OBJECT_EX || (*member).flags & PY_AUDIT_READ != 0 {
            return None;
        }
        Some((*member).offset)
    }
}

/// Read a slot at ``offset``, as ``getattr`` would: ``None`` when unset.
#[inline]
fn read_slot<'py>(obj: &B<'py>, offset: isize) -> Option<B<'py>> {
    unsafe {
        let field = *obj
            .as_ptr()
            .cast::<u8>()
            .offset(offset)
            .cast::<*mut ffi::PyObject>();
        (!field.is_null()).then(|| Bound::from_borrowed_ptr(obj.py(), field))
    }
}

/// ``getattr(obj, name)``, or ``None`` where that raises ``AttributeError``.
#[cfg(Py_3_13)]
fn optional_attr<'py>(obj: &B<'py>, name: &B<'py>) -> PyResult<Option<B<'py>>> {
    let mut found: *mut ffi::PyObject = std::ptr::null_mut();
    match unsafe { ffi::PyObject_GetOptionalAttr(obj.as_ptr(), name.as_ptr(), &mut found) } {
        0 => Ok(None),
        1 => Ok(Some(unsafe { Bound::from_owned_ptr(obj.py(), found) })),
        _ => Err(PyErr::fetch(obj.py())),
    }
}

/// ``getattr(obj, name)``, or ``None`` where that raises ``AttributeError``.
#[cfg(not(Py_3_13))]
fn optional_attr<'py>(obj: &B<'py>, name: &B<'py>) -> PyResult<Option<B<'py>>> {
    unsafe {
        let found = ffi::PyObject_GetAttr(obj.as_ptr(), name.as_ptr());
        if !found.is_null() {
            return Ok(Some(Bound::from_owned_ptr(obj.py(), found)));
        }
        if ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) != 0 {
            ffi::PyErr_Clear();
            return Ok(None);
        }
    }
    Err(PyErr::fetch(obj.py()))
}

/// The items in the order ``sorted`` gives them by their keys, each a tuple
/// of strs.
///
/// Keys of exact strs sort here, by code point. A ``repr`` may instead be a
/// str subclass whose ``<`` raises or is no total order, which Rust's sort
/// panics on, so any other key leaves the whole sort to ``sorted`` itself.
fn sorted_by_keys<'py, T, const N: usize>(
    py: Python<'py>,
    keyed: Vec<([B<'py>; N], T)>,
) -> PyResult<Vec<T>> {
    let (keys, items): (Vec<_>, Vec<_>) = keyed.into_iter().unzip();
    let exact: Option<Vec<[&[u8]; N]>> = keys
        .iter()
        .map(|key| {
            let parts = key.each_ref().map(|part| {
                if tp(part) == t_str() {
                    utf8(part)
                } else {
                    None
                }
            });
            parts
                .iter()
                .all(Option::is_some)
                .then(|| parts.map(Option::unwrap_or_default))
        })
        .collect();
    if let Some(exact) = exact {
        let mut zipped: Vec<_> = exact.into_iter().zip(items).collect();
        // UTF-8 byte order is code point order.
        zipped.sort_by(|a, b| a.0.cmp(&b.0));
        return Ok(zipped.into_iter().map(|(_, item)| item).collect());
    }
    let shown = PyList::empty(py);
    for key in &keys {
        shown.append(PyTuple::new(py, key)?)?;
    }
    let builtins = py.import("builtins")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("key", shown.getattr(intern!(py, "__getitem__"))?)?;
    let order = builtins.getattr(intern!(py, "sorted"))?.call(
        (builtins
            .getattr(intern!(py, "range"))?
            .call1((keys.len(),))?,),
        Some(&kwargs),
    )?;
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    let mut sorted = Vec::with_capacity(slots.len());
    for index in order.try_iter()? {
        let index: usize = index?.extract()?;
        sorted.push(
            slots
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| PyRuntimeError::new_err("sorted did not permute its range"))?,
        );
    }
    Ok(sorted)
}

/// ``_is_plain_key``: a dict key written as itself.
#[inline]
fn is_plain_key(key: &B<'_>) -> PyResult<bool> {
    Ok(tp(key) == t_str() && !str_starts_any(key, &["py/", "json://"])?)
}

/// A value that encodes to itself, or to a fixed ``py/float``, with no
/// registration and no Python code: a set of these orders by ``repr`` alone.
#[inline]
fn is_leaf_value(value: &B<'_>) -> bool {
    let kind = tp(value);
    kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() || kind == t_float()
}

// ------------------------------------------------------------------ walk --

const ROOT: u32 = 0;

/// The frame an escaped key's own parts hang from, Python's ``_IN_KEY``.
const IN_KEY: u32 = 1;

/// Where in its node a part is written.
#[derive(Clone)]
enum Slot<'py> {
    Index(usize),
    Key(B<'py>),
}

/// A node's place: its parent's frame and its own slot there.
struct Frame<'py> {
    parent: u32,
    step: Slot<'py>,
}

/// One entry of the walk's stack, Python's ``_Entry`` or ``_Checkpoint``.
enum Entry<'py> {
    /// A part still to encode, written into ``node`` at ``slot``. When
    /// ``raw``, the slot's key is the dict key itself, escaped when reached.
    Part {
        value: B<'py>,
        node: B<'py>,
        slot: Slot<'py>,
        raw: bool,
        parent: u32,
    },
    /// Where a reduce restarts by protocol when one of its parts fails.
    Check {
        value: B<'py>,
        mark: usize,
        node: B<'py>,
        slot: Slot<'py>,
        frame: u32,
    },
}

fn write<'py>(node: &B<'py>, slot: &Slot<'py>, done: B<'py>) -> PyResult<()> {
    // Every node a part is written into is a list or dict this walk built.
    match slot {
        Slot::Index(index) => unsafe { node.cast_unchecked::<PyList>() }.set_item(*index, done),
        Slot::Key(key) => unsafe { node.cast_unchecked::<PyDict>() }.set_item(key, done),
    }
}

struct Encoder<'py, 'g> {
    py: Python<'py>,
    g: &'g EncGlobals,
    dialect: B<'py>,
    python_dialect: bool,
    hooks: B<'py>,
    hooks_are_immutable_empty: bool,
    sort: Flag<'py>,
    frames: Vec<Frame<'py>>,
    stack: Vec<Entry<'py>>,
    seen: HashMap<usize, u32>,
    alive: Vec<B<'py>>,
    env: Env<'py>,
    /// The generation ``env`` was read at.
    generation: Option<u64>,
    classes: HashMap<usize, ClassInfo>,
    /// The ``seen`` and ``alive`` every Python walk this one runs shares,
    /// matching the first ``synced`` entries of ``alive``. Kept across those
    /// walks so each one sends only what was registered since the last.
    shared: Option<(Bound<'py, PyDict>, Bound<'py, PyList>)>,
    synced: usize,
    /// Each frame's Python linked place, once built or adopted.
    linked: Vec<Option<B<'py>>>,
    /// The frame of each place in ``linked``, by address; ``linked`` holds
    /// every place keyed here, so no address is reused.
    linked_frames: HashMap<usize, u32>,
}

impl<'py, 'g> Encoder<'py, 'g> {
    fn new(
        py: Python<'py>,
        g: &'g EncGlobals,
        dialect: B<'py>,
        python_dialect: bool,
        hooks: B<'py>,
        sort: Flag<'py>,
        classes: HashMap<usize, ClassInfo>,
    ) -> PyResult<Self> {
        let hooks_are_immutable_empty = hooks.is(g.default_hooks.bind(py)) && hooks.len()? == 0;
        let generation = watch::ENCODE.generation();
        let env = Env::read(py, g)?;
        let mut frames = Vec::with_capacity(64);
        frames.push(Frame {
            parent: ROOT,
            step: Slot::Index(0),
        });
        frames.push(Frame {
            parent: ROOT,
            step: Slot::Key(pystr(py, "")),
        });
        let mut encoder = Self {
            py,
            g,
            dialect,
            python_dialect,
            hooks,
            hooks_are_immutable_empty,
            sort,
            frames,
            stack: Vec::with_capacity(64),
            seen: HashMap::with_capacity(64),
            alive: Vec::with_capacity(64),
            env,
            generation,
            classes,
            shared: None,
            synced: 0,
            linked: Vec::new(),
            linked_frames: HashMap::new(),
        };
        encoder.remember_linked(IN_KEY, g.in_key.bind(py).clone());
        Ok(encoder)
    }

    #[inline(always)]
    fn b<T>(&self, value: &'g Py<T>) -> &'g Bound<'py, T> {
        value.bind(self.py)
    }

    fn tagged(&self, tag: &Bound<'py, PyString>, value: B<'py>) -> PyResult<B<'py>> {
        let node = PyDict::new(self.py);
        node.set_item(tag, value)?;
        Ok(node.into_any())
    }

    // -------------------------------------------------------- the stack --

    /// ``_encode``: encode ``value``, placed at ``frame``, and all it holds.
    fn walk(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let base = self.stack.len();
        let root = PyList::new(self.py, [self.py.None()])?;
        let result = self.drain(value, frame, &root, base);
        // A failed walk leaves its parts behind; they belong to no one now.
        self.stack.truncate(base);
        result?;
        root.get_item(0)
    }

    fn drain(
        &mut self,
        value: &B<'py>,
        frame: u32,
        root: &Bound<'py, PyList>,
        base: usize,
    ) -> PyResult<()> {
        // As in Python, a failure at the root itself has no checkpoint below it.
        let done = self.encode_node(value, frame, root.as_any(), &Slot::Index(0))?;
        root.set_item(0, done)?;
        while self.stack.len() > base {
            let Some(Entry::Part {
                value,
                node,
                slot,
                raw,
                parent,
            }) = self.stack.pop()
            else {
                continue;
            };
            match self.encode_part(&value, &node, slot, raw, parent) {
                Ok((slot, done)) => write(&node, &slot, done)?,
                Err(error) => {
                    let (node, slot, done) = self.recover(error, base)?;
                    write(&node, &slot, done)?;
                }
            }
        }
        Ok(())
    }

    fn encode_part(
        &mut self,
        value: &B<'py>,
        node: &B<'py>,
        slot: Slot<'py>,
        raw: bool,
        parent: u32,
    ) -> PyResult<(Slot<'py>, B<'py>)> {
        let slot = match slot {
            Slot::Key(key) if raw => Slot::Key(self.escape_key(&key)?),
            slot => slot,
        };
        if let Some(done) = self.leaf(value) {
            return Ok((slot, done?));
        }
        let frame = self.push_frame(parent, slot.clone());
        let done = self.encode_node(value, frame, node, &slot)?;
        Ok((slot, done))
    }

    /// ``_recover``: drop parts down to the innermost reduce and encode it by
    /// protocol instead, or raise if none is left in this walk.
    fn recover(&mut self, mut error: PyErr, base: usize) -> PyResult<(B<'py>, Slot<'py>, B<'py>)> {
        if !error.is_instance_of::<PyTypeError>(self.py) {
            return Err(error);
        }
        while self.stack.len() > base {
            let Some(Entry::Check {
                value,
                mark,
                node,
                slot,
                frame,
            }) = self.stack.pop()
            else {
                continue;
            };
            self.rollback(mark)?;
            match self.encode_unreduced(&value, frame) {
                Ok(done) => return Ok((node, slot, done)),
                Err(again) if again.is_instance_of::<PyTypeError>(self.py) => error = again,
                Err(again) => return Err(again),
            }
        }
        Err(error)
    }

    #[inline]
    fn push_frame(&mut self, parent: u32, step: Slot<'py>) -> u32 {
        self.frames.push(Frame { parent, step });
        (self.frames.len() - 1) as u32
    }

    /// Whether a part is already plain, as ``_push_items`` judges it.
    #[inline(always)]
    fn is_plain(&self, item: &B<'py>) -> bool {
        let kind = tp(item);
        if kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() {
            return true;
        }
        kind == t_float()
            && (self.python_dialect || unsafe { ffi::PyFloat_AS_DOUBLE(item.as_ptr()) }.is_finite())
    }

    /// ``_push_items``: queue each part of ``node`` not yet plain, the first last.
    fn push_items(&mut self, node: &Bound<'py, PyList>, parent: u32) -> PyResult<()> {
        for index in (0..node.len()).rev() {
            let item = node.get_item(index)?;
            if self.is_plain(&item) {
                continue;
            }
            self.stack.push(Entry::Part {
                value: item,
                node: node.clone().into_any(),
                slot: Slot::Index(index),
                raw: false,
                parent,
            });
        }
        Ok(())
    }

    /// A dict holding ``entries``, its values not yet plain queued as
    /// ``_push_values`` queues them: what building the dict then pushing
    /// its values does, without reading the dict back.
    ///
    /// Every key is an exact str, whose hash and ``==`` run no Python code,
    /// so the dict ends with one entry per key in first-seen order holding
    /// the last value set; and its iteration order is that order.
    fn node_from(
        &mut self,
        lead: Option<(&Bound<'py, PyString>, B<'py>)>,
        entries: Vec<(B<'py>, B<'py>)>,
        parent: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = self.py;
        let size = entries.len() + usize::from(lead.is_some());
        let node = unsafe {
            Bound::from_owned_ptr_or_err(py, _PyDict_NewPresized(size as ffi::Py_ssize_t))?
                .cast_into_unchecked::<PyDict>()
        };
        if let Some((tag, value)) = lead {
            node.set_item(tag, value)?;
        }
        for (key, value) in &entries {
            node.set_item(key, value)?;
        }
        if node.len() != size {
            // A repeated key leaves the dict holding fewer entries than were
            // set, in an order only the dict knows.
            self.push_values(&node, parent);
            return Ok(node);
        }
        let shared = node.clone().into_any();
        for (key, item) in entries.into_iter().rev() {
            if self.is_plain(&item) {
                continue;
            }
            self.stack.push(Entry::Part {
                value: item,
                node: shared.clone(),
                slot: Slot::Key(key),
                raw: false,
                parent,
            });
        }
        Ok(node)
    }

    /// ``_push_values``: queue each value of ``node`` not yet plain, the first last.
    fn push_values(&mut self, node: &Bound<'py, PyDict>, parent: u32) {
        for (key, item) in dict_entries(node).into_iter().rev() {
            if self.is_plain(&item) {
                continue;
            }
            self.stack.push(Entry::Part {
                value: item,
                node: node.clone().into_any(),
                slot: Slot::Key(key),
                raw: false,
                parent,
            });
        }
    }

    // ---------------------------------------------------- registrations --

    #[inline]
    fn register(&mut self, value: &B<'py>, frame: u32) {
        self.seen.insert(value.as_ptr() as usize, frame);
        self.alive.push(value.clone());
    }

    /// ``_rollback``: forget every registration since ``alive`` had ``mark``.
    fn rollback(&mut self, mark: usize) -> PyResult<()> {
        if mark < self.synced {
            self.unshare(mark)?;
        }
        for value in self.alive.drain(mark..) {
            self.seen.remove(&(value.as_ptr() as usize));
        }
        Ok(())
    }

    /// Forget every shared registration from position ``mark`` on.
    fn unshare(&mut self, mark: usize) -> PyResult<()> {
        if let Some((seen, alive)) = &self.shared {
            let end = alive.len();
            // A value repeated inside a dict key is registered once per copy.
            for index in mark..end {
                let id = alive.get_item(index)?.as_ptr() as usize;
                if seen.contains(id)? {
                    seen.del_item(id)?;
                }
            }
            alive.del_slice(mark, end)?;
        }
        self.synced = self.synced.min(mark);
        Ok(())
    }

    /// ``_reference``: a ``py/ref`` naming ``frame`` by the keys leading to it,
    /// or ``None`` when ``frame`` is inside an escaped key.
    fn reference(&self, mut frame: u32) -> PyResult<Option<B<'py>>> {
        let mut steps = Vec::new();
        while frame != ROOT {
            if frame == IN_KEY {
                return Ok(None);
            }
            let here = &self.frames[frame as usize];
            steps.push(match &here.step {
                Slot::Index(index) => index.into_pyobject(self.py)?.into_any(),
                Slot::Key(key) => key.clone(),
            });
            frame = here.parent;
        }
        steps.reverse();
        self.tagged(
            intern!(self.py, "py/ref"),
            PyList::new(self.py, steps)?.into_any(),
        )
        .map(Some)
    }

    /// ``_encloses``: whether ``place`` is ``here`` or above it, within the
    /// same key.
    fn encloses(&self, place: u32, mut here: u32) -> bool {
        while here != ROOT {
            if here == place {
                return true;
            }
            if here == IN_KEY {
                return false;
            }
            here = self.frames[here as usize].parent;
        }
        false
    }

    // ------------------------------------------------------- class facts --

    /// The current generation, with ``env`` read again if it has moved on.
    #[inline]
    fn current(&mut self) -> PyResult<Option<u64>> {
        let now = watch::ENCODE.generation();
        if now.is_none() || now != self.generation {
            // The generation is read before the dicts, so a change made after
            // this read is caught at the next check.
            self.env = Env::read(self.py, self.g)?;
            self.generation = now;
        }
        Ok(now)
    }

    fn known(&self, kind: &B<'py>, now: Option<u64>) -> Option<&ClassInfo> {
        let now = now?;
        self.classes
            .get(&(kind.as_ptr() as usize))
            .filter(|info| info.generation == now && info.stamp.holds(kind))
    }

    /// The facts kept for ``kind``, started afresh unless they still hold.
    /// ``None`` when ``kind`` changed since ``before`` was taken, while the
    /// fact about to be kept was being read, or when nothing can be kept.
    fn keep(&mut self, kind: &B<'py>, now: Option<u64>, before: &Stamp) -> Option<&mut ClassInfo> {
        let now = now?;
        if !before.holds(kind) {
            return None;
        }
        let key = kind.as_ptr() as usize;
        let fresh = self
            .classes
            .get(&key)
            .is_some_and(|info| info.generation == now && info.stamp.holds(kind));
        if !fresh {
            let info = ClassInfo {
                stamp: Stamp::always(kind),
                generation: now,
                encoder: None,
                path: None,
                layouts: [None, None],
                enum_member: None,
                dict_order: None,
            };
            let stale = self.classes.insert(key, info);
            drop(stale);
        }
        self.classes.get_mut(&key)
    }

    fn native_or_foreign(&self, func: B<'py>) -> Enc<'py> {
        match native_of(&self.g.natives, &func) {
            Some(nat) => Enc::Native(nat),
            None => Enc::Foreign(func),
        }
    }

    /// ``_encoder_for(kind)``.
    fn encoder_of(&mut self, kind: &B<'py>) -> PyResult<Enc<'py>> {
        let now = self.current()?;
        if self.env.encoders_own && self.env.encoder_for_own && self.g.encoders_all_native {
            let key = kind.as_ptr() as usize;
            if let Some((_, nat)) = self.g.exact.iter().find(|(each, _)| *each == key) {
                return Ok(Enc::Native(*nat));
            }
        }
        if let Some(info) = self.known(kind, now)
            && let Some(encoder) = &info.encoder
        {
            return Ok(encoder.bind(self.py));
        }
        let before = Stamp::always(kind);
        let (encoder, keepable) = self.resolve_encoder(kind)?;
        if keepable && let Some(info) = self.keep(kind, now, &before) {
            info.encoder = Some(encoder.kept());
        }
        Ok(encoder)
    }

    /// ``_ENCODERS.get(kind) or _ENCODER_CACHE.get(kind)``, else what the
    /// module's ``_encoder_for`` chooses and caches; and whether the answer
    /// came from watched state only, so it may be kept.
    fn resolve_encoder(&self, kind: &B<'py>) -> PyResult<(Enc<'py>, bool)> {
        let py = self.py;
        let env = &self.env;
        if !env.encoder_for_own {
            return Ok((
                self.native_or_foreign(env.encoder_for.call1((kind,))?),
                false,
            ));
        }
        let (listed, keepable) = if env.encoders_own && self.g.encoders_all_native {
            // The caller found ``kind`` among the native entries or not at all.
            (None, true)
        } else if let Some(dict) = &env.encoders_dict {
            (dict.get_item(kind)?, true)
        } else {
            let found = env.encoders.call_method1(intern!(py, "get"), (kind,))?;
            (Some(found), env.encoders_own)
        };
        if let Some(found) = listed
            && found.is_truthy()?
        {
            return Ok((self.native_or_foreign(found), keepable));
        }
        let Some(data) = &env.cache_data else {
            return Ok((
                self.native_or_foreign(env.encoder_for.call1((kind,))?),
                false,
            ));
        };
        if let Some(found) = weak_get(data, kind)?
            && found.is_truthy()?
        {
            return Ok((self.native_or_foreign(found), keepable));
        }
        // ``_encoder_for`` caches what it chooses, which moves the
        // generation on; the answer is read from the cache next time.
        Ok((
            self.native_or_foreign(env.encoder_for.call1((kind,))?),
            keepable,
        ))
    }

    /// ``_import_path(value)``, and whether ``_IMPORT_PATHS`` answered.
    fn import_path(&self, value: &B<'py>) -> PyResult<(B<'py>, bool)> {
        if let Some(paths) = &self.env.import_paths
            && let Some(cached) = paths.get_item(PyInt::new(self.py, value.as_ptr() as usize))?
            && !cached.is_none()
            && cached.get_item(0)?.is(value)
        {
            return Ok((cached.get_item(1)?, true));
        }
        Ok((self.env.import_path.call1((value,))?, false))
    }

    /// ``_import_path(kind)`` for a class, kept with the class's facts.
    fn class_path(&mut self, kind: &B<'py>) -> PyResult<B<'py>> {
        let now = self.current()?;
        if let Some(info) = self.known(kind, now)
            && let Some(path) = &info.path
        {
            return Ok(path.bind(self.py).clone());
        }
        let before = Stamp::always(kind);
        let (path, keepable) = self.import_path(kind)?;
        if keepable && let Some(info) = self.keep(kind, now, &before) {
            info.path = Some(path.clone().unbind());
        }
        Ok(path)
    }

    /// ``kind``'s slot names as ``attribute_names`` reads them from
    /// ``reflect``'s caches, or ``None`` where it would compute or store them.
    fn layout(&mut self, kind: &B<'py>, sort: bool) -> PyResult<Option<Rc<Layout>>> {
        let now = self.current()?;
        if let Some(info) = self.known(kind, now)
            && let Some(layout) = &info.layouts[usize::from(sort)]
        {
            return Ok(Some(layout.clone()));
        }
        let (Some(slot_cache), Some(sorted_cache)) = (&self.env.slot_cache, &self.env.sorted_cache)
        else {
            return Ok(None);
        };
        let before = Stamp::always(kind);
        let Some(unsorted) = weak_get(slot_cache, kind)?.filter(|names| !names.is_none()) else {
            return Ok(None);
        };
        let names = if sort {
            // ``_SORTED_SLOT_NAMES.get(kind) or _sorted_slots(kind)``: an empty
            // or missing entry is sorted and stored again, which gives the
            // same empty tuple only when there are no slots and the entry is
            // there to read.
            match weak_get(sorted_cache, kind)? {
                Some(names) if names.is_truthy()? || !unsorted.is_truthy()? => names,
                _ => return Ok(None),
            }
        } else {
            unsorted
        };
        let slots: Vec<B<'py>> = names.try_iter()?.collect::<PyResult<_>>()?;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let generic = generic_getattr(class);
        let offsets = slots
            .iter()
            .map(|name| generic.then(|| slot_offset(class, name)).flatten())
            .collect();
        let layout = Rc::new(Layout {
            stamp: Stamp::always(kind),
            names: names.unbind(),
            slots: slots.into_iter().map(Bound::unbind).collect(),
            offsets,
        });
        if let Some(info) = self.keep(kind, now, &before) {
            info.layouts[usize::from(sort)] = Some(layout.clone());
        }
        Ok(Some(layout))
    }

    /// Whether ``kind``'s instances reduce through ``Enum.__reduce_ex__``
    /// untouched: no other ``__reduce_ex__``, ``__class__``, ``_value_``, or
    /// attribute lookup between, and nothing ``_canonical_reduce`` rewrites.
    fn enum_member(&mut self, kind: &B<'py>) -> PyResult<bool> {
        let now = self.current()?;
        if !self.env.reduce_own {
            return Ok(false);
        }
        if let Some(info) = self.known(kind, now)
            && let Some(fast) = info.enum_member
        {
            return Ok(fast);
        }
        let before = Stamp::always(kind);
        let py = self.py;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let subtype = |base: &Py<PyAny>| unsafe {
            ffi::PyType_IsSubtype(class, base.as_ptr().cast::<ffi::PyTypeObject>()) != 0
        };
        let lookup = |name: &Bound<'py, PyString>| unsafe { _PyType_Lookup(class, name.as_ptr()) };
        let fast = subtype(&self.g.enum_type)
            && !subtype(&self.g.pure_path)
            && !subtype(&self.g.partial)
            && generic_getattr(class)
            && lookup(intern!(py, "__reduce_ex__")) == self.g.enum_reduce_ex.as_ptr()
            && lookup(intern!(py, "__class__")) == self.g.object_class.as_ptr()
            && lookup(intern!(py, "_value_")).is_null();
        if let Some(info) = self.keep(kind, now, &before) {
            info.enum_member = Some(fast);
        }
        Ok(fast)
    }

    /// ``_canonical_reduce(value)`` for an enum member, read from its own
    /// attributes; ``None`` where Python's own reading is needed.
    fn enum_recipe(&self, value: &B<'py>, kind: &B<'py>) -> PyResult<Option<Vec<B<'py>>>> {
        let py = self.py;
        let attrs = unsafe {
            Bound::from_owned_ptr_or_opt(
                py,
                PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()),
            )
        };
        let Some(attrs) = attrs else {
            unsafe { ffi::PyErr_Clear() };
            return Ok(None);
        };
        let Ok(attrs) = attrs.cast_into_exact::<PyDict>() else {
            return Ok(None);
        };
        if attrs.contains(intern!(py, "__reduce_ex__"))? {
            return Ok(None);
        }
        let Some(member) = attrs.get_item(intern!(py, "_value_"))? else {
            return Ok(None);
        };
        Ok(Some(vec![
            kind.clone(),
            PyTuple::new(py, [member])?.into_any(),
        ]))
    }

    // ------------------------------------------------------------- nodes --

    #[inline(always)]
    fn leaf(&self, value: &B<'py>) -> Option<PyResult<B<'py>>> {
        let kind = tp(value);
        if kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() {
            return Some(Ok(value.clone()));
        }
        if kind == t_float() {
            return Some(self.encode_float(value));
        }
        None
    }

    /// ``_encode_node``: a leaf, a ``py/ref`` repeat, a hook, or the encoder's
    /// output. ``node`` and ``slot`` are where the output is written, which a
    /// reduce's checkpoint needs.
    fn encode_node(
        &mut self,
        value: &B<'py>,
        frame: u32,
        node: &B<'py>,
        slot: &Slot<'py>,
    ) -> PyResult<B<'py>> {
        if let Some(done) = self.leaf(value) {
            return done;
        }
        if let Some(&at) = self.seen.get(&(value.as_ptr() as usize)) {
            if let Some(reference) = self.reference(at)? {
                return Ok(reference);
            }
            // A repeat inside a dict key is written again in full: a key is a
            // value, and a ``py/ref`` to a place inside one could not be read back.
            if self.encloses(at, frame) {
                return Err(PyTypeError::new_err(format!(
                    "Cannot serialize a dict key holding a cycle through {}.",
                    value.repr()?
                )));
            }
        }
        let kind = value.get_type().into_any();
        if !self.hooks_are_immutable_empty && self.hooks.contains(&kind)? {
            return self.encode_hooked(value, &kind, frame);
        }
        let encoder = self.encoder_of(&kind)?;
        self.run(encoder, value, &kind, frame, node, slot)
    }

    fn run(
        &mut self,
        encoder: Enc<'py>,
        value: &B<'py>,
        kind: &B<'py>,
        frame: u32,
        node: &B<'py>,
        slot: &Slot<'py>,
    ) -> PyResult<B<'py>> {
        let py = self.py;
        let nat = match encoder {
            Enc::Foreign(_) => return self.foreign(value, frame),
            Enc::Native(nat) => nat,
        };
        match nat {
            Nat::Float => self.encode_float(value),
            Nat::Complex => {
                let real = self.encode_float(&value.getattr(intern!(py, "real"))?)?;
                let imag = self.encode_float(&value.getattr(intern!(py, "imag"))?)?;
                self.tagged(
                    intern!(py, "py/complex"),
                    PyList::new(py, [real, imag])?.into_any(),
                )
            }
            Nat::Bytes => {
                let text = self
                    .b(&self.g.b64encode)
                    .call1((value,))?
                    .call_method0(intern!(py, "decode"))?;
                self.tagged(intern!(py, "py/b64"), text)
            }
            Nat::List => self.encode_list(value, frame),
            Nat::Tuple => {
                let items = self.snapshot(value)?;
                self.push_items(&items, frame)?;
                self.tagged(intern!(py, "py/tuple"), items.into_any())
            }
            Nat::Set => self.encode_set(value, frame),
            Nat::FrozenSet => {
                let items = self.ordered(value, frame)?;
                self.push_items(&items, frame)?;
                self.tagged(intern!(py, "py/frozenset"), items.into_any())
            }
            Nat::Dict => self.encode_dict(value, frame),
            Nat::MappingProxy => {
                self.register(value, frame);
                let node = PyDict::new(py);
                let tag = intern!(py, "py/mappingproxy").clone().into_any();
                node.set_item(&tag, py.None())?;
                let copied = self.b(&self.g.dict_type).call1((value,))?;
                self.stack.push(Entry::Part {
                    value: copied,
                    node: node.clone().into_any(),
                    slot: Slot::Key(tag),
                    raw: false,
                    parent: frame,
                });
                Ok(node.into_any())
            }
            Nat::Function => {
                let owner = optional_attr(value, intern!(py, "__self__").as_any())?
                    .unwrap_or_else(|| py.None().into_bound(py));
                if !(owner.is_none() || owner.is_instance(self.b(&self.g.module_type))?) {
                    return self.encode_reduce(value, kind, frame, node, slot);
                }
                match self.import_path(value) {
                    Ok((path, _)) => self.tagged(intern!(py, "py/function"), path),
                    // A static builtin method (``torch.relu``) has no
                    // importable path of its own; its reduce recipe names one.
                    Err(error)
                        if error.is_instance_of::<PyTypeError>(py)
                            && kind.is(self.b(&self.g.builtin_function_type)) =>
                    {
                        self.encode_reduce(value, kind, frame, node, slot)
                    }
                    Err(error) => Err(error),
                }
            }
            Nat::Type => {
                let path = if is_class(value) {
                    self.class_path(value)?
                } else {
                    self.import_path(value)?.0
                };
                self.tagged(intern!(py, "py/type"), path)
            }
            Nat::Reduce => self.encode_reduce(value, kind, frame, node, slot),
            Nat::Path => self.tagged(intern!(py, "py/path"), value.str()?.into_any()),
            Nat::Uuid => self.tagged(intern!(py, "py/uuid"), value.str()?.into_any()),
            Nat::Datetime => {
                let mut text = value.call_method0(intern!(py, "isoformat"))?;
                let zone = value.getattr(intern!(py, "tzinfo"))?;
                if zone.is_instance(self.b(&self.g.zoneinfo))? {
                    let key = zone.getattr(intern!(py, "key"))?;
                    text = pystr(
                        py,
                        &format!("{}[{}]", text.str()?.to_cow()?, key.str()?.to_cow()?),
                    );
                }
                self.tagged(intern!(py, "py/datetime"), text)
            }
            Nat::Object => self.encode_object(value, kind, frame),
            Nat::Hooked => self.encode_hooked(value, kind, frame),
            Nat::Unreduced => self.encode_unreduced(value, frame),
        }
    }

    // ---------------------------------------------------------- encoders --

    fn encode_float(&self, value: &B<'py>) -> PyResult<B<'py>> {
        let number = unsafe {
            if tp(value) == t_float() {
                ffi::PyFloat_AS_DOUBLE(value.as_ptr())
            } else {
                let number = ffi::PyFloat_AsDouble(value.as_ptr());
                if number == -1.0 && !ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(self.py));
                }
                number
            }
        };
        if number.is_finite() || self.python_dialect {
            return Ok(value.clone());
        }
        self.tagged(intern!(self.py, "py/float"), value.repr()?.into_any())
    }

    /// ``list(value)``.
    fn snapshot(&self, value: &B<'py>) -> PyResult<Bound<'py, PyList>> {
        unsafe { Bound::from_owned_ptr_or_err(self.py, ffi::PySequence_List(value.as_ptr()))? }
            .cast_into::<PyList>()
            .map_err(PyErr::from)
    }

    fn encode_list(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        self.register(value, frame);
        // Every item is read before any is encoded, so a hook that changes
        // the list mid-walk changes no output.
        let items = self.snapshot(value)?;
        self.push_items(&items, frame)?;
        Ok(items.into_any())
    }

    fn encode_set(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        self.register(value, frame);
        let items = self.ordered(value, frame)?;
        self.push_items(&items, frame)?;
        self.tagged(intern!(self.py, "py/set"), items.into_any())
    }

    /// ``_ordered``: a set's members sorted by ``(repr, encoded JSON)``.
    fn ordered(&mut self, members: &B<'py>, frame: u32) -> PyResult<Bound<'py, PyList>> {
        let all: Vec<B<'py>> = members.try_iter()?.collect::<PyResult<_>>()?;
        if all.iter().all(is_leaf_value) {
            // Distinct leaves have distinct reprs, except NaNs, which also
            // encode alike; so the JSON half of the key would decide nothing,
            // and it is skipped. Python still calls ``json.dumps`` for each
            // member, so only a replaced ``json.dumps`` sees a difference.
            let keyed: Vec<([B<'py>; 1], B<'py>)> = all
                .into_iter()
                .map(|member| Ok(([member.repr()?.into_any()], member)))
                .collect::<PyResult<_>>()?;
            return PyList::new(self.py, sorted_by_keys(self.py, keyed)?);
        }
        let mut keyed = Vec::with_capacity(all.len());
        for member in all {
            let (shown, text) = self.set_order_key(&member, frame)?;
            keyed.push(([shown, text], member));
        }
        PyList::new(self.py, sorted_by_keys(self.py, keyed)?)
    }

    /// ``_set_order_key``: encode the member, forgetting what it registers,
    /// then take its ``repr`` and the encoding's sorted JSON.
    fn set_order_key(&mut self, value: &B<'py>, frame: u32) -> PyResult<(B<'py>, B<'py>)> {
        let py = self.py;
        let _guard = Recursion::enter(py, c" while encoding")?;
        let mark = self.alive.len();
        let encoded = self.walk(value, frame);
        self.rollback(mark)?;
        let encoded = encoded?;
        let shown = value.repr()?.into_any();
        let kwargs = PyDict::new(py);
        kwargs.set_item("sort_keys", true)?;
        let text = self
            .b(&self.g.json)
            .getattr(intern!(py, "dumps"))?
            .call((encoded,), Some(&kwargs))?;
        Ok((shown, text))
    }

    fn encode_dict(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        self.register(value, frame);
        let (plain, entries) = if tp(value) == t_dict() {
            let entries = dict_entries(unsafe { value.cast_unchecked::<PyDict>() });
            let mut plain = true;
            for (key, _) in &entries {
                if !is_plain_key(key)? {
                    plain = false;
                    break;
                }
            }
            (plain, entries)
        } else {
            // Any other mapping is read as Python reads it: its keys by
            // iteration, until one is not plain, then its items.
            let mut plain = true;
            for key in value.try_iter()? {
                if !is_plain_key(&key?)? {
                    plain = false;
                    break;
                }
            }
            let mut entries = Vec::new();
            let mut items = MapItems::new(value)?;
            while let Some(entry) = items.next_item()? {
                entries.push(entry);
            }
            (plain, entries)
        };
        if plain {
            // A plain key is an exact str, as ``node_from`` needs.
            return Ok(self.node_from(None, entries, frame)?.into_any());
        }
        let node = PyDict::new(py);
        // An escaped key is encoded when its entry is reached, after every
        // earlier value, so a key naming an object written earlier becomes a
        // ``py/ref``. Every entry waits its turn, plain ones too, to keep the
        // keys in order.
        for (key, item) in entries.into_iter().rev() {
            let raw = !is_plain_key(&key)?;
            self.stack.push(Entry::Part {
                value: item,
                node: node.clone().into_any(),
                slot: Slot::Key(key),
                raw,
                parent: frame,
            });
        }
        Ok(node.into_any())
    }

    /// ``_encode_key``: a dict key, ``json://``-escaping non-strs and tag
    /// lookalikes. A key is text, so nothing in it has a path a later
    /// ``py/ref`` can name; what encoding it registers is forgotten. Its parts
    /// hang from ``IN_KEY``, which ``reference`` refuses to name.
    fn escape_key(&mut self, key: &B<'py>) -> PyResult<B<'py>> {
        let py = self.py;
        if is_plain_key(key)? {
            return Ok(key.clone());
        }
        let text = match leaf_key_text(key)? {
            Some(text) => text,
            None => {
                let _guard = Recursion::enter(py, c" while encoding")?;
                let mark = self.alive.len();
                let encoded = self.walk(key, IN_KEY);
                self.rollback(mark)?;
                let text = self
                    .b(&self.g.json)
                    .getattr(intern!(py, "dumps"))?
                    .call1((encoded?,))?;
                text.str()?.to_cow()?.into_owned()
            }
        };
        Ok(pystr(py, &format!("json://{text}")))
    }

    fn encode_object(&mut self, value: &B<'py>, kind: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        self.register(value, frame);
        let path = self.class_path(kind)?;
        let members = self.object_members(value, kind)?;
        // A member named ``py/object`` overwrites the tag in place, which
        // ``node_from`` cannot tell from a repeat; leave that to the dict.
        if members.iter().all(|(name, _)| tp(name) == t_str()) {
            let node = self.node_from(Some((intern!(py, "py/object"), path)), members, frame)?;
            return Ok(node.into_any());
        }
        let node = PyDict::new(py);
        node.set_item(intern!(py, "py/object"), path)?;
        for (name, member) in members {
            node.set_item(name, member)?;
        }
        self.push_values(&node, frame);
        Ok(node.into_any())
    }

    /// Each ``(name, getattr(value, name))`` for the names
    /// ``attribute_names(value, sort=sort)`` gives, leaving out what is
    /// missing or ``PENDING``.
    fn object_members(&mut self, value: &B<'py>, kind: &B<'py>) -> PyResult<Vec<(B<'py>, B<'py>)>> {
        let py = self.py;
        let layout = if self.env.names_own {
            let sort = self.sort.get()?;
            self.layout(kind, sort)?.map(|layout| (layout, sort))
        } else {
            None
        };
        let Some((layout, sort)) = layout else {
            let kwargs = PyDict::new(py);
            kwargs.set_item("sort", self.sort.object(py))?;
            let names = self.env.attribute_names.call((value,), Some(&kwargs))?;
            let mut members = Vec::new();
            for name in names.try_iter()? {
                let name = name?;
                if let Some(member) = self.member(value, kind, &name, None)? {
                    members.push((name, member));
                }
            }
            return Ok(members);
        };
        // ``attribute_names``' own reading of the instance: its ``__dict__``
        // keys that are not slots, after the slots, sorted together if asked.
        let names = layout.names.bind(py);
        let mut extra = Vec::new();
        let state = optional_attr(value, intern!(py, "__dict__").as_any())?;
        if layout.slots.is_empty()
            && let Some(state) = &state
            && let Some(members) = self.dict_members(value, kind, state, sort)?
        {
            return Ok(members);
        }
        if let Some(state) = state
            && state.is_truthy()?
        {
            for key in state.try_iter()? {
                let key = key?;
                if !names.contains(&key)? {
                    extra.push(key);
                }
            }
        }
        let mut members = Vec::with_capacity(layout.slots.len() + extra.len());
        if extra.is_empty() {
            for (index, name) in layout.slots.iter().enumerate() {
                let name = name.bind(py);
                if let Some(member) = self.member(value, kind, name, Some((&layout, index)))? {
                    members.push((name.clone(), member));
                }
            }
            return Ok(members);
        }
        let all = PyList::new(py, layout.slots.iter().map(|slot| slot.bind(py)))?;
        for key in extra {
            all.append(key)?;
        }
        if sort {
            all.sort()?;
        }
        for name in all.iter() {
            // Slots enter ``all`` as themselves, and a key equal to a slot
            // never does, so identity finds each slot.
            let slot = layout
                .slots
                .iter()
                .position(|slot| slot.bind(py).is(&name))
                .map(|index| (&*layout, index));
            if let Some(member) = self.member(value, kind, &name, slot)? {
                members.push((name, member));
            }
        }
        Ok(members)
    }

    /// The members of a slotless object read straight from its ``__dict__``,
    /// or ``None`` where ``getattr`` might answer otherwise.
    ///
    /// ``getattr`` returns the instance dict's entry unless the class has a
    /// data descriptor of that name, or its lookup is not ``object``'s; and
    /// only when ``__dict__`` is the dict that lookup reads. With exact str
    /// keys no Python code runs, so the dict cannot change while it is read.
    fn dict_members(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        state: &B<'py>,
        sort: bool,
    ) -> PyResult<Option<Vec<(B<'py>, B<'py>)>>> {
        let py = self.py;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        if tp(state) != t_dict() || !generic_getattr(class) {
            return Ok(None);
        }
        let generic = unsafe {
            Bound::from_owned_ptr_or_opt(
                py,
                PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()),
            )
        };
        match generic {
            Some(generic) if generic.is(state) => {}
            Some(_) => return Ok(None),
            None => {
                unsafe { ffi::PyErr_Clear() };
                return Ok(None);
            }
        }
        let entries = dict_entries(unsafe { state.cast_unchecked::<PyDict>() });
        // The order and the descriptor check are kept per class only while the
        // class is unchanged, so its keys need no fresh lookup.
        let now = self.current()?;
        let known = self
            .known(kind, now)
            .and_then(|info| info.dict_order.as_ref());
        if let Some(order) = known
            && order.keys.len() == entries.len()
            && order
                .keys
                .iter()
                .zip(&entries)
                .all(|(key, (each, _))| key.as_ptr() == each.as_ptr())
        {
            let pending = self.b(&self.g.pending);
            let mut members = Vec::with_capacity(entries.len());
            let pick = |index: usize| entries[index].clone();
            if sort {
                for &index in &order.sorted {
                    let (key, member) = pick(index);
                    if !member.is(pending) {
                        members.push((key, member));
                    }
                }
            } else {
                for index in 0..entries.len() {
                    let (key, member) = pick(index);
                    if !member.is(pending) {
                        members.push((key, member));
                    }
                }
            }
            return Ok(Some(members));
        }
        let before = Stamp::always(kind);
        let mut keyed = Vec::with_capacity(entries.len());
        for (index, (key, _)) in entries.iter().enumerate() {
            if tp(key) != t_str() {
                return Ok(None);
            }
            let found = unsafe { _PyType_Lookup(class, key.as_ptr()) };
            if !found.is_null() && unsafe { (*ob_type(found)).tp_descr_set.is_some() } {
                return Ok(None);
            }
            keyed.push(([key.clone()], index));
        }
        let sorted = sorted_by_keys(py, keyed)?;
        let order = DictOrder {
            keys: entries
                .iter()
                .map(|(key, _)| key.clone().unbind())
                .collect(),
            sorted: sorted.clone(),
        };
        let pending = self.b(&self.g.pending).clone();
        let indices: Vec<usize> = if sort {
            sorted
        } else {
            (0..entries.len()).collect()
        };
        let members = indices
            .into_iter()
            .map(|index| entries[index].clone())
            .filter(|(_, member)| !member.is(&pending))
            .collect();
        if let Some(info) = self.keep(kind, now, &before) {
            info.dict_order = Some(order);
        }
        Ok(Some(members))
    }

    /// ``getattr(value, name, PENDING)`` unless that is ``PENDING``, read
    /// from the slot's offset while the class is as the layout found it.
    fn member(
        &self,
        value: &B<'py>,
        kind: &B<'py>,
        name: &B<'py>,
        slot: Option<(&Layout, usize)>,
    ) -> PyResult<Option<B<'py>>> {
        let offset = slot
            .filter(|(layout, _)| layout.stamp.holds(kind))
            .and_then(|(layout, index)| layout.offsets[index]);
        let found = match offset {
            Some(offset) => read_slot(value, offset),
            None => optional_attr(value, name)?,
        };
        Ok(found.filter(|member| !member.is(self.b(&self.g.pending))))
    }

    fn encode_hooked(&mut self, value: &B<'py>, kind: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        let encode_hook = self.hooks.get_item(kind)?.get_item(0)?;
        self.register(value, frame);
        // The payload is arbitrary caller data, so it takes the same walk as
        // any other value: it may hold non-finite floats or tag-like keys.
        let path = self.class_path(kind)?;
        let raw = encode_hook.call1((value,))?;
        let node = PyList::new(py, [path, raw])?;
        self.push_items(&node, frame)?;
        self.tagged(intern!(py, "py/hook"), node.into_any())
    }

    fn encode_reduce(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        frame: u32,
        node: &B<'py>,
        slot: &Slot<'py>,
    ) -> PyResult<B<'py>> {
        let py = self.py;
        let recipe = if self.enum_member(kind)? {
            self.enum_recipe(value, kind)?
        } else {
            None
        };
        let mut parts = match recipe {
            Some(parts) => parts,
            None => {
                let recipe = self.env.canonical_reduce.call1((value,))?;
                if recipe.is_none() {
                    return self.encode_unreduced(value, frame);
                }
                // A bare-name reduce names a module global, written as an
                // import path.
                if is_str(&recipe) {
                    let module = kind.getattr(intern!(py, "__module__"))?;
                    let path = format!("{}.{}", module.str()?.to_cow()?, recipe.str()?.to_cow()?);
                    let path = self.env.verified_path.call1((path, value))?;
                    return self.tagged(intern!(py, "py/type"), path);
                }
                recipe.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
        };
        let mark = self.alive.len();
        // A hashable value with no state beyond its arguments (an enum
        // member, a Decimal) is a value: each occurrence is written in full.
        if is_unhashable(value)? || parts.iter().skip(2).any(|part| !part.is_none()) {
            self.register(value, frame);
        }
        while parts.last().is_some_and(|last| last.is_none()) {
            parts.pop();
        }
        if parts.is_empty() {
            return Err(PyIndexError::new_err("list index out of range"));
        }
        self.stack.push(Entry::Check {
            value: value.clone(),
            mark,
            node: node.clone(),
            slot: slot.clone(),
            frame,
        });
        let parts = PyList::new(py, parts)?;
        self.push_items(&parts, frame)?;
        self.tagged(intern!(py, "py/reduce"), parts.into_any())
    }

    /// ``_encode_unreduced``: a value with no usable reduce recipe, by the
    /// protocols it has.
    fn encode_unreduced(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        let kind = value.get_type();
        let name = kind.getattr(intern!(py, "__name__"))?;
        if kind.hasattr(intern!(py, "__dataclass_fields__"))?
            || kind
                .getattr(intern!(py, "__dict__"))?
                .contains(intern!(py, "__slots__"))?
        {
            return self.encode_object(value, kind.as_any(), frame);
        }
        if kind.is_subclass(self.b(&self.g.mapping))? {
            return self.encode_dict(value, frame);
        }
        if kind.is_subclass(self.b(&self.g.sequence))?
            && !kind.is_subclass(self.b(&self.g.str_bytes))?
        {
            return self.encode_list(value, frame);
        }
        if kind.is_subclass(self.b(&self.g.abstract_set))? {
            return self.encode_set(value, frame);
        }
        let shown = name.str()?.to_cow()?.into_owned();
        Err(PyTypeError::new_err(format!(
            "Cannot serialize leaf of type {}. Pass hooks={{{}: (encode, decode)}} to to_plain().",
            repr(&name)?,
            shown
        )))
    }

    // ----------------------------------------------------- python walks --

    /// A caller-installed encoder owns its subtree: Python's walk encodes it
    /// with this walk's state and shares its registrations back.
    fn foreign(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        let (seen, alive) = self.share()?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dialect", &self.dialect)?;
        kwargs.set_item("hooks", &self.hooks)?;
        kwargs.set_item("sort", self.sort.object(py))?;
        kwargs.set_item("here", self.linked_place(frame)?)?;
        kwargs.set_item("seen", &seen)?;
        kwargs.set_item("alive", &alive)?;
        let state = self.env.encoding.call((), Some(&kwargs))?;
        let mark = self.alive.len();
        // Python's walk registers into ``seen`` and ``alive`` in place.
        let result = match self.env.walk.call1((value, &state)) {
            Ok(result) => result,
            Err(error) => {
                self.unshare(mark)?;
                return Err(error);
            }
        };
        let added: Vec<B<'py>> = alive.get_slice(mark, alive.len()).iter().collect();
        for each in added {
            let place = seen.as_any().get_item(each.as_ptr() as usize)?;
            let at = self.frame_of_linked(&place)?;
            self.register(&each, at);
        }
        self.synced = self.alive.len();
        Ok(result)
    }

    /// The shared ``seen`` and ``alive``, brought up to date with this walk's.
    fn share(&mut self) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
        let py = self.py;
        let (seen, alive) = self
            .shared
            .get_or_insert_with(|| (PyDict::new(py), PyList::empty(py)))
            .clone();
        for index in self.synced..self.alive.len() {
            let value = self.alive[index].clone();
            let at = self.seen[&(value.as_ptr() as usize)];
            seen.set_item(value.as_ptr() as usize, self.linked_place(at)?)?;
            alive.append(&value)?;
        }
        self.synced = self.alive.len();
        Ok((seen, alive))
    }

    /// The native frame of a Python linked place, adopting each new step once.
    fn frame_of_linked(&mut self, place: &B<'py>) -> PyResult<u32> {
        let mut missing = Vec::new();
        let mut here = place.clone();
        let mut frame = loop {
            if here.is_none() {
                break ROOT;
            }
            if let Some(&frame) = self.linked_frames.get(&(here.as_ptr() as usize)) {
                break frame;
            }
            let parts = here.cast::<PyTuple>()?;
            let (parent, item) = (parts.get_item(0)?, parts.get_item(1)?);
            missing.push((here, item));
            here = parent;
        };
        for (place, item) in missing.into_iter().rev() {
            let step = if tp(&item) == t_int() {
                Slot::Index(item.extract::<usize>()?)
            } else {
                Slot::Key(item)
            };
            frame = self.push_frame(frame, step);
            self.remember_linked(frame, place);
        }
        Ok(frame)
    }

    /// ``frame``'s Python linked place, built from the nearest one already known.
    fn linked_place(&mut self, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        let mut missing = Vec::new();
        let mut here = frame;
        let mut place = loop {
            if here == ROOT {
                break py.None().into_bound(py);
            }
            if let Some(Some(place)) = self.linked.get(here as usize) {
                break place.clone();
            }
            missing.push(here);
            here = self.frames[here as usize].parent;
        };
        for frame in missing.into_iter().rev() {
            let step = match &self.frames[frame as usize].step {
                Slot::Index(index) => index.into_pyobject(py)?.into_any(),
                Slot::Key(key) => key.clone(),
            };
            place = PyTuple::new(py, [place, step])?.into_any();
            self.remember_linked(frame, place.clone());
        }
        Ok(place)
    }

    fn remember_linked(&mut self, frame: u32, place: B<'py>) {
        let slot = frame as usize;
        if self.linked.len() <= slot {
            self.linked.resize(slot + 1, None);
        }
        self.linked_frames.insert(place.as_ptr() as usize, frame);
        self.linked[slot] = Some(place);
    }
}

/// ``json.dumps`` of a key that encodes to itself, where that is known
/// without asking: an int is its ``repr``, as ``json`` writes it.
fn leaf_key_text(key: &B<'_>) -> PyResult<Option<String>> {
    let kind = tp(key);
    if kind == t_int() {
        return Ok(Some(key.repr()?.to_cow()?.into_owned()));
    }
    if kind == t_bool() {
        let truth = key.as_ptr() == unsafe { ffi::Py_True() };
        return Ok(Some(if truth { "true" } else { "false" }.to_owned()));
    }
    if kind == t_none() {
        return Ok(Some("null".to_owned()));
    }
    if kind == t_float() && unsafe { ffi::PyFloat_AS_DOUBLE(key.as_ptr()) }.is_finite() {
        return Ok(Some(key.repr()?.to_cow()?.into_owned()));
    }
    Ok(None)
}

// ---------------------------------------------------------------- freeze --

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
fn freeze<'py>(py: Python<'py>, tree: &B<'py>) -> PyResult<Option<B<'py>>> {
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

// ------------------------------------------------------------ entry point --

pub fn to_plain<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Option<B<'py>>,
    mutable: Option<B<'py>>,
    hooks: Option<B<'py>>,
    sort: Option<B<'py>>,
) -> PyResult<B<'py>> {
    let Some(globals) = ENC.cloned(py) else {
        return Err(PyTypeError::new_err(
            "treekle._rust.setup_encode was not called",
        ));
    };
    let g = &*globals;
    let dialect = dialect.unwrap_or_else(|| g.default_dialect.bind(py).clone());
    let python_dialect = match (tp(&dialect) == t_str()).then(|| utf8(&dialect)).flatten() {
        Some(b"json") => false,
        Some(b"python") => true,
        _ => {
            if !g.dialects.bind(py).contains(&dialect)? {
                return Err(PyValueError::new_err(format!(
                    "unknown dialect {}; expected 'json' or 'python'",
                    repr(&dialect)?
                )));
            }
            py_eq(&dialect, intern!(py, "python").as_any())?
        }
    };
    let kind = tp(obj);
    let tree = if kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() {
        obj.clone()
    } else {
        let hooks = hooks.unwrap_or_else(|| g.default_hooks.bind(py).clone());
        let classes = CLASSES.take(py).unwrap_or_default();
        let mut encoder = Encoder::new(
            py,
            g,
            dialect,
            python_dialect,
            hooks,
            Flag::new(sort, true),
            classes,
        )?;
        let tree = encoder.walk(obj, ROOT);
        let mut classes = std::mem::take(&mut encoder.classes);
        drop(encoder);
        if classes.len() > CLASS_LIMIT {
            classes.clear();
        }
        // A nested call may have put back its own map meanwhile; this one
        // replaces it, and the other drops after the swap.
        let stale = CLASSES.replace(py, Some(classes));
        drop(stale);
        tree?
    };
    // Read after the walk, as Python reads it.
    let mutable = match &mutable {
        Some(mutable) => mutable.is_truthy()?,
        None => true,
    };
    if mutable {
        return Ok(tree);
    }
    let immutable = interned_item(g.module.bind(py), intern!(py, "immutable"))?;
    if immutable.is(g.freeze.bind(py))
        && let Some(frozen) = freeze(py, &tree)?
    {
        return Ok(frozen);
    }
    immutable.call1((tree,))
}
