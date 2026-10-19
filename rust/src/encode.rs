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
//! while the class keeps its version tag and while ``sync::ENCODE``, which
//! watches every Python dict the fact was read from, stays at the generation
//! it was read at. So a module global, ``_ENCODER_CACHE``, ``_IMPORT_PATHS``,
//! or a ``reflect`` slot cache changed mid-walk is seen at the next node, as
//! Python sees it. An entry added under a new key is the one change that does
//! not move that generation on: every fact kept for another key is still
//! true, and the new key's own fact is read from the dict when first needed.

use std::ffi::c_void;
use std::rc::Rc;

use hashbrown::HashMap;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyString, PyTuple, PyWeakrefReference};

use crate::class::*;
use crate::freeze::freeze;
use crate::py::*;
use crate::sync;

// ---------------------------------------------------------------- globals --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nat {
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
pub enum Kept {
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
    verified_path: Py<PyAny>,
    resolve_import: Py<PyAny>,
    is_dataclass: Py<PyAny>,
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
    paths: PathGlobals,
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

/// What walks have learned about classes, by address. A call owns the table
/// while it runs and puts it back after, so a nested ``to_plain`` from a hook
/// starts with an empty one rather than finding it borrowed.
static CLASSES: GilCell<Option<Classes>> = GilCell::new(None);

/// Hashes an object's address: one multiply, its high bits spread by
/// hashbrown's own use of the top and bottom of the hash.
#[derive(Default, Clone, Copy)]
pub struct AddressHasher(u64);

impl std::hash::Hasher for AddressHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
    }

    #[inline(always)]
    fn write_usize(&mut self, address: usize) {
        // Objects are 16-byte aligned; the multiply moves the varying bits up.
        self.0 = ((address >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Where each registered object first sits, by address.
type Seen = HashMap<usize, u32, std::hash::BuildHasherDefault<AddressHasher>>;

/// The last walk's emptied buffers, so the next one reuses their capacity.
#[derive(Default)]
struct Buffers {
    frames: Vec<Frame<'static>>,
    stack: Vec<Entry<'static>>,
    seen: Seen,
    alive: Vec<B<'static>>,
}

static BUFFERS: GilCell<Option<Buffers>> = GilCell::new(None);

impl Buffers {
    /// Empty buffers bound to ``'py``; an empty `Vec` holds no `'static` value.
    fn take<'py>(py: Python<'py>) -> (Vec<Frame<'py>>, Vec<Entry<'py>>, Seen, Vec<B<'py>>) {
        let buffers = BUFFERS.take(py).unwrap_or_default();
        // SAFETY: each Vec is empty, so no value of the old lifetime survives.
        unsafe {
            (
                std::mem::transmute::<Vec<Frame<'static>>, Vec<Frame<'py>>>(buffers.frames),
                std::mem::transmute::<Vec<Entry<'static>>, Vec<Entry<'py>>>(buffers.stack),
                buffers.seen,
                std::mem::transmute::<Vec<B<'static>>, Vec<B<'py>>>(buffers.alive),
            )
        }
    }

    /// Empty ``frames``, ``stack``, ``seen`` and ``alive`` and keep them.
    fn give<'py>(
        py: Python<'py>,
        mut frames: Vec<Frame<'py>>,
        mut stack: Vec<Entry<'py>>,
        mut seen: Seen,
        mut alive: Vec<B<'py>>,
    ) {
        frames.clear();
        stack.clear();
        seen.clear();
        alive.clear();
        if frames.capacity() > BUFFER_LIMIT || alive.capacity() > BUFFER_LIMIT {
            return;
        }
        // SAFETY: each Vec is empty, so no value of lifetime `'py` is kept.
        let buffers = unsafe {
            Buffers {
                frames: std::mem::transmute::<Vec<Frame<'py>>, Vec<Frame<'static>>>(frames),
                stack: std::mem::transmute::<Vec<Entry<'py>>, Vec<Entry<'static>>>(stack),
                seen,
                alive: std::mem::transmute::<Vec<B<'py>>, Vec<B<'static>>>(alive),
            }
        };
        let stale = BUFFERS.replace(py, Some(buffers));
        drop(stale);
    }
}

const BUFFER_LIMIT: usize = 1 << 16;

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
        verified_path: get("_verified_path")?,
        resolve_import: get("resolve_import")?,
        is_dataclass: get("is_dataclass")?,
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
        paths: PathGlobals {
            module_type: get("ModuleType")?,
            sys_dict: py.import("sys")?.dict().unbind(),
        },
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
    let stale = (
        ENC.replace(py, Some(Rc::new(globals))),
        CLASSES.take(py),
        ENV.take(py),
        BUFFERS.take(py),
    );
    drop(stale);
    Ok(())
}

// -------------------------------------------------------- python state --

unsafe extern "C" {
    fn PyObject_GenericGetDict(obj: *mut ffi::PyObject, context: *mut c_void)
    -> *mut ffi::PyObject;
    /// A dict sized for ``minused`` entries, so filling it never resizes.
    fn _PyDict_NewPresized(minused: ffi::Py_ssize_t) -> *mut ffi::PyObject;
}

/// What the walk reads from ``treekle.serialize`` and ``treekle.reflect``,
/// read again whenever ``sync::ENCODE`` moves on, and shared by every call
/// while it stands still.
struct Env {
    encoders: Py<PyAny>,
    encoders_own: bool,
    /// ``_ENCODERS``, when it is a watched dict other than the module's own.
    encoders_dict: Option<Py<PyDict>>,
    encoder_for: Py<PyAny>,
    encoder_for_own: bool,
    /// ``_ENCODER_CACHE.data``, when the cache is a watched ``WeakKeyDictionary``.
    cache_data: Option<Py<PyDict>>,
    /// Whether ``_import_path``'s helpers (``_verified_path``,
    /// ``resolve_import``) and ``is_dataclass`` are the modules' own, so their
    /// answers can be read natively.
    helpers_own: bool,
    import_path: Py<PyAny>,
    /// ``_IMPORT_PATHS``, while ``_import_path`` is the module's own.
    import_paths: Option<Py<PyDict>>,
    canonical_reduce: Py<PyAny>,
    reduce_own: bool,
    verified_path: Py<PyAny>,
    attribute_names: Py<PyAny>,
    /// Whether ``attribute_names`` and the ``reflect`` functions it calls are
    /// the modules' own, so their caches can be read in their place.
    names_own: bool,
    slot_cache: Option<Py<PyDict>>,
    sorted_cache: Option<Py<PyDict>>,
    walk: Py<PyAny>,
    encoding: Py<PyAny>,
}

/// The ``Env`` the last call read, and the generation it was read at.
static ENV: GilCell<Option<(u64, Rc<Env>)>> = GilCell::new(None);

impl Env {
    /// ``Env::read``, reused while no watched dict was rewritten since.
    fn current(py: Python<'_>, g: &EncGlobals, now: Option<u64>) -> PyResult<Rc<Self>> {
        if let Some(now) = now
            && let Some((at, env)) = ENV.cloned(py)
            && at == now
        {
            return Ok(env);
        }
        let env = Rc::new(Self::read(py, g)?);
        if let Some(now) = now {
            let stale = ENV.replace(py, Some((now, env.clone())));
            drop(stale);
        }
        Ok(env)
    }

    /// Whether ``_ENCODERS`` and ``_encoder_for`` are the module's own, with
    /// every entry native, so a builtin container's encoder is known.
    fn native_encoders(&self, g: &EncGlobals) -> bool {
        self.encoders_own && self.encoder_for_own && g.encoders_all_native
    }
}

/// A ``WeakKeyDictionary``'s ``data``, watched along with the instance's own
/// attributes so a rebound ``data`` is seen too.
fn weak_data<'py>(g: &EncGlobals, cache: &B<'py>) -> Option<Bound<'py, PyDict>> {
    let py = cache.py();
    if !cache.get_type().is(g.weak_key_dict.bind(py)) {
        return None;
    }
    // An exact ``WeakKeyDictionary`` always has an instance dict.
    let attrs = unsafe {
        Bound::from_owned_ptr(
            py,
            PyObject_GenericGetDict(cache.as_ptr(), std::ptr::null_mut()),
        )
    };
    let attrs = attrs.cast_into_exact::<PyDict>().ok()?;
    if !sync::ENCODE.watch(py, &attrs) {
        return None;
    }
    let data = attrs.get_item(intern!(py, "data")).ok().flatten()?;
    let data = data.cast_into_exact::<PyDict>().ok()?;
    sync::ENCODE.watch_seeded(py, &data).then_some(data)
}

/// ``WeakKeyDictionary.get(kind)`` read from its ``data``.
///
/// A live weakref equals every other to the same referent and hashes as it
/// does, so any of a heap class's own weakrefs looks the entry up; and a class
/// with none cannot be a key.
fn weak_get<'py>(data: &Bound<'py, PyDict>, kind: &B<'py>) -> PyResult<Option<B<'py>>> {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    if unsafe { ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) } == 0 {
        return data.get_item(PyWeakrefReference::new(kind)?);
    }
    // A ``WeakKeyDictionary`` key carries the dict's ``_remove`` callback;
    // a class none of whose weakrefs has a callback is no key of any.
    let mut each = unsafe { (*class).tp_weaklist }.cast::<ffi::PyWeakReference>();
    let mut keyed = false;
    while !each.is_null() {
        if unsafe { !(*each).wr_callback.is_null() } {
            keyed = true;
            break;
        }
        each = unsafe { (*each).wr_next };
    }
    if !keyed {
        return Ok(None);
    }
    let head = unsafe { (*class).tp_weaklist };
    data.get_item(unsafe { Bound::from_borrowed_ptr(kind.py(), head) })
}

fn watched<'py>(value: B<'py>) -> Option<Bound<'py, PyDict>> {
    let dict = value.cast_into_exact::<PyDict>().ok()?;
    sync::ENCODE.watch_seeded(dict.py(), &dict).then_some(dict)
}

impl Env {
    fn read<'py>(py: Python<'py>, g: &EncGlobals) -> PyResult<Self> {
        let module = g.module.bind(py);
        let reflect = g.reflect.bind(py);
        sync::ENCODE.watch(py, module);
        sync::ENCODE.watch(py, reflect);
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
        let helpers_own = global(intern!(py, "_verified_path"))?.is(g.verified_path.bind(py))
            && global(intern!(py, "resolve_import"))?.is(g.resolve_import.bind(py))
            && global(intern!(py, "is_dataclass"))?.is(g.is_dataclass.bind(py));
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
            encoders: encoders.unbind(),
            encoders_own,
            encoders_dict: encoders_dict.map(Bound::unbind),
            encoder_for: encoder_for.unbind(),
            cache_data: cache_data.map(Bound::unbind),
            helpers_own,
            import_path: import_path.unbind(),
            import_paths: import_paths.map(Bound::unbind),
            reduce_own: canonical_reduce.is(g.canonical_reduce.bind(py)),
            canonical_reduce: canonical_reduce.unbind(),
            verified_path: global(intern!(py, "_verified_path"))?.unbind(),
            attribute_names: attribute_names.unbind(),
            names_own,
            slot_cache: reflect_cache(intern!(py, "_SLOT_NAMES"))?.map(Bound::unbind),
            sorted_cache: reflect_cache(intern!(py, "_SORTED_SLOT_NAMES"))?.map(Bound::unbind),
            walk: global(intern!(py, "_encode"))?.unbind(),
            encoding: global(intern!(py, "_Encoding"))?.unbind(),
        })
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

/// ``_is_plain_key`` for a raw key, answered with no Python code; ``false``
/// also for a str whose ``startswith`` only Python can answer (one with lone
/// surrogates), which ``is_plain_key`` then reads.
///
/// # Safety
/// ``key`` is a live object.
#[inline(always)]
unsafe fn plain_key_raw(key: *mut ffi::PyObject) -> bool {
    unsafe {
        if ob_type(key) != t_str() {
            return false;
        }
        let text = if compact_ascii_str(key) {
            ascii_bytes(key)
        } else {
            match utf8_raw(key) {
                Some(text) => text,
                None => return false,
            }
        };
        !(text.starts_with(b"py/") || text.starts_with(b"json://"))
    }
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

/// How deep parts are encoded in place, on the native stack, before the
/// rest wait on the walk's own stack.
const IN_PLACE_DEPTH: usize = 48;

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
    /// Whether ``_ENCODERS`` and ``_encoder_for`` are the module's own, with
    /// every entry native, so a builtin container's encoder is known.
    native_containers: bool,
    sort: Flag<'py>,
    /// How many parts are being encoded in place, each inside the last.
    depth: usize,
    frames: Vec<Frame<'py>>,
    stack: Vec<Entry<'py>>,
    seen: Seen,
    alive: Vec<B<'py>>,
    env: Rc<Env>,
    /// The generation ``env`` was read at.
    generation: Option<u64>,
    classes: Classes,
    /// The last module an import path was read from: its name, the
    /// ``sys.modules`` generation then, and its namespace.
    module_memo: ModuleMemo<'py>,
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
        classes: Classes,
    ) -> PyResult<Self> {
        let hooks_are_immutable_empty = hooks.is(g.default_hooks.bind(py)) && hooks.len()? == 0;
        let generation = sync::ENCODE.rewrites();
        let env = Env::current(py, g, generation)?;
        let (mut frames, stack, seen, alive) = Buffers::take(py);
        frames.push(Frame {
            parent: ROOT,
            step: Slot::Index(0),
        });
        frames.push(Frame {
            parent: ROOT,
            step: Slot::Key(intern!(py, "").clone().into_any()),
        });
        let native_containers = env.native_encoders(g);
        let encoder = Self {
            py,
            g,
            dialect,
            python_dialect,
            hooks,
            hooks_are_immutable_empty,
            native_containers,
            sort,
            depth: 0,
            frames,
            stack,
            seen,
            alive,
            env,
            generation,
            classes,
            module_memo: std::cell::RefCell::new(None),
            shared: None,
            synced: 0,
            linked: Vec::new(),
            linked_frames: HashMap::new(),
        };
        Ok(encoder)
    }

    #[inline(always)]
    fn b<T>(&self, value: &'g Py<T>) -> &'g Bound<'py, T> {
        value.bind(self.py)
    }

    fn tagged(&self, tag: &Bound<'py, PyString>, value: B<'py>) -> PyResult<B<'py>> {
        // An interned exact str key into a new dict: neither can fail but on
        // memory, and no Python code runs.
        unsafe {
            let node = Bound::from_owned_ptr_or_err(self.py, _PyDict_NewPresized(1))?;
            if ffi::PyDict_SetItem(node.as_ptr(), tag.as_ptr(), value.as_ptr()) < 0 {
                return Err(PyErr::fetch(self.py));
            }
            Ok(node)
        }
    }

    // -------------------------------------------------------- the stack --

    /// ``_encode``: encode ``value``, placed at ``frame``, and all it holds.
    fn walk(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        // A leaf needs no walk, and so no root to write it into.
        if let Some(done) = self.leaf(value) {
            return done;
        }
        self.walk_node(value, frame)
    }

    /// ``walk`` for a value that is no leaf.
    fn walk_node(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let base = self.stack.len();
        let result = self.drain(value, frame, base);
        // A failed walk leaves its parts behind; they belong to no one now.
        self.stack.truncate(base);
        result
    }

    /// Encode ``value`` and every part it queues. The root is written into
    /// no node, so a reduce at the root keeps its checkpoint's node as
    /// ``ROOT_NODE``, which ``write`` never sees: a failure at the root has
    /// no checkpoint below it, as in Python.
    fn drain(&mut self, value: &B<'py>, frame: u32, base: usize) -> PyResult<B<'py>> {
        let root = unsafe { Bound::from_borrowed_ptr(self.py, ffi::Py_None()) };
        let done = match self.encode_node(value, frame, &root, &Slot::Index(0)) {
            Ok(done) => done,
            // Only the root's own reduce can have checked in below its parts.
            Err(error) => self.recover(error, base)?.2,
        };
        self.drain_from(done, base)
    }

    /// Encode every part queued above ``base``, writing each into its node;
    /// ``done`` is what the node they belong to encoded to, replaced if a
    /// failure restarts that node's own reduce.
    fn drain_from(&mut self, mut done: B<'py>, base: usize) -> PyResult<B<'py>> {
        let root = unsafe { Bound::from_borrowed_ptr(self.py, ffi::Py_None()) };
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
                Ok((slot, part)) => write(&node, &slot, part)?,
                Err(error) => {
                    let (node, slot, part) = self.recover(error, base)?;
                    // A reduce at the root restarts into the root itself.
                    if node.is(&root) {
                        done = part;
                    } else {
                        write(&node, &slot, part)?;
                    }
                }
            }
        }
        Ok(done)
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
        unsafe { self.plain_raw(item.as_ptr()) }
    }

    /// ``is_plain`` for a borrowed pointer, read with no reference taken.
    ///
    /// # Safety
    /// ``raw`` is a live object.
    #[inline(always)]
    unsafe fn plain_raw(&self, raw: *mut ffi::PyObject) -> bool {
        let kind = unsafe { ob_type(raw) };
        if kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() {
            return true;
        }
        kind == t_float()
            && (self.python_dialect || unsafe { ffi::PyFloat_AS_DOUBLE(raw) }.is_finite())
    }

    /// One part of ``node``, at ``slot`` under ``parent``, unless already
    /// plain: encoded now and written in place while the walk is shallow,
    /// else queued for the stack. Its own walk finishes all it queues before
    /// returning, so parts done now are done first to last, as popping them
    /// would; a caller queuing several reverses them after (``queued``).
    #[inline]
    fn part(&mut self, node: &B<'py>, slot: Slot<'py>, item: B<'py>, parent: u32) -> PyResult<()> {
        if self.is_plain(&item) {
            return Ok(());
        }
        if self.depth < IN_PLACE_DEPTH {
            let frame = self.push_frame(parent, slot.clone());
            self.depth += 1;
            let base = self.stack.len();
            // Its own walk's root: a reduce here checks in against ``None``,
            // so a failure below restarts into ``done``, as ``drain`` does. A
            // part that queues nothing (the walk is shallow, so most queue
            // nothing) needs no drain; a reduce queues its checkpoint.
            let root = unsafe { Bound::from_borrowed_ptr(self.py, ffi::Py_None()) };
            let done = match self.encode_node(&item, frame, &root, &Slot::Index(0)) {
                Ok(done) if self.stack.len() == base => Ok(done),
                Ok(done) => self.drain_from(done, base),
                Err(error) => self.recover(error, base).map(|(_, _, done)| done),
            };
            self.stack.truncate(base);
            self.depth -= 1;
            return write(node, &slot, done?);
        }
        self.stack.push(Entry::Part {
            value: item,
            node: node.clone(),
            slot,
            raw: false,
            parent,
        });
        Ok(())
    }

    /// Put the parts queued since ``base`` in the order the walk takes them:
    /// the first last, so it is popped first.
    #[inline]
    fn queued(&mut self, base: usize) {
        self.stack[base..].reverse();
    }

    /// ``_push_items``: each part of ``node``, as ``part`` takes it.
    fn push_items(&mut self, node: &Bound<'py, PyList>, parent: u32) -> PyResult<()> {
        let base = self.stack.len();
        let shared = node.as_any();
        // ``node`` is this walk's own list, which no code but it sees, so it
        // keeps its size and each item until that item is written.
        let list = node.as_ptr();
        for index in 0..unsafe { ffi::PyList_GET_SIZE(list) } {
            let raw = unsafe { ffi::PyList_GET_ITEM(list, index) };
            // A plain item stays as it is; skipping it here takes no
            // reference, so its refcount line is never written.
            if unsafe { self.plain_raw(raw) } {
                continue;
            }
            let item = unsafe { Bound::from_borrowed_ptr(self.py, raw) };
            self.part(shared, Slot::Index(index as usize), item, parent)?;
        }
        self.queued(base);
        Ok(())
    }

    /// A dict holding ``entries``, its parts taken as ``_push_values`` takes
    /// them: what building the dict then pushing its values does, without
    /// reading the dict back.
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
            self.push_values(&node, parent)?;
            return Ok(node);
        }
        let base = self.stack.len();
        let shared = node.as_any();
        for (key, item) in entries {
            self.part(shared, Slot::Key(key), item, parent)?;
        }
        self.queued(base);
        Ok(node)
    }

    /// ``_push_values``: each value of ``node``, as ``part`` takes it.
    fn push_values(&mut self, node: &Bound<'py, PyDict>, parent: u32) -> PyResult<()> {
        let base = self.stack.len();
        let shared = node.as_any();
        for (key, item) in dict_entries(node) {
            self.part(shared, Slot::Key(key), item, parent)?;
        }
        self.queued(base);
        Ok(())
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
        // Called only once registrations were shared, so ``shared`` is set.
        let Some((seen, alive)) = &self.shared else {
            unreachable!("unshare runs after share");
        };
        let end = alive.len();
        // A value repeated inside a dict key is registered once per copy.
        for index in mark..end {
            let id = alive.get_item(index)?.as_ptr() as usize;
            if seen.contains(id)? {
                seen.del_item(id)?;
            }
        }
        alive.del_slice(mark, end)?;
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
        loop {
            if here == place {
                return true;
            }
            if here == IN_KEY || here == ROOT {
                return false;
            }
            here = self.frames[here as usize].parent;
        }
    }

    // ------------------------------------------------------- class facts --

    /// The current generation, with ``env`` read again if it has moved on.
    #[inline]
    fn current(&mut self) -> PyResult<Option<u64>> {
        let now = sync::ENCODE.rewrites();
        if now.is_none() || now != self.generation {
            // The generation is read before the dicts, so a change made after
            // this read is caught at the next check.
            self.env = Env::current(self.py, self.g, now)?;
            self.generation = now;
            self.native_containers = self.env.native_encoders(self.g);
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

    /// ``value`` encoded as ``_encode_object`` with its slots read in place,
    /// when its class is one whose every fact ``encode_object`` would read is
    /// known, or read now in one pass; ``None`` leaves it to ``encoder_of``.
    #[inline]
    fn encode_planned(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        frame: u32,
    ) -> PyResult<Option<B<'py>>> {
        let now = self.current()?;
        if !self.env.names_own {
            return Ok(None);
        }
        let sort = self.sort.get()?;
        let py = self.py;
        let plan = move |info: &ClassInfo| {
            let layout = info.layouts[usize::from(sort)].as_ref()?;
            match (&info.encoder, &info.path) {
                (Some(Kept::Native(Nat::Object)), Some(path)) if layout.direct => {
                    Some((path.bind(py).clone(), layout.clone()))
                }
                _ => None,
            }
        };
        let planned = match (self.known(kind, now), now) {
            (Some(info), _) => plan(info),
            (None, Some(now)) => return self.first_sight(value, kind, frame, now, sort),
            (None, None) => return Ok(None),
        };
        let Some((path, layout)) = planned else {
            return Ok(None);
        };
        self.register(value, frame);
        self.direct_object(value, path, &layout, frame).map(Some)
    }

    fn native_or_foreign(&self, func: B<'py>) -> Enc<'py> {
        match native_of(&self.g.natives, &func) {
            Some(nat) => Enc::Native(nat),
            None => Enc::Foreign(func),
        }
    }

    /// Every fact a plain slotted dataclass straight under ``object`` needs,
    /// read in one pass: its encoder, import path and layout, each exactly as
    /// ``encoder_of``, ``class_path`` and ``layout`` would find it, then used
    /// for ``value``. Kept only for a class met before. Anything else is left
    /// to them.
    fn first_sight(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        frame: u32,
        now: u64,
        sort: bool,
    ) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        let env = &self.env;
        let (true, Some(cache), Some(paths), Some(slots), Some(sorted)) = (
            env.encoders_own
                && env.encoder_for_own
                && env.helpers_own
                && self.g.encoders_all_native,
            &env.cache_data,
            &env.import_paths,
            &env.slot_cache,
            &env.sorted_cache,
        ) else {
            return Ok(None);
        };
        let key = kind.as_ptr() as usize;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        if !plain_heap_class(kind) || !generic_getattr(class) {
            return Ok(None);
        }
        // What Python's caches hold for ``kind`` answers first, as it does
        // for ``_encoder_for``, ``_import_path`` and ``attribute_names``. A
        // class no watched cache ever took a key for is in none of them.
        if sync::planted(key)
            && (weak_get(cache.bind(py), kind)?.is_some()
                || weak_get(slots.bind(py), kind)?.is_some()
                || weak_get(sorted.bind(py), kind)?.is_some()
                || paths.bind(py).contains(PyInt::new(py, key))?)
        {
            return Ok(None);
        }
        let mut offsets = [None; STACK_SLOTS];
        let Some(read) = (unsafe { own_offsets(class, sort, py, &mut offsets) }) else {
            return Ok(None);
        };
        // More slots than the stack holds: read again into the heap.
        let spilled = (read.count > STACK_SLOTS)
            .then(|| own_layout(class, &Stamp::none(), sort, py))
            .flatten();
        let offsets = match &spilled {
            Some(layout) => &layout.offsets[..],
            None if read.count <= STACK_SLOTS => &offsets[..read.count],
            None => return Ok(None),
        };
        let Some(path) = self.native_import_path(kind)? else {
            return Ok(None);
        };
        // The slots were read from the class's own dict; a dataclass is one
        // whose MRO finds ``__dataclass_fields__``, which for a class
        // straight under ``object`` means its own dict.
        if unsafe { type_lookup(class, intern!(py, "__dataclass_fields__").as_ptr()) }.is_null() {
            return Ok(None);
        }
        // Most classes a cold walk meets are met once: one is kept only when
        // met again, in this walk or a later one. No Python code ran since
        // the slots were read, so a stamp taken now holds for them.
        let layout = if self.classes.met_before(key) {
            let stamp = Stamp::always(kind);
            let Some(layout) =
                (unsafe { layout_from(py, stamp.clone_in(py), sort, &read, offsets) })
            else {
                return Ok(None);
            };
            let layout = Rc::new(layout);
            let mut layouts = [None, None];
            layouts[usize::from(sort)] = Some(layout.clone());
            let info = ClassInfo {
                stamp,
                generation: now,
                encoder: Some(Kept::Native(Nat::Object)),
                path: Some(path.clone().unbind()),
                layouts,
                enum_member: None,
                dict_order: None,
            };
            let stale = self.classes.insert(key, info);
            drop(stale);
            Some(layout)
        } else {
            None
        };
        // Unsorted, the output follows ``__slots__``' order, which only a
        // layout holds.
        let layout = match layout {
            None if !sort => {
                unsafe { layout_from(py, Stamp::none(), false, &read, offsets) }.map(Rc::new)
            }
            layout => layout,
        };
        if !sort && layout.is_none() {
            return Ok(None);
        }
        self.register(value, frame);
        let done = match &layout {
            Some(layout) => self.direct_object(value, path, layout, frame),
            None => self.direct_slots(value, path, read.names, offsets, frame),
        };
        done.map(Some)
    }

    /// ``_encoder_for(kind)``.
    fn encoder_of(&mut self, kind: &B<'py>) -> PyResult<Enc<'py>> {
        let now = self.current()?;
        if self.env.native_encoders(self.g) {
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
        let encoder_for = env.encoder_for.bind(py);
        if !env.encoder_for_own {
            return Ok((self.native_or_foreign(encoder_for.call1((kind,))?), false));
        }
        let (listed, keepable) = if env.encoders_own && self.g.encoders_all_native {
            // The caller found ``kind`` among the native entries or not at all.
            (None, true)
        } else if let Some(dict) = &env.encoders_dict {
            (dict.bind(py).get_item(kind)?, true)
        } else {
            let found = env
                .encoders
                .bind(py)
                .call_method1(intern!(py, "get"), (kind,))?;
            (Some(found), env.encoders_own)
        };
        if let Some(found) = listed
            && found.is_truthy()?
        {
            return Ok((self.native_or_foreign(found), keepable));
        }
        let Some(data) = &env.cache_data else {
            return Ok((self.native_or_foreign(encoder_for.call1((kind,))?), false));
        };
        if let Some(found) = weak_get(data.bind(py), kind)?
            && found.is_truthy()?
        {
            return Ok((self.native_or_foreign(found), keepable));
        }
        if env.helpers_own
            && env.encoders_own
            && self.g.encoders_all_native
            && self.plain_dataclass_encoder(kind).is_some()
        {
            return Ok((Enc::Native(Nat::Object), keepable));
        }
        // ``_encoder_for`` caches what it chooses under ``kind``, a key no
        // kept fact read, so the walk keeps its other facts.
        let chosen = sync::ENCODE.expecting(kind.as_ptr(), || encoder_for.call1((kind,)))?;
        Ok((self.native_or_foreign(chosen), keepable))
    }

    /// ``_encode_object``, when ``_encoder_for`` would choose it for ``kind``
    /// with no Python code run: a heap class under a plain ``type``
    /// metaclass, neither a ``type`` nor an ``Enum``, whose MRO finds
    /// ``__dataclass_fields__`` (``is_dataclass``). Such a class is none of the
    /// ``Path``, ``UUID`` and ``datetime`` classes the chooser tests before it.
    fn plain_dataclass_encoder(&self, kind: &B<'py>) -> Option<&'g Py<PyAny>> {
        let py = self.py;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let plain = plain_heap_class(kind)
            && unsafe {
                !type_lookup(class, intern!(py, "__dataclass_fields__").as_ptr()).is_null()
            };
        if !plain {
            return None;
        }
        self.g
            .natives
            .iter()
            .find(|(_, nat)| *nat == Nat::Object)
            .map(|(func, _)| func)
    }

    /// ``_import_path(value)`` for a class whose path reads with no Python
    /// code: a heap class under a plain ``type`` metaclass, its ``__module__``
    /// an exact str in its own dict, its ``__qualname__`` free of
    /// ``<locals>``, its module in ``sys.modules`` and reached from there, by
    /// plain dict entries of the module and of each class on the way, as
    /// ``resolve_import`` walks. ``None`` to ask Python.
    fn native_import_path(&self, kind: &B<'py>) -> PyResult<Option<B<'py>>> {
        if !self.env.helpers_own {
            return Ok(None);
        }
        native_import_path_in(self.py, &self.g.paths, &self.module_memo, kind)
    }

    /// ``_import_path(value)``, and whether ``_IMPORT_PATHS`` answered.
    fn import_path(&self, value: &B<'py>) -> PyResult<(B<'py>, bool)> {
        let py = self.py;
        if let Some(paths) = &self.env.import_paths
            && let Some(cached) = paths
                .bind(py)
                .get_item(PyInt::new(py, value.as_ptr() as usize))?
            && !cached.is_none()
            && cached.get_item(0)?.is(value)
        {
            return Ok((cached.get_item(1)?, true));
        }
        // Kept with the class's facts rather than in ``_IMPORT_PATHS``.
        if self.env.import_paths.is_some()
            && let Some(path) = self.native_import_path(value)?
        {
            return Ok((path, true));
        }
        // ``_import_path`` caches the path under ``id(value)``, which only
        // this lookup read, so the walk keeps its other facts.
        let path = sync::ENCODE.expecting(value.as_ptr(), || {
            self.env.import_path.bind(py).call1((value,))
        })?;
        Ok((path, false))
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
        let py = self.py;
        let (Some(slot_cache), Some(sorted_cache)) = (&self.env.slot_cache, &self.env.sorted_cache)
        else {
            return Ok(None);
        };
        let (slot_cache, sorted_cache) =
            (slot_cache.bind(py).clone(), sorted_cache.bind(py).clone());
        let before = Stamp::always(kind);
        // ``attribute_names(sort=True)`` reads ``_SORTED_SLOT_NAMES`` first;
        // any other answer starts from ``slot_names``, which reads
        // ``_SLOT_NAMES`` first.
        let sorted = match sort {
            true => {
                weak_get(&sorted_cache, kind)?.filter(|names| names.is_truthy().unwrap_or(false))
            }
            false => None,
        };
        let from_sorted = sorted.is_some();
        let cached = match sorted {
            Some(names) => Some(names),
            None => weak_get(&slot_cache, kind)?.filter(|names| !names.is_none()),
        };
        let names = match cached {
            Some(names) if !sort || from_sorted => names,
            // Anything else is named by ``reflect`` itself, which caches it. An
            // `if let` match guard needs Rust 1.95, above the 1.88 floor.
            cached => {
                if cached.is_none()
                    && let Some(layout) = self.kept_layout(kind, now, &before, sort)
                {
                    return Ok(Some(layout));
                }
                let reflect = self.g.reflect.bind(self.py);
                let name = if sort { "_sorted_slots" } else { "slot_names" };
                let names = interned_item(reflect, &PyString::intern(self.py, name))?;
                sync::ENCODE.expecting(kind.as_ptr(), || names.call1((kind,)))?
            }
        };
        let slots: Vec<B<'py>> = names.try_iter()?.collect::<PyResult<_>>()?;
        Ok(Some(self.keep_layout(kind, now, &before, sort, slots)?))
    }

    /// ``class_layout``'s reading, kept with the class's facts.
    fn kept_layout(
        &mut self,
        kind: &B<'py>,
        now: Option<u64>,
        before: &Stamp,
        sort: bool,
    ) -> Option<Rc<Layout>> {
        let layout = self.class_layout(kind, before, sort)?;
        if let Some(info) = self.keep(kind, now, before) {
            info.layouts[usize::from(sort)] = Some(layout.clone());
        }
        Some(layout)
    }

    /// The layout of a class whose instances encode from their slots alone,
    /// read with no Python code; ``None`` for ``layout``'s general reading.
    fn class_layout(&self, kind: &B<'py>, before: &Stamp, sort: bool) -> Option<Rc<Layout>> {
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { ob_type(kind.as_ptr()) } != &raw mut ffi::PyType_Type || !generic_getattr(class)
        {
            return None;
        }
        own_layout(class, before, sort, self.py).map(Rc::new)
    }

    fn keep_layout(
        &mut self,
        kind: &B<'py>,
        now: Option<u64>,
        before: &Stamp,
        sort: bool,
        slots: Vec<B<'py>>,
    ) -> PyResult<Rc<Layout>> {
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let generic = generic_getattr(class);
        let slots: Vec<(B<'py>, Option<isize>)> = slots
            .into_iter()
            .map(|name| {
                let offset = generic.then(|| slot_offset(class, &name)).flatten();
                (name, offset)
            })
            .collect();
        let py = self.py;
        let dictless = generic && dictless(class, py);
        let direct = dictless
            && slots.iter().all(|(name, offset)| {
                offset.is_some()
                    && tp(name) == t_str()
                    && utf8(name).is_some_and(|text| {
                        !text.starts_with(b"py/") && !text.starts_with(b"json://")
                    })
            });
        let layout = Rc::new(Layout::new(py, Stamp::always(kind), slots, direct)?);
        if let Some(info) = self.keep(kind, now, before) {
            info.layouts[usize::from(sort)] = Some(layout.clone());
        }
        Ok(layout)
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
        // ``Enum`` has no ``__slots__``, so every member has an instance dict.
        let attrs = unsafe {
            Bound::from_owned_ptr(
                py,
                PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()),
            )
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
        if !self.seen.is_empty()
            && let Some(&at) = self.seen.get(&(value.as_ptr() as usize))
        {
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
        // The builtin containers, while no hook and no rebound ``_ENCODERS``
        // could claim them: ``_encoder_for`` gives their native encoders.
        if self.hooks_are_immutable_empty
            && self.native_containers
            && sync::ENCODE.rewrites() == self.generation
        {
            let exact = tp(value);
            if exact == t_list() {
                return self.encode_list(value, frame);
            }
            if exact == t_dict() {
                return self.encode_dict(value, frame);
            }
            if exact == t_tuple() {
                return self.encode_tuple(value, frame);
            }
        }
        let kind = value.get_type().into_any();
        if !self.hooks_are_immutable_empty && self.hooks.contains(&kind)? {
            return self.encode_hooked(value, &kind, frame);
        }
        if let Some(done) = self.encode_planned(value, &kind, frame)? {
            return Ok(done);
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
            Nat::Tuple => self.encode_tuple(value, frame),
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
                self.part(node.as_any(), Slot::Key(tag), copied, frame)?;
                Ok(node.into_any())
            }
            Nat::Function => {
                let owner = optional_attr(value, intern!(py, "__self__").as_any())?
                    .unwrap_or_else(|| py.None().into_bound(py));
                if !(owner.is_none() || owner.is_instance(self.b(&self.g.paths.module_type))?) {
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
        if tp(value) == t_list() {
            let size = unsafe { ffi::PyList_GET_SIZE(value.as_ptr()) };
            return unsafe {
                Bound::from_owned_ptr_or_err(self.py, ffi::PyList_GetSlice(value.as_ptr(), 0, size))
                    .map(|list| list.cast_into_unchecked::<PyList>())
            };
        }
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

    /// ``_encode_tuple``: ``{"py/tuple": list(value)}``, its parts taken.
    fn encode_tuple(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        let items = if tp(value) == t_tuple() {
            // An exact tuple's items, copied straight into a list of its size.
            let source = value.as_ptr();
            let size = unsafe { ffi::PyTuple_GET_SIZE(source) };
            let list = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyList_New(size))? };
            for index in 0..size {
                unsafe {
                    let item = ffi::PyTuple_GET_ITEM(source, index);
                    ffi::Py_INCREF(item);
                    ffi::PyList_SET_ITEM(list.as_ptr(), index, item);
                }
            }
            unsafe { list.cast_into_unchecked::<PyList>() }
        } else {
            self.snapshot(value)?
        };
        self.push_items(&items, frame)?;
        self.tagged(intern!(py, "py/tuple"), items.into_any())
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

    /// ``encode_dict`` for an exact dict whose keys are all plain: copied
    /// whole, as Python's comprehension copies it, then each value that is
    /// not plain encoded into the copy in place. ``None`` when a key is not
    /// plain, which leaves the dict to ``encode_dict``'s general reading.
    fn plain_dict(&mut self, value: &B<'py>, frame: u32) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        let source = value.as_ptr();
        let mut position: ffi::Py_ssize_t = 0;
        let mut key = std::ptr::null_mut();
        let mut item = std::ptr::null_mut();
        // Exact str keys run no Python code to read, so the dict holds still.
        while unsafe { ffi::PyDict_Next(source, &mut position, &mut key, &mut item) } != 0 {
            if !unsafe { plain_key_raw(key) } {
                return Ok(None);
            }
        }
        self.register(value, frame);
        let node = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyDict_Copy(source))? };
        let base = self.stack.len();
        // ``node`` is this walk's own dict, which no other code sees: setting
        // a key it holds never resizes it, so its iteration stays valid.
        let mut position: ffi::Py_ssize_t = 0;
        while unsafe { ffi::PyDict_Next(node.as_ptr(), &mut position, &mut key, &mut item) } != 0 {
            if unsafe { self.plain_raw(item) } {
                continue;
            }
            let part = unsafe { Bound::from_borrowed_ptr(py, item) };
            let key = unsafe { Bound::from_borrowed_ptr(py, key) };
            self.part(&node, Slot::Key(key), part, frame)?;
        }
        self.queued(base);
        Ok(Some(node))
    }

    fn encode_dict(&mut self, value: &B<'py>, frame: u32) -> PyResult<B<'py>> {
        let py = self.py;
        if tp(value) == t_dict()
            && let Some(node) = self.plain_dict(value, frame)?
        {
            return Ok(node);
        }
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
        if self.env.names_own {
            let sort = self.sort.get()?;
            if let Some(layout) = self.layout(kind, sort)?
                && layout.direct
                && layout.stamp.holds(kind)
            {
                return self.direct_object(value, path, &layout, frame);
            }
        }
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
        self.push_values(&node, frame)?;
        Ok(node.into_any())
    }

    /// ``encode_object`` for an instance whose members are exactly its
    /// class's slots (``Layout::direct``): each read from its offset into a
    /// presized dict, its parts taken in order.
    #[inline]
    fn direct_object(
        &mut self,
        value: &B<'py>,
        path: B<'py>,
        layout: &Layout,
        frame: u32,
    ) -> PyResult<B<'py>> {
        self.direct_slots(value, path, layout.names.as_ptr(), &layout.offsets, frame)
    }

    /// ``direct_object`` from the slots' names, a tuple, and their offsets.
    #[inline]
    fn direct_slots(
        &mut self,
        value: &B<'py>,
        path: B<'py>,
        names: *mut ffi::PyObject,
        offsets: &[Option<isize>],
        frame: u32,
    ) -> PyResult<B<'py>> {
        let py = self.py;
        let size = offsets.len() + 1;
        let node = unsafe {
            Bound::from_owned_ptr_or_err(py, _PyDict_NewPresized(size as ffi::Py_ssize_t))?
                .cast_into_unchecked::<PyDict>()
        };
        node.set_item(intern!(py, "py/object"), path)?;
        let pending = self.g.pending.as_ptr();
        let base = self.stack.len();
        let shared = node.as_any();
        for (index, offset) in offsets.iter().enumerate() {
            let Some(offset) = offset else {
                continue;
            };
            let member = unsafe {
                *value
                    .as_ptr()
                    .cast::<u8>()
                    .offset(*offset)
                    .cast::<*mut ffi::PyObject>()
            };
            if member.is_null() || member == pending {
                continue;
            }
            // The layout holds its names, and the walk holds the instance,
            // which holds the member while no Python code runs.
            let name = unsafe { ffi::PyTuple_GET_ITEM(names, index as ffi::Py_ssize_t) };
            if unsafe { ffi::PyDict_SetItem(node.as_ptr(), name, member) } < 0 {
                return Err(PyErr::fetch(py));
            }
            if unsafe { self.plain_raw(member) } {
                continue;
            }
            let key = unsafe { Bound::from_borrowed_ptr(py, name) };
            let member = unsafe { Bound::from_borrowed_ptr(py, member) };
            self.part(shared, Slot::Key(key), member, frame)?;
        }
        self.queued(base);
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
            let names = self
                .env
                .attribute_names
                .bind(py)
                .call((value,), Some(&kwargs))?;
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
        let mut extra = Vec::new();
        // ``getattr(value, "__dict__", None)``: a plain class with no
        // ``__dict__`` slot and no ``__dict__`` in its MRO has none to give.
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let dictless = generic_getattr(class) && dictless(class, py);
        let state = if dictless {
            None
        } else {
            optional_attr(value, intern!(py, "__dict__").as_any())?
        };
        if layout.len() == 0
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
                // ``key in names``, as ``tuple.__contains__`` compares.
                let mut known = false;
                for (name, _) in layout.slots(py) {
                    if py_eq(&name, &key)? {
                        known = true;
                        break;
                    }
                }
                if !known {
                    extra.push(key);
                }
            }
        }
        let mut members = Vec::with_capacity(layout.len() + extra.len());
        if extra.is_empty() {
            for (index, (name, _)) in layout.slots(py).enumerate() {
                if let Some(member) = self.member(value, kind, &name, Some((&layout, index)))? {
                    members.push((name, member));
                }
            }
            return Ok(members);
        }
        let all = PyList::new(py, layout.slots(py).map(|(name, _)| name))?;
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
                .slots(py)
                .position(|(slot, _)| slot.is(&name))
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
        let native = match recipe {
            Some(parts) => Some(parts),
            None if self.env.reduce_own => self.native_reduce(value)?,
            None => None,
        };
        let mut parts = match native {
            Some(parts) if parts.is_empty() => return self.encode_unreduced(value, frame),
            // A bare-name reduce, its path verified natively.
            Some(mut parts) if parts.len() == 1 => {
                return self.tagged(intern!(py, "py/type"), parts.remove(0));
            }
            Some(parts) => parts,
            None => {
                let recipe = self.env.canonical_reduce.bind(py).call1((value,))?;
                if recipe.is_none() {
                    return self.encode_unreduced(value, frame);
                }
                // A bare-name reduce names a module global, written as an
                // import path.
                if is_str(&recipe) {
                    let module = kind.getattr(intern!(py, "__module__"))?;
                    let path = format!("{}.{}", module.str()?.to_cow()?, recipe.str()?.to_cow()?);
                    let path = self.env.verified_path.bind(py).call1((path, value))?;
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
        // A recipe starts with a callable, so trailing ``None``s never empty it.
        while parts.last().is_some_and(|last| last.is_none()) {
            parts.pop();
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

    /// ``_canonical_reduce(value)``, read natively: ``Some(empty)`` where it
    /// gives ``None``, ``None`` where Python must answer (a str recipe, a
    /// ``PurePath`` or ``partial``, which it rewrites).
    fn native_reduce(&self, value: &B<'py>) -> PyResult<Option<Vec<B<'py>>>> {
        let py = self.py;
        let class = tp(value);
        let subtype = |base: &Py<PyAny>| unsafe {
            ffi::PyType_IsSubtype(class, base.as_ptr().cast::<ffi::PyTypeObject>()) != 0
        };
        if subtype(&self.g.pure_path) || subtype(&self.g.partial) {
            return Ok(None);
        }
        let reduced = match value.call_method1(intern!(py, "__reduce_ex__"), (2,)) {
            Ok(reduced) => reduced,
            Err(error)
                if error.is_instance_of::<PyAttributeError>(py)
                    || error.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                    || error.is_instance_of::<PyTypeError>(py)
                    || error.is_instance_of::<PyValueError>(py) =>
            {
                return Ok(Some(Vec::new()));
            }
            Err(error) => return Err(error),
        };
        if is_str(&reduced) {
            return Ok(self.global_path(value, &reduced)?.map(|path| vec![path]));
        }
        if !is_tuple(&reduced) {
            return Ok(Some(Vec::new()));
        }
        let reduced = unsafe { reduced.cast_unchecked::<PyTuple>() };
        let size = reduced.len();
        if !(2..=5).contains(&size) {
            return Ok(Some(Vec::new()));
        }
        let func = reduced.get_item(0)?;
        let args = reduced.get_item(1)?;
        if !is_callable(&func) || !is_tuple(&args) {
            return Ok(Some(Vec::new()));
        }
        let mut parts = Vec::with_capacity(size);
        parts.push(func);
        parts.push(args);
        for index in 2..size {
            let part = reduced.get_item(index)?;
            // ``list(part)`` for the list and dict item iterators.
            let part = if (index == 3 || index == 4) && !part.is_none() {
                let listed = unsafe { ffi::PySequence_List(part.as_ptr()) };
                unsafe { Bound::from_owned_ptr_or_err(py, listed)? }
            } else {
                part
            };
            parts.push(part);
        }
        Ok(Some(parts))
    }

    /// ``_verified_path(f"{type(value).__module__}.{name}", value)`` where it
    /// reads natively: the module is in ``sys.modules`` as a plain module
    /// whose own dict holds ``name`` as ``value`` itself, and no longer
    /// dotted prefix of the path is loaded. ``None`` to ask Python.
    fn global_path(&self, value: &B<'py>, name: &B<'py>) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        if !self.env.helpers_own || tp(name) != t_str() {
            return Ok(None);
        }
        let kind = value.get_type();
        let Some(module) = kind.as_any().getattr_opt(intern!(py, "__module__"))? else {
            return Ok(None);
        };
        let (Some(module_text), Some(name_text)) = (
            (tp(&module) == t_str()).then(|| utf8(&module)).flatten(),
            utf8(name),
        ) else {
            return Ok(None);
        };
        // ``resolve_import`` walks from the longest loaded prefix; a dotted
        // name could find a longer one than the module.
        if name_text.contains(&b'.') {
            return Ok(None);
        }
        let modules = self
            .g
            .paths
            .sys_dict
            .bind(py)
            .get_item(intern!(py, "modules"))?;
        let Some(modules) = modules.and_then(|modules| modules.cast_into_exact::<PyDict>().ok())
        else {
            return Ok(None);
        };
        let Some(found) = modules.get_item(&module)? else {
            return Ok(None);
        };
        if !found.get_type().is(self.g.paths.module_type.bind(py)) {
            return Ok(None);
        }
        let namespace = unsafe {
            Bound::from_borrowed_ptr(py, ffi::PyModule_GetDict(found.as_ptr()))
                .cast_into_unchecked::<PyDict>()
        };
        match namespace.get_item(name)? {
            Some(walked) if walked.is(value) => {}
            _ => return Ok(None),
        }
        let mut path = Vec::with_capacity(module_text.len() + 1 + name_text.len());
        path.extend_from_slice(module_text);
        path.push(b'.');
        path.extend_from_slice(name_text);
        // Both halves are a str's own UTF-8.
        let path = unsafe { std::str::from_utf8_unchecked(&path) };
        Ok(Some(PyString::new(py, path).into_any()))
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
        let state = self.env.encoding.bind(py).call((), Some(&kwargs))?;
        let mark = self.alive.len();
        // Python's walk registers into ``seen`` and ``alive`` in place.
        let result = match self.env.walk.bind(py).call1((value, &state)) {
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
        self.link_in_key();
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
        self.link_in_key();
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

    /// ``_IN_KEY`` as ``IN_KEY``'s linked place, once a Python walk needs one.
    fn link_in_key(&mut self) {
        if self.linked_frames.is_empty() {
            self.remember_linked(IN_KEY, self.g.in_key.bind(self.py).clone());
        }
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
        let sort = Flag::new(sort, true);
        let classes = CLASSES.take(py).unwrap_or_default();
        let mut encoder = Encoder::new(py, g, dialect, python_dialect, hooks, sort, classes)?;
        let tree = encoder.walk(obj, ROOT);
        // Moved out whole: a default ``Classes`` left in their place would
        // allocate and zero its tables only to drop them.
        let Encoder {
            classes,
            frames,
            stack,
            seen,
            alive,
            ..
        } = encoder;
        Buffers::give(py, frames, stack, seen, alive);
        // A nested call may have put back its own map meanwhile; this one
        // replaces it, and the other drops after the swap.
        let stale = CLASSES.replace(py, Some(classes));
        drop(stale);
        tree?
    };
    finish(py, g, tree, mutable.as_ref())
}

/// ``to_plain``'s last step: the tree, frozen unless ``mutable``.
fn finish<'py>(
    py: Python<'py>,
    g: &EncGlobals,
    tree: B<'py>,
    mutable: Option<&B<'py>>,
) -> PyResult<B<'py>> {
    // Read after the walk, as Python reads it.
    let mutable = match mutable {
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
