//! ``from_plain``: plain trees to Python values, matching ``treekle.deserialize``.
//!
//! Each function mirrors the Python function of the same name. Type facts are
//! compiled once into a ``Plan`` per target and cached across calls: classes by
//! weak reference, field types only when Python's own ``_FIELD_TYPES`` caches them.

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

use hashbrown::{HashMap, HashSet};
use pyo3::exceptions::{
    PyAttributeError, PyImportError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyDict, PyFrozenSet, PyIterator, PyList, PySet, PyString, PyTuple, PyWeakrefReference,
};

use crate::common::*;
use crate::paths::{Loc, Paths, PlacedTable, ROOT, StepRef};

// ---------------------------------------------------------------- globals --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Any,
    None,
    Bool,
    Int,
    Float,
    Complex,
    Str,
    Leaf,
    Literal,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
    MappingProxy,
    Type,
    Union,
    Object,
    TypedDict,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tag {
    Type,
    Function,
    Tuple,
    Set,
    FrozenSet,
    MappingProxy,
    B64,
    Float,
    Complex,
    Path,
    Uuid,
    Datetime,
    Reduce,
    Hook,
    Object,
}

/// Every tag, in the precedence ``_TAGS`` gives a node carrying several.
const TAGS: [(&str, Tag); 15] = [
    ("py/type", Tag::Type),
    ("py/function", Tag::Function),
    ("py/tuple", Tag::Tuple),
    ("py/set", Tag::Set),
    ("py/frozenset", Tag::FrozenSet),
    ("py/mappingproxy", Tag::MappingProxy),
    ("py/b64", Tag::B64),
    ("py/float", Tag::Float),
    ("py/complex", Tag::Complex),
    ("py/path", Tag::Path),
    ("py/uuid", Tag::Uuid),
    ("py/datetime", Tag::Datetime),
    ("py/reduce", Tag::Reduce),
    ("py/hook", Tag::Hook),
    ("py/object", Tag::Object),
];

fn tag_named(name: &[u8]) -> Option<Tag> {
    TAGS.iter()
        .find(|(text, _)| text.as_bytes() == name)
        .map(|(_, tag)| *tag)
}

fn tag_text(tag: Tag) -> &'static str {
    TAGS.iter()
        .find(|(_, each)| *each == tag)
        .map(|(text, _)| *text)
        .unwrap_or("")
}

pub struct DecGlobals {
    module: Py<PyDict>,
    read_error: Py<PyAny>,
    invalid: Py<PyAny>,
    pending: Py<PyAny>,
    absent: Py<PyAny>,
    object: Py<PyAny>,
    mappingproxy: Py<PyAny>,
    mapping: Py<PyAny>,
    sequence: Py<PyAny>,
    real: Py<PyAny>,
    decimal: Py<PyAny>,
    enum_type: Py<PyAny>,
    literal: Py<PyAny>,
    union_type: Py<PyAny>,
    generic_alias: Py<PyAny>,
    typing_union: Py<PyAny>,
    get_origin: Py<PyAny>,
    get_args: Py<PyAny>,
    is_typeddict: Py<PyAny>,
    is_dataclass: Py<PyAny>,
    fields: Py<PyAny>,
    /// ``dataclasses._FIELD``, while ``fields`` is ``dataclasses.fields``, so
    /// ``fields`` can be read natively.
    field_kind: Option<Py<PyAny>>,
    field_layout: Option<FieldLayout>,
    missing: Py<PyAny>,
    json_loads: Py<PyAny>,
    json_error: Py<PyAny>,
    lax_bools: Py<PyAny>,
    resolve_alias: Py<PyAny>,
    decoder_for: Py<PyAny>,
    union_candidates: Py<PyAny>,
    fits: Py<PyAny>,
    is_subclass: Py<PyAny>,
    is_protocol: Py<PyAny>,
    is_any: Py<PyAny>,
    field_types: Py<PyAny>,
    field_types_cache: Py<PyAny>,
    required_keys: Py<PyAny>,
    import_name: Py<PyAny>,
    object_setattr: Py<PyAny>,
    complex_type: Py<PyAny>,
    float_type: Py<PyAny>,
    int_type: Py<PyAny>,
    str_type: Py<PyAny>,
    type_type: Py<PyAny>,
    tuple_type: Py<PyAny>,
    sorted: Py<PyAny>,
    shown: Py<PyAny>,
    /// The longest leaf repr ``_shown`` keeps whole.
    shown_limit: usize,
    decoders: Vec<(Py<PyAny>, Kind)>,
    tag_leaves: Vec<(Tag, Py<PyAny>)>,
}

/// Shared, so a call keeps the globals it started with even if a hook reloads
/// ``treekle.deserialize`` and so replaces them.
static DEC: GilCell<Option<Rc<DecGlobals>>> = GilCell::new(None);

/// The ``dataclasses.Field`` attributes the decoder reads.
#[derive(Clone, Copy)]
enum FieldSlot {
    Name,
    Default,
    DefaultFactory,
    Init,
    Kind,
}

const FIELD_SLOTS: [FieldSlot; 5] = [
    FieldSlot::Name,
    FieldSlot::Default,
    FieldSlot::DefaultFactory,
    FieldSlot::Init,
    FieldSlot::Kind,
];

impl FieldSlot {
    fn name<'py>(self, py: Python<'py>) -> &'py Bound<'py, PyString> {
        match self {
            Self::Name => pyo3::intern!(py, "name"),
            Self::Default => pyo3::intern!(py, "default"),
            Self::DefaultFactory => pyo3::intern!(py, "default_factory"),
            Self::Init => pyo3::intern!(py, "init"),
            Self::Kind => pyo3::intern!(py, "_field_type"),
        }
    }
}

/// Where ``dataclasses.Field`` instances hold each attribute the decoder
/// reads, for a ``Field`` whose class is exactly ``kind``. An offset is kept
/// only for a plain object member that generic lookup would read directly.
struct FieldLayout {
    kind: Py<PyAny>,
    /// ``Field``'s state when the offsets were read; they serve only while
    /// it holds, so a ``Field`` class changed later is read by ``getattr``.
    stamp: Stamp,
    offsets: [Option<isize>; 5],
}

fn field_layout(kind: &B<'_>) -> Option<FieldLayout> {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    // A class whose attribute lookup is not ``object``'s could answer
    // ``getattr`` without its slots; and an immutable type cannot change them.
    let generic = unsafe { (*class).tp_getattro }.map(|f| f as usize) == generic_getattr_slot();
    if !generic {
        return None;
    }
    let py = kind.py();
    let stamp = Stamp::always(kind);
    let offsets = FIELD_SLOTS.map(|slot| member_offset(class, slot.name(py)));
    Some(FieldLayout {
        kind: kind.clone().unbind(),
        stamp,
        offsets,
    })
}

unsafe extern "C" {
    fn _PyType_Lookup(kind: *mut ffi::PyTypeObject, name: *mut ffi::PyObject)
    -> *mut ffi::PyObject;
}

fn member_offset(class: *mut ffi::PyTypeObject, name: &Bound<'_, PyString>) -> Option<isize> {
    const PY_AUDIT_READ: std::ffi::c_int = 2;
    unsafe {
        let found = _PyType_Lookup(class, name.as_ptr());
        if found.is_null() || ob_type(found) != &raw mut ffi::PyMemberDescr_Type {
            return None;
        }
        let descriptor = found.cast::<ffi::PyMemberDescrObject>();
        if ffi::PyType_IsSubtype(class, (*descriptor).d_common.d_type) == 0 {
            return None;
        }
        let member = (*descriptor).d_member.cast::<ffi::PyMemberDef>();
        if (*member).type_code != ffi::Py_T_OBJECT_EX || (*member).flags & PY_AUDIT_READ != 0 {
            return None;
        }
        Some((*member).offset)
    }
}

/// Each field slot's member offset in ``kind``, where storing through it is
/// what ``object.__setattr__`` would do: ``kind`` is a plain-``type`` class
/// whose ``__setattr__`` is ``object``'s, and the name finds a writable
/// object member of ``kind``'s own layout. Stamped before it is read.
fn store_offsets(kind: &B<'_>, info: &DcInfo) -> (Stamp, Vec<Option<isize>>) {
    let py = kind.py();
    let stamp = Stamp::always(kind);
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    let generic = unsafe {
        ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
            && (*class).tp_setattro.map(|f| f as usize)
                == ffi::PyBaseObject_Type.tp_setattro.map(|f| f as usize)
    };
    let Some(names) = info.names.as_ref().filter(|_| generic) else {
        return (stamp, Vec::new());
    };
    let offsets = names
        .iter()
        .map(|name| {
            let name = PyString::new(py, std::str::from_utf8(name).ok()?);
            writable_offset(class, &name)
        })
        .collect();
    (stamp, offsets)
}

/// The offset of ``name``'s object member, when ``_PyType_Lookup`` finds it
/// first and it is writable: ``object.__setattr__`` then stores through it.
fn writable_offset(class: *mut ffi::PyTypeObject, name: &Bound<'_, PyString>) -> Option<isize> {
    unsafe {
        let found = _PyType_Lookup(class, name.as_ptr());
        if found.is_null() || ob_type(found) != &raw mut ffi::PyMemberDescr_Type {
            return None;
        }
        let descriptor = found.cast::<ffi::PyMemberDescrObject>();
        if ffi::PyType_IsSubtype(class, (*descriptor).d_common.d_type) == 0 {
            return None;
        }
        let member = (*descriptor).d_member.cast::<ffi::PyMemberDef>();
        if (*member).type_code != ffi::Py_T_OBJECT_EX || (*member).flags & ffi::Py_READONLY != 0 {
            return None;
        }
        Some((*member).offset)
    }
}

fn item<'py>(module: &Bound<'py, PyDict>, name: &str) -> PyResult<B<'py>> {
    module
        .get_item(name)?
        .ok_or_else(|| PyAttributeError::new_err(format!("treekle.deserialize has no {name}")))
}

pub fn setup(module: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = module.py();
    let get = |name: &str| item(module, name).map(Bound::unbind);
    let builtins = py.import("builtins")?;
    let typing = item(module, "typing")?;
    let json = item(module, "json")?;
    let decoders = [
        ("_decode_any", Kind::Any),
        ("_decode_none", Kind::None),
        ("_decode_bool", Kind::Bool),
        ("_decode_int", Kind::Int),
        ("_decode_float", Kind::Float),
        ("_decode_complex", Kind::Complex),
        ("_decode_str", Kind::Str),
        ("_decode_bytes", Kind::Leaf),
        ("_decode_path", Kind::Leaf),
        ("_decode_uuid", Kind::Leaf),
        ("_decode_datetime", Kind::Leaf),
        ("_decode_enum", Kind::Leaf),
        ("_decode_literal", Kind::Literal),
        ("_decode_list", Kind::List),
        ("_decode_tuple", Kind::Tuple),
        ("_decode_set", Kind::Set),
        ("_decode_frozenset", Kind::FrozenSet),
        ("_decode_dict", Kind::Dict),
        ("_decode_mappingproxy", Kind::MappingProxy),
        ("_decode_type", Kind::Type),
        ("_decode_union", Kind::Union),
        ("_decode_object", Kind::Object),
        ("_decode_typeddict", Kind::TypedDict),
    ]
    .into_iter()
    .map(|(name, kind)| Ok((get(name)?, kind)))
    .collect::<PyResult<Vec<_>>>()?;
    let tag_leaves = [
        (Tag::B64, "_decode_bytes"),
        (Tag::Path, "_decode_path"),
        (Tag::Uuid, "_decode_uuid"),
        (Tag::Datetime, "_decode_datetime"),
    ]
    .into_iter()
    .map(|(tag, name)| Ok((tag, get(name)?)))
    .collect::<PyResult<Vec<_>>>()?;
    let dataclasses = py.import("dataclasses")?;
    let field_kind = if item(module, "fields")?.is(dataclasses.getattr("fields")?) {
        dataclasses.getattr("_FIELD").ok().map(Bound::unbind)
    } else {
        None
    };
    let shown = item(module, "_SHOWN")?;
    let mut shown_limit = usize::MAX;
    for name in ["maxstring", "maxlong", "maxother"] {
        shown_limit = shown_limit.min(shown.getattr(name)?.extract::<usize>()?);
    }
    let globals = DecGlobals {
        module: module.clone().unbind(),
        read_error: get("ReadError")?,
        invalid: get("Invalid")?,
        pending: get("PENDING")?,
        absent: get("ABSENT")?,
        object: builtins.getattr("object")?.unbind(),
        mappingproxy: get("MappingProxyType")?,
        mapping: get("Mapping")?,
        sequence: get("Sequence")?,
        real: get("Real")?,
        decimal: get("Decimal")?,
        enum_type: get("Enum")?,
        literal: get("Literal")?,
        union_type: get("UnionType")?,
        generic_alias: py.import("types")?.getattr("GenericAlias")?.unbind(),
        typing_union: typing.getattr("Union")?.unbind(),
        get_origin: get("get_origin")?,
        get_args: get("get_args")?,
        is_typeddict: get("is_typeddict")?,
        is_dataclass: get("is_dataclass")?,
        fields: get("fields")?,
        field_layout: field_layout(&dataclasses.getattr("Field")?),
        field_kind,
        missing: get("MISSING")?,
        json_loads: json.getattr("loads")?.unbind(),
        json_error: json.getattr("JSONDecodeError")?.unbind(),
        lax_bools: get("_LAX_BOOLS")?,
        resolve_alias: get("_resolve_alias")?,
        decoder_for: get("_decoder_for")?,
        union_candidates: get("_union_candidates")?,
        fits: get("_fits")?,
        is_subclass: get("_is_subclass")?,
        is_protocol: get("_is_protocol")?,
        is_any: get("_is_any")?,
        field_types: get("_field_types")?,
        field_types_cache: get("_FIELD_TYPES")?,
        required_keys: get("_required_keys")?,
        import_name: get("_import_name")?,
        object_setattr: builtins.getattr("object")?.getattr("__setattr__")?.unbind(),
        complex_type: builtins.getattr("complex")?.unbind(),
        float_type: builtins.getattr("float")?.unbind(),
        int_type: builtins.getattr("int")?.unbind(),
        str_type: builtins.getattr("str")?.unbind(),
        type_type: builtins.getattr("type")?.unbind(),
        tuple_type: builtins.getattr("tuple")?.unbind(),
        sorted: builtins.getattr("sorted")?.unbind(),
        shown: get("_shown")?,
        shown_limit,
        decoders,
        tag_leaves,
    };
    let stale = (DEC.replace(py, Some(Rc::new(globals))), CACHES.take(py));
    drop(stale);
    Ok(())
}

// ------------------------------------------------------------------ plans --

enum Fits {
    Always,
    Union,
    Literal,
    Type,
    Python,
    /// ``_has_members(value, origin)``: the origin is a Protocol class.
    Protocol(Py<PyAny>),
    /// ``isinstance(value, origin)``; ``None`` means the target class itself.
    IsInstance(Option<Py<PyAny>>),
}

struct UnionInfo {
    import_names: Vec<Option<Py<PyAny>>>,
    /// Member order for None, bool, int, float, str, mapping, and sequence
    /// data, each asked of ``_union_candidates`` when that kind first arrives.
    orders: [OnceCell<Vec<usize>>; 7],
}

struct DcInfo {
    index: Py<PyDict>,
    /// Each field's name as UTF-8, in ``index``'s order, when every name is
    /// an exact str: then an exact str key matches a name exactly when its
    /// bytes do, as the index dict's own lookup would find.
    names: Option<Vec<Box<[u8]>>>,
    /// ``names`` inverted: each name's slot.
    slots: HashMap<Box<[u8]>, usize>,
    /// Slots by name bytes: ``to_plain`` sorts a record's keys by default,
    /// so a sorted record meets its fields in this order.
    sorted: Vec<usize>,
    targets: Vec<Py<PyAny>>,
    plans: Vec<RefCell<Option<Rc<Plan>>>>,
    persistent: bool,
    /// Per field slot, the offset ``object.__setattr__`` writes when it stores
    /// into an instance of exactly the class these facts were read from;
    /// filled on first use, ``None`` where only ``setattr`` itself will do.
    stores: OnceCell<(Stamp, Vec<Option<isize>>)>,
}

impl DcInfo {
    /// The slot of the field named ``key``, searched from ``cursor``, which
    /// moves past the match: records list fields in order, so each key is
    /// usually found at the first slot tried. ``None`` where no field has
    /// that name, or names are not all exact strs (the dict decides then).
    fn slot_of(&self, key: &[u8], cursor: &mut usize) -> Option<usize> {
        let names = self.names.as_ref()?;
        if let Some(&slot) = self.sorted.get(*cursor)
            && *names[slot] == *key
        {
            *cursor += 1;
            return Some(slot);
        }
        let slot = *self.slots.get(key)?;
        *cursor = self
            .sorted
            .iter()
            .position(|each| *each == slot)
            .map_or(0, |at| at + 1);
        Some(slot)
    }
}

struct Plan {
    kind: Kind,
    is_object: bool,
    /// The alias-free target, when it is not the target itself.
    resolved: Option<Py<PyAny>>,
    args: Py<PyTuple>,
    origin_is_tuple: bool,
    origin_is_type: bool,
    fits: Fits,
    persistent: bool,
    union: Option<UnionInfo>,
    leaf: Option<Py<PyAny>>,
    import_name: Option<Py<PyAny>>,
    is_dataclass: bool,
    info: OnceCell<Rc<DcInfo>>,
    /// The target's state when compiled, where the target is a class.
    own: Option<Stamp>,
    /// Every other class the plan's facts were read from.
    deps: Vec<(Py<PyAny>, Stamp)>,
}

impl Plan {
    /// Whether every class this plan was compiled from is unchanged.
    #[inline]
    fn fresh(&self, target: &B<'_>) -> bool {
        self.own.as_ref().is_none_or(|stamp| stamp.holds(target))
            && self
                .deps
                .iter()
                .all(|(class, stamp)| stamp.holds(class.bind(target.py())))
    }
}

fn depend(deps: &mut Vec<(Py<PyAny>, Stamp)>, target: &B<'_>, class: &B<'_>) {
    if !is_class(class)
        || class.is(target)
        || deps.iter().any(|(each, _)| each.as_ptr() == class.as_ptr())
    {
        return;
    }
    if let Some(stamp) = Stamp::of(class) {
        deps.push((class.clone().unbind(), stamp));
    }
}

/// Parts of one value still to judge, as Python's ``_Parts``.
enum Parts<'py> {
    /// The value against each union member; one must fit.
    Members {
        value: B<'py>,
        members: Bound<'py, PyTuple>,
        next: usize,
        persistent: bool,
    },
    /// Each key, then its member, against the key and member types; all must fit.
    Entries {
        items: MapItems<'py>,
        key: (B<'py>, Rc<Plan>),
        member: (B<'py>, Rc<Plan>),
        held: Option<B<'py>>,
    },
    /// Each item against the item type; all must fit.
    Items {
        iter: Bound<'py, PyIterator>,
        item: (B<'py>, Rc<Plan>),
    },
}

/// ``_shallow_fit``'s answer: a verdict, or whether all parts must fit and the parts.
enum Shallow<'py> {
    Verdict(bool),
    Parts(bool, Parts<'py>),
}

/// One value being judged by its parts, as Python's ``_Judging``.
struct Judging<'py> {
    needs_all: bool,
    parts: Parts<'py>,
    key: (usize, usize),
    /// Holds the pair, so no other object takes either ``id`` in ``key``.
    _pair: (B<'py>, B<'py>),
}

// ------------------------------------------------------------------ frames --

/// ``_start``'s answer: a value read at once, or a frame that reads its parts.
///
/// Frames are boxed from where they start to where they finish. A ``Frame``
/// is ~180 bytes, and passing it by value through ``Start``, the walk's
/// stack, and ``finished`` copied it at each hop; that copying cost a tenth
/// of decoding a list of dataclasses.
enum Start<'py> {
    Value(B<'py>),
    Frame(Box<Frame<'py>>),
}

// Python's parent keeps the part it handed out (``pending_raw``, ``pending_at``)
// to record it bad if it fails; here the part's own frame carries its node and
// place, which are the same, and hands them up when it fails.
/// Python's ``_Frame``: a container being read.
struct Frame<'py> {
    /// Where its parts sit below.
    at: u32,
    /// Where the node itself sits; a proxy's dict sits one step below it.
    place: u32,
    /// The node this frame started from, a tag's envelope included.
    raw: B<'py>,
    /// The target a tag-decoded value must fit, with its plan.
    check: Option<(B<'py>, Rc<Plan>)>,
    /// A ``py/mappingproxy``'s dict and proxy, filled once the entries are read.
    then: Option<(Bound<'py, PyDict>, B<'py>)>,
    /// The address of the input container this frame reads, if any.
    reading: Option<usize>,
    body: Body<'py>,
}

/// What a frame that failed hands to the frame below it.
struct Failed<'py> {
    place: u32,
    raw: B<'py>,
    reading: Option<usize>,
}

/// How a sequence's items are typed.
enum Types<'py> {
    Repeat(B<'py>, Rc<Plan>),
    /// A fixed tuple's item types, planned as the tuple's plan is kept.
    Fixed(Bound<'py, PyTuple>, bool),
}

/// What a sequence's items are read into.
enum Built<'py> {
    /// The placed list itself.
    List(Bound<'py, PyList>),
    Tuple(Vec<B<'py>>),
    FrozenSet(Vec<B<'py>>),
    /// The placed set itself, refusing an unhashable member.
    Set(Bound<'py, PySet>),
}

/// What a ``_Steps`` frame builds from its parts.
enum Done<'py> {
    Complex,
    Hooked { rebuild: B<'py>, index: usize },
}

enum Body<'py> {
    /// ``_Items`` and ``_SetItems``.
    Items {
        raws: SeqItems<'py>,
        index: usize,
        types: Types<'py>,
        built: Built<'py>,
    },
    /// ``_Entries``.
    Entries {
        items: MapItems<'py>,
        key: (B<'py>, Rc<Plan>),
        value: (B<'py>, Rc<Plan>),
        plain_keys: bool,
        result: Bound<'py, PyDict>,
        pending_key: Option<B<'py>>,
    },
    /// ``_Fields`` for a ``TypedDict``: each declared key by its hint.
    Fields {
        items: MapItems<'py>,
        hints: B<'py>,
        members: Bound<'py, PyDict>,
        pending_name: Option<B<'py>>,
    },
    /// ``_Object``: a dataclass record, built once its fields have read.
    Object {
        entries: std::vec::IntoIter<(B<'py>, B<'py>, usize)>,
        info: Rc<DcInfo>,
        target: B<'py>,
        stored: bool,
        value: B<'py>,
        index: usize,
        before: usize,
        members: Vec<(B<'py>, B<'py>, usize)>,
        pending: Option<(B<'py>, usize)>,
    },
    /// ``_Stored``: a ``py/object`` record's attributes, set once read.
    Stored {
        items: MapItems<'py>,
        value: B<'py>,
        members: Vec<(B<'py>, B<'py>)>,
        pending_name: Option<B<'py>>,
    },
    /// ``_Steps``: a fixed list of parts whose failure is the container's own.
    Steps {
        jobs: std::vec::IntoIter<(B<'py>, B<'py>, Rc<Plan>, usize)>,
        values: Vec<B<'py>>,
        done: Done<'py>,
    },
    /// ``_Reduce``: the callable and arguments, the call, then the state.
    Reduce {
        parts: Vec<B<'py>>,
        part: usize,
        target: B<'py>,
        plan: Rc<Plan>,
        stateful: bool,
        index: usize,
        func: Option<B<'py>>,
        value: Option<B<'py>>,
    },
    /// ``_Union``: each member in turn until one reads with no bad part.
    Union {
        data: B<'py>,
        target: B<'py>,
        plan: Rc<Plan>,
        order: std::vec::IntoIter<usize>,
        key: (usize, usize),
        references: usize,
        marks: Mark,
        value: Option<B<'py>>,
    },
}

impl<'py> Frame<'py> {
    fn new(at: u32, raw: &B<'py>, body: Body<'py>) -> Box<Self> {
        Box::new(Self {
            at,
            place: at,
            raw: raw.clone(),
            check: None,
            then: None,
            reading: None,
            body,
        })
    }

    fn failed(self) -> Failed<'py> {
        Failed {
            place: self.place,
            raw: self.raw,
            reading: self.reading,
        }
    }
}

/// ``type(value) in _PLAIN_LEAVES``: an exact plain leaf, read as ``object`` unchanged.
#[inline(always)]
fn is_leaf_kind(kind: *mut ffi::PyTypeObject) -> bool {
    kind == t_str() || kind == t_int() || kind == t_float() || kind == t_bool() || kind == t_none()
}

// A call owns these while it runs, taken from ``CACHES`` and put back at the
// end, so a nested ``from_plain`` (from a hook) starts afresh instead of
// finding them borrowed. Entries leave the maps before they are dropped:
// dropping one may run Python code.
struct Caches {
    object: Rc<Plan>,
    classes: HashMap<usize, (Py<PyWeakrefReference>, Rc<Plan>)>,
    objects: HashMap<usize, (Py<PyWeakrefReference>, Rc<Plan>)>,
}

static CACHES: GilCell<Option<Caches>> = GilCell::new(None);

// A full map is emptied rather than searched for an entry to evict: the
// maps keep no order to evict by, and a search on every insert past the
// limit would cost the whole map each time. Live targets compile again.
const CLASS_LIMIT: usize = 1 << 14;
const OBJECT_LIMIT: usize = 1 << 10;

impl Caches {
    /// Empty maps around an existing ``object`` plan.
    fn around(object: Rc<Plan>) -> Self {
        Self {
            object,
            classes: HashMap::new(),
            objects: HashMap::new(),
        }
    }
}

fn new_caches(py: Python<'_>) -> Caches {
    Caches::around(Rc::new(Plan {
        kind: Kind::Any,
        is_object: true,
        resolved: None,
        args: PyTuple::empty(py).unbind(),
        origin_is_tuple: false,
        origin_is_type: false,
        fits: Fits::Always,
        persistent: true,
        union: None,
        leaf: None,
        import_name: Some(pystr(py, "builtins.object").unbind()),
        is_dataclass: false,
        info: OnceCell::new(),
        own: None,
        deps: Vec::new(),
    }))
}

// ----------------------------------------------------------------- errors --

pub enum DErr {
    Read(String, Option<PyErr>),
    Py(PyErr),
}

impl From<PyErr> for DErr {
    fn from(error: PyErr) -> Self {
        Self::Py(error)
    }
}

type DR<T> = Result<T, DErr>;

fn fail<T>(message: String) -> DR<T> {
    Err(DErr::Read(message, None))
}

fn fail_from<T>(message: String, cause: PyErr) -> DR<T> {
    Err(DErr::Read(message, Some(cause)))
}

/// Link ``cause`` to ``error`` as Python's ``raise error from cause`` does
/// inside ``except``: as both its ``__cause__`` and its ``__context__``.
fn chain(py: Python<'_>, error: &PyErr, cause: PyErr) {
    let cause = cause.into_value(py);
    unsafe {
        ffi::PyException_SetContext(error.value(py).as_ptr(), cause.clone_ref(py).into_ptr());
        ffi::PyException_SetCause(error.value(py).as_ptr(), cause.into_ptr());
    }
}

// ---------------------------------------------------------------- decoder --

pub struct Decoder<'py, 'g> {
    py: Python<'py>,
    g: &'g DecGlobals,
    hooks: Option<B<'py>>,
    allow_imports: Flag<'py>,
    strict: Flag<'py>,
    untagged: bool,
    built: Vec<B<'py>>,
    paths: Paths<'py>,
    placed: PlacedTable<'py>,
    bad: Option<Bound<'py, PyDict>>,
    bad_log: Vec<B<'py>>,
    resolved: Bound<'py, PyDict>,
    /// Plans for targets that are neither classes nor cached hints, each held
    /// so no other target takes its address during the walk.
    memo: HashMap<usize, (B<'py>, Rc<Plan>)>,
    /// Plans for lasting targets that cannot be weakly referenced, such as
    /// ``X | Y`` before 3.14, held for this call like ``memo``'s.
    unweakable: HashMap<usize, (B<'py>, Rc<Plan>)>,
    /// Field facts for dataclasses whose field types Python does not cache,
    /// each kept with the plan it was read for.
    infos: HashMap<usize, (Rc<Plan>, Rc<DcInfo>)>,
    object_plan: Rc<Plan>,
    /// The cross-call plan caches, owned for the length of the call.
    caches: Caches,
    /// How many references have resolved so far.
    references: usize,
    /// Each ``(data, union)`` that failed with no reference read, keyed by
    /// address and held so neither address is reused.
    unread: HashMap<(usize, usize), (B<'py>, B<'py>)>,
    reading: HashSet<usize>,
    /// How many containers ``quick`` is reading on the Rust stack now.
    quick_depth: u32,
}

type Mark = (usize, usize, usize);

/// How deep ``quick`` recurses before handing nodes to the frames, which
/// keep the Rust stack flat however deep the tree is.
const QUICK_DEPTH: u32 = 24;

/// What ``start`` would make of an exact dict, judged by its keys alone.
enum DictShape<'py> {
    /// Exact str keys, none a tag, reference or escaped key.
    Plain,
    /// ``{"py/tuple": payload}``.
    Tuple(B<'py>),
    /// Anything else, left to the frames.
    Other,
}

fn dict_shape<'py>(dict: &Bound<'py, PyDict>) -> DictShape<'py> {
    let entries = dict_entries(dict);
    if let [(key, payload)] = entries.as_slice()
        && tp(key) == t_str()
        && utf8(key) == Some(b"py/tuple".as_slice())
    {
        return DictShape::Tuple(payload.clone());
    }
    for (key, _) in &entries {
        let Some(text) = (tp(key) == t_str()).then(|| utf8(key)).flatten() else {
            return DictShape::Other;
        };
        if text.starts_with(b"py/") || text.starts_with(b"json://") {
            return DictShape::Other;
        }
    }
    DictShape::Plain
}

impl<'py, 'g> Decoder<'py, 'g> {
    // ------------------------------------------------------------ helpers --

    fn new(
        py: Python<'py>,
        g: &'g DecGlobals,
        hooks: Option<B<'py>>,
        allow_imports: Flag<'py>,
        strict: Flag<'py>,
        untagged: bool,
        caches: Caches,
    ) -> PyResult<Self> {
        let resolved = item(g.module.bind(py), "_RESOLVED")?
            .cast_into::<PyDict>()
            .map_err(PyErr::from)?;
        Ok(Self {
            py,
            g,
            hooks,
            allow_imports,
            strict,
            untagged,
            built: Vec::with_capacity(64),
            paths: Paths::new(py),
            placed: PlacedTable::new(),
            bad: None,
            bad_log: Vec::new(),
            resolved,
            memo: HashMap::new(),
            unweakable: HashMap::new(),
            infos: HashMap::new(),
            object_plan: caches.object.clone(),
            caches,
            references: 0,
            unread: HashMap::new(),
            reading: HashSet::new(),
            quick_depth: 0,
        })
    }

    #[inline(always)]
    fn b<T>(&self, value: &'g Py<T>) -> &'g Bound<'py, T> {
        value.bind(self.py)
    }

    fn is_read(&self, error: &DErr) -> bool {
        match error {
            DErr::Read(..) => true,
            DErr::Py(error) => error.is_instance(self.py, self.b(&self.g.read_error)),
        }
    }

    fn message(&self, error: &DErr) -> PyResult<String> {
        match error {
            DErr::Read(message, _) => Ok(message.clone()),
            DErr::Py(error) => err_text(self.py, error),
        }
    }

    pub fn pyerr(&self, error: DErr) -> PyErr {
        match error {
            DErr::Py(error) => error,
            DErr::Read(message, cause) => {
                let made = match self.b(&self.g.read_error).call1((message,)) {
                    Ok(made) => PyErr::from_value(made),
                    Err(error) => return error,
                };
                if let Some(cause) = cause {
                    chain(self.py, &made, cause);
                }
                made
            }
        }
    }

    fn is_mapping(&self, data: &B<'py>) -> PyResult<bool> {
        let kind = tp(data);
        if kind == t_dict() || kind == t_proxy() {
            return Ok(true);
        }
        if kind == t_list()
            || kind == t_tuple()
            || kind == t_str()
            || kind == t_int()
            || kind == t_float()
            || kind == t_bool()
            || kind == t_none()
        {
            return Ok(false);
        }
        data.is_instance(self.b(&self.g.mapping))
    }

    fn is_sequence(&self, data: &B<'py>) -> PyResult<bool> {
        let kind = tp(data);
        if kind == t_list() || kind == t_tuple() || kind == t_str() {
            return Ok(true);
        }
        if kind == t_dict()
            || kind == t_int()
            || kind == t_float()
            || kind == t_bool()
            || kind == t_none()
        {
            return Ok(false);
        }
        data.is_instance(self.b(&self.g.sequence))
    }

    /// ``_is_plain``: a leaf, mapping, or sequence, but not bytes.
    fn is_plain(&self, data: &B<'py>) -> PyResult<bool> {
        let kind = tp(data);
        if kind == t_none()
            || kind == t_str()
            || kind == t_int()
            || kind == t_float()
            || kind == t_bool()
            || kind == t_dict()
            || kind == t_list()
            || kind == t_tuple()
        {
            return Ok(true);
        }
        if is_str(data) || is_int(data) || is_float(data) {
            return Ok(true);
        }
        if unsafe {
            ffi::PyBytes_Check(data.as_ptr()) != 0 || ffi::PyByteArray_Check(data.as_ptr()) != 0
        } {
            return Ok(false);
        }
        Ok(self.is_mapping(data)? || data.is_instance(self.b(&self.g.sequence))?)
    }

    fn contains(&self, data: &B<'py>, key: &Bound<'py, PyString>) -> PyResult<bool> {
        if tp(data) == t_dict() {
            return unsafe { data.cast_unchecked::<PyDict>() }.contains(key);
        }
        data.contains(key)
    }

    fn get(&self, data: &B<'py>, key: &Bound<'py, PyString>) -> PyResult<B<'py>> {
        data.get_item(key)
    }

    /// ``Mapping.get(key)``, ``None`` when absent.
    fn get_or_none(&self, data: &B<'py>, key: &Bound<'py, PyString>) -> PyResult<B<'py>> {
        if tp(data) == t_dict() {
            return Ok(unsafe { data.cast_unchecked::<PyDict>() }
                .get_item(key)?
                .unwrap_or_else(|| self.py.None().into_bound(self.py)));
        }
        data.call_method1(pyo3::intern!(self.py, "get"), (key,))
    }

    /// ``_shown``: a value's repr for a message, elided past a fixed size.
    fn shown(&self, value: &B<'py>) -> PyResult<String> {
        // ``reprlib`` writes these leaves as ``repr`` does until it truncates
        // one, which only happens past the limit, so they skip the Python call.
        let kind = tp(value);
        if (kind == t_none()
            || kind == t_bool()
            || kind == t_int()
            || kind == t_float()
            || kind == t_str())
            && let Ok(shown) = value.repr()
        {
            let shown = shown.to_string_lossy();
            if shown.len() <= self.g.shown_limit {
                return Ok(shown.into_owned());
            }
        }
        text(&self.b(&self.g.shown).call1((value,))?)
    }

    /// ``_shown(dict(data))``, which differs from ``_shown(data)`` for a proxy.
    fn shown_dict(&self, data: &B<'py>) -> PyResult<String> {
        if tp(data) == t_dict() {
            return self.shown(data);
        }
        let copied = PyDict::new(self.py);
        copied.call_method1(pyo3::intern!(self.py, "update"), (data,))?;
        self.shown(copied.as_any())
    }

    /// ``_shown(list(data))``.
    fn shown_list(&self, data: &B<'py>) -> PyResult<String> {
        if tp(data) == t_list() {
            return self.shown(data);
        }
        let mut items = SeqItems::new(data)?;
        let list = PyList::empty(self.py);
        while let Some(each) = items.next_item()? {
            list.append(each)?;
        }
        self.shown(list.as_any())
    }

    fn target_text(&self, target: &B<'py>) -> PyResult<String> {
        text(target)
    }

    fn class_name(&self, target: &B<'py>) -> PyResult<String> {
        if is_class(target) {
            return text(&target.getattr(pyo3::intern!(self.py, "__name__"))?);
        }
        text(target)
    }

    #[inline]
    fn frame(&mut self, at: Loc<'_, 'py>) -> PyResult<u32> {
        self.paths.frame(at)
    }

    fn place(&mut self, frame: u32, value: &B<'py>) -> PyResult<()> {
        self.built.push(value.clone());
        self.placed.set(&self.paths, frame, value)
    }

    fn place_only(&mut self, frame: u32, value: &B<'py>) -> PyResult<()> {
        self.placed.set(&self.paths, frame, value)
    }

    fn replace(&mut self, index: usize, frame: u32, value: &B<'py>) -> PyResult<()> {
        self.built[index] = value.clone();
        self.placed.set(&self.paths, frame, value)
    }

    fn pending(&self) -> B<'py> {
        self.b(&self.g.pending).clone()
    }

    fn invalid(&mut self, raw: &B<'py>, reason: &str, at: Loc<'_, 'py>) -> DR<B<'py>> {
        let py = self.py;
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "raw"), raw)?;
        kwargs.set_item(pyo3::intern!(py, "reason"), reason)?;
        let made = self.b(&self.g.invalid).call((), Some(&kwargs))?;
        let frame = self.frame(at)?;
        let key = self.paths.tuple(py, frame)?.into_any();
        let bad = self.bad.get_or_insert_with(|| PyDict::new(py)).clone();
        if !bad.contains(&key)? {
            self.bad_log.push(key.clone());
        }
        bad.set_item(key, &made)?;
        Ok(made)
    }

    fn bad_len(&self) -> usize {
        self.bad.as_ref().map_or(0, |bad| bad.len())
    }

    fn mark(&self) -> Mark {
        (self.built.len(), self.bad_log.len(), self.placed.mark())
    }

    fn rollback(&mut self, mark: Mark) -> PyResult<()> {
        self.built.truncate(mark.0);
        if let Some(bad) = &self.bad {
            for key in self.bad_log.drain(mark.1..) {
                bad.del_item(key)?;
            }
        }
        self.placed.rollback(mark.2);
        Ok(())
    }

    fn setattr(&self, value: &B<'py>, name: &B<'py>, member: &B<'py>) -> PyResult<()> {
        unsafe {
            if (*tp(value)).tp_setattro.map(|f| f as usize)
                == (*t_object()).tp_setattro.map(|f| f as usize)
            {
                if ffi::PyObject_GenericSetAttr(value.as_ptr(), name.as_ptr(), member.as_ptr()) < 0
                {
                    return Err(PyErr::fetch(self.py));
                }
                return Ok(());
            }
        }
        self.b(&self.g.object_setattr)
            .call1((value, name, member))?;
        Ok(())
    }

    /// ``object.__setattr__(value, name, member)`` for a dataclass field.
    ///
    /// When ``value``'s class is exactly ``kind``, which is unchanged since
    /// ``info.stores`` was read, uses ``object``'s ``__setattr__``, and finds a
    /// plain object slot for the field, the store is that slot's write:
    /// ``PyMember_SetOne`` for ``Py_T_OBJECT_EX`` swaps the pointer.
    fn store(
        &self,
        info: &DcInfo,
        kind: &B<'py>,
        value: &B<'py>,
        slot: usize,
        name: &B<'py>,
        member: &B<'py>,
    ) -> PyResult<()> {
        if tp(value) == kind.as_ptr().cast() {
            let (stamp, offsets) = info.stores.get_or_init(|| store_offsets(kind, info));
            if stamp.holds(kind)
                && let Some(Some(offset)) = offsets.get(slot)
            {
                unsafe {
                    let field = value
                        .as_ptr()
                        .cast::<u8>()
                        .offset(*offset)
                        .cast::<*mut ffi::PyObject>();
                    let old = *field;
                    *field = member.clone().into_ptr();
                    if !old.is_null() {
                        ffi::Py_DECREF(old);
                    }
                }
                return Ok(());
            }
        }
        self.setattr(value, name, member)
    }

    fn allocate(&self, kind: &B<'py>) -> PyResult<B<'py>> {
        let py = self.py;
        // ``kind.__new__(kind)`` is ``object.__new__(kind)`` when a plain
        // ``type`` metaclass finds ``object``'s own ``__new__`` and ``kind``
        // inherits its slot; that call is the slot with no arguments.
        unsafe {
            let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
            let name = pyo3::intern!(py, "__new__").as_ptr();
            let base = &raw mut ffi::PyBaseObject_Type;
            if ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
                && (*class).tp_new.map(|f| f as usize) == (*base).tp_new.map(|f| f as usize)
                && let Some(new) = (*base).tp_new
                && !_PyType_Lookup(class, name).is_null()
                && _PyType_Lookup(class, name) == _PyType_Lookup(base, name)
            {
                let made = new(class, PyTuple::empty(py).as_ptr(), std::ptr::null_mut());
                return Bound::from_owned_ptr_or_err(py, made);
            }
        }
        kind.getattr(pyo3::intern!(py, "__new__"))?.call1((kind,))
    }

    // -------------------------------------------------------------- plans --

    fn plan_for(&mut self, target: &B<'py>, persistent: bool) -> DR<Rc<Plan>> {
        let py = self.py;
        if target.is(self.b(&self.g.object)) {
            return Ok(self.object_plan.clone());
        }
        let key = target.as_ptr() as usize;
        if is_class(target) {
            if let Some((weak, plan)) = self.caches.classes.get(&key)
                && weak
                    .bind(py)
                    .upgrade()
                    .is_some_and(|alive| alive.is(target))
                && plan.fresh(target)
            {
                return Ok(plan.clone());
            }
            let plan = Rc::new(self.compile(target, true)?);
            let mut stale = Vec::new();
            if let Ok(weak) = PyWeakrefReference::new(target) {
                let cache = &mut self.caches;
                if cache.classes.len() >= CLASS_LIMIT {
                    stale.extend(cache.classes.drain().map(|(_, entry)| entry));
                }
                stale.extend(cache.classes.insert(key, (weak.unbind(), plan.clone())));
            }
            drop(stale);
            return Ok(plan);
        }
        if persistent {
            if let Some((weak, plan)) = self.caches.objects.get(&key)
                && weak
                    .bind(py)
                    .upgrade()
                    .is_some_and(|alive| alive.is(target))
                && plan.fresh(target)
            {
                return Ok(plan.clone());
            }
            if let Some((_, plan)) = self.unweakable.get(&key)
                && plan.fresh(target)
            {
                return Ok(plan.clone());
            }
            let plan = Rc::new(self.compile(target, true)?);
            let mut stale = Vec::new();
            let Ok(weak) = PyWeakrefReference::new(target) else {
                // Without a weak reference the plan cannot outlast the call
                // safely, but it still serves every use within it.
                self.unweakable.insert(key, (target.clone(), plan.clone()));
                return Ok(plan);
            };
            {
                let cache = &mut self.caches;
                if cache.objects.len() >= OBJECT_LIMIT {
                    stale.extend(cache.objects.drain().map(|(_, entry)| entry));
                }
                stale.extend(cache.objects.insert(key, (weak.unbind(), plan.clone())));
            }
            drop(stale);
            return Ok(plan);
        }
        if let Some((_, plan)) = self.memo.get(&key)
            && plan.fresh(target)
        {
            return Ok(plan.clone());
        }
        let plan = Rc::new(self.compile(target, false)?);
        self.memo.insert(key, (target.clone(), plan.clone()));
        Ok(plan)
    }

    fn compile(&self, target: &B<'py>, persistent: bool) -> PyResult<Plan> {
        let py = self.py;
        let g = self.g;
        // Each class is stamped before any fact is read from it, so a change
        // made while compiling leaves the plan stale rather than wrong.
        let own = if is_class(target) {
            Stamp::of(target)
        } else {
            None
        };
        let mut deps = Vec::new();
        let resolved = self.b(&g.resolve_alias).call1((target,))?;
        depend(&mut deps, target, &resolved);
        let raw_origin = self.b(&g.get_origin).call1((&resolved,))?;
        let origin_is_tuple = raw_origin.is(self.b(&g.tuple_type));
        let origin = if raw_origin.is_none() {
            resolved.clone()
        } else {
            raw_origin
        };
        depend(&mut deps, target, &origin);
        let decoder = self.b(&g.decoder_for).call1((&resolved,))?;
        let kind = g
            .decoders
            .iter()
            .find(|(each, _)| each.bind(py).is(&decoder))
            .map(|(_, kind)| *kind)
            .ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "unknown decoder {}",
                    repr(&decoder).unwrap_or_default()
                ))
            })?;
        let args = self
            .b(&g.get_args)
            .call1((&resolved,))?
            .cast_into::<PyTuple>()
            .map_err(PyErr::from)?;
        if kind == Kind::Union {
            for member in args.iter() {
                depend(&mut deps, target, &member);
            }
        }
        let fits = self.compile_fits(&resolved, &origin, target)?;
        let union = if kind == Kind::Union {
            Some(self.compile_union(&args)?)
        } else {
            None
        };
        let leaf = match kind {
            Kind::Leaf => Some(decoder.clone().unbind()),
            Kind::Literal => {
                let mut enum_option = false;
                for option in args.iter() {
                    if option.is_instance(self.b(&g.enum_type))? {
                        enum_option = true;
                    }
                }
                enum_option.then(|| decoder.clone().unbind())
            }
            _ => None,
        };
        let import_name = if is_class(&resolved) {
            Some(self.b(&g.import_name).call1((&resolved,))?.unbind())
        } else {
            None
        };
        let is_dataclass =
            is_class(&resolved) && self.b(&g.is_dataclass).call1((&resolved,))?.is_truthy()?;
        Ok(Plan {
            kind,
            is_object: resolved.is(self.b(&g.object)),
            resolved: if resolved.is(target) {
                None
            } else {
                Some(resolved.clone().unbind())
            },
            args: args.unbind(),
            origin_is_tuple,
            origin_is_type: origin.is(self.b(&g.type_type)),
            fits,
            persistent,
            union,
            leaf,
            import_name,
            is_dataclass,
            info: OnceCell::new(),
            own,
            deps,
        })
    }

    fn compile_fits(&self, resolved: &B<'py>, origin: &B<'py>, target: &B<'py>) -> PyResult<Fits> {
        let g = self.g;
        if origin.is(self.b(&g.union_type)) || origin.is(self.b(&g.typing_union)) {
            return Ok(Fits::Union);
        }
        if origin.is(self.b(&g.literal)) {
            return Ok(Fits::Literal);
        }
        if origin.is(self.b(&g.type_type)) {
            return Ok(Fits::Type);
        }
        if !is_class(origin)
            || origin.is(self.b(&g.object))
            || self.b(&g.is_any).call1((origin,))?.is_truthy()?
        {
            return Ok(Fits::Always);
        }
        if self.b(&g.is_typeddict).call1((origin,))?.is_truthy()? {
            return Ok(Fits::Python);
        }
        if self.b(&g.is_protocol).call1((origin,))?.is_truthy()? {
            return Ok(Fits::Protocol(origin.clone().unbind()));
        }
        if origin.is(resolved) && resolved.is(target) {
            return Ok(Fits::IsInstance(None));
        }
        Ok(Fits::IsInstance(Some(origin.clone().unbind())))
    }

    fn compile_union(&self, members: &Bound<'py, PyTuple>) -> PyResult<UnionInfo> {
        let g = self.g;
        let mut import_names = Vec::with_capacity(members.len());
        for member in members.iter() {
            import_names.push(if is_class(&member) {
                Some(self.b(&g.import_name).call1((&member,))?.unbind())
            } else {
                None
            });
        }
        Ok(UnionInfo {
            import_names,
            orders: Default::default(),
        })
    }

    /// ``_union_candidates(data, members)`` as indices into ``members``.
    fn union_order(&self, data: &B<'py>, members: &Bound<'py, PyTuple>) -> PyResult<Vec<usize>> {
        let picked = self.b(&self.g.union_candidates).call1((data, members))?;
        let mut order = Vec::with_capacity(members.len());
        for member in picked.try_iter()? {
            let member = member?;
            order.push(
                members
                    .iter()
                    .position(|each| each.is(&member))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("_union_candidates returned a non-member")
                    })?,
            );
        }
        Ok(order)
    }

    fn dc_info(&mut self, target: &B<'py>, plan: &Rc<Plan>) -> DR<Rc<DcInfo>> {
        if let Some(info) = plan.info.get() {
            return Ok(info.clone());
        }
        let key = target.as_ptr() as usize;
        if let Some((owner, info)) = self.infos.get(&key)
            && Rc::ptr_eq(owner, plan)
        {
            return Ok(info.clone());
        }
        let py = self.py;
        let g = self.g;
        let types = self.b(&g.field_types).call1((target,))?;
        let index = PyDict::new(py);
        let mut targets = Vec::new();
        let mut names = Some(Vec::new());
        let mut items = MapItems::new(&types)?;
        while let Some((name, hint)) = items.next_item()? {
            index.set_item(&name, targets.len())?;
            targets.push(hint.unbind());
            names = names.and_then(|mut names| {
                let text = (tp(&name) == t_str()).then(|| utf8(&name)).flatten()?;
                names.push(Box::<[u8]>::from(text));
                Some(names)
            });
        }
        // A repeated name keeps only its last slot in ``index``.
        let names = names.filter(|names| names.len() == index.len());
        let slots = names
            .iter()
            .flatten()
            .enumerate()
            .map(|(slot, name)| (name.clone(), slot))
            .collect();
        let mut sorted: Vec<usize> = (0..names.as_ref().map_or(0, Vec::len)).collect();
        if let Some(names) = &names {
            sorted.sort_by(|a, b| names[*a].cmp(&names[*b]));
        }
        let persistent = self.b(&g.field_types_cache).contains(target)?;
        let plans = (0..targets.len()).map(|_| RefCell::new(None)).collect();
        let info = Rc::new(DcInfo {
            index: index.unbind(),
            names,
            slots,
            sorted,
            targets,
            plans,
            persistent,
            stores: OnceCell::new(),
        });
        if persistent && plan.persistent {
            let _ = plan.info.set(info.clone());
        } else {
            self.infos.insert(key, (plan.clone(), info.clone()));
        }
        Ok(info)
    }

    fn field_plan(&mut self, info: &DcInfo, slot: usize) -> DR<Rc<Plan>> {
        let target = info.targets[slot].bind(self.py).clone();
        if let Some(plan) = info.plans[slot].borrow().as_ref()
            && plan.fresh(&target)
        {
            return Ok(plan.clone());
        }
        let plan = self.plan_for(&target, info.persistent)?;
        let stale = info.plans[slot].replace(Some(plan.clone()));
        drop(stale);
        Ok(plan)
    }

    // ------------------------------------------------------------ resolve --

    fn resolve(&mut self, path: &B<'py>) -> DR<B<'py>> {
        if !self.allow_imports.get()? {
            return fail(format!("{} needs allow_imports=True", repr(path)?));
        }
        if let Some(found) = self.resolved.get_item(path)? {
            return Ok(found);
        }
        let resolve_import = item(self.g.module.bind(self.py), "resolve_import")?;
        match resolve_import.call1((path,)) {
            Ok(found) => {
                self.resolved.set_item(path, &found)?;
                Ok(found)
            }
            Err(error)
                if error.is_instance_of::<PyAttributeError>(self.py)
                    || error.is_instance_of::<PyImportError>(self.py) =>
            {
                let message = format!(
                    "cannot import {}: {}",
                    repr(path)?,
                    err_text(self.py, &error)?
                );
                fail_from(message, error)
            }
            Err(error) => Err(error.into()),
        }
    }

    // ------------------------------------------------------------- decode --

    pub fn run(&mut self, data: &B<'py>, target: &B<'py>) -> DR<B<'py>> {
        // Built-in union and generic-alias objects do not change in place.
        // Their plans can be reused across top-level calls, just as field
        // plans are, without caching arbitrary mutable typing constructs.
        let kind = target.get_type();
        let persistent =
            kind.is(self.b(&self.g.union_type)) || kind.is(self.b(&self.g.generic_alias));
        let plan = self.plan_for(target, persistent)?;
        if let Some(value) = self.quick(data, target, &plan, Loc::At(ROOT))? {
            return Ok(value);
        }
        self.walk(data, target, &plan, Loc::At(ROOT))
    }

    // Containers wait on a stack of frames rather than the call stack, as in
    // Python, so a tree of any depth reads natively.
    /// ``_decode``: ``data`` as ``target``, one frame per container still being read.
    fn walk(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<B<'py>> {
        let root = match self.start(data, target, plan, at)? {
            Start::Value(value) => return Ok(value),
            Start::Frame(frame) => frame,
        };
        let mut stack = Vec::with_capacity(16);
        stack.push(root);
        loop {
            let top = stack
                .last_mut()
                .expect("the walk ends when its stack empties");
            let (failed, error) = match self.step(top) {
                Ok(Some(child)) => {
                    stack.push(child);
                    continue;
                }
                Ok(None) => {
                    let frame = stack.pop().expect("a frame was just stepped");
                    match self.finished(frame) {
                        (failed, Err(error)) => (failed, error),
                        (done, Ok(value)) => {
                            self.close(done.reading);
                            let Some(parent) = stack.last_mut() else {
                                return Ok(value);
                            };
                            match self.take(parent, value) {
                                Ok(()) => continue,
                                Err(error) => (stack.pop().expect("checked above").failed(), error),
                            }
                        }
                    }
                }
                Err(error) => (
                    stack.pop().expect("a frame was just stepped").failed(),
                    error,
                ),
            };
            self.unwind(&mut stack, failed, error)?;
        }
    }

    /// ``_finished``: a frame's container built, then converted and checked.
    // Boxed on purpose; see ``Start``.
    #[allow(clippy::boxed_local)]
    fn finished(&mut self, frame: Box<Frame<'py>>) -> (Failed<'py>, DR<B<'py>>) {
        let Frame {
            at,
            place,
            raw,
            check,
            then,
            reading,
            body,
        } = *frame;
        let result = match self.finish(at, body) {
            Ok(value) => self.checked(value, then, check),
            Err(error) => Err(error),
        };
        (
            Failed {
                place,
                raw,
                reading,
            },
            result,
        )
    }

    fn checked(
        &mut self,
        value: B<'py>,
        then: Option<(Bound<'py, PyDict>, B<'py>)>,
        check: Option<(B<'py>, Rc<Plan>)>,
    ) -> DR<B<'py>> {
        let value = match then {
            Some((inner, proxy)) => {
                inner.call_method1(pyo3::intern!(self.py, "update"), (value,))?;
                proxy
            }
            None => value,
        };
        if let Some((target, plan)) = check
            && !self.fits(&value, &target, &plan)?
        {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(&value)?,
                self.target_text(&target)?
            ));
        }
        Ok(value)
    }

    /// ``_unwind``: a failed frame's error handed to the nearest frame below
    /// that recovers, else returned.
    // Boxed on purpose; see ``Start``.
    #[allow(clippy::vec_box)]
    fn unwind(
        &mut self,
        stack: &mut Vec<Box<Frame<'py>>>,
        mut failed: Failed<'py>,
        mut error: DErr,
    ) -> DR<()> {
        loop {
            self.close(failed.reading);
            if !self.is_read(&error) {
                return Err(error);
            }
            let Some(parent) = stack.last_mut() else {
                return Err(error);
            };
            match self.recover(parent, error, &failed) {
                Ok(()) => return Ok(()),
                Err(again) => {
                    error = again;
                    failed = stack.pop().expect("checked above").failed();
                }
            }
        }
    }

    /// ``_close``: forget that a frame's input container is being read.
    #[inline]
    fn close(&mut self, reading: Option<usize>) {
        if let Some(identity) = reading {
            self.reading.remove(&identity);
        }
    }

    // A union rereads its data through each member, so only the member's frame counts.
    /// ``_start``: ``data`` as ``target``: its value, or a frame that reads its parts.
    fn start(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let mut started = self.start_node(data, target, plan, at)?;
        if let Start::Frame(frame) = &mut started {
            // A tag's frame reads its payload; a parent records the envelope.
            frame.raw = data.clone();
            if !matches!(frame.body, Body::Union { .. }) {
                let identity = data.as_ptr() as usize;
                if !self.reading.insert(identity) {
                    return fail("plain data cannot hold a cycle".to_owned());
                }
                frame.reading = Some(identity);
            }
        }
        Ok(started)
    }

    /// ``_start_part``: a container's part; one that does not read becomes its ``Invalid``.
    #[inline]
    fn start_part(
        &mut self,
        raw: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        match self.quick(raw, target, plan, at) {
            Ok(Some(value)) => return Ok(Start::Value(value)),
            Ok(None) => {}
            Err(error) if self.is_read(&error) => {
                let reason = self.message(&error)?;
                return Ok(Start::Value(self.invalid(raw, &reason, at)?));
            }
            Err(error) => return Err(error),
        }
        match self.start(raw, target, plan, at) {
            Err(error) if self.is_read(&error) => {
                let reason = self.message(&error)?;
                Ok(Start::Value(self.invalid(raw, &reason, at)?))
            }
            started => started,
        }
    }

    /// ``_start_node``: one node by its target and tags, before cycle bookkeeping.
    fn start_node(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let kind = tp(data);
        // A leaf of exactly its declared kind reads as itself under every decoder.
        if is_leaf_kind(kind) && target.as_ptr() == kind.cast::<ffi::PyObject>() {
            return Ok(Start::Value(data.clone()));
        }
        if plan.is_object && target.is(self.b(&self.g.object)) {
            if is_leaf_kind(kind) {
                return Ok(Start::Value(data.clone()));
            }
            if !self.untagged {
                if kind == t_dict() {
                    return self.decode_tagged(unsafe { data.cast_unchecked::<PyDict>() }, at);
                }
                if kind == t_list() {
                    let object = self.object_plan.clone();
                    return self.decode_items(data, target, &object, at);
                }
            }
        }
        self.start_typed(data, target, plan, at)
    }

    fn start_typed(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        let target = resolved_of(py, plan, target);
        let target = &target;
        if !self.is_plain(data)? {
            if self.fits(data, target, plan)? {
                return Ok(Start::Value(data.clone()));
            }
            if target.is(self.b(&self.g.float_type))
                && (data.is_instance(self.b(&self.g.real))?
                    || data.is_instance(self.b(&self.g.decimal))?)
            {
                return Ok(Start::Value(self.b(&self.g.float_type).call1((data,))?));
            }
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        if plan.kind == Kind::Union || !self.is_mapping(data)? {
            return self.run_kind(data, target, plan, at);
        }
        // A dataclass record read straight into its fields, its tag and
        // references checked in the same pass, wherever ``decode_object``
        // would be the decoder: ``py/object`` or no tag at all.
        if plan.kind == Kind::Object
            && plan.is_dataclass
            && is_class(target)
            && let Some(started) = self.decode_exact_object(data, target, plan, at)?
        {
            return Ok(started);
        }
        let value = if self.contains(data, pyo3::intern!(py, "py/ref"))? {
            self.decode_path_reference(data)?
        } else if self.contains(data, pyo3::intern!(py, "py/id"))? {
            self.decode_reference(data)?
        } else {
            let tag = self.tag_of(data)?;
            let Some(tag) = tag else {
                return self.run_kind(data, target, plan, at);
            };
            if (plan.kind == Kind::Dict && tag != Tag::MappingProxy)
                || (tag == Tag::Object && plan.kind == Kind::Any && !self.allow_imports.get()?)
            {
                return self.run_kind(data, target, plan, at);
            }
            if tag != Tag::Object && data.len()? != 1 {
                return fail(format!(
                    "invalid {} envelope: {}",
                    tag_text(tag),
                    self.shown_dict(data)?
                ));
            }
            let payload = if tag == Tag::Object {
                data.clone()
            } else {
                self.get(data, &PyString::intern(py, tag_text(tag)))?
            };
            match self.run_tag(tag, &payload, target, plan, at)? {
                Start::Frame(mut frame) => {
                    frame.check = Some((target.clone(), plan.clone()));
                    return Ok(Start::Frame(frame));
                }
                Start::Value(value) => value,
            }
        };
        if !self.fits(&value, target, plan)? {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(&value)?,
                self.target_text(target)?
            ));
        }
        Ok(Start::Value(value))
    }

    /// ``_decode_tagged``: a dict read as ``object``, by its tag, else as a plain dict.
    fn decode_tagged(&mut self, data: &Bound<'py, PyDict>, at: Loc<'_, 'py>) -> DR<Start<'py>> {
        let py = self.py;
        let any = data.as_any();
        if data.contains(pyo3::intern!(py, "py/ref"))? {
            return Ok(Start::Value(self.decode_path_reference(any)?));
        }
        if data.contains(pyo3::intern!(py, "py/id"))? {
            return Ok(Start::Value(self.decode_reference(any)?));
        }
        if data.contains(pyo3::intern!(py, "py/object"))? {
            if !self.allow_imports.get()? {
                let object = self.object_plan.clone();
                return self.decode_entries(any, &object, &object, at);
            }
            return self.decode_stored(any, at);
        }
        if data.len() == 1 {
            let mut items = DictItems::new(data)?;
            if let Some((key, payload)) = items.next_item()?
                && is_str(&key)
                && let Some(tag) = utf8(&key).and_then(tag_named)
            {
                let object = self.object_plan.clone();
                let target = self.b(&self.g.object).clone();
                return self.run_tag(tag, &payload, &target, &object, at);
            }
        }
        match self.tag_of(any)? {
            None => {
                let object = self.object_plan.clone();
                self.decode_entries(any, &object, &object, at)
            }
            Some(tag) => fail(format!(
                "invalid {} envelope: {}",
                tag_text(tag),
                self.shown(any)?
            )),
        }
    }

    fn run_kind(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let value = match plan.kind {
            Kind::Any => return self.decode_any(data, at),
            Kind::None => {
                if !data.is_none() {
                    return fail(format!("cannot read {} as None", self.shown(data)?));
                }
                data.clone()
            }
            Kind::Bool => self.decode_bool(data)?,
            Kind::Int => self.decode_int(data)?,
            Kind::Float => self.decode_float(data)?,
            Kind::Complex => return self.decode_complex(data, at),
            Kind::Str => {
                if !is_str(data) {
                    return fail(format!("cannot read {} as str", self.shown(data)?));
                }
                data.clone()
            }
            Kind::Leaf => self.leaf(plan.leaf.as_ref(), data, target)?,
            Kind::Literal => match &plan.leaf {
                Some(_) => self.leaf(plan.leaf.as_ref(), data, target)?,
                None => self.decode_literal(data, target, plan)?,
            },
            Kind::List => return self.decode_list(data, target, plan, at),
            Kind::Tuple => return self.decode_tuple(data, target, plan, at),
            Kind::Set => return self.decode_set(data, target, plan, at),
            Kind::FrozenSet => return self.decode_frozenset(data, target, plan, at),
            Kind::Dict => return self.decode_dict(data, target, plan, at),
            Kind::MappingProxy => return self.decode_mappingproxy(data, target, plan, at),
            Kind::Type => self.decode_type(data, plan)?,
            Kind::Union => return self.decode_union(data, target, plan, at),
            Kind::Object => return self.decode_object(data, target, plan, at),
            Kind::TypedDict => return self.decode_typeddict(data, target, at),
        };
        Ok(Start::Value(value))
    }

    fn run_tag(
        &mut self,
        tag: Tag,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let value = match tag {
            Tag::Float => self.decode_float(data)?,
            Tag::Complex => return self.decode_complex(data, at),
            Tag::B64 | Tag::Path | Tag::Uuid | Tag::Datetime => {
                let func = self
                    .g
                    .tag_leaves
                    .iter()
                    .find(|(each, _)| *each == tag)
                    .map(|(_, func)| func);
                self.leaf(func, data, target)?
            }
            Tag::Tuple => return self.decode_tuple(data, target, plan, at),
            Tag::Set => return self.decode_set(data, target, plan, at),
            Tag::FrozenSet => return self.decode_frozenset(data, target, plan, at),
            Tag::MappingProxy => return self.decode_mappingproxy(data, target, plan, at),
            Tag::Type => self.decode_type(data, plan)?,
            Tag::Function => self.decode_function(data)?,
            Tag::Object => return self.decode_object(data, target, plan, at),
            Tag::Hook => return self.decode_hooked(data, at),
            Tag::Reduce => return self.decode_reduce(data, target, plan, at),
        };
        Ok(Start::Value(value))
    }

    // -------------------------------------------------------------- quick --

    // The frames exist so a tree of any depth reads with a flat Rust stack.
    // Most trees are shallow and regular, so ``quick`` reads the common nodes
    // by recursion instead, taking each step ``start`` and the frames would
    // take, in the same order. It answers ``None`` before any effect for a
    // node it does not read that way; past that point every part reads as
    // ``_start_part`` would, through ``part``.
    /// A node read at once, recursively, else ``None`` for the frames.
    fn quick(
        &mut self,
        raw: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Option<B<'py>>> {
        let kind = tp(raw);
        let object = plan.is_object && target.is(self.b(&self.g.object));
        if is_leaf_kind(kind) {
            let own = target.as_ptr() == kind.cast::<ffi::PyObject>();
            return Ok((own || object).then(|| raw.clone()));
        }
        // ``start`` refuses a container already being read only after
        // starting it; the frames reproduce that.
        if self.quick_depth >= QUICK_DEPTH || self.reading.contains(&(raw.as_ptr() as usize)) {
            return Ok(None);
        }
        if kind == t_dict() {
            let dict = unsafe { raw.cast_unchecked::<PyDict>() }.clone();
            if object {
                if self.untagged {
                    return Ok(None);
                }
                let any = self.object_plan.clone();
                return match dict_shape(&dict) {
                    DictShape::Plain => {
                        let pair = (target.clone(), any);
                        self.quick_entries(raw, &dict, pair.clone(), pair, at)
                    }
                    DictShape::Tuple(payload) => {
                        self.quick_tuple(raw, &payload, target, &any, false, at)
                    }
                    DictShape::Other => Ok(None),
                };
            }
            if plan.resolved.is_some() {
                return Ok(None);
            }
            if plan.kind == Kind::Object && plan.is_dataclass && is_class(target) {
                return self.quick_record(raw, &dict, target, plan, at);
            }
            return match (plan.kind, dict_shape(&dict)) {
                (Kind::Dict, DictShape::Plain) => {
                    let args = plan.args.bind(self.py).clone();
                    let (key, value) = if args.len() == 2 {
                        let (key, value) = (args.get_item(0)?, args.get_item(1)?);
                        let key_plan = self.plan_for(&key, plan.persistent)?;
                        let value_plan = self.plan_for(&value, plan.persistent)?;
                        ((key, key_plan), (value, value_plan))
                    } else {
                        let any = (self.b(&self.g.object).clone(), self.object_plan.clone());
                        (any.clone(), any)
                    };
                    self.quick_entries(raw, &dict, key, value, at)
                }
                (Kind::Tuple, DictShape::Tuple(payload)) => {
                    self.quick_tuple(raw, &payload, target, plan, true, at)
                }
                _ => Ok(None),
            };
        }
        if kind == t_list() {
            if object {
                if self.untagged {
                    return Ok(None);
                }
                let any = self.object_plan.clone();
                return self.quick_items(raw, target, &any, at);
            }
            if plan.resolved.is_some() {
                return Ok(None);
            }
            return match plan.kind {
                Kind::List => {
                    let (item, item_plan) = self.element(plan)?;
                    self.quick_items(raw, &item, &item_plan, at)
                }
                Kind::Tuple => self.quick_tuple(raw, raw, target, plan, false, at),
                _ => Ok(None),
            };
        }
        Ok(None)
    }

    /// ``_start_part``, run to its value: a part that does not read is
    /// recorded and becomes its ``Invalid``.
    fn part(
        &mut self,
        raw: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<B<'py>> {
        let read = match self.quick(raw, target, plan, at) {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => self.walk(raw, target, plan, at),
            Err(error) => Err(error),
        };
        match read {
            Err(error) if self.is_read(&error) => {
                let reason = self.message(&error)?;
                self.invalid(raw, &reason, at)
            }
            read => read,
        }
    }

    #[inline]
    fn enter(&mut self, raw: &B<'py>) {
        self.reading.insert(raw.as_ptr() as usize);
        self.quick_depth += 1;
    }

    #[inline]
    fn leave(&mut self, raw: &B<'py>) {
        self.reading.remove(&(raw.as_ptr() as usize));
        self.quick_depth -= 1;
    }

    /// ``decode_exact_object`` and the ``_Object`` frame it starts, at once.
    fn quick_record(
        &mut self,
        raw: &B<'py>,
        dict: &Bound<'py, PyDict>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Option<B<'py>>> {
        let py = self.py;
        if !matches!(self.allow_imports, Flag::Known(_)) {
            return Ok(None);
        }
        let info = self.dc_info(target, plan)?;
        if info.names.is_none() {
            return Ok(None);
        }
        let own = plan
            .import_name
            .as_ref()
            .expect("classes have import names")
            .bind(py);
        let Some(own) = (tp(own) == t_str()).then(|| utf8(own)).flatten() else {
            return Ok(None);
        };
        let mut entries = Vec::with_capacity(dict.len());
        let mut stored = false;
        let mut cursor = 0;
        for (key, value) in dict_entries(dict) {
            let Some(text) = (tp(&key) == t_str()).then(|| utf8(&key)).flatten() else {
                return Ok(None);
            };
            if text == b"py/object" {
                if (tp(&value) == t_str()).then(|| utf8(&value)).flatten() != Some(own) {
                    return Ok(None);
                }
                stored = true;
                continue;
            }
            match info.slot_of(text, &mut cursor) {
                Some(slot) => entries.push((key, value, slot)),
                None => return Ok(None),
            }
        }
        let position = self.built.len();
        let frame = self.frame(at)?;
        let reserved = if stored {
            self.allocate(target)?
        } else {
            self.pending()
        };
        self.place(frame, &reserved)?;
        let before = self.bad_len();
        self.enter(raw);
        let built = match self.quick_fields(&info, frame, entries) {
            Ok(members) => self.build_object(
                frame, &info, target, stored, reserved, position, before, members,
            ),
            Err(error) => Err(error),
        };
        self.leave(raw);
        built.map(Some)
    }

    /// The ``_Object`` frame's ``step``: each field read as its declared type.
    #[allow(clippy::type_complexity)]
    fn quick_fields(
        &mut self,
        info: &DcInfo,
        frame: u32,
        entries: Vec<(B<'py>, B<'py>, usize)>,
    ) -> DR<Vec<(B<'py>, B<'py>, usize)>> {
        let object = self.b(&self.g.object).clone();
        let mut members = Vec::with_capacity(entries.len());
        for (name, raw, slot) in entries {
            let target = info.targets[slot].bind(self.py).clone();
            if is_leaf_kind(tp(&raw)) && target.is(&object) {
                members.push((name, raw, slot));
                continue;
            }
            let plan = self.field_plan(info, slot)?;
            let member = self.part(&raw, &target, &plan, Loc::Child(frame, StepRef::Key(&name)))?;
            members.push((name, member, slot));
        }
        Ok(members)
    }

    /// ``decode_items`` and its list's frame, at once.
    fn quick_items(
        &mut self,
        raw: &B<'py>,
        item: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Option<B<'py>>> {
        let py = self.py;
        let frame = self.frame(at)?;
        if item.is(self.b(&self.g.object)) && self.all_leaves(raw)? {
            let leaves =
                unsafe { Bound::from_owned_ptr_or_err(py, ffi::PySequence_List(raw.as_ptr()))? };
            self.place(frame, &leaves)?;
            return Ok(Some(leaves));
        }
        let result = PyList::empty(py);
        self.place(frame, result.as_any())?;
        let mut built = Built::List(result.clone());
        let types = Types::Repeat(item.clone(), plan.clone());
        self.enter(raw);
        let read = self.quick_seq(raw, frame, &types, &mut built);
        self.leave(raw);
        read.map(|()| Some(result.into_any()))
    }

    /// ``decode_tuple`` and its frame, at once, checked against ``target``
    /// when a tag decoded it.
    fn quick_tuple(
        &mut self,
        raw: &B<'py>,
        payload: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        check: bool,
        at: Loc<'_, 'py>,
    ) -> DR<Option<B<'py>>> {
        let py = self.py;
        let kind = tp(payload);
        if kind != t_list() && kind != t_tuple() {
            return Ok(None);
        }
        let args = plan.args.bind(py).clone();
        let types = if !plan.origin_is_tuple {
            let (item, item_plan) = self.element(plan)?;
            Types::Repeat(item, item_plan)
        } else if args.len() == 2 && args.get_item(1)?.is(py.Ellipsis()) {
            let first = args.get_item(0)?;
            let first_plan = self.plan_for(&first, plan.persistent)?;
            Types::Repeat(first, first_plan)
        } else {
            if args.len() != payload.len()? {
                return Ok(None);
            }
            Types::Fixed(args, plan.persistent)
        };
        let frame = self.frame(at)?;
        let mut built = Built::Tuple(Vec::new());
        self.enter(raw);
        let read = self.quick_seq(payload, frame, &types, &mut built);
        self.leave(raw);
        read?;
        let Built::Tuple(values) = built else {
            unreachable!("a tuple's items");
        };
        let value = PyTuple::new(py, values)?.into_any();
        if check && !self.fits(&value, target, plan)? {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(&value)?,
                self.target_text(target)?
            ));
        }
        Ok(Some(value))
    }

    /// The ``_Items`` frame's ``step``, run to the end.
    fn quick_seq(
        &mut self,
        data: &B<'py>,
        frame: u32,
        types: &Types<'py>,
        built: &mut Built<'py>,
    ) -> DR<()> {
        let object = self.b(&self.g.object).clone();
        let mut raws = SeqItems::new(data)?;
        let mut index = 0;
        while let Some(raw) = raws.next_item()? {
            let here = index;
            let fixed: (B<'py>, Rc<Plan>);
            let (target, plan) = match types {
                Types::Repeat(target, plan) => (target, plan),
                Types::Fixed(types, persistent) => {
                    if here >= types.len() {
                        break;
                    }
                    let target = types.get_item(here)?;
                    let plan = self.plan_for(&target, *persistent)?;
                    fixed = (target, plan);
                    (&fixed.0, &fixed.1)
                }
            };
            index += 1;
            if is_leaf_kind(tp(&raw)) && target.is(&object) {
                self.put(built, raw)?;
                continue;
            }
            let value = self.part(&raw, target, plan, Loc::Child(frame, StepRef::Index(here)))?;
            self.put(built, value)?;
        }
        Ok(())
    }

    /// ``decode_entries_typed`` and its frame, at once.
    fn quick_entries(
        &mut self,
        raw: &B<'py>,
        dict: &Bound<'py, PyDict>,
        key: (B<'py>, Rc<Plan>),
        value: (B<'py>, Rc<Plan>),
        at: Loc<'_, 'py>,
    ) -> DR<Option<B<'py>>> {
        let plain_keys = key.1.is_object || key.0.is(self.b(&self.g.str_type));
        if !plain_keys {
            return Ok(None);
        }
        let frame = self.frame(at)?;
        let result = PyDict::new(self.py);
        self.place(frame, result.as_any())?;
        self.enter(raw);
        let read = self.quick_pairs(dict, frame, &key, &value, &result);
        self.leave(raw);
        read.map(|()| Some(result.into_any()))
    }

    /// The ``_Entries`` frame's ``step``, run to the end.
    fn quick_pairs(
        &mut self,
        dict: &Bound<'py, PyDict>,
        frame: u32,
        key: &(B<'py>, Rc<Plan>),
        value: &(B<'py>, Rc<Plan>),
        result: &Bound<'py, PyDict>,
    ) -> DR<()> {
        let object = self.b(&self.g.object).clone();
        let mut items = DictItems::new(dict)?;
        while let Some((name, raw)) = items.next_item()? {
            let here = Loc::Child(frame, StepRef::Key(&name));
            let decoded = if tp(&name) == t_str() && !str_starts(&name, "json://")? {
                name.clone()
            } else {
                match self.decode_key(&name, &key.0, &key.1, here) {
                    Ok(decoded) => decoded,
                    Err(error) if self.is_read(&error) => {
                        let reason = self.message(&error)?;
                        self.invalid(&name, &reason, here)?;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            };
            if is_leaf_kind(tp(&raw)) && value.0.is(&object) {
                result.set_item(decoded, raw)?;
                continue;
            }
            let member = self.part(&raw, &value.0, &value.1, here)?;
            result.set_item(decoded, member)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------- frames --

    /// ``_Frame.step``: parts read until one needs a frame of its own.
    fn step(&mut self, frame: &mut Frame<'py>) -> DR<Option<Box<Frame<'py>>>> {
        let py = self.py;
        let at = frame.at;
        match &mut frame.body {
            Body::Items {
                raws,
                index,
                types,
                built,
            } => {
                while let Some(raw) = raws.next_item()? {
                    let here = *index;
                    let fixed: (B<'py>, Rc<Plan>);
                    let (target, plan) = match types {
                        Types::Repeat(target, plan) => (&*target, &*plan),
                        Types::Fixed(types, persistent) => {
                            if here >= types.len() {
                                break;
                            }
                            let target = types.get_item(here)?;
                            let plan = self.plan_for(&target, *persistent)?;
                            fixed = (target, plan);
                            (&fixed.0, &fixed.1)
                        }
                    };
                    *index += 1;
                    if is_leaf_kind(tp(&raw)) && target.is(self.b(&self.g.object)) {
                        self.put(built, raw)?;
                        continue;
                    }
                    let here = Loc::Child(at, StepRef::Index(here));
                    match self.start_part(&raw, target, plan, here)? {
                        Start::Value(value) => self.put(built, value)?,
                        Start::Frame(child) => return Ok(Some(child)),
                    }
                }
                Ok(None)
            }
            Body::Entries {
                items,
                key,
                value,
                plain_keys,
                result,
                pending_key,
            } => {
                let object = self.b(&self.g.object);
                while let Some((name, raw)) = items.next_item()? {
                    let here = Loc::Child(at, StepRef::Key(&name));
                    let decoded =
                        if *plain_keys && tp(&name) == t_str() && !str_starts(&name, "json://")? {
                            name.clone()
                        } else {
                            match self.decode_key(&name, &key.0, &key.1, here) {
                                Ok(decoded) => decoded,
                                Err(error) if self.is_read(&error) => {
                                    let reason = self.message(&error)?;
                                    self.invalid(&name, &reason, here)?;
                                    continue;
                                }
                                Err(error) => return Err(error),
                            }
                        };
                    if is_leaf_kind(tp(&raw)) && value.0.is(object) {
                        result.set_item(decoded, raw)?;
                        continue;
                    }
                    match self.start_part(&raw, &value.0, &value.1, here)? {
                        Start::Value(member) => result.set_item(decoded, member)?,
                        Start::Frame(child) => {
                            *pending_key = Some(decoded);
                            return Ok(Some(child));
                        }
                    }
                }
                Ok(None)
            }
            Body::Fields {
                items,
                hints,
                members,
                pending_name,
            } => {
                let object = self.b(&self.g.object);
                while let Some((name, raw)) = items.next_item()? {
                    if !hints.contains(&name)? {
                        continue;
                    }
                    let hint = hints.get_item(&name)?;
                    if is_leaf_kind(tp(&raw)) && hint.is(object) {
                        members.set_item(name, raw)?;
                        continue;
                    }
                    let plan = self.plan_for(&hint, false)?;
                    let here = Loc::Child(at, StepRef::Key(&name));
                    match self.start_part(&raw, &hint, &plan, here)? {
                        Start::Value(member) => members.set_item(name, member)?,
                        Start::Frame(child) => {
                            *pending_name = Some(name);
                            return Ok(Some(child));
                        }
                    }
                }
                Ok(None)
            }
            Body::Object {
                entries,
                info,
                members,
                pending,
                ..
            } => {
                let object = self.b(&self.g.object);
                for (name, raw, slot) in entries.by_ref() {
                    let target = info.targets[slot].bind(py).clone();
                    if is_leaf_kind(tp(&raw)) && target.is(object) {
                        members.push((name, raw, slot));
                        continue;
                    }
                    let plan = self.field_plan(info, slot)?;
                    let here = Loc::Child(at, StepRef::Key(&name));
                    match self.start_part(&raw, &target, &plan, here)? {
                        Start::Value(member) => members.push((name, member, slot)),
                        Start::Frame(child) => {
                            *pending = Some((name, slot));
                            return Ok(Some(child));
                        }
                    }
                }
                Ok(None)
            }
            Body::Stored {
                items,
                members,
                pending_name,
                ..
            } => {
                let key_object = pyo3::intern!(py, "py/object");
                let object = self.b(&self.g.object);
                while let Some((name, raw)) = items.next_item()? {
                    let skip = if tp(&name) == t_str() {
                        utf8(&name) == Some(b"py/object".as_slice())
                    } else {
                        py_eq(&name, key_object.as_any())?
                    };
                    if skip {
                        continue;
                    }
                    if is_leaf_kind(tp(&raw)) {
                        members.push((name, raw));
                        continue;
                    }
                    let plan = self.object_plan.clone();
                    let here = Loc::Child(at, StepRef::Key(&name));
                    match self.start_part(&raw, object, &plan, here)? {
                        Start::Value(member) => members.push((name, member)),
                        Start::Frame(child) => {
                            *pending_name = Some(name);
                            return Ok(Some(child));
                        }
                    }
                }
                Ok(None)
            }
            Body::Steps { jobs, values, .. } => {
                for (raw, target, plan, index) in jobs.by_ref() {
                    let here = Loc::Child(at, StepRef::Index(index));
                    match self.start(&raw, &target, &plan, here)? {
                        Start::Value(value) => values.push(value),
                        Start::Frame(child) => return Ok(Some(child)),
                    }
                }
                Ok(None)
            }
            Body::Reduce { .. } => self.step_reduce(frame),
            Body::Union { .. } => self.step_union(frame),
        }
    }

    fn step_reduce(&mut self, frame: &mut Frame<'py>) -> DR<Option<Box<Frame<'py>>>> {
        let at = frame.at;
        loop {
            let Body::Reduce {
                parts,
                part,
                target,
                plan,
                ..
            } = &mut frame.body
            else {
                unreachable!("a reduce frame");
            };
            let Some(raw) = parts.get(*part).cloned() else {
                return Ok(None);
            };
            let here = *part;
            let value = if here == 0 && is_class(target) && self.names_target(&raw, target, plan)? {
                target.clone()
            } else if here >= 2 && raw.is_none() {
                *part += 1;
                continue;
            } else {
                let object = self.b(&self.g.object).clone();
                let object_plan = self.object_plan.clone();
                let here = Loc::Child(at, StepRef::Index(here));
                match self.start(&raw, &object, &object_plan, here)? {
                    Start::Value(value) => value,
                    Start::Frame(child) => return Ok(Some(child)),
                }
            };
            self.take(frame, value)?;
        }
    }

    /// Whether a recipe's callable is ``{"py/type": ...}`` naming class ``target``.
    fn names_target(&self, raw: &B<'py>, target: &B<'py>, plan: &Plan) -> PyResult<bool> {
        let py = self.py;
        let own = match &plan.import_name {
            Some(name) => name.bind(py).clone(),
            None => self.b(&self.g.import_name).call1((target,))?,
        };
        let expected = PyDict::new(py);
        expected.set_item(pyo3::intern!(py, "py/type"), own)?;
        py_eq(raw, expected.as_any())
    }

    fn step_union(&mut self, frame: &mut Frame<'py>) -> DR<Option<Box<Frame<'py>>>> {
        let at = frame.at;
        loop {
            let Body::Union {
                data,
                plan,
                order,
                marks,
                ..
            } = &mut frame.body
            else {
                unreachable!("a union frame");
            };
            let Some(index) = order.next() else {
                return Ok(None);
            };
            let (data, plan) = (data.clone(), plan.clone());
            let mark = self.mark();
            *marks = mark;
            let member = plan.args.bind(self.py).get_item(index)?;
            let started = match self.plan_for(&member, plan.persistent) {
                Ok(member_plan) => self.start(&data, &member, &member_plan, Loc::At(at)),
                Err(error) => Err(error),
            };
            match started {
                Ok(Start::Frame(child)) => return Ok(Some(child)),
                Ok(Start::Value(value)) => self.take(frame, value)?,
                Err(error) if self.is_read(&error) => self.rollback(mark)?,
                Err(error) => return Err(error),
            }
        }
    }

    /// ``_Frame.take``: the value the part handed out last was read as.
    fn take(&mut self, frame: &mut Frame<'py>, value: B<'py>) -> DR<()> {
        let at = frame.at;
        match &mut frame.body {
            Body::Items { built, .. } => self.put(built, value)?,
            Body::Entries {
                result,
                pending_key,
                ..
            } => result.set_item(pending_key.take().expect("an entry is pending"), value)?,
            Body::Fields {
                members,
                pending_name,
                ..
            } => members.set_item(pending_name.take().expect("a field is pending"), value)?,
            Body::Object {
                members, pending, ..
            } => {
                let (name, slot) = pending.take().expect("a field is pending");
                members.push((name, value, slot));
            }
            Body::Stored {
                members,
                pending_name,
                ..
            } => members.push((pending_name.take().expect("a field is pending"), value)),
            Body::Steps { values, .. } => values.push(value),
            Body::Reduce {
                part,
                stateful,
                index,
                func,
                value: built,
                ..
            } => {
                let here = *part;
                *part += 1;
                match here {
                    0 => *func = Some(value),
                    1 => {
                        let made = self.reduce_call(func.as_ref(), &value)?;
                        if *stateful {
                            self.replace(*index, at, &made)?;
                        } else if made
                            .get_type()
                            .getattr(pyo3::intern!(self.py, "__hash__"))?
                            .is_none()
                        {
                            // ``to_plain`` names a mutable stateless value by
                            // path only; it wrote no ``py/id`` for one.
                            self.place_only(at, &made)?;
                        }
                        *built = Some(made);
                    }
                    2 => self.apply_state(built.as_ref().expect("the call came first"), &value)?,
                    3 => {
                        self.reduce_extend(built.as_ref().expect("the call came first"), &value)?
                    }
                    _ => {
                        self.reduce_set_items(
                            built.as_ref().expect("the call came first"),
                            &value,
                        )?;
                    }
                }
            }
            Body::Union {
                marks,
                order,
                value: chosen,
                ..
            } => {
                if self.bad_len() == marks.1 {
                    *chosen = Some(value);
                    *order = Vec::new().into_iter();
                } else {
                    let mark = *marks;
                    self.rollback(mark)?;
                }
            }
        }
        Ok(())
    }

    /// A sequence frame's next item, into the list, set, or values it builds.
    fn put(&self, built: &mut Built<'py>, value: B<'py>) -> DR<()> {
        match built {
            Built::List(list) => list.append(value)?,
            Built::Tuple(values) | Built::FrozenSet(values) => values.push(value),
            Built::Set(set) => {
                if let Err(error) = set.add(&value) {
                    if error.is_instance_of::<PyTypeError>(self.py) {
                        return fail_from(
                            format!("set member {} is unhashable", self.shown(&value)?),
                            error,
                        );
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }

    /// ``_Frame.fail``: recover from the part handed out last failing, else pass
    /// the error on.
    fn recover(&mut self, frame: &mut Frame<'py>, error: DErr, failed: &Failed<'py>) -> DR<()> {
        match &frame.body {
            Body::Union { marks, .. } => {
                let mark = *marks;
                self.rollback(mark)?;
                Ok(())
            }
            Body::Steps { .. } | Body::Reduce { .. } => Err(error),
            Body::Items { .. }
            | Body::Entries { .. }
            | Body::Fields { .. }
            | Body::Object { .. }
            | Body::Stored { .. } => {
                let reason = self.message(&error)?;
                let invalid = self.invalid(&failed.raw, &reason, Loc::At(failed.place))?;
                self.take(frame, invalid)
            }
        }
    }

    /// ``_Frame.finish``: the container built from its parts.
    fn finish(&mut self, at: u32, body: Body<'py>) -> DR<B<'py>> {
        let py = self.py;
        match body {
            Body::Items { built, .. } => match built {
                Built::List(list) => Ok(list.into_any()),
                Built::Set(set) => Ok(set.into_any()),
                Built::Tuple(values) => Ok(PyTuple::new(py, values)?.into_any()),
                Built::FrozenSet(values) => {
                    let members = PyList::new(py, values)?;
                    match PyFrozenSet::new(py, members.iter()) {
                        Ok(result) => Ok(result.into_any()),
                        Err(error) if error.is_instance_of::<PyTypeError>(py) => fail_from(
                            format!(
                                "frozenset members {} are unhashable",
                                self.shown(members.as_any())?
                            ),
                            error,
                        ),
                        Err(error) => Err(error.into()),
                    }
                }
            },
            Body::Entries { result, .. } => Ok(result.into_any()),
            Body::Fields { members, .. } => Ok(members.into_any()),
            Body::Object {
                info,
                target,
                stored,
                value,
                index,
                before,
                members,
                ..
            } => self.build_object(at, &info, &target, stored, value, index, before, members),
            Body::Stored { value, members, .. } => {
                for (attribute, member) in members {
                    self.setattr(&value, &attribute, &member)?;
                }
                Ok(value)
            }
            Body::Steps { values, done, .. } => match done {
                Done::Complex => {
                    let [real, imag] = values.as_slice() else {
                        let message = if values.len() > 2 {
                            if cfg!(Py_3_14) {
                                format!(
                                    "too many values to unpack (expected 2, got {})",
                                    values.len()
                                )
                            } else {
                                "too many values to unpack (expected 2)".to_owned()
                            }
                        } else {
                            format!(
                                "not enough values to unpack (expected 2, got {})",
                                values.len()
                            )
                        };
                        return Err(PyValueError::new_err(message).into());
                    };
                    Ok(self.b(&self.g.complex_type).call1((real, imag))?)
                }
                Done::Hooked { rebuild, index } => {
                    let decoded = values.into_iter().next().expect("a hook reads one payload");
                    let value = rebuild.call1((decoded,))?;
                    self.replace(index, at, &value)?;
                    Ok(value)
                }
            },
            Body::Reduce { value, .. } => Ok(value.unwrap_or_else(|| py.None().into_bound(py))),
            Body::Union {
                data,
                target,
                key,
                references,
                value,
                ..
            } => {
                if let Some(value) = value {
                    return Ok(value);
                }
                if self.references == references {
                    let message = format!(
                        "cannot read {} as {}",
                        self.shown(&data)?,
                        self.target_text(&target)?
                    );
                    self.unread.insert(key, (data, target));
                    return fail(message);
                }
                fail(format!(
                    "cannot read {} as {}",
                    self.shown(&data)?,
                    self.target_text(&target)?
                ))
            }
        }
    }

    fn leaf(&mut self, func: Option<&Py<PyAny>>, data: &B<'py>, target: &B<'py>) -> DR<B<'py>> {
        let py = self.py;
        let func = func.expect("leaf decoders have a function").bind(py);
        Ok(func.call1((data, target, py.None(), py.None()))?)
    }

    /// ``_tag_of``: the highest-precedence tag on a mapping, if any.
    fn tag_of(&self, data: &B<'py>) -> DR<Option<Tag>> {
        // An exact dict of exact str keys answers in one pass: no key's
        // ``__eq__`` or ``__hash__`` is Python code, so membership is bytes.
        if tp(data) == t_dict() {
            let mut best: Option<usize> = None;
            let mut tagged = false;
            let mut exact = true;
            let mut position: ffi::Py_ssize_t = 0;
            let mut key = std::ptr::null_mut();
            let mut value = std::ptr::null_mut();
            while unsafe { ffi::PyDict_Next(data.as_ptr(), &mut position, &mut key, &mut value) }
                != 0
            {
                let key = unsafe { Bound::from_borrowed_ptr(self.py, key) };
                let Some(text) = (tp(&key) == t_str()).then(|| utf8(&key)).flatten() else {
                    exact = false;
                    break;
                };
                if text.starts_with(b"py/") {
                    tagged = true;
                    if let Some(rank) = TAGS.iter().position(|(name, _)| name.as_bytes() == text) {
                        best = Some(best.map_or(rank, |seen| seen.min(rank)));
                    }
                }
            }
            if exact {
                return match (tagged, best) {
                    (false, _) => Ok(None),
                    (true, Some(rank)) => Ok(Some(TAGS[rank].1)),
                    (true, None) => self.unknown_tag(data),
                };
            }
        }
        if !self.has_tag_key(data)? {
            return Ok(None);
        }
        static NAMES: PyOnceLock<Vec<Py<PyString>>> = PyOnceLock::new();
        let names = NAMES.get_or_init(self.py, || {
            TAGS.iter()
                .map(|(name, _)| PyString::intern(self.py, name).unbind())
                .collect()
        });
        for ((_, tag), name) in TAGS.iter().zip(names) {
            if self.contains(data, name.bind(self.py))? {
                return Ok(Some(*tag));
            }
        }
        self.unknown_tag(data)
    }

    /// ``_tag_of``'s refusal of a node whose ``py/`` keys name no tag.
    fn unknown_tag(&self, data: &B<'py>) -> DR<Option<Tag>> {
        let names = PyList::empty(self.py);
        for key in data.try_iter()? {
            names.append(key?.str()?)?;
        }
        let names = self.b(&self.g.sorted).call1((names,))?;
        fail(format!("unknown tag among {}", repr(&names)?))
    }

    fn has_tag_key(&self, data: &B<'py>) -> PyResult<bool> {
        if tp(data) == t_dict() {
            let mut items = DictItems::new(unsafe { data.cast_unchecked::<PyDict>() })?;
            while let Some((key, _)) = items.next_item()? {
                if str_starts(&key, "py/")? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        for key in data.try_iter()? {
            if str_starts(&key?, "py/")? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // Judged with a stack, as Python's ``_fits`` is: a referenced or hooked
    // value is as deep as its data, and the C stack is not. A pair met again
    // while still being judged fits; the rest of the walk decides.
    /// ``_fits``: whether a tag-decoded or referenced ``value`` is acceptable as ``target``.
    fn fits(&mut self, value: &B<'py>, target: &B<'py>, plan: &Rc<Plan>) -> DR<bool> {
        // An instance of exactly the target class fits it with no Python
        // code: ``isinstance`` answers an exact type before any
        // ``__instancecheck__``, and a class target has no parts to judge.
        if matches!(plan.fits, Fits::IsInstance(None))
            && tp(value) == target.as_ptr().cast()
            && plan.args.bind(self.py).is_empty()
        {
            return Ok(true);
        }
        let mut stack: Vec<Judging<'py>> = Vec::new();
        let mut judging: HashSet<(usize, usize)> = HashSet::new();
        let mut pair = (value.clone(), target.clone());
        let mut verdict = self.shallow_fit(value, target, plan)?;
        loop {
            match verdict {
                Shallow::Verdict(fit) => {
                    while stack.last().is_some_and(|top| top.needs_all != fit) {
                        let done = stack.pop().expect("checked above");
                        judging.remove(&done.key);
                    }
                    if stack.is_empty() {
                        return Ok(fit);
                    }
                }
                Shallow::Parts(needs_all, parts) => {
                    let key = (pair.0.as_ptr() as usize, pair.1.as_ptr() as usize);
                    judging.insert(key);
                    stack.push(Judging {
                        needs_all,
                        parts,
                        key,
                        _pair: pair.clone(),
                    });
                }
            }
            let top = stack.last_mut().expect("a frame is left to judge");
            match self.next_part(&mut top.parts)? {
                None => {
                    let done = stack.pop().expect("a frame is left to judge");
                    judging.remove(&done.key);
                    verdict = Shallow::Verdict(done.needs_all);
                }
                Some((value, target, plan)) => {
                    let key = (value.as_ptr() as usize, target.as_ptr() as usize);
                    verdict = if judging.contains(&key) {
                        Shallow::Verdict(true)
                    } else {
                        self.shallow_fit(&value, &target, &plan)?
                    };
                    pair = (value, target);
                }
            }
        }
    }

    /// ``_shallow_fit``: ``value`` against ``target`` alone, or the parts left to judge.
    fn shallow_fit(
        &mut self,
        value: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
    ) -> DR<Shallow<'py>> {
        let py = self.py;
        if value.is_instance(self.b(&self.g.invalid))? {
            return Ok(Shallow::Verdict(true));
        }
        let fit = match &plan.fits {
            Fits::Always => true,
            Fits::Union => {
                return Ok(Shallow::Parts(
                    false,
                    Parts::Members {
                        value: value.clone(),
                        members: plan.args.bind(py).clone(),
                        next: 0,
                        persistent: plan.persistent,
                    },
                ));
            }
            Fits::Literal => plan.args.bind(py).contains(value)?,
            Fits::Type => {
                let args = plan.args.bind(py);
                is_class(value)
                    && (args.is_empty()
                        || self
                            .b(&self.g.is_subclass)
                            .call1((value, args.get_item(0)?))?
                            .is_truthy()?)
            }
            Fits::Python => self.b(&self.g.fits).call1((value, target))?.is_truthy()?,
            Fits::Protocol(origin) => {
                // Read live each time, as ``_has_members`` does.
                let members = origin
                    .bind(py)
                    .getattr(pyo3::intern!(py, "__dict__"))?
                    .get_item(pyo3::intern!(py, "__protocol_attrs__"))?;
                let mut fit = true;
                for name in members.try_iter()? {
                    let name = name?;
                    // A name that is not a str gets ``hasattr``'s own TypeError.
                    let has = match name.cast::<PyString>() {
                        Ok(name) => value.hasattr(name)?,
                        Err(_) => py
                            .import("builtins")?
                            .getattr(pyo3::intern!(py, "hasattr"))?
                            .call1((value, &name))?
                            .is_truthy()?,
                    };
                    if !has {
                        fit = false;
                        break;
                    }
                }
                fit
            }
            Fits::IsInstance(origin) => {
                let origin = match origin {
                    Some(origin) => origin.bind(py).clone(),
                    None => target.clone(),
                };
                if !self.is_instance(value, &origin)? {
                    return Ok(Shallow::Verdict(false));
                }
                return self.item_parts(value, plan);
            }
        };
        Ok(Shallow::Verdict(fit))
    }

    /// ``isinstance(value, origin)``.
    ///
    /// ``isinstance`` asks ``type(origin).__instancecheck__``; ``type``'s own
    /// answers true at once when ``type(value)`` subclasses ``origin``, and
    /// only otherwise reads ``value.__class__``, which can run Python code.
    /// Under a metaclass that inherits ``type``'s check, that first answer is
    /// decided by the MRO here.
    fn is_instance(&self, value: &B<'py>, origin: &B<'py>) -> PyResult<bool> {
        if is_class(origin) {
            let meta = unsafe { ob_type(origin.as_ptr()) };
            let check = unsafe {
                _PyType_Lookup(meta, pyo3::intern!(self.py, "__instancecheck__").as_ptr())
            };
            let own = unsafe {
                _PyType_Lookup(
                    &raw mut ffi::PyType_Type,
                    pyo3::intern!(self.py, "__instancecheck__").as_ptr(),
                )
            };
            if !check.is_null()
                && check == own
                && unsafe { ffi::PyType_IsSubtype(tp(value), origin.as_ptr().cast()) } != 0
            {
                return Ok(true);
            }
        }
        value.is_instance(origin)
    }

    /// A container's items against its element types, as ``_shallow_fit`` queues them.
    fn item_parts(&mut self, value: &B<'py>, plan: &Rc<Plan>) -> DR<Shallow<'py>> {
        let py = self.py;
        let args = plan.args.bind(py).clone();
        if args.is_empty() {
            return Ok(Shallow::Verdict(true));
        }
        if args.len() == 2 && self.is_mapping(value)? {
            let (key_target, member_target) = (args.get_item(0)?, args.get_item(1)?);
            let key_plan = self.plan_for(&key_target, plan.persistent)?;
            let member_plan = self.plan_for(&member_target, plan.persistent)?;
            let key_target = resolved_of(py, &key_plan, &key_target);
            let member_target = resolved_of(py, &member_plan, &member_target);
            return Ok(Shallow::Parts(
                true,
                Parts::Entries {
                    items: MapItems::new(value)?,
                    key: (key_target, key_plan),
                    member: (member_target, member_plan),
                    held: None,
                },
            ));
        }
        if is_list(value) || unsafe { ffi::PySet_Check(value.as_ptr()) != 0 } {
            let item_target = args.get_item(0)?;
            let item_plan = self.plan_for(&item_target, plan.persistent)?;
            let item_target = resolved_of(py, &item_plan, &item_target);
            return Ok(Shallow::Parts(
                true,
                Parts::Items {
                    iter: value.try_iter()?,
                    item: (item_target, item_plan),
                },
            ));
        }
        Ok(Shallow::Verdict(true))
    }

    /// The next ``(value, target, plan)`` of ``parts`` to judge, if any is left.
    #[allow(clippy::type_complexity)]
    fn next_part(&mut self, parts: &mut Parts<'py>) -> DR<Option<(B<'py>, B<'py>, Rc<Plan>)>> {
        match parts {
            Parts::Members {
                value,
                members,
                next,
                persistent,
            } => {
                if *next >= members.len() {
                    return Ok(None);
                }
                let member = members.get_item(*next)?;
                *next += 1;
                let member_plan = self.plan_for(&member, *persistent)?;
                let member_target = resolved_of(self.py, &member_plan, &member);
                Ok(Some((value.clone(), member_target, member_plan)))
            }
            Parts::Entries {
                items,
                key,
                member,
                held,
            } => {
                if let Some(value) = held.take() {
                    return Ok(Some((value, member.0.clone(), member.1.clone())));
                }
                match items.next_item()? {
                    Some((name, value)) => {
                        *held = Some(value);
                        Ok(Some((name, key.0.clone(), key.1.clone())))
                    }
                    None => Ok(None),
                }
            }
            Parts::Items { iter, item } => Ok(iter
                .next()
                .transpose()?
                .map(|value| (value, item.0.clone(), item.1.clone()))),
        }
    }

    fn decode_reference(&mut self, node: &B<'py>) -> DR<B<'py>> {
        let index = self.get(node, pyo3::intern!(self.py, "py/id"))?;
        let valid = node.len()? == 1 && is_int(&index) && tp(&index) != t_bool();
        let position = if valid {
            index.extract::<i64>().ok()
        } else {
            None
        };
        let Some(position) =
            position.filter(|value| *value >= 0 && (*value as usize) < self.built.len())
        else {
            return fail(format!(
                "invalid py/id reference: {}",
                self.shown_dict(node)?
            ));
        };
        self.references += 1;
        let value = self.built[position as usize].clone();
        if value.is(self.b(&self.g.pending)) {
            return fail(format!(
                "py/id {} names a value still being built",
                text(&index)?
            ));
        }
        Ok(value)
    }

    fn decode_path_reference(&mut self, node: &B<'py>) -> DR<B<'py>> {
        let path = self.get(node, pyo3::intern!(self.py, "py/ref"))?;
        let mut valid = node.len()? == 1
            && !is_str(&path)
            && !self.is_mapping(&path)?
            && self.is_sequence(&path)?;
        let mut steps = Vec::new();
        if valid {
            let mut items = SeqItems::new(&path)?;
            while let Some(step) = items.next_item()? {
                let kind = tp(&step);
                if kind != t_str() && kind != t_int() {
                    valid = false;
                    break;
                }
                steps.push(step);
            }
        }
        if !valid {
            return fail(format!(
                "invalid py/ref reference: {}",
                self.shown_dict(node)?
            ));
        }
        self.references += 1;
        let Some(value) = self.placed.get(&self.paths, &steps)?.cloned() else {
            return fail(format!(
                "py/ref {} names no earlier value",
                self.shown_list(&path)?
            ));
        };
        if value.is(self.b(&self.g.pending)) {
            return fail(format!(
                "py/ref {} names a value still being built",
                self.shown_list(&path)?
            ));
        }
        Ok(value)
    }

    fn decode_bool(&mut self, data: &B<'py>) -> DR<B<'py>> {
        if tp(data) == t_bool() {
            return Ok(data.clone());
        }
        if !self.strict.get()? {
            let mut value = data.clone();
            if is_str(data) {
                let lowered = data.call_method0(pyo3::intern!(self.py, "lower"))?;
                value = self
                    .b(&self.g.lax_bools)
                    .call_method1(pyo3::intern!(self.py, "get"), (lowered, data))?;
            }
            if is_int(&value) {
                let zero = pyo3::types::PyInt::new(self.py, 0).into_any();
                let one = pyo3::types::PyInt::new(self.py, 1).into_any();
                if py_eq(&value, &zero)? || py_eq(&value, &one)? {
                    return Ok(pyo3::types::PyBool::new(self.py, value.is_truthy()?)
                        .to_owned()
                        .into_any());
                }
            }
        }
        fail(format!("cannot read {} as bool", self.shown(data)?))
    }

    fn decode_int(&mut self, data: &B<'py>) -> DR<B<'py>> {
        let is_bool = tp(data) == t_bool();
        if is_int(data) && !is_bool {
            return Ok(data.clone());
        }
        if !self.strict.get()? && !is_bool {
            // Integer text reads exactly; through a float it would round past 2**53.
            if is_str(data) {
                match self.b(&self.g.int_type).call1((data,)) {
                    Ok(number) => return Ok(number),
                    Err(error) if error.is_instance_of::<PyValueError>(self.py) => {}
                    Err(error) => return Err(error.into()),
                }
            }
            let number = if is_str(data) {
                match self.b(&self.g.float_type).call1((data,)) {
                    Ok(number) => Some(number),
                    Err(error) if error.is_instance_of::<PyValueError>(self.py) => None,
                    Err(error) => return Err(error.into()),
                }
            } else if is_float(data) {
                Some(data.clone())
            } else {
                None
            };
            if let Some(number) = number
                && number
                    .call_method0(pyo3::intern!(self.py, "is_integer"))?
                    .is_truthy()?
            {
                // Float-form text reads exactly as decimal text, as in Python.
                if is_str(data) {
                    let exact = self.b(&self.g.decimal).call1((data,))?;
                    let integral =
                        exact.call_method0(pyo3::intern!(self.py, "to_integral_value"))?;
                    if exact.eq(&integral)? {
                        return Ok(self.b(&self.g.int_type).call1((exact,))?);
                    }
                }
                return Ok(self.b(&self.g.int_type).call1((number,))?);
            }
        }
        fail(format!("cannot read {} as int", self.shown(data)?))
    }

    fn decode_float(&mut self, data: &B<'py>) -> DR<B<'py>> {
        if is_float(data) {
            return Ok(data.clone());
        }
        if is_int(data) && tp(data) != t_bool() {
            return Ok(self.b(&self.g.float_type).call1((data,))?);
        }
        if is_str(data) {
            let stripped = data.call_method0(pyo3::intern!(self.py, "strip"))?;
            let number = match self.b(&self.g.float_type).call1((stripped,)) {
                Ok(number) => Some(number),
                Err(error) if error.is_instance_of::<PyValueError>(self.py) => None,
                Err(error) => return Err(error.into()),
            };
            if let Some(number) = number {
                let value: f64 = number.extract()?;
                if !value.is_finite() || !self.strict.get()? {
                    return Ok(number);
                }
            }
        }
        fail(format!("cannot read {} as float", self.shown(data)?))
    }

    fn decode_complex(&mut self, data: &B<'py>, at: Loc<'_, 'py>) -> DR<Start<'py>> {
        if (is_int(data) || is_float(data)) && tp(data) != t_bool() {
            return Ok(Start::Value(self.b(&self.g.complex_type).call1((data,))?));
        }
        if self.is_sequence(data)? && !is_str(data) && data.len()? == 2 {
            // Each part is a tree of its own: a non-finite one arrives as
            // ``py/float``. Python lists the parts before reading any, and
            // unpacks them once read, so a ``len`` the items disagree with
            // fails the unpack.
            let frame = self.frame(at)?;
            let float = self.b(&self.g.float_type).clone();
            let plan = self.plan_for(&float, true)?;
            let mut jobs = Vec::with_capacity(2);
            let mut items = SeqItems::new(data)?;
            while let Some(part) = items.next_item()? {
                let index = jobs.len();
                jobs.push((part, float.clone(), plan.clone(), index));
            }
            return Ok(Start::Frame(Frame::new(
                frame,
                data,
                Body::Steps {
                    jobs: jobs.into_iter(),
                    values: Vec::with_capacity(2),
                    done: Done::Complex,
                },
            )));
        }
        fail(format!("cannot read {} as complex", self.shown(data)?))
    }

    fn decode_literal(&mut self, data: &B<'py>, target: &B<'py>, plan: &Rc<Plan>) -> DR<B<'py>> {
        for option in plan.args.bind(self.py).iter() {
            if tp(&option) == tp(data) && py_eq(&option, data)? {
                return Ok(option);
            }
        }
        fail(format!(
            "cannot read {} as {}",
            self.shown(data)?,
            self.target_text(target)?
        ))
    }

    fn element(&mut self, plan: &Rc<Plan>) -> DR<(B<'py>, Rc<Plan>)> {
        let args = plan.args.bind(self.py);
        if args.len() == 1 {
            let target = args.get_item(0)?;
            let item_plan = self.plan_for(&target, plan.persistent)?;
            return Ok((target, item_plan));
        }
        Ok((self.b(&self.g.object).clone(), self.object_plan.clone()))
    }

    fn decode_list(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        if !self.is_sequence(data)? || is_str(data) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let (item, item_plan) = self.element(plan)?;
        self.decode_items(data, &item, &item_plan, at)
    }

    /// ``_decode_items``: a sequence's items as ``item`` into a list numbered for ``py/id``.
    fn decode_items(
        &mut self,
        data: &B<'py>,
        item: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        let frame = self.frame(at)?;
        if item.is(self.b(&self.g.object)) && self.all_leaves(data)? {
            let leaves =
                unsafe { Bound::from_owned_ptr_or_err(py, ffi::PySequence_List(data.as_ptr()))? };
            self.place(frame, &leaves)?;
            return Ok(Start::Value(leaves));
        }
        let result = PyList::empty(py);
        self.place(frame, result.as_any())?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Items {
                raws: SeqItems::new(data)?,
                index: 0,
                types: Types::Repeat(item.clone(), plan.clone()),
                built: Built::List(result),
            },
        )))
    }

    /// ``all(type(raw) in _PLAIN_LEAVES for raw in data)``.
    fn all_leaves(&self, data: &B<'py>) -> PyResult<bool> {
        let mut items = SeqItems::new(data)?;
        while let Some(raw) = items.next_item()? {
            if !is_leaf_kind(tp(&raw)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn decode_tuple(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        if !self.is_sequence(data)? || is_str(data) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let args = plan.args.bind(py).clone();
        let types = if !plan.origin_is_tuple {
            let (item, item_plan) = self.element(plan)?;
            Types::Repeat(item, item_plan)
        } else if args.len() == 2 && args.get_item(1)?.is(py.Ellipsis()) {
            let first = args.get_item(0)?;
            let first_plan = self.plan_for(&first, plan.persistent)?;
            Types::Repeat(first, first_plan)
        } else {
            let size = data.len()?;
            if args.len() != size {
                return fail(format!(
                    "expected {} items for {}, got {}",
                    args.len(),
                    self.target_text(target)?,
                    size
                ));
            }
            Types::Fixed(args, plan.persistent)
        };
        let frame = self.frame(at)?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Items {
                raws: SeqItems::new(data)?,
                index: 0,
                types,
                built: Built::Tuple(Vec::new()),
            },
        )))
    }

    fn decode_set(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        if !self.is_sequence(data)? || is_str(data) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let frame = self.frame(at)?;
        let result = PySet::empty(self.py)?;
        self.place(frame, result.as_any())?;
        let (item, item_plan) = self.element(plan)?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Items {
                raws: SeqItems::new(data)?,
                index: 0,
                types: Types::Repeat(item, item_plan),
                built: Built::Set(result),
            },
        )))
    }

    fn decode_frozenset(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        if !self.is_sequence(data)? || is_str(data) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let (item, item_plan) = self.element(plan)?;
        let frame = self.frame(at)?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Items {
                raws: SeqItems::new(data)?,
                index: 0,
                types: Types::Repeat(item, item_plan),
                built: Built::FrozenSet(Vec::new()),
            },
        )))
    }

    fn decode_dict(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        if !self.is_mapping(data)? {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let args = plan.args.bind(self.py).clone();
        if args.len() == 2 {
            let (key, value) = (args.get_item(0)?, args.get_item(1)?);
            let key_plan = self.plan_for(&key, plan.persistent)?;
            let value_plan = self.plan_for(&value, plan.persistent)?;
            return self.decode_entries_typed(data, (key, key_plan), (value, value_plan), at);
        }
        let object = self.object_plan.clone();
        self.decode_entries(data, &object, &object, at)
    }

    fn decode_entries(
        &mut self,
        data: &B<'py>,
        key_plan: &Rc<Plan>,
        value_plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let object = self.b(&self.g.object).clone();
        self.decode_entries_typed(
            data,
            (object.clone(), key_plan.clone()),
            (object, value_plan.clone()),
            at,
        )
    }

    /// ``_decode_entries``: a mapping's keys and values into a dict numbered for ``py/id``.
    fn decode_entries_typed(
        &mut self,
        data: &B<'py>,
        key: (B<'py>, Rc<Plan>),
        value: (B<'py>, Rc<Plan>),
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let frame = self.frame(at)?;
        let result = PyDict::new(self.py);
        self.place(frame, result.as_any())?;
        let plain_keys = key.1.is_object || key.0.is(self.b(&self.g.str_type));
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Entries {
                items: MapItems::new(data)?,
                key,
                value,
                plain_keys,
                result,
                pending_key: None,
            },
        )))
    }

    /// ``_decode_key``: a dict key as ``target``, unescaping a ``json://`` key.
    fn decode_key(
        &mut self,
        key: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<B<'py>> {
        let py = self.py;
        if !is_str(key) {
            let resolved = resolved_of(py, plan, target);
            if self.fits(key, &resolved, plan)? {
                return Ok(key.clone());
            }
            return fail(format!(
                "cannot read key {} as {}",
                self.shown(key)?,
                self.target_text(target)?
            ));
        }
        let mut raw = key.clone();
        if str_starts(key, "json://")? {
            let body = key.call_method1(pyo3::intern!(py, "removeprefix"), ("json://",))?;
            match self.b(&self.g.json_loads).call1((body,)) {
                Ok(loaded) => raw = loaded,
                Err(error) if error.is_instance(py, self.b(&self.g.json_error)) => {
                    return fail_from(format!("invalid escaped key {}", self.shown(key)?), error);
                }
                Err(error) => return Err(error.into()),
            }
        }
        // A key reads with a walk of its own, as Python's does; only keys
        // escaped inside escaped keys nest it.
        let decoded = {
            let _guard = Recursion::enter(py, c" while decoding")?;
            self.walk(&raw, target, plan, at)?
        };
        // Hashing is the test the dict insert makes; the hash slot passes a
        // tuple holding a list.
        if let Err(error) = decoded.hash() {
            if error.is_instance_of::<PyTypeError>(py) {
                return fail_from(
                    format!("key {} is unhashable", self.shown(&decoded)?),
                    error,
                );
            }
            return Err(error.into());
        }
        Ok(decoded)
    }

    // ``to_plain`` registers the proxy at its own path and the copied dict one
    // step below it, under ``py/mappingproxy``; a ``py/ref`` may name either.
    fn decode_mappingproxy(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        let inner = PyDict::new(py);
        let value = self.b(&self.g.mappingproxy).call1((&inner,))?;
        let place = self.frame(at)?;
        self.place_only(place, &value)?;
        let key = pystr(py, "py/mappingproxy");
        let below = Loc::Child(place, StepRef::Key(&key));
        let Start::Frame(mut frame) = self.decode_dict(data, target, plan, below)? else {
            unreachable!("a mapping's entries read in a frame");
        };
        frame.place = place;
        frame.then = Some((inner, value));
        Ok(Start::Frame(frame))
    }

    fn decode_type(&mut self, data: &B<'py>, plan: &Rc<Plan>) -> DR<B<'py>> {
        if !is_str(data) {
            return fail(format!("cannot read {} as a class path", self.shown(data)?));
        }
        let resolved = self.resolve(data)?;
        if plan.origin_is_type && !is_class(&resolved) {
            return fail(format!("{} does not name a class", self.shown(data)?));
        }
        Ok(resolved)
    }

    fn decode_function(&mut self, data: &B<'py>) -> DR<B<'py>> {
        if !is_str(data) {
            return fail(format!(
                "cannot read {} as a function path",
                self.shown(data)?
            ));
        }
        let resolved = self.resolve(data)?;
        if !is_callable(&resolved) {
            return fail(format!("{} does not name a callable", self.shown(data)?));
        }
        Ok(resolved)
    }

    /// ``_decode_object``: the ``py/object`` class, else the target's dataclass.
    fn decode_object(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        if !self.is_mapping(data)? {
            return fail(format!(
                "expected object for {}, got {}",
                self.class_name(target)?,
                self.shown(data)?
            ));
        }
        if plan.is_dataclass
            && is_class(target)
            && let Some(started) = self.decode_exact_object(data, target, plan, at)?
        {
            return Ok(started);
        }
        let key_object = pyo3::intern!(py, "py/object");
        let stored = self.contains(data, key_object)?;
        let mut data = data.clone();
        if plan.is_dataclass && is_class(target) {
            let names = !self.allow_imports.get()? || {
                let named = self.get_or_none(&data, key_object)?;
                let own = plan
                    .import_name
                    .as_ref()
                    .expect("classes have import names")
                    .bind(py);
                py_eq(&named, own)?
            };
            // Python copies every record it reads by the target's own fields,
            // keeping each key ``!= "py/object"``. For an exact dict without
            // the tag whose keys are all exact ``str``, those comparisons run
            // no Python code and keep every entry, so the copy is skipped.
            if names && (stored || !is_str_keyed_dict(&data)) {
                let fields_only = PyDict::new(py);
                let mut items = MapItems::new(&data)?;
                while let Some((key, raw)) = items.next_item()? {
                    if py_ne(&key, key_object.as_any())? {
                        fields_only.set_item(key, raw)?;
                    }
                }
                data = fields_only.into_any();
            }
        }
        if self.contains(&data, key_object)? {
            return self.decode_stored(&data, at);
        }
        if !(plan.is_dataclass && is_class(target)) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown_dict(&data)?,
                self.class_name(target)?
            ));
        }
        let info = self.dc_info(target, plan)?;
        let index = info.index.bind(py).clone();
        let mut unknown = Vec::new();
        let mut items = MapItems::new(&data)?;
        let mut entries = Vec::new();
        // An exact str key matches a field by its bytes, which is what the
        // index dict's lookup decides for it, and records list fields in
        // declared order, so the search starts after the last match.
        let mut cursor = 0;
        while let Some((key, raw)) = items.next_item()? {
            let slot = match (tp(&key) == t_str()).then(|| utf8(&key)).flatten() {
                Some(text) => info.slot_of(text, &mut cursor),
                None => index
                    .get_item(&key)?
                    .map(|slot| slot.extract::<usize>())
                    .transpose()?,
            };
            match slot {
                Some(slot) => entries.push((key, raw, slot)),
                None => unknown.push(key),
            }
        }
        if !unknown.is_empty() {
            let sorted = self.b(&self.g.sorted);
            let unknown = sorted.call1((PyList::new(py, unknown)?,))?;
            let valid = sorted.call1((&index,))?;
            return fail(format!(
                "{}: unknown field(s) {}; valid: {}",
                self.class_name(target)?,
                repr(&unknown)?,
                repr(&valid)?
            ));
        }
        let position = self.built.len();
        let frame = self.frame(at)?;
        // A ``py/object`` record is state ``to_plain`` took from a built
        // object, so it is restored as is, allocated before its fields read so
        // a field can refer back to it. An untagged record is constructor
        // input, built through ``__init__`` once its fields have read.
        let reserved = if stored {
            self.allocate(target)?
        } else {
            self.pending()
        };
        self.place(frame, &reserved)?;
        let before = self.bad_len();
        let members = Vec::with_capacity(entries.len());
        Ok(Start::Frame(Frame::new(
            frame,
            &data,
            Body::Object {
                entries: entries.into_iter(),
                info,
                target: target.clone(),
                stored,
                value: reserved,
                index: position,
                before,
                members,
                pending: None,
            },
        )))
    }

    /// ``decode_object`` for an exact dict whose keys are all exact strs and
    /// whose ``py/object``, if any, names ``target``: one pass over the dict,
    /// matching keys to fields by their bytes. Such keys run no Python code
    /// when hashed or compared, so the copy without the tag, the second pass
    /// and the tag's ``==`` that ``decode_object`` makes can be skipped.
    /// ``None`` hands the record to ``decode_object``, which also words every
    /// refusal.
    fn decode_exact_object(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Option<Start<'py>>> {
        let py = self.py;
        if tp(data) != t_dict() {
            return Ok(None);
        }
        // A flag that is not a bool runs its ``__bool__`` each time it is
        // asked; ``decode_object`` asks once, so only a real bool comes here.
        if !matches!(self.allow_imports, Flag::Known(_)) {
            return Ok(None);
        }
        let info = self.dc_info(target, plan)?;
        if info.names.is_none() {
            return Ok(None);
        }
        let own = plan
            .import_name
            .as_ref()
            .expect("classes have import names")
            .bind(py);
        let Some(own) = (tp(own) == t_str()).then(|| utf8(own)).flatten() else {
            return Ok(None);
        };
        let dict = unsafe { data.cast_unchecked::<PyDict>() };
        let mut entries = Vec::with_capacity(dict.len());
        let mut stored = false;
        let mut cursor = 0;
        for (key, raw) in dict_entries(dict) {
            let Some(text) = (tp(&key) == t_str()).then(|| utf8(&key)).flatten() else {
                return Ok(None);
            };
            if text == b"py/object" {
                let named = (tp(&raw) == t_str()).then(|| utf8(&raw)).flatten();
                if named != Some(own) {
                    return Ok(None);
                }
                stored = true;
                continue;
            }
            match info.slot_of(text, &mut cursor) {
                Some(slot) => entries.push((key, raw, slot)),
                None => return Ok(None),
            }
        }
        // A record naming its class whose imports are off reads its fields
        // just the same, since only the target's own name was accepted.
        let position = self.built.len();
        let frame = self.frame(at)?;
        let reserved = if stored {
            self.allocate(target)?
        } else {
            self.pending()
        };
        self.place(frame, &reserved)?;
        let before = self.bad_len();
        let members = Vec::with_capacity(entries.len());
        Ok(Some(Start::Frame(Frame::new(
            frame,
            data,
            Body::Object {
                entries: entries.into_iter(),
                info,
                target: target.clone(),
                stored,
                value: reserved,
                index: position,
                before,
                members,
                pending: None,
            },
        ))))
    }

    /// ``_Object.finish``: the dataclass built from its read fields.
    #[allow(clippy::too_many_arguments)]
    fn build_object(
        &mut self,
        at: u32,
        info: &DcInfo,
        target: &B<'py>,
        stored: bool,
        reserved: B<'py>,
        position: usize,
        before: usize,
        members: Vec<(B<'py>, B<'py>, usize)>,
    ) -> DR<B<'py>> {
        let py = self.py;
        let (value, rest) = if stored || self.bad_len() > before {
            let value = if stored {
                reserved
            } else {
                self.allocate(target)?
            };
            let rest = if self.declares_exactly(target, &members) {
                for (key, member, slot) in &members {
                    self.store(info, target, &value, *slot, key, member)?;
                }
                self.replace(position, at, &value)?;
                return Ok(value);
            } else {
                self.with_defaults(members, target)?
            };
            (value, rest)
        } else {
            let kwargs = PyDict::new(py);
            let mut rest = Vec::new();
            let mut init = Vec::new();
            for field in self.dataclass_fields(target)? {
                if self.field_slot(&field, FieldSlot::Init)?.is_truthy()? {
                    init.push(self.field_slot(&field, FieldSlot::Name)?);
                }
            }
            // With exact str keys and names, membership is a byte comparison
            // and runs no ``__eq__``; any other key asks the set Python builds.
            let exact = init
                .iter()
                .all(|name| tp(name) == t_str() && utf8(name).is_some());
            let set = if exact {
                None
            } else {
                let set = PySet::empty(py)?;
                for name in &init {
                    set.add(name)?;
                }
                Some(set)
            };
            for (key, member, _) in members {
                let inside = match &set {
                    Some(set) => set.contains(&key)?,
                    None if tp(&key) == t_str() && utf8(&key).is_some() => {
                        let text = utf8(&key);
                        init.iter().any(|name| name.is(&key) || utf8(name) == text)
                    }
                    None => {
                        let set = PySet::new(py, &init)?;
                        set.contains(&key)?
                    }
                };
                if inside {
                    kwargs.set_item(key, member)?;
                } else {
                    rest.push((key, member));
                }
            }
            match target.call((), Some(&kwargs)) {
                Ok(value) => (value, rest),
                Err(error)
                    if error.is_instance_of::<PyTypeError>(py)
                        || error.is_instance_of::<PyValueError>(py) =>
                {
                    let message = format!(
                        "cannot build {}: {}",
                        self.class_name(target)?,
                        err_text(py, &error)?
                    );
                    return fail_from(message, error);
                }
                Err(error) => return Err(error.into()),
            }
        };
        for (key, member) in &rest {
            self.setattr(&value, key, member)?;
        }
        self.replace(position, at, &value)?;
        Ok(value)
    }

    /// ``fields(target)``, read natively while ``fields`` is ``dataclasses``' own:
    /// the values of ``__dataclass_fields__`` whose ``_field_type`` is ``_FIELD``.
    fn dataclass_fields(&self, target: &B<'py>) -> PyResult<Vec<B<'py>>> {
        let py = self.py;
        let Some(field_kind) = &self.g.field_kind else {
            return self
                .b(&self.g.fields)
                .call1((target,))?
                .try_iter()?
                .collect();
        };
        let declared = match target.getattr(pyo3::intern!(py, "__dataclass_fields__")) {
            Ok(declared) => declared,
            Err(error) if error.is_instance_of::<PyAttributeError>(py) => {
                return Err(PyTypeError::new_err(
                    "must be called with a dataclass type or instance",
                ));
            }
            Err(error) => return Err(error),
        };
        // ``fields`` iterates ``.values()``; an exact dict yields them in the
        // order ``PyDict_Next`` does, and reading it runs no Python code.
        if tp(&declared) == t_dict() {
            let mut found = Vec::new();
            for (_, field) in dict_entries(unsafe { declared.cast_unchecked::<PyDict>() }) {
                if self
                    .field_slot(&field, FieldSlot::Kind)?
                    .is(field_kind.bind(py))
                {
                    found.push(field);
                }
            }
            return Ok(found);
        }
        let values = declared.call_method0(pyo3::intern!(py, "values"))?;
        let mut found = Vec::new();
        for field in values.try_iter()? {
            let field = field?;
            if self
                .field_slot(&field, FieldSlot::Kind)?
                .is(field_kind.bind(py))
            {
                found.push(field);
            }
        }
        Ok(found)
    }

    /// ``field.<name>`` for a ``dataclasses.Field``, read from its slot when
    /// the field is exactly that class, else by ``getattr``.
    fn field_slot(&self, field: &B<'py>, slot: FieldSlot) -> PyResult<B<'py>> {
        let py = self.py;
        if let Some(layout) = &self.g.field_layout
            && tp(field) == layout.kind.bind(py).as_ptr().cast()
            && layout.stamp.holds(layout.kind.bind(py))
            && let Some(offset) = layout.offsets[slot as usize]
        {
            let read = unsafe {
                *field
                    .as_ptr()
                    .cast::<u8>()
                    .offset(offset)
                    .cast::<*mut ffi::PyObject>()
            };
            if !read.is_null() {
                return Ok(unsafe { Bound::from_borrowed_ptr(py, read) });
            }
        }
        field.getattr(slot.name(py))
    }

    /// Whether ``members`` holds exactly the entries of ``target``'s
    /// ``__dataclass_fields__``, found as ``getattr`` would with no Python code
    /// run: then ``with_defaults`` fills nothing and keeps ``members``' order.
    /// Read without iterating the fields, which is what makes it cheap.
    fn declares_exactly(&self, target: &B<'py>, members: &[(B<'py>, B<'py>, usize)]) -> bool {
        let py = self.py;
        if self.g.field_kind.is_none() {
            return false;
        }
        let name = pyo3::intern!(py, "__dataclass_fields__").as_ptr();
        unsafe {
            let class = target.as_ptr().cast::<ffi::PyTypeObject>();
            let meta = ob_type(target.as_ptr());
            let type_getattro = ffi::PyType_Type.tp_getattro.map(|f| f as usize);
            if (*meta).tp_getattro.map(|f| f as usize) != type_getattro
                || !_PyType_Lookup(meta, name).is_null()
            {
                return false;
            }
            let declared = _PyType_Lookup(class, name);
            if declared.is_null() || ob_type(declared) != t_dict() {
                return false;
            }
            if ffi::PyDict_Size(declared) != members.len() as ffi::Py_ssize_t {
                return false;
            }
            // Each member is a field the record read, and fields are
            // distinct, so a count equal to the dict's means every entry is a
            // field read; a ``ClassVar`` or ``InitVar`` entry leaves it short.
            members.iter().all(|(key, _, _)| {
                tp(key) == t_str() && ffi::PyDict_Contains(declared, key.as_ptr()) == 1
            })
        }
    }

    fn with_defaults(
        &self,
        members: Vec<(B<'py>, B<'py>, usize)>,
        target: &B<'py>,
    ) -> DR<Vec<(B<'py>, B<'py>)>> {
        let py = self.py;
        let declared = self.dataclass_fields(target)?;
        let mut names = Vec::with_capacity(declared.len());
        for field in &declared {
            names.push(self.field_slot(field, FieldSlot::Name)?);
        }
        // A record carrying every field, each key an exact str equal to its
        // field's name by identity or by bytes, needs no default: comparing
        // such strs runs no Python code, so the set Python builds is skipped.
        let exact = |key: &B<'py>| tp(key) == t_str();
        if members.len() >= names.len()
            && names.iter().all(exact)
            && members.iter().all(|(key, _, _)| exact(key))
            && names.iter().all(|name| {
                members.iter().any(|(key, _, _)| {
                    key.is(name) || (utf8(key).is_some() && utf8(key) == utf8(name))
                })
            })
        {
            return Ok(members
                .into_iter()
                .map(|(key, member, _)| (key, member))
                .collect());
        }
        let present = PySet::empty(py)?;
        let mut filled = Vec::with_capacity(members.len());
        for (key, member, _) in members {
            present.add(&key)?;
            filled.push((key, member));
        }
        let mut missing = Vec::new();
        for (field, name) in declared.iter().zip(names) {
            if present.contains(&name)? {
                continue;
            }
            let default = self.field_slot(field, FieldSlot::Default)?;
            let factory = self.field_slot(field, FieldSlot::DefaultFactory)?;
            if !default.is(self.b(&self.g.missing)) {
                filled.push((name, default));
            } else if !factory.is(self.b(&self.g.missing)) {
                filled.push((name, factory.call0()?));
            } else if self.field_slot(field, FieldSlot::Init)?.is_truthy()? {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            let missing = PyList::new(py, missing)?;
            return fail(format!(
                "cannot build {}: missing field(s) {}",
                self.class_name(target)?,
                repr(missing.as_any())?
            ));
        }
        Ok(filled)
    }

    /// ``_decode_stored``: the class a ``py/object`` record names, attributes untyped.
    fn decode_stored(&mut self, data: &B<'py>, at: Loc<'_, 'py>) -> DR<Start<'py>> {
        let py = self.py;
        let path = self.get(data, pyo3::intern!(py, "py/object"))?;
        if !is_str(&path) {
            return fail(format!("invalid py/object path: {}", self.shown(&path)?));
        }
        let kind = self.resolve(&path)?;
        if !is_class(&kind) {
            return fail(format!("{} does not name a class", self.shown(&path)?));
        }
        let value = self.allocate(&kind)?;
        let frame = self.frame(at)?;
        self.place(frame, &value)?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Stored {
                items: MapItems::new(data)?,
                value,
                members: Vec::new(),
                pending_name: None,
            },
        )))
    }

    fn decode_hooked(&mut self, data: &B<'py>, at: Loc<'_, 'py>) -> DR<Start<'py>> {
        let py = self.py;
        if !self.is_sequence(data)? || is_str(data) || data.len()? != 2 {
            return fail(format!("invalid py/hook payload: {}", self.shown(data)?));
        }
        let (path, payload) = (data.get_item(0)?, data.get_item(1)?);
        let mut hook = None;
        if let Some(hooks) = self.hooks.clone() {
            let mut items = MapItems::new(&hooks)?;
            while let Some((kind, pair)) = items.next_item()? {
                let module = kind.getattr(pyo3::intern!(py, "__module__"))?;
                let qualname = kind.getattr(pyo3::intern!(py, "__qualname__"))?;
                let name = pystr(py, &format!("{}.{}", text(&module)?, text(&qualname)?));
                if py_eq(&name, &path)? {
                    hook = Some(pair);
                    break;
                }
            }
        }
        let Some(hook) = hook else {
            return fail(format!("hook {} is not registered", self.shown(&path)?));
        };
        // The encoder numbers the hooked value before its payload, so the slot
        // is reserved in that order and filled once the hook rebuilds it.
        let position = self.built.len();
        let frame = self.frame(at)?;
        let pending = self.pending();
        self.place(frame, &pending)?;
        let rebuild = hook.get_item(1)?;
        let object = self.b(&self.g.object).clone();
        let object_plan = self.object_plan.clone();
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Steps {
                jobs: vec![(payload, object, object_plan, 1)].into_iter(),
                values: Vec::with_capacity(1),
                done: Done::Hooked {
                    rebuild,
                    index: position,
                },
            },
        )))
    }

    /// ``_decode_reduce``: a ``py/reduce`` recipe; one naming ``target`` itself needs no import.
    fn decode_reduce(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        if !self.is_sequence(data)? || is_str(data) {
            return fail(format!("invalid py/reduce payload: {}", self.shown(data)?));
        }
        let size = data.len()?;
        if !(2..=5).contains(&size) {
            return fail("py/reduce requires two to five elements".to_owned());
        }
        let parts: Vec<B<'py>> = (0..size)
            .map(|index| data.get_item(index))
            .collect::<PyResult<_>>()?;
        let stateful = parts[2..].iter().any(|part| !part.is_none());
        let position = self.built.len();
        let frame = self.frame(at)?;
        if stateful {
            let pending = self.pending();
            self.place(frame, &pending)?;
        }
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Reduce {
                parts,
                part: 0,
                target: target.clone(),
                plan: plan.clone(),
                stateful,
                index: position,
                func: None,
                value: None,
            },
        )))
    }

    /// ``_Reduce._call``: the recipe's callable called on its arguments.
    fn reduce_call(&self, func: Option<&B<'py>>, args: &B<'py>) -> DR<B<'py>> {
        let py = self.py;
        let func = func.filter(|func| is_callable(func));
        let (Some(func), true) = (func, is_tuple(args)) else {
            return fail("py/reduce needs a callable and an argument tuple".to_owned());
        };
        match func.call1(unsafe { args.cast_unchecked::<PyTuple>() }.clone()) {
            Ok(value) => Ok(value),
            Err(error)
                if error.is_instance_of::<PyTypeError>(py)
                    || error.is_instance_of::<PyValueError>(py) =>
            {
                let message = format!("py/reduce call failed: {}", err_text(py, &error)?);
                fail_from(message, error)
            }
            Err(error) => Err(error.into()),
        }
    }

    /// ``_Reduce._extend``: the built value extended with the recipe's list items.
    fn reduce_extend(&self, value: &B<'py>, items: &B<'py>) -> DR<()> {
        let extend = getattr_or_none(value, pyo3::intern!(self.py, "extend"))?;
        let Some(extend) = extend.filter(|extend| is_callable(extend) && is_list(items)) else {
            return fail(format!(
                "reduce target cannot take list items: {}",
                self.shown(value)?
            ));
        };
        extend.call1((items,))?;
        Ok(())
    }

    /// ``_Reduce._set_items``: the recipe's dict items set on the built value.
    fn reduce_set_items(&self, value: &B<'py>, pairs: &B<'py>) -> DR<()> {
        let setitem = getattr_or_none(value, pyo3::intern!(self.py, "__setitem__"))?;
        let Some(setitem) = setitem.filter(|setitem| is_callable(setitem) && is_list(pairs)) else {
            return fail(format!(
                "reduce target cannot take dict items: {}",
                self.shown(value)?
            ));
        };
        for pair in pairs.try_iter()? {
            let pair = pair?;
            let entry = if is_tuple(&pair) {
                Some(unsafe { pair.cast_unchecked::<PyTuple>() }.clone())
            } else {
                None
            };
            let Some(entry) = entry.filter(|entry| entry.len() == 2) else {
                return fail(format!("invalid reduce dict item: {}", self.shown(&pair)?));
            };
            setitem.call1((entry.get_item(0)?, entry.get_item(1)?))?;
        }
        Ok(())
    }

    fn apply_state(&mut self, value: &B<'py>, attributes: &B<'py>) -> PyResult<()> {
        let py = self.py;
        if let Some(setstate) = getattr_or_none(value, pyo3::intern!(py, "__setstate__"))?
            && is_callable(&setstate)
        {
            setstate.call1((attributes,))?;
            return Ok(());
        }
        let mut chunks = vec![attributes.clone()];
        if is_tuple(attributes) && attributes.len()? == 2 {
            chunks = vec![attributes.get_item(0)?, attributes.get_item(1)?];
        }
        for chunk in chunks {
            if is_dict(&chunk) {
                let mut items = MapItems::new(&chunk)?;
                while let Some((key, member)) = items.next_item()? {
                    self.b(&self.g.object_setattr).call1((value, key, member))?;
                }
            }
        }
        Ok(())
    }

    /// ``_decode_any``: untagged data for ``object``, ``Any``, or a Protocol, as is.
    fn decode_any(&mut self, data: &B<'py>, at: Loc<'_, 'py>) -> DR<Start<'py>> {
        if self.untagged || data.is_none() || is_str(data) || is_int(data) || is_float(data) {
            return Ok(Start::Value(data.clone()));
        }
        let object = self.object_plan.clone();
        if self.is_mapping(data)? {
            return self.decode_entries(data, &object, &object, at);
        }
        let target = self.b(&self.g.object).clone();
        self.decode_items(data, &target, &object, at)
    }

    /// The order ``_decode_union`` tries ``plan``'s members in for ``data``.
    fn union_members(&mut self, data: &B<'py>, plan: &Rc<Plan>) -> DR<Vec<usize>> {
        let py = self.py;
        let members = plan.args.bind(py).clone();
        let info = plan.union.as_ref().expect("unions compile their members");
        let tagged = self.is_mapping(data)? && self.has_tag_key(data)?;
        let order: Vec<usize> = if tagged {
            // The member a ``py/object`` tag names goes first, so member order
            // cannot pick a sibling whose fields happen to fit too.
            let path = self.get_or_none(data, pyo3::intern!(py, "py/object"))?;
            let mut keyed = Vec::with_capacity(members.len());
            for (index, name) in info.import_names.iter().enumerate() {
                let differs = match name {
                    Some(name) => !py_eq(name.bind(py), &path)?,
                    None => !path.is_none(),
                };
                keyed.push((differs, index));
            }
            keyed.sort_by_key(|(differs, _)| *differs);
            keyed.into_iter().map(|(_, index)| index).collect()
        } else {
            let kind = tp(data);
            let slot = if kind == t_none() {
                Some(0)
            } else if kind == t_bool() {
                Some(1)
            } else if kind == t_int() {
                Some(2)
            } else if kind == t_float() {
                Some(3)
            } else if kind == t_str() {
                Some(4)
            } else if is_str(data) || is_int(data) || is_float(data) {
                None
            } else if self.is_mapping(data)? {
                Some(5)
            } else {
                Some(6)
            };
            // ``_union_candidates`` looks only at the data's kind, so one
            // answer per slot serves every later datum of that kind.
            match slot {
                Some(slot) => match info.orders[slot].get() {
                    Some(order) => order.clone(),
                    None => {
                        let order = self.union_order(data, &members)?;
                        let _ = info.orders[slot].set(order.clone());
                        order
                    }
                },
                None => self.union_order(data, &members)?,
            }
        };
        Ok(order)
    }

    /// ``_decode_union``: the member the data's kind selects, else the first that fits.
    fn decode_union(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        // Each member a parent union tries reaches this node again; remembering
        // the failure keeps nested unions linear instead of exponential in
        // depth. A read that resolved a reference may succeed elsewhere, where
        // it names another value, so it is never remembered.
        let key = (data.as_ptr() as usize, target.as_ptr() as usize);
        if self.unread.contains_key(&key) {
            return fail(format!(
                "cannot read {} as {}",
                self.shown(data)?,
                self.target_text(target)?
            ));
        }
        let order = self.union_members(data, plan)?;
        let frame = self.frame(at)?;
        let mark = self.mark();
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Union {
                data: data.clone(),
                target: target.clone(),
                plan: plan.clone(),
                order: order.into_iter(),
                key,
                references: self.references,
                marks: mark,
                value: None,
            },
        )))
    }

    fn decode_typeddict(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        at: Loc<'_, 'py>,
    ) -> DR<Start<'py>> {
        let py = self.py;
        let name = || -> PyResult<String> { text(&target.getattr(pyo3::intern!(py, "__name__"))?) };
        if !self.is_mapping(data)? {
            return fail(format!(
                "expected object for {}, got {}",
                name()?,
                self.shown(data)?
            ));
        }
        let hints = self.b(&self.g.field_types).call1((target,))?;
        let required = self.b(&self.g.required_keys).call1((target,))?;
        let keys = PySet::empty(py)?;
        for key in data.try_iter()? {
            keys.add(key?)?;
        }
        let missing = required.call_method1(pyo3::intern!(py, "difference"), (keys,))?;
        let missing = self.b(&self.g.sorted).call1((missing,))?;
        if missing.len()? > 0 {
            return fail(format!(
                "{}: missing required field(s) {}",
                name()?,
                repr(&missing)?
            ));
        }
        let frame = self.frame(at)?;
        let result = PyDict::new(py);
        self.place(frame, result.as_any())?;
        Ok(Start::Frame(Frame::new(
            frame,
            data,
            Body::Fields {
                items: MapItems::new(data)?,
                hints,
                members: result,
                pending_name: None,
            },
        )))
    }
}

fn resolved_of<'py>(py: Python<'py>, plan: &Plan, target: &B<'py>) -> B<'py> {
    match &plan.resolved {
        Some(resolved) => resolved.bind(py).clone(),
        None => target.clone(),
    }
}

fn getattr_or_none<'py>(obj: &B<'py>, name: &Bound<'py, PyString>) -> PyResult<Option<B<'py>>> {
    match obj.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyAttributeError>(obj.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

// ------------------------------------------------------------ holds tags --

/// ``_holds_tags``: whether any mapping in ``data`` has a tag or escaped key.
fn holds_tags<'py>(g: &DecGlobals, data: &B<'py>) -> PyResult<bool> {
    let py = data.py();
    let mut stack = vec![data.clone()];
    let mut seen = HashSet::new();
    // Each node is kept, not just its address: a custom container may build
    // its parts afresh on each read, and one dropped here frees its address
    // for the next.
    let mut kept = Vec::new();
    while let Some(node) = stack.pop() {
        let kind = tp(&node);
        if is_str(&node) || is_int(&node) || is_float(&node) || kind == t_none() {
            continue;
        }
        if !seen.insert(node.as_ptr() as usize) {
            continue;
        }
        kept.push(node.clone());
        if kind == t_dict() {
            let dict = unsafe { node.cast_unchecked::<PyDict>() };
            let mut items = DictItems::new(dict)?;
            while let Some((key, value)) = items.next_item()? {
                if is_str(&key) && str_starts_any(&key, &["py/", "json://"])? {
                    return Ok(true);
                }
                stack.push(value);
            }
            continue;
        }
        if kind == t_list() || kind == t_tuple() {
            let mut items = SeqItems::new(&node)?;
            while let Some(value) = items.next_item()? {
                stack.push(value);
            }
            continue;
        }
        if node.is_instance(g.mapping.bind(py))? {
            for key in node.try_iter()? {
                let key = key?;
                if is_str(&key) && str_starts_any(&key, &["py/", "json://"])? {
                    return Ok(true);
                }
            }
            for value in node.call_method0(pyo3::intern!(py, "values"))?.try_iter()? {
                stack.push(value?);
            }
            continue;
        }
        if node.is_instance(g.sequence.bind(py))?
            && unsafe {
                ffi::PyBytes_Check(node.as_ptr()) == 0 && ffi::PyByteArray_Check(node.as_ptr()) == 0
            }
        {
            for value in node.try_iter()? {
                stack.push(value?);
            }
        }
    }
    Ok(false)
}

// ------------------------------------------------------------ entry point --

pub fn from_plain<'py>(
    py: Python<'py>,
    data: &B<'py>,
    target: &B<'py>,
    default: Option<B<'py>>,
    hooks: Option<B<'py>>,
    allow_imports: Flag<'py>,
    strict: Flag<'py>,
) -> PyResult<B<'py>> {
    let Some(globals) = DEC.cloned(py) else {
        return Err(PyTypeError::new_err(
            "treekle._rust.setup_decode was not called",
        ));
    };
    let g = &*globals;
    if data.is_none()
        && let Some(default) = &default
        && !default.is(g.absent.bind(py))
    {
        return Ok(default.clone());
    }
    let untagged = !holds_tags(g, data)?;
    let caches = CACHES.take(py).unwrap_or_else(|| new_caches(py));
    let mut decoder = Decoder::new(py, g, hooks, allow_imports, strict, untagged, caches)?;
    let result = decoder.run(data, target);
    let kept = std::mem::replace(
        &mut decoder.caches,
        Caches::around(decoder.object_plan.clone()),
    );
    // Nested calls may have returned their caches meanwhile. Keep the outer
    // plans only if setup has not replaced the helpers they were built for;
    // a reentrant setup clears CACHES and must not be undone here.
    if DEC
        .cloned(py)
        .is_some_and(|current| Rc::ptr_eq(&current, &globals))
    {
        let stale = CACHES.replace(py, Some(kept));
        drop(stale);
    }
    let value = match result {
        Ok(value) => value,
        Err(error) if decoder.is_read(&error) => {
            let message = decoder.message(&error)?;
            let inner = decoder.pyerr(error);
            let kwargs = PyDict::new(py);
            kwargs.set_item("raw", data)?;
            kwargs.set_item("reason", &message)?;
            let whole = g.invalid.bind(py).call((), Some(&kwargs))?;
            let bad = PyDict::new(py);
            if let Some(found) = &decoder.bad {
                bad.update(found.as_mapping())?;
            }
            bad.set_item(PyTuple::empty(py), &whole)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("partial", &whole)?;
            kwargs.set_item("bad", g.mappingproxy.bind(py).call1((bad,))?)?;
            let outer = PyErr::from_value(g.read_error.bind(py).call((message,), Some(&kwargs))?);
            chain(py, &outer, inner);
            return Err(outer);
        }
        Err(error) => return Err(decoder.pyerr(error)),
    };
    if let Some(bad) = &decoder.bad
        && !bad.is_empty()
    {
        let mut items = DictItems::new(bad)?;
        let (at, first) = items.next_item()?.expect("bad is not empty");
        let at = PyList::new(py, at.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let message = format!(
            "{} part(s) do not read; first at {}: {}",
            bad.len(),
            repr(at.as_any())?,
            text(&first.getattr("reason")?)?
        );
        let kwargs = PyDict::new(py);
        kwargs.set_item("partial", &value)?;
        kwargs.set_item("bad", g.mappingproxy.bind(py).call1((bad,))?)?;
        return Err(PyErr::from_value(
            g.read_error.bind(py).call((message,), Some(&kwargs))?,
        ));
    }
    Ok(value)
}
