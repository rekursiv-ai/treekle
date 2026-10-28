//! What the encoder reads once per class and keeps while the class is
//! unchanged: which encoder it takes, its attribute layout, its field types,
//! and its import path. A class's stamp (version tag and qualname) decides
//! whether a kept fact still holds; nothing outside the class is watched.

use std::cell::RefCell;
use std::rc::Rc;

use hashbrown::HashMap;
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::class::{PathCheck, Stamp, dict_storage, member_of, type_lookup};
use crate::py::*;

/// Which of ``serialize``'s encoders a class takes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Enc {
    Float,
    Complex,
    Bytes,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
    Proxy,
    Function,
    Type,
    Reduce,
    Path,
    Uuid,
    Datetime,
    Object,
}

/// ``_ENCODERS``: the encoder of each class it names exactly.
pub fn exact_encoder(kind: *mut ffi::PyTypeObject) -> Option<Enc> {
    {
        Some(if kind == &raw mut ffi::PyFloat_Type {
            Enc::Float
        } else if kind == &raw mut ffi::PyList_Type {
            Enc::List
        } else if kind == &raw mut ffi::PyDict_Type {
            Enc::Dict
        } else if kind == &raw mut ffi::PyTuple_Type {
            Enc::Tuple
        } else if kind == &raw mut ffi::PySet_Type {
            Enc::Set
        } else if kind == &raw mut ffi::PyFrozenSet_Type {
            Enc::FrozenSet
        } else if kind == &raw mut ffi::PyComplex_Type {
            Enc::Complex
        } else if kind == &raw mut ffi::PyBytes_Type {
            Enc::Bytes
        } else if kind == &raw mut ffi::PyDictProxy_Type {
            Enc::Proxy
        } else if kind == &raw mut ffi::PyFunction_Type || kind == &raw mut ffi::PyCFunction_Type {
            Enc::Function
        } else {
            return None;
        })
    }
}

/// One member name a class's instances share, with what is read about it
/// once: where a slot holds it, its JSON key text, and its field type.
pub struct Slot {
    pub name: Py<PyString>,
    /// Where the slot's object is held, where ``getattr`` reads it from there.
    pub offset: Option<isize>,
    /// Quoted, where it needs no escape; made when a JSON walk first needs it.
    text: std::cell::OnceCell<Option<Box<[u8]>>>,
    /// Read when a part here first needs it; see ``SlotHints``.
    hint: std::cell::OnceCell<Hint>,
}

impl Slot {
    pub fn text(&self, py: Python<'_>) -> Option<&[u8]> {
        self.text
            .get_or_init(|| crate::json::plain_key_text(self.name.bind(py).as_any()))
            .as_deref()
    }
}

/// A class's slots, or a kept instance dict's keys, in one block; a part
/// that names one holds the block, so evicting the class frees nothing a walk
/// still reads.
pub type Slots = Rc<[Slot]>;

/// ``Slots`` of ``names``, each with its offset.
pub fn slots(names: impl IntoIterator<Item = (Py<PyString>, Option<isize>)>) -> Slots {
    names
        .into_iter()
        .map(|(name, offset)| Slot {
            name,
            offset,
            text: std::cell::OnceCell::new(),
            hint: std::cell::OnceCell::new(),
        })
        .collect()
}

/// A class's state attributes as ``reflect.attribute_names`` reads them, and
/// how to read each one without ``getattr`` where that gives the same.
pub struct Layout {
    /// Every slot, in the order ``slot_names`` gives or by name.
    pub slots: Slots,
    /// Each output position's slot, sorted then unsorted; empty where a
    /// position is its slot's own index.
    orders: [std::cell::OnceCell<Vec<u32>>; 2],
    /// ``__slots__`` as declared, which the unsorted order is read from
    /// when first asked, where ``slots`` holds them by name.
    declared: Option<Py<PyAny>>,
    /// No instance holds ``__dict__`` state.
    pub dictless: bool,
    /// Attribute lookup is ``object``'s own and the instance dict is
    /// ``__dict__``, so a slot, or a dict entry no data descriptor shadows,
    /// is what ``getattr`` gives.
    pub plain: bool,
    /// Every slot is an exact str other than ``py/object``, read at its offset.
    pub plain_slots: bool,
    /// How the last instance dict read was named, sorted then unsorted.
    pub plans: [RefCell<Option<Rc<DictPlan>>>; 2],
    /// For a dataclass, its slots' field types as each is read; ``None``
    /// within where only ``_field_types`` can say.
    pub hints: std::cell::OnceCell<Option<SlotHints>>,
}

impl Layout {
    /// The slot at output position ``index``.
    #[inline]
    pub fn at(&self, py: Python<'_>, sort: bool, index: usize) -> u32 {
        let order = self.order(py, sort);
        if order.is_empty() {
            index as u32
        } else {
            order[index]
        }
    }

    /// Each output position's slot; empty where it is the position itself.
    pub fn order(&self, py: Python<'_>, sort: bool) -> &[u32] {
        self.orders[usize::from(!sort)].get_or_init(|| {
            let Some(declared) = &self.declared else {
                return Vec::new();
            };
            // ``slots`` holds each declared name itself, by name; the
            // ``__weakref__`` slot carries no state.
            let weakref = intern!(py, "__weakref__").as_ptr();
            unsafe { borrowed_items(declared.bind(py)) }
                .iter()
                .filter(|&&name| name != weakref)
                .map(|&name| {
                    self.slots
                        .iter()
                        .position(|each| each.name.as_ptr() == name)
                        .expect("own_layout proved every declared name a slot")
                        as u32
                })
                .collect()
        })
    }

    /// Every output position's slot, in order.
    pub fn positions(&self, py: Python<'_>, sort: bool) -> Vec<u32> {
        (0..self.slots.len())
            .map(|index| self.at(py, sort, index))
            .collect()
    }
}

/// The members of instances whose dicts hold ``keys``, in order, each one an
/// exact str other than ``py/object`` that names no slot and that no data
/// descriptor on the class shadows: so ``getattr`` gives the dict's value.
pub struct DictPlan {
    pub keys: Slots,
    /// Each member in ``attribute_names`` order.
    pub order: Vec<Pick>,
}

#[derive(Clone, Copy)]
pub enum Pick {
    Slot(u32),
    Key(u32),
}

/// A dataclass's field types, as ``_field_types`` gives them.
pub struct Types {
    pub mapping: Py<PyAny>,
    /// Each exact str name to its type, ``object`` included.
    pub by_name: Py<PyDict>,
    /// Each slot's type by ``Layout`` index, ``None`` for ``object``; empty
    /// unless every slot is a key, so every slotted member is within.
    pub slots: Vec<Option<Py<PyAny>>>,
}

impl Types {
    /// ``types.get(name, object)`` for an exact str ``name``: ``None`` where
    /// it is no key, ``Some(None)`` for ``object``.
    #[inline]
    pub fn get<'py>(&self, py: Python<'py>, name: &B<'py>) -> Option<Option<B<'py>>> {
        // An exact dict and an exact str key: the lookup runs no code.
        let found = unsafe { ffi::PyDict_GetItem(self.by_name.as_ptr(), name.as_ptr()) };
        if found.is_null() {
            return None;
        }
        let found = unsafe { Bound::from_borrowed_ptr(py, found) };
        Some((!found.is(py.get_type::<PyAny>())).then_some(found))
    }
}

/// A class or function's verified import path, and the native check that
/// it still names the same object, where there is one.
pub struct PathFact {
    pub path: Py<PyString>,
    pub check: Option<Check>,
}

/// How a kept path is proved to name its object again.
pub enum Check {
    /// A top-level class: ``getattr(sys.modules[module], qualname)``, where
    /// ``qualname`` is the class's own, which its stamp holds to; with the
    /// ``MODULES`` generation it last held at, every dict it read watched.
    Top(Py<PyString>, std::cell::Cell<u64>),
    /// Any other path, walked as ``resolve_import`` walks it.
    Walk(PathCheck),
}

impl Check {
    /// Whether the path names ``value`` now; ``None`` where Python must say.
    pub fn holds(&self, g: &crate::class::PathGlobals, value: &B<'_>) -> PyResult<Option<bool>> {
        match self {
            Self::Walk(check) => check.holds(g, value),
            Self::Top(module, held) => {
                let now = crate::sync::MODULES.generation();
                if now.is_some_and(|now| now == held.get()) {
                    return Ok(Some(true));
                }
                let Some(namespace) = g.module_namespace(module.bind(value.py()))? else {
                    return Ok(None);
                };
                let class = value.as_ptr().cast::<ffi::PyTypeObject>();
                let qualname = unsafe { (*class.cast::<ffi::PyHeapTypeObject>()).ht_qualname };
                let entry = unsafe { g.module_entry(namespace.as_ptr(), qualname) };
                if entry.is_null() {
                    return Ok(None);
                }
                let holds = entry == value.as_ptr();
                // Kept once the namespace read is watched, as ``module_namespace``
                // watches what it keeps.
                if holds
                    && let Some(now) = now
                    && g.last
                        .borrow()
                        .as_ref()
                        .is_some_and(|(_, at, _)| *at == now)
                {
                    held.set(now);
                }
                Ok(Some(holds))
            }
        }
    }
}

impl Facts {
    /// No facts yet about ``kind``, stamped as it is now.
    pub fn new(kind: &B<'_>) -> Self {
        Self {
            stamp: Stamp::always(kind),
            enc: None,
            dataclass: None,
            layout: None,
            types: None,
            path: None,
            leaf: None,
        }
    }
}

/// Dicts, each with its size when a fact was read: the fact holds while
/// none gains or loses a name.
pub type Guard = Vec<(Py<PyDict>, isize)>;

/// Everything kept about one class.
pub struct Facts {
    pub stamp: Stamp,
    pub enc: Option<Enc>,
    pub dataclass: Option<bool>,
    pub layout: Option<Rc<Layout>>,
    /// With the namespaces it holds while each keeps its size; none for good.
    pub types: Option<(Rc<Types>, Guard)>,
    pub path: Option<Rc<PathFact>>,
    pub leaf: Option<Leaf>,
}

/// What a declared type may write a class's instances as: text it reads
/// back (``_untagged_leaf``), or nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Leaf {
    None,
    Enum,
    Datetime,
    Text,
}

/// Classes' facts by address; a stamp that no longer holds is read again.
///
/// The classes form a ring of ``TABLE_LIMIT``; past it each new class takes
/// the oldest one's place, so none is kept forever and none is dropped in
/// bulk. Facts replaced in a walk are dropped when it ends, as dropping one
/// may run Python code.
pub struct Table {
    classes: Vec<(usize, Facts)>,
    index: HashMap<usize, u32>,
    /// The ring place the next kept class takes once the ring is full.
    next: usize,
    /// The entry last found, which the next lookup most often wants.
    last: (usize, u32),
    stale: Vec<Facts>,
    /// Functions' and other objects' paths, each kept with its object so its
    /// address is never reused while the entry stands.
    pub paths: Bounded<usize, (Py<PyAny>, Rc<PathFact>)>,
    /// Builtins whose own path failed, with the ``sys.modules`` generation
    /// then; each kept with its object, as ``paths`` keeps them.
    pub unpathed: Bounded<usize, (Py<PyAny>, u64)>,
}

/// How many classes' facts are kept.
pub const TABLE_LIMIT: usize = 1 << 14;

/// How many functions' and globals' paths are kept.
const PATH_LIMIT: usize = 1 << 12;

/// ``last`` before any lookup.
const NONE: u32 = u32::MAX;

impl Default for Table {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
            index: HashMap::new(),
            next: 0,
            last: (0, NONE),
            stale: Vec::new(),
            paths: Bounded::new(PATH_LIMIT),
            unpathed: Bounded::new(PATH_LIMIT),
        }
    }
}

impl Table {
    /// The facts of ``kind``, fresh ones where none hold.
    #[inline]
    pub fn of(&mut self, kind: &B<'_>) -> &mut Facts {
        let address = kind.as_ptr() as usize;
        let at = match self.find(address) {
            Some(at) if self.facts_at(at).stamp.holds(kind) => at,
            found => self.hold(address, found, Facts::new(kind)),
        };
        self.last = (address, at);
        self.facts_at(at)
    }

    /// Where the facts kept for ``address`` stand, stamp unchecked.
    #[inline]
    fn find(&self, address: usize) -> Option<u32> {
        if self.last.0 == address && self.last.1 != NONE {
            return Some(self.last.1);
        }
        self.index.get(&address).copied()
    }

    #[inline]
    fn facts_at(&mut self, at: u32) -> &mut Facts {
        &mut self.classes[at as usize].1
    }

    /// Hold ``facts`` for the class at ``address``, in place of any it had
    /// at ``found``; where it stands now.
    fn hold(&mut self, address: usize, found: Option<u32>, facts: Facts) -> u32 {
        let Some(at) = found else {
            return self.keep(address, facts);
        };
        let replaced = std::mem::replace(self.facts_at(at), facts);
        self.stale.push(replaced);
        at
    }

    /// Keep ``facts`` in the ring, in the oldest class's place once full.
    fn keep(&mut self, address: usize, facts: Facts) -> u32 {
        if self.classes.len() < TABLE_LIMIT {
            let at = self.classes.len() as u32;
            self.classes.push((address, facts));
            self.index.insert(address, at);
            return at;
        }
        let at = self.next;
        self.next = (at + 1) % TABLE_LIMIT;
        let (old, replaced) = std::mem::replace(&mut self.classes[at], (address, facts));
        self.index.remove(&old);
        self.index.insert(address, at as u32);
        self.stale.push(replaced);
        at as u32
    }

    /// Drop what the walk replaced, once it has ended.
    pub fn settle(&mut self) {
        self.stale.clear();
    }
}

/// ``reflect.slot_names(kind)`` read natively, or ``None`` where Python must
/// read it: a ``__slots__`` that is not a str or an exact tuple or list of
/// exact strs, a private slot (whose mangling Python works out), or a base
/// whose ``vars`` a metaclass answers.
pub fn slot_names(py: Python<'_>, kind: &B<'_>) -> Option<Vec<Py<PyString>>> {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    let mut names: Vec<Py<PyString>> = Vec::new();
    unsafe {
        let mro = (*class).tp_mro;
        if mro.is_null() {
            return None;
        }
        let type_dict = type_lookup(&raw mut ffi::PyType_Type, intern!(py, "__dict__").as_ptr());
        for index in 0..ffi::PyTuple_GET_SIZE(mro) {
            let base = ffi::PyTuple_GET_ITEM(mro, index);
            let meta = ob_type(base);
            if meta != &raw mut ffi::PyType_Type
                && type_lookup(meta, intern!(py, "__dict__").as_ptr()) != type_dict
            {
                return None;
            }
            let own = Bound::from_owned_ptr(py, ffi::PyType_GetDict(base.cast()));
            let slots = ffi::PyDict_GetItem(own.as_ptr(), intern!(py, "__slots__").as_ptr());
            if slots.is_null() {
                continue;
            }
            let slots = Bound::from_borrowed_ptr(py, slots);
            let one = [slots.as_ptr()];
            let declared: &[*mut ffi::PyObject] = if tp(&slots) == t_str() {
                &one
            } else if tp(&slots) == t_tuple() || tp(&slots) == t_list() {
                borrowed_items(&slots)
            } else {
                return None;
            };
            // Names repeat only across bases, never within one.
            let earlier = names.len();
            for &slot in declared {
                let slot = Bound::from_borrowed_ptr(py, slot);
                if tp(&slot) != t_str() {
                    return None;
                }
                let text = utf8(&slot)?;
                if text == b"__weakref__" || text == b"__dict__" {
                    continue;
                }
                if text.starts_with(b"__") && !text.ends_with(b"__") {
                    return None;
                }
                let seen = names[..earlier].iter().any(|known| {
                    known.as_ptr() == slot.as_ptr()
                        || ffi::PyUnicode_Compare(known.as_ptr(), slot.as_ptr()) == 0
                });
                if !seen {
                    names.push(slot.cast_into_unchecked::<PyString>().unbind());
                }
            }
        }
    }
    Some(names)
}

/// ``kind``'s ``Layout`` over slot ``names``. ``dict_getter`` is the getter
/// of a plain class's ``__dict__`` descriptor.
pub fn layout(
    py: Python<'_>,
    kind: &B<'_>,
    names: Vec<Py<PyString>>,
    dict_getter: Option<usize>,
) -> Layout {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    let generic = unsafe { (*class).tp_getattro }.map(|f| f as usize) == generic_getattr_slot();
    let offset = |name: &Py<PyString>| {
        if !generic {
            return None;
        }
        unsafe {
            let member = member_of(class, type_lookup(class, name.as_ptr()));
            if member.is_null()
                || (*member).type_code != ffi::Py_T_OBJECT_EX
                || (*member).flags & crate::class::PY_AUDIT_READ != 0
            {
                return None;
            }
            Some((*member).offset)
        }
    };
    let slots = slots(names.into_iter().map(|name| {
        let offset = offset(&name);
        (name, offset)
    }));
    let mut sorted: Vec<u32> = (0..slots.len() as u32).collect();
    sorted.sort_by(|&a, &b| unsafe {
        let (a, b) = (&slots[a as usize].name, &slots[b as usize].name);
        ffi::PyUnicode_Compare(a.as_ptr(), b.as_ptr()).cmp(&0)
    });
    let orders = [
        std::cell::OnceCell::from(sorted),
        std::cell::OnceCell::from(Vec::new()),
    ];
    let dict_name = intern!(py, "__dict__").as_ptr();
    let dictless = !dict_storage(class) && unsafe { type_lookup(class, dict_name).is_null() };
    let plain_dict = dictless
        || unsafe {
            let found = type_lookup(class, dict_name);
            !found.is_null()
                && ffi::Py_TYPE(found) == &raw mut ffi::PyGetSetDescr_Type
                && (*(*found.cast::<ffi::PyGetSetDescrObject>()).d_getset)
                    .get
                    .map(|f| f as usize)
                    == dict_getter
        };
    let plain_slots = generic
        && slots.iter().all(|slot| {
            slot.offset.is_some()
                && utf8(slot.name.bind(py).as_any()).is_some_and(|text| text != b"py/object")
        });
    Layout {
        slots,
        orders,
        declared: None,
        dictless,
        plain: generic && plain_dict,
        plain_slots,
        plans: [RefCell::new(None), RefCell::new(None)],
        hints: std::cell::OnceCell::new(),
    }
}

/// What ``plain_field_types`` reads hints with.
pub struct HintGlobals {
    /// ``type.__dict__["__annotations__"]``: a class's own annotations.
    pub annotations: Py<PyAny>,
    pub union_type: Py<PyAny>,
    pub generic_alias: Py<PyAny>,
    /// ``dataclasses._FIELD``: a regular field's ``_field_type``.
    pub field: Py<PyAny>,
    pub alias: AliasParts,
}

/// ``deserialize._field_types(kind)`` for a dataclass whose every
/// annotation is a plain hint (``_plain_hints``), read with no Python code
/// of treekle's run; ``None`` where Python must read it.
pub fn plain_field_types<'py>(
    py: Python<'py>,
    g: &HintGlobals,
    kind: &B<'py>,
    slots: &[Slot],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if no_type_check(py, kind)? {
        return Ok(None);
    }
    let mro = unsafe {
        Bound::from_borrowed_ptr(py, (*kind.as_ptr().cast::<ffi::PyTypeObject>()).tp_mro)
    };
    // Each heap base's own annotations, most derived first.
    let mut sources: Vec<Bound<'py, PyDict>> = Vec::new();
    for &base in unsafe { borrowed_items(&mro) } {
        if unsafe { ffi::PyType_HasFeature(base.cast(), ffi::Py_TPFLAGS_HEAPTYPE) } == 0 {
            continue;
        }
        let Some(own) = own_annotations(py, g, &unsafe { Bound::from_borrowed_ptr(py, base) })
        else {
            return Ok(None);
        };
        let mut position = 0;
        let mut key = std::ptr::null_mut();
        let mut hint = std::ptr::null_mut();
        while unsafe { ffi::PyDict_Next(own.as_ptr(), &mut position, &mut key, &mut hint) } != 0 {
            if !plain_hint(py, g, &unsafe { Bound::from_borrowed_ptr(py, hint) })? {
                return Ok(None);
            }
        }
        sources.push(own);
    }
    let fields_name = intern!(py, "__dataclass_fields__").as_ptr();
    let crate::class::ClassAttr::Found(fields, _) =
        (unsafe { crate::class::class_attr(kind.as_ptr(), fields_name) })
    else {
        return Ok(None);
    };
    let Ok(fields) = unsafe { B::from_borrowed_ptr(py, fields) }.cast_into_exact::<PyDict>() else {
        return Ok(None);
    };
    let object = py.get_type::<PyAny>().into_any();
    let none = py.None().into_bound(py).get_type().into_any();
    let types = PyDict::new(py);
    // ``get_type_hints`` reads ``None`` as its class, and every other plain
    // hint as one equal to it, so the annotation itself serves.
    let add = |name: *mut ffi::PyObject| -> PyResult<()> {
        let mut hint = object.as_ptr();
        for own in &sources {
            let found = unsafe { ffi::PyDict_GetItem(own.as_ptr(), name) };
            if !found.is_null() {
                hint = if found == unsafe { ffi::Py_None() } {
                    none.as_ptr()
                } else {
                    found
                };
                break;
            }
        }
        if unsafe { ffi::PyDict_SetItem(types.as_ptr(), name, hint) } < 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(())
    };
    let field_type = intern!(py, "_field_type");
    let mut position = 0;
    let mut name = std::ptr::null_mut();
    let mut field = std::ptr::null_mut();
    while unsafe { ffi::PyDict_Next(fields.as_ptr(), &mut position, &mut name, &mut field) } != 0 {
        let field = unsafe { Bound::from_borrowed_ptr(py, field) };
        if field.getattr(field_type)?.is(g.field.bind(py)) {
            add(name)?;
        }
    }
    for slot in slots {
        let name = slot.name.as_ptr();
        if unsafe { ffi::PyDict_Contains(types.as_ptr(), name) } == 0 {
            add(name)?;
        }
    }
    Ok(Some(types))
}

/// ``_is_plain_hint``: whether ``hint`` is its own evaluated,
/// ``Annotated``-free form.
fn plain_hint(py: Python<'_>, g: &HintGlobals, hint: &B<'_>) -> PyResult<bool> {
    if hint.is_none() || hint.is(py.Ellipsis()) || tp(hint) == &raw mut ffi::PyType_Type {
        return Ok(true);
    }
    let kind = hint.get_type();
    let union = kind.is(g.union_type.bind(py));
    let generic = kind.is(g.generic_alias.bind(py));
    if !(union || generic) {
        return Ok(false);
    }
    if generic && g.alias.unpacked(hint)? {
        return Ok(false);
    }
    let args = if generic {
        g.alias.args(hint)?
    } else {
        g.alias.union_args(hint)?
    };
    if tp(&args) != t_tuple() {
        return Ok(false);
    }
    for &arg in unsafe { borrowed_items(&args) } {
        if !plain_hint(py, g, &unsafe { Bound::from_borrowed_ptr(py, arg) })? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// What ``record_built`` reads ``from_plain``'s dispatch with.
pub struct RecordGlobals {
    /// ``deserialize._DECODERS``, the table ``_dispatch`` returns.
    pub dispatch: Py<PyAny>,
    pub union_type: Py<PyAny>,
    pub enum_type: Py<PyAny>,
    pub pure_path: Py<PyAny>,
}

/// ``serialize._record_built(declared, {})``, read natively: ``Some(None)``
/// where no record goes untagged, ``Some(Some(class))`` for the record
/// class that does, ``None`` where only Python can say. A plain dataclass
/// builds itself; a union builds its first plain dataclass where only
/// ``None`` precedes it and only plain classes but ``dict`` follow, which
/// ``_union_candidates`` would not move ahead of it.
pub fn record_built<'py>(
    py: Python<'py>,
    g: &RecordGlobals,
    alias: &AliasParts,
    declared: &B<'py>,
) -> PyResult<Option<Option<B<'py>>>> {
    if tp(declared) == &raw mut ffi::PyType_Type {
        return Ok(if plain_record(py, g, declared)? {
            Some(Some(declared.clone()))
        } else {
            None
        });
    }
    // A builtin generic builds its own class, which is no record.
    if tp(declared) == alias_type(py) {
        let origin = alias.origin(declared)?;
        let builtin = tp(&origin) == &raw mut ffi::PyType_Type
            && unsafe { ffi::PyType_HasFeature(origin.as_ptr().cast(), ffi::Py_TPFLAGS_HEAPTYPE) }
                == 0
            && g.dispatch.bind(py).contains(&origin)?;
        return Ok(builtin.then_some(None));
    }
    if !declared.get_type().is(g.union_type.bind(py)) {
        return Ok(None);
    }
    let none = py.None().into_bound(py).get_type().into_any();
    let mut first = None;
    let args = alias.union_args(declared)?;
    if tp(&args) != t_tuple() {
        return Ok(None);
    }
    for &member in unsafe { borrowed_items(&args) } {
        let member = unsafe { Bound::from_borrowed_ptr(py, member) };
        if member.is(&none) {
            continue;
        }
        if first.is_none() {
            if !plain_record(py, g, &member)? {
                return Ok(None);
            }
            first = Some(member);
            continue;
        }
        let stays = tp(&member) == &raw mut ffi::PyType_Type
            && tp_ptr(&member) != &raw mut ffi::PyDict_Type
            && unsafe { type_lookup(member.as_ptr().cast(), intern!(py, "__value__").as_ptr()) }
                .is_null();
        if !stays {
            return Ok(None);
        }
    }
    Ok(Some(first))
}

/// ``types.GenericAlias``.
fn alias_type(py: Python<'_>) -> *mut ffi::PyTypeObject {
    static ALIAS: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    ALIAS
        .get_or_init(py, || {
            py.import("types")
                .and_then(|types| types.getattr("GenericAlias"))
                .expect("types.GenericAlias exists")
                .unbind()
        })
        .as_ptr()
        .cast()
}

/// ``member`` as a class pointer.
fn tp_ptr(member: &B<'_>) -> *mut ffi::PyTypeObject {
    member.as_ptr().cast()
}

/// Whether ``member`` is a dataclass ``from_plain`` reads by
/// ``_decode_object``: an exact ``type`` instance, no alias, no decoder of
/// its own, neither an enum nor a path.
fn plain_record(py: Python<'_>, g: &RecordGlobals, member: &B<'_>) -> PyResult<bool> {
    if tp(member) != &raw mut ffi::PyType_Type {
        return Ok(false);
    }
    let class = tp_ptr(member);
    let has = |name: &Bound<'_, PyString>| unsafe { !type_lookup(class, name.as_ptr()).is_null() };
    let subclass =
        |base: &Py<PyAny>| unsafe { ffi::PyType_IsSubtype(class, base.as_ptr().cast()) != 0 };
    Ok(has(intern!(py, "__dataclass_fields__"))
        && !has(intern!(py, "__value__"))
        && !subclass(&g.enum_type)
        && !subclass(&g.pure_path)
        && !g.dispatch.bind(py).contains(member)?)
}

/// What one slot's field type is, read from the class's annotations.
pub enum Hint {
    /// ``object``: nothing declared.
    Object,
    Type(Py<PyAny>),
    /// Not a plain hint: ``_field_types`` must evaluate it.
    Ask,
}

/// A slotted dataclass's field types, by slot, read from each heap base's
/// own annotations, most derived first. ``_field_types`` names every slot,
/// so its keys never need reading here.
pub struct SlotHints {
    /// The class's own annotations, then each heap base's, if any.
    own: Py<PyDict>,
    inherited: Vec<Py<PyDict>>,
}

impl SlotHints {
    /// Each of ``slots``' hints as ``_plain_hints`` reads it, or ``None``
    /// where Python must read the field types.
    pub fn read(py: Python<'_>, g: &HintGlobals, kind: &B<'_>) -> PyResult<Option<Self>> {
        if no_type_check(py, kind)? {
            return Ok(None);
        }
        let mro = unsafe {
            Bound::from_borrowed_ptr(py, (*kind.as_ptr().cast::<ffi::PyTypeObject>()).tp_mro)
        };
        let mut own = None;
        let mut inherited = Vec::new();
        for &base in unsafe { borrowed_items(&mro) } {
            if unsafe { ffi::PyType_HasFeature(base.cast(), ffi::Py_TPFLAGS_HEAPTYPE) } == 0 {
                continue;
            }
            let Some(found) =
                own_annotations(py, g, &unsafe { Bound::from_borrowed_ptr(py, base) })
            else {
                return Ok(None);
            };
            if own.is_none() {
                own = Some(found.unbind());
            } else {
                inherited.push(found.unbind());
            }
        }
        Ok(own.map(|own| Self { own, inherited }))
    }

    /// The field type of ``slot``, one of the class's.
    pub fn get(&self, py: Python<'_>, g: &HintGlobals, slot: &Slot) -> PyResult<Hint> {
        let hint = match slot.hint.get() {
            Some(hint) => hint,
            None => {
                let read = self.read_hint(py, g, &slot.name)?;
                slot.hint.get_or_init(|| read)
            }
        };
        Ok(match hint {
            Hint::Object => Hint::Object,
            Hint::Type(hint) => Hint::Type(hint.clone_ref(py)),
            Hint::Ask => Hint::Ask,
        })
    }

    fn read_hint(&self, py: Python<'_>, g: &HintGlobals, name: &Py<PyString>) -> PyResult<Hint> {
        for own in std::iter::once(&self.own).chain(&self.inherited) {
            let found = unsafe { ffi::PyDict_GetItem(own.as_ptr(), name.as_ptr()) };
            if found.is_null() {
                continue;
            }
            let found = unsafe { Bound::from_borrowed_ptr(py, found) };
            return Ok(if !plain_hint(py, g, &found)? {
                Hint::Ask
            } else if found.is_none() {
                Hint::Type(found.get_type().into_any().unbind())
            } else if found.is(py.get_type::<PyAny>()) {
                Hint::Object
            } else {
                Hint::Type(found.unbind())
            });
        }
        Ok(Hint::Object)
    }
}

/// ``inspect.get_annotations(base)`` for a heap class: its own
/// annotations, an exact dict, or ``None`` where Python must read them.
fn own_annotations<'py>(
    py: Python<'py>,
    g: &HintGlobals,
    base: &B<'py>,
) -> Option<Bound<'py, PyDict>> {
    // ``type``'s ``__annotations__`` getter answers from the class's own
    // ``__annotations__``, else its ``__annotations_cache__``, before it
    // would call ``__annotate__``: an exact dict found there is the answer.
    if ob_type(base.as_ptr()) == &raw mut ffi::PyType_Type {
        let namespace = unsafe { (*base.as_ptr().cast::<ffi::PyTypeObject>()).tp_dict };
        for name in [
            intern!(py, "__annotations__"),
            intern!(py, "__annotations_cache__"),
        ] {
            let found = unsafe { ffi::PyDict_GetItem(namespace, name.as_ptr()) };
            if !found.is_null() {
                if ob_type(found) == t_dict() {
                    return Some(unsafe {
                        Bound::from_borrowed_ptr(py, found).cast_into_unchecked()
                    });
                }
                break;
            }
        }
    }
    // ``type``'s own ``__annotations__`` is what a plain class answers;
    // another metaclass's may answer first, so its descriptor is asked.
    let own = if ob_type(base.as_ptr()) == &raw mut ffi::PyType_Type {
        base.getattr(intern!(py, "__annotations__"))
    } else {
        g.annotations
            .bind(py)
            .call_method1(intern!(py, "__get__"), (base, base.get_type()))
    };
    own.ok()?.cast_into_exact::<PyDict>().ok()
}

/// ``getattr(kind, "__no_type_check__", None)`` is truthy: read through
/// the type attribute cache where ``kind``'s metaclass is ``type``, whose own
/// namespace holds no such name, and the value found is ``True`` or ``False``.
fn no_type_check(py: Python<'_>, kind: &B<'_>) -> PyResult<bool> {
    let name = intern!(py, "__no_type_check__");
    if ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type {
        let found = unsafe { type_lookup(kind.as_ptr().cast(), name.as_ptr()) };
        if found.is_null() {
            return Ok(false);
        }
        if ob_type(found) == t_bool() {
            return Ok(found == unsafe { ffi::Py_True() });
        }
    }
    match optional_attr(kind, name)? {
        Some(flag) => flag.is_truthy(),
        None => Ok(false),
    }
}

/// ``layout`` for the common class read with no dict lookup: a heap class
/// straight under ``object``, no instance dict, generic attribute lookup,
/// and its namespace as ``type`` left it, ``__slots__`` then each slot's
/// member descriptor in ``ht_slots`` order (as ``dataclass(slots=True)``
/// builds it). ``ht_slots`` is sorted, so it is the sorted order. ``None``
/// for any other class, which ``layout`` reads.
pub fn own_layout(py: Python<'_>, kind: &B<'_>) -> Option<Layout> {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    unsafe {
        let mro = (*class).tp_mro;
        if ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0
            || mro.is_null()
            || ffi::PyTuple_GET_SIZE(mro) != 2
            || ffi::PyTuple_GET_ITEM(mro, 1) != (&raw mut ffi::PyBaseObject_Type).cast()
            || dict_storage(class)
            || (*class).tp_getattro.map(|f| f as usize) != generic_getattr_slot()
        {
            return None;
        }
        let own = (*class).tp_dict;
        let sorted_names = (*class.cast::<ffi::PyHeapTypeObject>()).ht_slots;
        let members = (*class).tp_members;
        if sorted_names.is_null() || members.is_null() {
            return None;
        }
        let count = ffi::PyTuple_GET_SIZE(sorted_names);
        let weakref = (*class).tp_weaklistoffset != 0;
        // The namespace ends with ``__slots__``, the members, and the
        // ``__weakref__`` slot if there is one. A dict that lost an entry
        // holds its live ones past ``used`` positions, so ``__slots__`` is
        // searched from where a dict with no holes would hold it.
        let tail = count + 1 + ffi::Py_ssize_t::from(weakref);
        let mut position = ffi::PyDict_Size(own) - tail;
        let mut key = std::ptr::null_mut();
        let mut entry = std::ptr::null_mut();
        let slots_key = intern!(py, "__slots__").as_ptr();
        loop {
            if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0 {
                return None;
            }
            if same_name(key, slots_key) {
                break;
            }
        }
        let declared = entry;
        if ob_type(declared) != t_tuple()
            || ffi::PyTuple_GET_SIZE(declared) != count + ffi::Py_ssize_t::from(weakref)
        {
            return None;
        }
        for index in 0..count {
            if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0 {
                return None;
            }
            let member = members.offset(index);
            let name = ffi::PyTuple_GET_ITEM(sorted_names, index);
            if !same_name(key, name) || ob_type(entry) != &raw mut ffi::PyMemberDescr_Type {
                return None;
            }
            let descriptor = entry.cast::<ffi::PyMemberDescrObject>();
            if (*descriptor).d_common.d_type != class
                || (*descriptor).d_member.cast::<ffi::PyMemberDef>() != member
                || (*member).type_code != ffi::Py_T_OBJECT_EX
                || (*member).flags & crate::class::PY_AUDIT_READ != 0
            {
                return None;
            }
            // A private slot is mangled in ``ht_slots`` but not in
            // ``__slots__``; a tag-like one writes otherwise.
            let text = utf8_ptr(name)?;
            if text.starts_with(b"__") || text.starts_with(b"py/") || text.starts_with(b"json://") {
                return None;
            }
        }
        if weakref
            && (ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0
                || !same_name(key, intern!(py, "__weakref__").as_ptr()))
        {
            return None;
        }
        if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) != 0 {
            return None;
        }
        // Each declared name is one ``ht_slots`` holds itself, its order
        // the unsorted output's: a private one, mangled there, is refused.
        for &name in borrowed_items(&Bound::from_borrowed_ptr(py, declared)) {
            if same_name(name, intern!(py, "__weakref__").as_ptr()) {
                continue;
            }
            if utf8_ptr(name)?.starts_with(b"__") {
                return None;
            }
        }
        let slots = slots((0..count).map(|index| {
            let name =
                Py::<PyString>::from_borrowed_ptr(py, ffi::PyTuple_GET_ITEM(sorted_names, index));
            (name, Some((*members.offset(index)).offset))
        }));
        Some(Layout {
            slots,
            orders: [
                std::cell::OnceCell::from(Vec::new()),
                std::cell::OnceCell::new(),
            ],
            declared: Some(Py::from_borrowed_ptr(py, declared)),
            dictless: true,
            plain: true,
            plain_slots: true,
            plans: [RefCell::new(None), RefCell::new(None)],
            hints: std::cell::OnceCell::new(),
        })
    }
}

/// Whether ``key`` is the exact str ``name``, by identity or by text.
unsafe fn same_name(key: *mut ffi::PyObject, name: *mut ffi::PyObject) -> bool {
    unsafe {
        key == name
            || (ob_type(key) == t_str()
                && utf8_ptr(key).is_some_and(|text| Some(text) == utf8_ptr(name)))
    }
}

/// Where ``types.GenericAlias`` and ``types.UnionType`` hold the parts a
/// hint is read by, so they are read as ``getattr`` would read those member
/// descriptors, with no lookup: ``GenericAlias.__getattribute__`` checks a
/// list of names before every one.
pub struct AliasParts {
    origin: Option<isize>,
    args: Option<isize>,
    unpacked: Option<isize>,
    union_args: Option<isize>,
}

/// ``_Py_T_OBJECT``: a member read as ``None`` where unset.
const T_OBJECT: std::ffi::c_int = 6;

impl AliasParts {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let types = py.import("types")?;
        let member = |kind: &str, name: &str, code: std::ffi::c_int| -> PyResult<Option<isize>> {
            let found = types.getattr(kind)?.getattr("__dict__")?.get_item(name)?;
            if ob_type(found.as_ptr()) != &raw mut ffi::PyMemberDescr_Type {
                return Ok(None);
            }
            let def = unsafe {
                (*found.as_ptr().cast::<ffi::PyMemberDescrObject>())
                    .d_member
                    .cast::<ffi::PyMemberDef>()
            };
            Ok(unsafe { ((*def).type_code == code).then_some((*def).offset) })
        };
        Ok(Self {
            origin: member("GenericAlias", "__origin__", T_OBJECT)?,
            args: member("GenericAlias", "__args__", T_OBJECT)?,
            unpacked: member("GenericAlias", "__unpacked__", ffi::Py_T_BOOL)?,
            union_args: member("UnionType", "__args__", T_OBJECT)?,
        })
    }

    /// A ``GenericAlias``'s ``__origin__``.
    pub fn origin<'py>(&self, hint: &B<'py>) -> PyResult<B<'py>> {
        read(hint, self.origin, intern!(hint.py(), "__origin__"))
    }

    /// A ``GenericAlias``'s ``__args__``.
    pub fn args<'py>(&self, hint: &B<'py>) -> PyResult<B<'py>> {
        read(hint, self.args, intern!(hint.py(), "__args__"))
    }

    /// A ``UnionType``'s ``__args__``.
    pub fn union_args<'py>(&self, hint: &B<'py>) -> PyResult<B<'py>> {
        read(hint, self.union_args, intern!(hint.py(), "__args__"))
    }

    /// A ``GenericAlias``'s ``__unpacked__``.
    pub fn unpacked(&self, hint: &B<'_>) -> PyResult<bool> {
        match self.unpacked {
            Some(offset) => Ok(unsafe { *hint.as_ptr().cast::<u8>().offset(offset) } != 0),
            None => hint
                .getattr(intern!(hint.py(), "__unpacked__"))?
                .is_truthy(),
        }
    }
}

/// ``getattr(hint, name)`` for the object member at ``offset``.
fn read<'py>(
    hint: &B<'py>,
    offset: Option<isize>,
    name: &Bound<'py, PyString>,
) -> PyResult<B<'py>> {
    let Some(offset) = offset else {
        return hint.getattr(name);
    };
    let raw = unsafe {
        *hint
            .as_ptr()
            .cast::<u8>()
            .offset(offset)
            .cast::<*mut ffi::PyObject>()
    };
    let py = hint.py();
    Ok(if raw.is_null() {
        py.None().into_bound(py)
    } else {
        unsafe { Bound::from_borrowed_ptr(py, raw) }
    })
}
