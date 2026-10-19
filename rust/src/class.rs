//! What is read from a class, natively and with no Python code run: its
//! stamp, its slot layout and its import path; and the table that keeps
//! these facts across calls while the class and the Python state they were
//! read from stand still.

use std::rc::Rc;

use hashbrown::HashMap;
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::encode::Kept;
use crate::py::*;
use crate::sync;

unsafe extern "C" {
    pub fn _PyType_Lookup(
        kind: *mut ffi::PyTypeObject,
        name: *mut ffi::PyObject,
    ) -> *mut ffi::PyObject;
}

// ------------------------------------------------------------------ stamps --

unsafe extern "C" {
    fn PyUnstable_Type_AssignVersionTag(kind: *mut ffi::PyTypeObject) -> std::ffi::c_int;
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
        unsafe {
            if ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_IMMUTABLETYPE) != 0 {
                return None;
            }
            if (*kind).tp_version_tag == 0 {
                // A class out of tags stays at 0, which never matches.
                PyUnstable_Type_AssignVersionTag(kind);
            }
            let qualname = heap_qualname(kind);
            Some(Self {
                tag: (*kind).tp_version_tag,
                qualname: (!qualname.is_null())
                    .then(|| Py::from_borrowed_ptr(class.py(), qualname)),
            })
        }
    }

    /// The class's current state, immutable classes included. A fact kept
    /// across calls by class address needs this: version tags are never
    /// reused, so a new class that takes a dead one's address never matches.
    pub fn always(class: &B<'_>) -> Self {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        unsafe {
            if (*kind).tp_version_tag == 0 {
                PyUnstable_Type_AssignVersionTag(kind);
            }
            let qualname = heap_qualname(kind);
            Self {
                tag: (*kind).tp_version_tag,
                qualname: (!qualname.is_null())
                    .then(|| Py::from_borrowed_ptr(class.py(), qualname)),
            }
        }
    }

    /// A stamp no class holds: for a fact used at once and never kept.
    pub fn none() -> Self {
        Self {
            tag: 0,
            qualname: None,
        }
    }

    /// The same stamp, for a second holder.
    pub fn clone_in(&self, py: Python<'_>) -> Self {
        Self {
            tag: self.tag,
            qualname: self.qualname.as_ref().map(|name| name.clone_ref(py)),
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

// ------------------------------------------------------------------- table --

/// How many classes' facts are kept; past it each new class takes the
/// oldest one's slot.
pub const CLASS_LIMIT: usize = 1 << 12;

/// At most ``CLASS_LIMIT`` classes' facts, each slot reused in turn once the
/// table is full, so no class is kept forever and none is dropped in bulk.
pub struct Classes {
    pub index: HashMap<usize, u32>,
    pub slots: Vec<(usize, ClassInfo)>,
    pub next: usize,
    /// The classes met and not kept, by address, one per slot: a class
    /// pushed out by another is only read again unkept.
    pub met: Box<[usize]>,
}

impl Default for Classes {
    fn default() -> Self {
        Self {
            index: HashMap::new(),
            slots: Vec::new(),
            next: 0,
            met: vec![0; MET_SLOTS].into_boxed_slice(),
        }
    }
}

/// How many classes met once are remembered, so a second sighting is known.
const MET_SLOTS: usize = 1 << 10;

impl Classes {
    /// Whether the class at ``key`` was met before; it is marked met now.
    #[inline]
    pub fn met_before(&mut self, key: usize) -> bool {
        let slot = (key >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - 10);
        let held = &mut self.met[slot];
        if *held == key {
            return true;
        }
        *held = key;
        false
    }

    #[inline]
    pub fn get(&self, key: &usize) -> Option<&ClassInfo> {
        let slot = *self.index.get(key)?;
        Some(&self.slots[slot as usize].1)
    }

    #[inline]
    pub fn get_mut(&mut self, key: &usize) -> Option<&mut ClassInfo> {
        let slot = *self.index.get(key)?;
        Some(&mut self.slots[slot as usize].1)
    }

    /// Put ``info`` under ``key``, replacing its old facts or the oldest
    /// class's; what was replaced is returned, to drop after the write.
    pub fn insert(&mut self, key: usize, info: ClassInfo) -> Option<ClassInfo> {
        if let Some(&slot) = self.index.get(&key) {
            return Some(std::mem::replace(&mut self.slots[slot as usize].1, info));
        }
        if self.slots.len() < CLASS_LIMIT {
            self.index.insert(key, self.slots.len() as u32);
            self.slots.push((key, info));
            return None;
        }
        let slot = self.next;
        self.next = (slot + 1) % CLASS_LIMIT;
        let (old, stale) = std::mem::replace(&mut self.slots[slot], (key, info));
        self.index.remove(&old);
        self.index.insert(key, slot as u32);
        Some(stale)
    }
}

/// What walks learned about one class, each fact read at ``generation``.
pub struct ClassInfo {
    pub stamp: Stamp,
    pub generation: u64,
    pub encoder: Option<Kept>,
    /// The class's own verified import path.
    pub path: Option<Py<PyAny>>,
    /// How its instances' attributes are named and read, unsorted then sorted.
    pub layouts: [Option<Rc<Layout>>; 2],
    /// Whether it reduces as ``Enum.__reduce_ex__`` does, read natively.
    pub enum_member: Option<bool>,
    /// The ``__dict__`` keys of the last instance read, by identity, and the
    /// order sorting gave them; instances of one class usually share both.
    pub dict_order: Option<DictOrder>,
}

/// The keys one instance's ``__dict__`` held, and the order they sort in.
pub struct DictOrder {
    pub keys: Vec<Py<PyAny>>,
    pub sorted: Vec<usize>,
}

/// A class's slots, as ``attribute_names`` names them, and where each lives.
pub struct Layout {
    /// Taken when the offsets were read, so a class changed while an
    /// instance is being read stops them being used mid-object.
    pub stamp: Stamp,
    /// Each slot's name, in output order: the class's own ``ht_slots`` where
    /// that is the order, so no name is copied.
    pub names: Py<PyTuple>,
    /// Each slot's offset in the instance where ``getattr`` would only read
    /// it, in ``names`` order.
    pub offsets: Box<[Option<isize>]>,
    /// When each slot has an offset, no slot name is a tag, and instances
    /// carry no ``__dict__``: the members are then exactly the slots, in order.
    pub direct: bool,
}

/// ``_PyType_Lookup``: the first of the MRO's dicts holding ``name``. Read
/// here dict by dict, as it reads them, without filling the type cache a
/// class seen once never reuses. Borrowed, as the class's dicts hold it.
#[inline]
pub unsafe fn type_lookup(
    kind: *mut ffi::PyTypeObject,
    name: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // A ready class's MRO is a tuple of classes, and an exact str's hash
    // and ``==`` cannot fail. A static builtin type keeps its dict per
    // interpreter rather than in ``tp_dict``, and every MRO ends in one,
    // ``object``: from there ``_PyType_Lookup`` reads on.
    unsafe {
        let mro = (*kind).tp_mro;
        for index in 0..ffi::PyTuple_GET_SIZE(mro) {
            let base = ffi::PyTuple_GET_ITEM(mro, index).cast::<ffi::PyTypeObject>();
            let namespace = (*base).tp_dict;
            if namespace.is_null() {
                break;
            }
            let found = dict_borrowed(namespace, name);
            if !found.is_null() {
                return found;
            }
        }
        _PyType_Lookup(kind, name)
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

impl Layout {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Each slot's name with its offset.
    pub fn slots<'a, 'py>(
        &'a self,
        py: Python<'py>,
    ) -> impl ExactSizeIterator<Item = (B<'py>, Option<isize>)> + 'a
    where
        'py: 'a,
    {
        let names = self.names.bind(py);
        self.offsets.iter().enumerate().map(move |(index, offset)| {
            (
                unsafe { names.get_borrowed_item_unchecked(index) }.to_owned(),
                *offset,
            )
        })
    }

    pub fn new(
        py: Python<'_>,
        stamp: Stamp,
        slots: Vec<(B<'_>, Option<isize>)>,
        direct: bool,
    ) -> PyResult<Self> {
        let (names, offsets): (Vec<_>, Vec<_>) = slots.into_iter().unzip();
        Ok(Self {
            stamp,
            names: PyTuple::new(py, names)?.unbind(),
            offsets: offsets.into_boxed_slice(),
            direct,
        })
    }
}

pub const PY_AUDIT_READ: std::ffi::c_int = 2;

#[inline]
pub fn generic_getattr(kind: *mut ffi::PyTypeObject) -> bool {
    unsafe { (*kind).tp_getattro }.map(|f| f as usize) == generic_getattr_slot()
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
        let found = type_lookup(kind, name.as_ptr());
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
        if (*member).type_code != ffi::Py_T_OBJECT_EX || (*member).flags & refused != 0 {
            return None;
        }
        Some((*member).offset)
    }
}

/// Where ``kind``'s instances hold slot ``name``, if generic attribute lookup
/// would read it straight from there: a plain object member, nothing audited.
pub fn slot_offset(kind: *mut ffi::PyTypeObject, name: &B<'_>) -> Option<isize> {
    member_offset(kind, name, PY_AUDIT_READ)
}

/// Where ``object.__setattr__`` stores ``name`` in ``kind``'s instances: a
/// writable plain object member.
pub fn store_offset(kind: *mut ffi::PyTypeObject, name: &B<'_>) -> Option<isize> {
    member_offset(kind, name, ffi::Py_READONLY)
}

/// Each of ``names``' store offset in ``kind``, where storing through it is
/// what ``object.__setattr__`` would do; stamped before it is read. Empty
/// where ``kind``'s ``__setattr__`` is not ``object``'s.
pub fn store_offsets(kind: &B<'_>, names: &[Box<[u8]>]) -> (Stamp, Vec<Option<isize>>) {
    let py = kind.py();
    let stamp = Stamp::always(kind);
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    let generic = unsafe {
        ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
            && (*class).tp_setattro.map(|f| f as usize)
                == ffi::PyBaseObject_Type.tp_setattro.map(|f| f as usize)
    };
    if !generic {
        return (stamp, Vec::new());
    }
    let offsets = names
        .iter()
        .map(|name| {
            let text = std::str::from_utf8(name).ok()?;
            store_offset(class, PyString::intern(py, text).as_any())
        })
        .collect();
    (stamp, offsets)
}

/// Whether ``kind`` is a heap class under a plain ``type`` metaclass, and
/// not itself a metaclass: one whose reads here answer as ``getattr`` would.
#[inline]
pub fn plain_heap_class(kind: &B<'_>) -> bool {
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    unsafe {
        ob_type(kind.as_ptr()) == &raw mut ffi::PyType_Type
            && ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) != 0
            && ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_TYPE_SUBCLASS) == 0
    }
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

/// Whether ``kind``'s instances have no ``__dict__``: no dict slot, and no
/// ``__dict__`` anywhere in its MRO.
pub fn dictless(kind: *mut ffi::PyTypeObject, py: Python<'_>) -> bool {
    unsafe {
        (*kind).tp_dictoffset == 0
            && ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_MANAGED_DICT) == 0
            && type_lookup(kind, intern!(py, "__dict__").as_ptr()).is_null()
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
            && !_PyType_Lookup(class, name).is_null()
            && _PyType_Lookup(class, name) == _PyType_Lookup(base, name);
        plain.then_some(new)
    }
}

/// The layout of a class ``type`` built from its own ``__slots__`` tuple,
/// straight under ``object``, read in one pass over the end of its dict.
///
/// ``type`` keeps the slot names sorted in ``ht_slots`` (``sorted``'s order)
/// and lays out ``tp_members`` in that order; it adds ``__slots__`` to the
/// namespace, then one member descriptor per name, then ``__weakref__`` when
/// that is a slot. The pass proves each entry is still there, so each member
/// is what ``getattr`` reads, and that no ``__dict__`` slot was made
/// (``ht_slots`` would then hold one name fewer than ``__slots__``). Unsorted,
/// the names keep ``__slots__``' order. ``None`` for any other class, which
/// ``slot_names`` then reads in Python.
/// Whether the dict key ``key`` is the exact str ``name``: the same object,
/// as an interned key is, or the same characters.
///
/// # Safety
/// Both are live; ``name`` is an exact str.
#[inline]
unsafe fn is_name(key: *mut ffi::PyObject, name: *mut ffi::PyObject) -> bool {
    unsafe {
        key == name
            || (ob_type(key) == t_str()
                && utf8_raw(key).is_some_and(|text| Some(text) == utf8_raw(name)))
    }
}

pub fn own_layout(
    class: *mut ffi::PyTypeObject,
    before: &Stamp,
    sort: bool,
    py: Python<'_>,
) -> Option<Layout> {
    unsafe {
        if ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
            return None;
        }
        let names = (*class.cast::<ffi::PyHeapTypeObject>()).ht_slots;
        if names.is_null() {
            return None;
        }
        let mut offsets = vec![None; ffi::PyTuple_GET_SIZE(names) as usize];
        let read = own_offsets(class, sort, py, &mut offsets)?;
        layout_from(py, before.clone_in(py), sort, &read, &offsets[..read.count])
    }
}

/// Up to this many slots are read into buffers on the stack.
pub const STACK_SLOTS: usize = 32;

/// What ``own_offsets`` proved of a class: its ``ht_slots`` names, its
/// ``__slots__`` entry, and how many slots it has.
pub struct OwnSlots {
    pub names: *mut ffi::PyObject,
    declared: *mut ffi::PyObject,
    pub count: usize,
}

/// ``own_layout``'s reading, each slot's offset written to ``offsets`` in
/// ``ht_slots`` order. ``count`` past ``offsets``' length writes nothing and
/// proves nothing; the caller reads the class with room for it.
///
/// # Safety
/// ``class`` is a live class.
pub unsafe fn own_offsets(
    class: *mut ffi::PyTypeObject,
    sort: bool,
    py: Python<'_>,
    offsets: &mut [Option<isize>],
) -> Option<OwnSlots> {
    unsafe {
        let mro = (*class).tp_mro;
        if ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0
            || mro.is_null()
            || ob_type(mro) != t_tuple()
            || ffi::PyTuple_GET_SIZE(mro) != 2
            || ffi::PyTuple_GET_ITEM(mro, 1) != (&raw mut ffi::PyBaseObject_Type).cast()
            || (*class).tp_dictoffset != 0
            || ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_MANAGED_DICT) != 0
        {
            return None;
        }
        let own = (*class).tp_dict;
        let names = (*class.cast::<ffi::PyHeapTypeObject>()).ht_slots;
        let members = (*class).tp_members;
        if names.is_null() || members.is_null() {
            return None;
        }
        let count = ffi::PyTuple_GET_SIZE(names);
        if count as usize > offsets.len() {
            return Some(OwnSlots {
                names,
                declared: std::ptr::null_mut(),
                count: count as usize,
            });
        }
        let used = ffi::PyDict_Size(own);
        let weakref = (*class).tp_weaklistoffset != 0;
        // ``__slots__``, the members, and ``__weakref__`` when it is a slot.
        let tail = count + 1 + ffi::Py_ssize_t::from(weakref);
        // A dict that lost an entry (``dataclass(slots=True)`` rebuilds its
        // namespace) holds its live entries past ``used`` positions, so the
        // tail is found by its first key, ``__slots__``, searched from where
        // a dict with no holes would hold it.
        let slots_key = intern!(py, "__slots__").as_ptr();
        // ``PyDict_Next`` finds nothing from a negative position.
        let mut position = used - tail;
        let mut key = std::ptr::null_mut();
        let mut entry = std::ptr::null_mut();
        loop {
            if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0 {
                return None;
            }
            if is_name(key, slots_key) {
                break;
            }
        }
        // Sorted, ``__slots__`` itself is never read: a ``__dict__`` slot
        // would have set the dict offset refused above, and the pass below
        // proves every name ``ht_slots`` holds, in order. Unsorted, its order
        // is the output's.
        let declared = entry;
        if !sort
            && (ob_type(declared) != t_tuple()
                || ffi::PyTuple_GET_SIZE(declared) != count + ffi::Py_ssize_t::from(weakref))
        {
            return None;
        }
        // The entries first, then each checked: every descriptor is an
        // object a new class has never brought in, and gathering them before
        // reading any lets those loads overlap rather than wait in turn. On
        // the stack: a heap buffer is memory a cold walk last touched long
        // ago, and the allocator misses on it.
        let mut held = [(std::ptr::null_mut(), std::ptr::null_mut()); STACK_SLOTS];
        let mut spilled = Vec::new();
        let entries = if count as usize <= STACK_SLOTS {
            &mut held[..count as usize]
        } else {
            spilled.resize(count as usize, (std::ptr::null_mut(), std::ptr::null_mut()));
            &mut spilled[..]
        };
        for each in entries.iter_mut() {
            if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0 {
                return None;
            }
            *each = (key, entry);
        }
        for (index, (&(key, entry), offset)) in entries.iter().zip(offsets.iter_mut()).enumerate() {
            let index = index as ffi::Py_ssize_t;
            let member = members.offset(index);
            let name = ffi::PyTuple_GET_ITEM(names, index);
            if !is_name(key, name) || ob_type(entry) != &raw mut ffi::PyMemberDescr_Type {
                return None;
            }
            let descriptor = entry.cast::<ffi::PyMemberDescrObject>();
            if (*descriptor).d_common.d_type != class
                || (*descriptor).d_member.cast::<ffi::PyMemberDef>() != member
                || (*member).type_code != ffi::Py_T_OBJECT_EX
                || (*member).flags & PY_AUDIT_READ != 0
            {
                return None;
            }
            let text = utf8_raw(name)?;
            if text.starts_with(b"__") || text.starts_with(b"py/") || text.starts_with(b"json://") {
                return None;
            }
            *offset = Some((*member).offset);
        }
        if weakref
            && (ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) == 0
                || !is_name(key, intern!(py, "__weakref__").as_ptr()))
        {
            return None;
        }
        // Nothing after: an entry added since would shadow no slot, but a
        // dict read this way is one ``type`` left as it built it.
        if ffi::PyDict_Next(own, &mut position, &mut key, &mut entry) != 0 {
            return None;
        }
        Some(OwnSlots {
            names,
            declared,
            count: count as usize,
        })
    }
}

/// The ``Layout`` of what ``own_offsets`` read, in ``__slots__``' order
/// unless ``sort``.
///
/// # Safety
/// ``read`` came from ``own_offsets`` with ``offsets`` written, its class
/// alive and unchanged since.
pub unsafe fn layout_from(
    py: Python<'_>,
    stamp: Stamp,
    sort: bool,
    read: &OwnSlots,
    offsets: &[Option<isize>],
) -> Option<Layout> {
    unsafe {
        let names = Bound::from_borrowed_ptr(py, read.names).cast_into_unchecked::<PyTuple>();
        if sort {
            return Some(Layout {
                stamp,
                names: names.unbind(),
                offsets: offsets.into(),
                direct: true,
            });
        }
        // ``__slots__``' order: each name is one ``ht_slots`` holds, or the
        // ``__weakref__`` slot, which carries no state.
        let declared = Bound::from_borrowed_ptr(py, read.declared).cast_into_unchecked::<PyTuple>();
        let mut order = Vec::with_capacity(read.count);
        let mut kept = Vec::with_capacity(read.count);
        for name in declared.iter() {
            if name.as_ptr() == intern!(py, "__weakref__").as_ptr() {
                continue;
            }
            let at = names.iter().position(|each| each.is(&name))?;
            order.push(name);
            kept.push(offsets[at]);
        }
        Some(Layout {
            stamp,
            names: PyTuple::new(py, order).ok()?.unbind(),
            offsets: kept.into_boxed_slice(),
            direct: true,
        })
    }
}

// ----------------------------------------------------------- import paths --

/// The last module an import path was read from: its name, the
/// ``sys.modules`` generation then, and its namespace.
pub type ModuleMemo<'py> = std::cell::RefCell<Option<(B<'py>, u64, Bound<'py, PyDict>)>>;

/// What the path reader reads besides the class: ``sys.__dict__``, where
/// ``sys.modules`` is bound, and ``types.ModuleType``.
pub struct PathGlobals {
    pub sys_dict: Py<PyDict>,
    pub module_type: Py<PyAny>,
}

/// ``_import_path(kind)`` for a class whose path reads with no Python code:
/// ``f"{__module__}.{__qualname__}"``, verified to resolve to the class
/// itself through plain module and class dicts. ``None`` to ask Python.
pub fn native_import_path_in<'py>(
    py: Python<'py>,
    g: &PathGlobals,
    memo: &ModuleMemo<'py>,
    kind: &B<'py>,
) -> PyResult<Option<B<'py>>> {
    if let Some(path) = memo_path(py, memo, kind) {
        return Ok(Some(path));
    }
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    unsafe {
        if ob_type(kind.as_ptr()) != &raw mut ffi::PyType_Type
            || ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0
        {
            return Ok(None);
        }
    }
    // A heap class's namespace is always an exact dict.
    let own =
        unsafe { Bound::from_borrowed_ptr(py, (*class).tp_dict).cast_into_unchecked::<PyDict>() };
    let Some(module) = own.get_item(intern!(py, "__module__"))? else {
        return Ok(None);
    };
    let qualname = unsafe {
        Bound::from_borrowed_ptr(py, (*class.cast::<ffi::PyHeapTypeObject>()).ht_qualname)
    };
    let (Some(module_text), Some(qual_text)) = (
        (tp(&module) == t_str()).then(|| utf8(&module)).flatten(),
        (tp(&qualname) == t_str())
            .then(|| utf8(&qualname))
            .flatten(),
    ) else {
        return Ok(None);
    };
    if qual_text.contains(&b'<') && qual_text.windows(8).any(|window| window == b"<locals>") {
        return Ok(None);
    }
    // ``utf8`` gives a str's own UTF-8 bytes, valid by construction.
    let (module_text, qual_text) = unsafe {
        (
            std::str::from_utf8_unchecked(module_text),
            std::str::from_utf8_unchecked(qual_text),
        )
    };
    let top_level = !qual_text.as_bytes().contains(&b'.');
    // A top-level class whose module the walk last read, while
    // ``sys.modules`` is unchanged: one namespace read, one str built.
    if top_level
        && let Some(generation) = sync::MODULES.generation()
        && let Some((name, at, namespace)) = &*memo.borrow()
        && name.is(&module)
        && *at == generation
    {
        return Ok(match namespace.get_item(&qualname)? {
            Some(walked) if walked.is(kind) => Some(dotted(py, module_text, qual_text)),
            _ => None,
        });
    }
    let mut path = String::with_capacity(module_text.len() + 1 + qual_text.len());
    path.push_str(module_text);
    path.push('.');
    path.push_str(qual_text);
    let parts: Vec<&str> = if top_level {
        Vec::new()
    } else {
        path.split('.').collect()
    };
    // ``resolve_import`` walks from the longest dotted prefix loaded in
    // ``sys.modules``; it must be exactly the class's own module here.
    // ``sys.modules`` read from ``sys``'s own dict, by an interned key.
    let modules = g.sys_dict.bind(py).get_item(intern!(py, "modules"))?;
    let Some(modules) = modules.and_then(|modules| modules.cast_into_exact::<PyDict>().ok()) else {
        return Ok(None);
    };
    let module_parts = if top_level {
        0
    } else {
        module_text.split('.').count()
    };
    if !top_level {
        for split in (module_parts + 1..parts.len()).rev() {
            if modules.contains(parts[..split].join("."))? {
                return Ok(None);
            }
        }
    }
    let Some(found) = modules.get_item(&module)? else {
        return Ok(None);
    };
    // A module's attributes are its ``__dict__`` entries, unless its class
    // overrides attribute lookup; a class's are its MRO's, read as
    // ``getattr`` reads them when the metaclass is plain ``type``.
    if found.get_type().as_ptr() != g.module_type.as_ptr() {
        return Ok(None);
    }
    // A top-level class: its ``__qualname__`` is the one key to read, and
    // the str already holds its hash.
    if top_level {
        // ``sys.modules[module]``'s namespace, kept once per module per walk
        // while ``sys.modules`` is unchanged.
        // The watcher is registered on first use: ``generation`` is ``None``
        // until ``watch`` has run once.
        let watched = sync::MODULES.watch(py, &modules);
        let namespace = module_dict(&found);
        // Read after watching, so a change from here on is seen.
        if watched && let Some(generation) = sync::MODULES.generation() {
            *memo.borrow_mut() = Some((module.clone(), generation, namespace.clone()));
        }
        return Ok(match namespace.get_item(&qualname)? {
            Some(walked) if walked.is(kind) => Some(dotted(py, module_text, qual_text)),
            _ => None,
        });
    }
    let mut walked = found;
    for part in &parts[module_parts..] {
        let next = if walked.get_type().as_ptr() == g.module_type.as_ptr() {
            module_dict(&walked).get_item(*part)?
        } else if unsafe { ob_type(walked.as_ptr()) } == &raw mut ffi::PyType_Type {
            let name = PyString::new(py, part);
            let found = unsafe { _PyType_Lookup(walked.as_ptr().cast(), name.as_ptr()) };
            // A descriptor found there could answer ``getattr`` itself.
            if found.is_null() {
                return Ok(None);
            }
            let found = unsafe { Bound::from_borrowed_ptr(py, found) };
            if !is_class(&found) {
                return Ok(None);
            }
            Some(found)
        } else {
            return Ok(None);
        };
        let Some(next) = next else {
            return Ok(None);
        };
        walked = next;
    }
    if !walked.is(kind) {
        return Ok(None);
    }
    Ok(Some(PyString::new(py, &path).into_any()))
}

/// An exact ``types.ModuleType``'s namespace, which is always its own dict.
fn module_dict<'py>(module: &B<'py>) -> Bound<'py, PyDict> {
    unsafe {
        Bound::from_borrowed_ptr(module.py(), ffi::PyModule_GetDict(module.as_ptr()))
            .cast_into_unchecked::<PyDict>()
    }
}

/// ``native_import_path_in`` for the common case, on raw pointers: a
/// top-level class of the module the walk last read, while ``sys.modules``
/// is unchanged, with ASCII names. ``None`` leaves every other case, and
/// any doubt, to the full lookup.
fn memo_path<'py>(py: Python<'py>, memo: &ModuleMemo<'py>, kind: &B<'py>) -> Option<B<'py>> {
    let generation = sync::MODULES.generation()?;
    let memo = memo.borrow();
    let (name, at, namespace) = memo.as_ref()?;
    if *at != generation {
        return None;
    }
    let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
    unsafe {
        if ob_type(kind.as_ptr()) != &raw mut ffi::PyType_Type
            || ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) == 0
        {
            return None;
        }
        // The class dict, its qualname and their headers are each a line a
        // new class has never brought in; ask for all of them at once.
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let qualname = (*class.cast::<ffi::PyHeapTypeObject>()).ht_qualname;
            _mm_prefetch::<_MM_HINT_T0>(qualname.cast());
            let own = (*class).tp_dict.cast::<ffi::PyDictObject>();
            _mm_prefetch::<_MM_HINT_T0>(own.cast());
        }
        let own = (*class).tp_dict;
        // A class statement stores ``__module__`` first; the key is the
        // interned str, so a pointer compare finds it.
        let mut position: ffi::Py_ssize_t = 0;
        let mut key = std::ptr::null_mut();
        let mut module = std::ptr::null_mut();
        if ffi::PyDict_Next(own, &mut position, &mut key, &mut module) == 0
            || key != intern!(py, "__module__").as_ptr()
            || module != name.as_ptr()
        {
            return None;
        }
        let qualname = (*class.cast::<ffi::PyHeapTypeObject>()).ht_qualname;
        if ob_type(qualname) != t_str()
            || !compact_ascii_str(qualname)
            || !compact_ascii_str(module)
        {
            return None;
        }
        let qual_text = ascii_bytes(qualname);
        // Top level: no ``.``, and so no ``<locals>`` either.
        if qual_text.contains(&b'.') {
            return None;
        }
        // An exact str key's lookup in an exact dict cannot fail.
        let found = dict_borrowed(namespace.as_ptr(), qualname);
        if found != kind.as_ptr() {
            return None;
        }
        ascii_dotted(py, ascii_bytes(module), qual_text)
    }
}

/// ``f"{module}.{qualname}"``: written straight into a compact str when both
/// are ASCII, else formatted.
fn dotted<'py>(py: Python<'py>, module: &str, qualname: &str) -> B<'py> {
    if module.is_ascii()
        && qualname.is_ascii()
        && let Some(made) = ascii_dotted(py, module.as_bytes(), qualname.as_bytes())
    {
        return made;
    }
    PyString::new(py, &format!("{module}.{qualname}")).into_any()
}

/// ``f"{module}.{qualname}"`` from two ASCII names, written straight into
/// one new compact ASCII str; ``None`` if it cannot be allocated.
fn ascii_dotted<'py>(py: Python<'py>, module: &[u8], qualname: &[u8]) -> Option<B<'py>> {
    let size = module.len() + 1 + qualname.len();
    unsafe {
        let raw = ffi::PyUnicode_New(size as ffi::Py_ssize_t, 127);
        let Some(made) = Bound::from_owned_ptr_or_opt(py, raw) else {
            drop(PyErr::fetch(py));
            return None;
        };
        // A new compact ASCII str's bytes follow its header, NUL-terminated.
        let data = raw.cast::<ffi::PyASCIIObject>().add(1).cast::<u8>();
        std::ptr::copy_nonoverlapping(module.as_ptr(), data, module.len());
        *data.add(module.len()) = b'.';
        std::ptr::copy_nonoverlapping(
            qualname.as_ptr(),
            data.add(module.len() + 1),
            qualname.len(),
        );
        Some(made)
    }
}
