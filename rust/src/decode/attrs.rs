//! Attribute writes and probes resolved once per class, as CPython would
//! resolve them on every call while the class is unchanged.
//!
//! CPython gives a class a new version tag whenever it or a base changes,
//! and never reuses one; its own attribute cache trusts a lookup for as long
//! as the tag holds. The tables here do the same: each is kept with the tag
//! of the class it was read from, and answers only while that tag holds.

use std::ffi::{c_char, c_int};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::class::{dict_storage, member_of, type_lookup, version_tag};
use crate::py::*;

/// ``PyMemberDef.flags`` bit that makes a read raise an audit event.
const PY_AUDIT_READ: c_int = 2;
/// ``PyMemberDef.flags`` bit for an offset not yet made absolute.
const PY_RELATIVE_OFFSET: c_int = 8;

// Perf: exp000 from_plain, paired: SetTable, HasTable, FieldSlots and HintGuard together
// 1.18-1.21x.
/// How ``object.__setattr__(value, name, member)`` lands for one class.
pub struct SetTable {
    tag: u32,
    /// Each name met, with the member descriptor's slot it sets, or null
    /// where the generic path decides.
    names: Vec<(Py<PyString>, *mut ffi::PyMemberDef)>,
}

/// Names a table keeps before it stops learning new ones.
const NAMES_LIMIT: usize = 256;

// Perf: exp000 from_plain of a json.loads tree, lockstep: 0.89x -> 1.23-1.26x of base; address-only
// matching missed every fresh key.
/// The entry for ``name``: the guessed one by address or value, then any by
/// address, then any by value. Names read from a
/// fresh document (``json.loads`` output) are new strings on every call, so an
/// address-only match would miss each time and fill the table with copies.
#[inline]
fn find_name<'t, T>(
    names: &'t [(Py<PyString>, T)],
    name: &Bound<'_, PyString>,
    guess: usize,
) -> Option<&'t T> {
    let at = name.as_ptr();
    // Both are exact str, so the comparison runs no Python code and cannot fail.
    let equal = |known: &Py<PyString>| unsafe { ffi::PyUnicode_Compare(known.as_ptr(), at) == 0 };
    if let Some((known, entry)) = names.get(guess)
        && (known.as_ptr() == at || equal(known))
    {
        return Some(entry);
    }
    if let Some((_, entry)) = names.iter().find(|(known, _)| known.as_ptr() == at) {
        return Some(entry);
    }
    names
        .iter()
        .find(|(known, _)| equal(known))
        .map(|(_, entry)| entry)
}

impl SetTable {
    /// A table for instances of ``kind``. Only heap classes qualify: for them
    /// ``object.__setattr__``'s ``hackcheck`` always passes, so the call is
    /// exactly ``PyObject_GenericSetAttr``. Other types keep the Python call.
    pub fn new(kind: *mut ffi::PyTypeObject) -> Option<Self> {
        if unsafe { ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_HEAPTYPE) } == 0 {
            return None;
        }
        let tag = version_tag(kind);
        if tag == 0 {
            return None;
        }
        Some(Self {
            tag,
            names: Vec::new(),
        })
    }

    #[inline]
    pub fn holds(&self, kind: *mut ffi::PyTypeObject) -> bool {
        unsafe { (*kind).tp_version_tag == self.tag }
    }

    /// The slot ``name`` sets on instances of ``kind``, or null; ``guess`` is
    /// where the name likely sits, as records of one class repeat their order.
    #[inline]
    fn member(
        &mut self,
        kind: *mut ffi::PyTypeObject,
        name: &Bound<'_, PyString>,
        guess: usize,
    ) -> *mut ffi::PyMemberDef {
        if let Some(member) = find_name(&self.names, name, guess) {
            return *member;
        }
        let member = unsafe { member_of(kind, type_lookup(kind, name.as_ptr())) };
        if self.names.len() < NAMES_LIMIT {
            self.names.push((name.clone().unbind(), member));
        }
        member
    }

    /// ``object.__setattr__(value, name, member)`` for ``value`` of this
    /// table's class, or ``None`` if only the Python call can tell.
    #[inline]
    pub fn set(
        &mut self,
        value: &B<'_>,
        name: &B<'_>,
        member: &B<'_>,
        guess: usize,
    ) -> Option<PyResult<()>> {
        let kind = tp(value);
        if tp(name) != t_str() || !self.holds(kind) {
            return None;
        }
        let name = unsafe { name.cast_unchecked::<PyString>() };
        let slot = self.member(kind, name, guess);
        let done = unsafe {
            if slot.is_null() {
                ffi::PyObject_GenericSetAttr(value.as_ptr(), name.as_ptr(), member.as_ptr())
            } else if (*slot).type_code == ffi::Py_T_OBJECT_EX
                && (*slot).flags & (ffi::Py_READONLY | PY_RELATIVE_OFFSET) == 0
            {
                // ``PyMember_SetOne``'s whole path for a writable object slot.
                let at = value
                    .as_ptr()
                    .cast::<u8>()
                    .offset((*slot).offset)
                    .cast::<*mut ffi::PyObject>();
                let old = *at;
                *at = ffi::Py_NewRef(member.as_ptr());
                ffi::Py_XDECREF(old);
                0
            } else {
                // ``member_set``: its instance check holds for every value of
                // the class, as checked when the slot was found.
                ffi::PyMember_SetOne(value.as_ptr().cast::<c_char>(), slot, member.as_ptr())
            }
        };
        Some(if done < 0 {
            Err(PyErr::fetch(value.py()))
        } else {
            Ok(())
        })
    }
}

/// What ``hasattr(value, name)`` answers for every value of one class.
#[derive(Clone, Copy)]
enum Has {
    Yes,
    No,
    /// Whether this ``__slots__`` slot is set.
    Slot(ffi::Py_ssize_t),
    /// Depends on more than the class; ask Python. A descriptor that runs
    /// code (a property, say) is asked for every value, since it may answer
    /// differently for each.
    Ask,
}

/// ``hasattr`` answers for one class, by name.
pub struct HasTable {
    tag: u32,
    names: Vec<(Py<PyString>, Has)>,
}

impl HasTable {
    pub fn new(kind: *mut ffi::PyTypeObject) -> Option<Self> {
        let tag = version_tag(kind);
        (tag != 0).then(|| Self {
            tag,
            names: Vec::new(),
        })
    }

    #[inline]
    pub fn holds(&self, kind: *mut ffi::PyTypeObject) -> bool {
        unsafe { (*kind).tp_version_tag == self.tag }
    }

    fn learn(
        kind: *mut ffi::PyTypeObject,
        name: *mut ffi::PyObject,
        getattr: *mut ffi::PyObject,
    ) -> Has {
        unsafe {
            if (*kind).tp_getattro.map(|f| f as usize) != generic_getattr_slot() {
                return Has::Ask;
            }
            let found = type_lookup(kind, name);
            // A ``__getattr__`` turns a failed or raising lookup into anything.
            let fallback = !type_lookup(kind, getattr).is_null();
            if found.is_null() {
                return if dict_storage(kind) || fallback {
                    Has::Ask
                } else {
                    Has::No
                };
            }
            let found_type = ffi::Py_TYPE(found);
            if (*found_type).tp_descr_set.is_some() {
                // A data descriptor answers for itself; a slot's answers by
                // whether the slot is set, with no code run.
                let member = member_of(kind, found);
                if fallback
                    || member.is_null()
                    || (*member).type_code != ffi::Py_T_OBJECT_EX
                    || (*member).flags & PY_AUDIT_READ != 0
                {
                    return Has::Ask;
                }
                return Has::Slot((*member).offset);
            }
            // An instance dict could only add the name. A plain class
            // attribute, or a function bound on access, is always there.
            if (*found_type).tp_descr_get.is_none() || found_type == &raw mut ffi::PyFunction_Type {
                return Has::Yes;
            }
            Has::Ask
        }
    }

    /// ``hasattr(value, name)`` for ``value`` of this table's class.
    #[inline]
    pub fn has(
        &mut self,
        value: &B<'_>,
        name: &Bound<'_, PyString>,
        guess: usize,
    ) -> PyResult<bool> {
        let kind = tp(value);
        let answer = if tp(name.as_any()) != t_str() || !self.holds(kind) {
            Has::Ask
        } else if let Some(has) = find_name(&self.names, name, guess) {
            *has
        } else {
            let getattr = pyo3::intern!(value.py(), "__getattr__");
            let has = Self::learn(kind, name.as_ptr(), getattr.as_ptr());
            if self.names.len() < NAMES_LIMIT {
                self.names.push((name.clone().unbind(), has));
            }
            has
        };
        match answer {
            Has::Yes => Ok(true),
            Has::No => Ok(false),
            Has::Slot(offset) => Ok(unsafe {
                !(*value
                    .as_ptr()
                    .cast::<u8>()
                    .offset(offset)
                    .cast::<*mut ffi::PyObject>())
                .is_null()
            }),
            Has::Ask => value.hasattr(name),
        }
    }
}

/// Whether ``kind.__new__`` keeps returning the same callable while ``kind``'s
/// version tag holds, so one read of it can be replayed. True when attribute
/// lookup on the class is ``type.__getattribute__``'s and the attribute is a
/// plain function or staticmethod in the class's own MRO.
pub fn new_callable_stable(
    py: Python<'_>,
    kind: *mut ffi::PyTypeObject,
    slot_getattro: usize,
    staticmethod_type: *mut ffi::PyTypeObject,
) -> bool {
    unsafe {
        let meta = ffi::Py_TYPE(kind.cast::<ffi::PyObject>());
        let getattro = (*meta).tp_getattro.map(|f| f as usize);
        if getattro != ffi::PyType_Type.tp_getattro.map(|f| f as usize)
            && getattro != Some(slot_getattro)
        {
            return false;
        }
        let new = pyo3::intern!(py, "__new__").as_ptr();
        let getattr = pyo3::intern!(py, "__getattribute__").as_ptr();
        // A metaclass ``__getattribute__`` other than ``type``'s, or a data
        // descriptor ``__new__`` on the metaclass, answers for itself.
        if type_lookup(meta, getattr) != type_lookup(&raw mut ffi::PyType_Type, getattr) {
            return false;
        }
        let found = type_lookup(meta, new);
        if !found.is_null() && (*ffi::Py_TYPE(found)).tp_descr_set.is_some() {
            return false;
        }
        let found = type_lookup(kind, new);
        if found.is_null() {
            return false;
        }
        let found_type = ffi::Py_TYPE(found);
        (*found_type).tp_descr_get.is_none()
            || found_type == &raw mut ffi::PyFunction_Type
            || found_type == staticmethod_type
    }
}

// Perf: exp000 from_plain, paired: part of SetTable's 1.18-1.21x.
/// Where ``dataclasses.Field`` keeps the attributes ``fields`` and
/// ``_with_defaults`` read, so they can be read without a lookup.
pub struct FieldSlots {
    field: Py<PyAny>,
    tag: u32,
    marker: Py<PyAny>,
    name: ffi::Py_ssize_t,
    default: ffi::Py_ssize_t,
    factory: ffi::Py_ssize_t,
    init: ffi::Py_ssize_t,
    field_type: ffi::Py_ssize_t,
}

/// One ``Field``'s attributes, read when asked as Python reads them.
pub struct FieldView<'py> {
    field: B<'py>,
}

impl FieldSlots {
    /// The slots, if ``fields`` is ``dataclasses.fields`` and ``Field``
    /// keeps each attribute in a plain slot.
    pub fn new(py: Python<'_>, fields: &B<'_>) -> PyResult<Option<Self>> {
        let dataclasses = py.import("dataclasses")?;
        if !dataclasses.getattr("fields")?.is(fields) {
            return Ok(None);
        }
        let field = dataclasses.getattr("Field")?;
        if !is_class(&field) {
            return Ok(None);
        }
        let kind = field.as_ptr().cast::<ffi::PyTypeObject>();
        let tag = version_tag(kind);
        let offset = |name: &str| -> Option<ffi::Py_ssize_t> {
            let name = PyString::new(py, name);
            let member = unsafe { member_of(kind, type_lookup(kind, name.as_ptr())) };
            (!member.is_null() && unsafe { (*member).type_code } == ffi::Py_T_OBJECT_EX)
                .then(|| unsafe { (*member).offset })
        };
        let (Some(name), Some(default), Some(factory), Some(init), Some(field_type)) = (
            offset("name"),
            offset("default"),
            offset("default_factory"),
            offset("init"),
            offset("_field_type"),
        ) else {
            return Ok(None);
        };
        if tag == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            field: field.unbind(),
            tag,
            marker: dataclasses.getattr("_FIELD")?.unbind(),
            name,
            default,
            factory,
            init,
            field_type,
        }))
    }

    #[inline]
    unsafe fn read<'py>(
        py: Python<'py>,
        object: *mut ffi::PyObject,
        offset: ffi::Py_ssize_t,
    ) -> Option<B<'py>> {
        unsafe {
            let slot = *object
                .cast::<u8>()
                .offset(offset)
                .cast::<*mut ffi::PyObject>();
            (!slot.is_null()).then(|| Bound::from_borrowed_ptr(py, slot))
        }
    }

    /// ``dataclasses.fields(target)``, or ``None`` where that call must say
    /// what happens (no fields, a stand-in ``Field``, an unset attribute).
    pub fn fields<'py>(&self, target: &B<'py>) -> PyResult<Option<Vec<FieldView<'py>>>> {
        let py = target.py();
        let kind = self.field.bind(py).as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { (*kind).tp_version_tag } != self.tag {
            return Ok(None);
        }
        let Ok(declared) = target.getattr(pyo3::intern!(py, "__dataclass_fields__")) else {
            return Ok(None);
        };
        if tp(&declared) != t_dict() {
            return Ok(None);
        }
        let marker = self.marker.bind(py).as_ptr();
        let mut views = Vec::new();
        let mut position: ffi::Py_ssize_t = 0;
        let mut key = std::ptr::null_mut();
        let mut value = std::ptr::null_mut();
        // No Python code runs in this loop, so the dict cannot change under it.
        unsafe {
            while ffi::PyDict_Next(declared.as_ptr(), &mut position, &mut key, &mut value) != 0 {
                if ffi::Py_TYPE(value) != kind {
                    return Ok(None);
                }
                let Some(field_type) = Self::read(py, value, self.field_type) else {
                    return Ok(None);
                };
                if field_type.as_ptr() == marker {
                    views.push(FieldView {
                        field: Bound::from_borrowed_ptr(py, value),
                    });
                }
            }
        }
        Ok(Some(views))
    }

    /// ``field``'s attribute at ``offset``, or its ``getattr`` where
    /// ``field`` is not a ``Field``, the slot is unset, or ``Field`` changed.
    #[inline]
    fn get<'py>(&self, field: &B<'py>, offset: ffi::Py_ssize_t, name: &str) -> PyResult<B<'py>> {
        let py = field.py();
        let kind = self.field.bind(py).as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { (*kind).tp_version_tag } == self.tag
            && tp(field) == kind
            && let Some(found) = unsafe { Self::read(py, field.as_ptr(), offset) }
        {
            return Ok(found);
        }
        field.getattr(name)
    }

    /// ``dataclasses._FIELD``, the ``_field_type`` of a regular field.
    pub fn marker<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.marker.bind(py)
    }

    /// ``field._field_type`` for any object ``field``.
    pub fn field_type<'py>(&self, field: &B<'py>) -> PyResult<B<'py>> {
        self.get(field, self.field_type, "_field_type")
    }

    pub fn name<'py>(&self, view: &FieldView<'py>) -> PyResult<B<'py>> {
        self.get(&view.field, self.name, "name")
    }

    pub fn default<'py>(&self, view: &FieldView<'py>) -> PyResult<B<'py>> {
        self.get(&view.field, self.default, "default")
    }

    pub fn factory<'py>(&self, view: &FieldView<'py>) -> PyResult<B<'py>> {
        self.get(&view.field, self.factory, "default_factory")
    }

    pub fn init<'py>(&self, view: &FieldView<'py>) -> PyResult<B<'py>> {
        self.get(&view.field, self.init, "init")
    }
}
