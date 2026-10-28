//! Dataclass fields as the encoder and decoder read them: their attributes
//! without a lookup, and the rule that leaves a field holding what its
//! default rebuilds out of the written tree.

use std::cell::{OnceCell, RefCell};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::class::{member_of, type_lookup, version_tag};
use crate::py::*;

// Perf: exp000 from_plain, paired: part of SetTable's 1.18-1.21x.
/// Where ``dataclasses.Field`` keeps the attributes ``fields`` and
/// ``_with_defaults`` read, so they can be read without a lookup.
pub struct FieldSlots {
    field: Py<PyAny>,
    tag: u32,
    marker: Py<PyAny>,
    /// ``dataclasses.MISSING`` and ``types.GenericAlias``, for ``native_default``.
    missing: Py<PyAny>,
    generic_alias: Py<PyAny>,
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

impl<'py> FieldView<'py> {
    /// The view of ``field``, any object ``FieldSlots`` reads as a ``Field``.
    pub fn of(field: B<'py>) -> Self {
        Self { field }
    }

    /// The ``Field`` object itself.
    pub fn field(&self) -> &B<'py> {
        &self.field
    }
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
            missing: dataclasses.getattr("MISSING")?.unbind(),
            generic_alias: py.import("types")?.getattr("GenericAlias")?.unbind(),
            name,
            default,
            factory,
            init,
            field_type,
        }))
    }

    /// The object at ``offset`` in ``object``, borrowed; null where unset.
    #[inline]
    unsafe fn read_raw(object: *mut ffi::PyObject, offset: ffi::Py_ssize_t) -> *mut ffi::PyObject {
        unsafe {
            *object
                .cast::<u8>()
                .offset(offset)
                .cast::<*mut ffi::PyObject>()
        }
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
        let name = pyo3::intern!(py, "__dataclass_fields__");
        let attr = if is_class(target) {
            unsafe { crate::class::class_attr(target.as_ptr(), name.as_ptr()) }
        } else {
            crate::class::ClassAttr::Ask
        };
        let declared = match attr {
            crate::class::ClassAttr::Found(found, _) => unsafe {
                Bound::from_borrowed_ptr(py, found)
            },
            crate::class::ClassAttr::Missing => return Ok(None),
            crate::class::ClassAttr::Ask => match target.getattr(name) {
                Ok(found) => found,
                Err(_) => return Ok(None),
            },
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
        match self.get_native(field, offset) {
            Some(found) => Ok(found),
            None => field.getattr(name),
        }
    }

    /// ``field``'s attribute at ``offset``; ``None`` where only ``getattr``
    /// can say.
    #[inline]
    fn get_native<'py>(&self, field: &B<'py>, offset: ffi::Py_ssize_t) -> Option<B<'py>> {
        let py = field.py();
        let kind = self.field.bind(py).as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { (*kind).tp_version_tag } != self.tag || tp(field) != kind {
            return None;
        }
        unsafe { Self::read(py, field.as_ptr(), offset) }
    }

    /// What ``_place_defaults`` builds for ``field``, where that runs no
    /// Python code: its default, or the empty container a builtin factory
    /// builds. ``None`` where it has neither, or only Python can say.
    pub fn native_default<'py>(&self, field: &B<'py>) -> Option<B<'py>> {
        let py = field.py();
        if !self
            .get_native(field, self.field_type)?
            .is(self.marker.bind(py))
        {
            return None;
        }
        let missing = self.missing.bind(py);
        let default = self.get_native(field, self.default)?;
        if !default.is(missing) {
            return Some(default);
        }
        let factory = self.get_native(field, self.factory)?;
        if factory.is(missing) {
            return None;
        }
        let kind = builtin_factory(&factory, self.generic_alias.bind(py)).ok()??;
        empty_of(py, kind).ok()
    }

    /// Another handle on the same slots.
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            field: self.field.clone_ref(py),
            tag: self.tag,
            marker: self.marker.clone_ref(py),
            missing: self.missing.clone_ref(py),
            generic_alias: self.generic_alias.clone_ref(py),
            name: self.name,
            default: self.default,
            factory: self.factory,
            init: self.init,
            field_type: self.field_type,
        }
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

// ----------------------------------------------------------- defaults --

/// What ``dataclasses`` and ``treekle`` the default rule reads.
pub struct DefaultGlobals {
    pub slots: Option<FieldSlots>,
    /// ``dataclasses.fields``, asked where ``slots`` cannot answer.
    pub fields: Py<PyAny>,
    pub missing: Py<PyAny>,
    pub enum_type: Py<PyAny>,
    pub generic_alias: Py<PyAny>,
}

/// What a field's default reads back as.
pub enum Sample<'py> {
    /// An empty instance of this builtin container: what a factory that is
    /// the class itself, or a builtin generic of it, builds.
    Empty(*mut ffi::PyTypeObject),
    /// The default itself, or what the factory built.
    Value(B<'py>),
    /// No default, or the factory raised: the field is written.
    Written,
}

/// What a factory built, kept so it runs once.
enum Built {
    Empty(*mut ffi::PyTypeObject),
    Value(Py<PyAny>),
    Raised,
}

impl Built {
    fn sample<'py>(&self, py: Python<'py>) -> Sample<'py> {
        match self {
            Self::Empty(kind) => Sample::Empty(*kind),
            Self::Value(value) => Sample::Value(value.bind(py).clone()),
            Self::Raised => Sample::Written,
        }
    }
}

/// ``_Default``: a field with a default, as a member is compared with it.
pub struct FieldDefault {
    pub name: Py<PyString>,
    /// The ``Field``, its default read anew each time, as ``from_plain`` reads it.
    field: Py<PyAny>,
    /// The factory last called, and what it built.
    made: RefCell<Option<(Py<PyAny>, Built)>>,
}

impl FieldDefault {
    fn new(name: Py<PyString>, field: Py<PyAny>) -> Self {
        Self {
            name,
            field,
            made: RefCell::new(None),
        }
    }
}

/// ``_defaults``: a dataclass's fields with a default, kept with its facts.
pub struct ClassDefaults {
    pub fields: Vec<FieldDefault>,
    /// Each layout slot's index in ``fields``, ``u32::MAX`` for none.
    pub by_slot: OnceCell<Vec<u32>>,
}

impl ClassDefaults {
    /// The index in ``fields`` of the field named ``name``, an exact str.
    pub fn find(&self, name: &B<'_>) -> Option<usize> {
        let ptr = name.as_ptr();
        if let Some(found) = self
            .fields
            .iter()
            .position(|each| each.name.as_ptr() == ptr)
        {
            return Some(found);
        }
        let bytes = utf8(name)?;
        let py = name.py();
        self.fields
            .iter()
            .position(|each| utf8(each.name.bind(py).as_any()) == Some(bytes))
    }
}

impl DefaultGlobals {
    /// ``_defaults``: ``kind``'s fields with a default or a default factory,
    /// in order; ``None`` where it has none.
    pub fn defaulted(&self, kind: &B<'_>) -> PyResult<Option<ClassDefaults>> {
        let py = kind.py();
        let views = match self.slots.as_ref().map(|slots| slots.fields(kind)) {
            Some(found) => found?,
            None => None,
        };
        let views = match views {
            Some(views) => views,
            None => self
                .fields
                .bind(py)
                .call1((kind,))?
                .try_iter()?
                .map(|field| field.map(FieldView::of))
                .collect::<PyResult<_>>()?,
        };
        let missing = self.missing.bind(py);
        let mut fields = Vec::new();
        for view in views {
            if self.attribute(&view, "default")?.is(missing)
                && self.attribute(&view, "default_factory")?.is(missing)
            {
                continue;
            }
            let name = self.attribute(&view, "name")?.cast_into::<PyString>()?;
            fields.push(FieldDefault::new(
                name.unbind(),
                view.field().clone().unbind(),
            ));
        }
        Ok((!fields.is_empty()).then(|| ClassDefaults {
            fields,
            by_slot: OnceCell::new(),
        }))
    }

    /// ``defaulted`` for a class whose fields are among ``names``, its slots,
    /// read with no Python code run and ``by_slot`` set to match ``names``:
    /// ``Some(None)`` where no field has a default, ``None`` where only
    /// ``defaulted`` can say.
    pub fn defaulted_slots<'a>(
        &self,
        kind: &B<'_>,
        names: impl ExactSizeIterator<Item = &'a Py<PyString>>,
    ) -> Option<Option<ClassDefaults>> {
        let py = kind.py();
        let slots = self.slots.as_ref()?;
        let field_kind = slots.field.bind(py).as_ptr().cast::<ffi::PyTypeObject>();
        if unsafe { (*field_kind).tp_version_tag } != slots.tag {
            return None;
        }
        let name = pyo3::intern!(py, "__dataclass_fields__");
        let crate::class::ClassAttr::Found(declared, _) =
            (unsafe { crate::class::class_attr(kind.as_ptr(), name.as_ptr()) })
        else {
            return None;
        };
        if ob_type(declared) != t_dict() {
            return None;
        }
        let missing = self.missing.as_ptr();
        let marker = slots.marker.as_ptr();
        // Each ``Field``'s attributes, borrowed: no Python code runs while
        // they are read, so the dict holds them.
        let defaulted = |field: *mut ffi::PyObject| -> Option<bool> {
            if ob_type(field) != field_kind {
                return None;
            }
            let read = |offset| unsafe { FieldSlots::read_raw(field, offset) };
            let (field_type, value, factory) = (
                read(slots.field_type),
                read(slots.default),
                read(slots.factory),
            );
            if field_type.is_null() || value.is_null() || factory.is_null() {
                return None;
            }
            Some(field_type == marker && (value != missing || factory != missing))
        };
        let mut fields = Vec::new();
        let mut position: ffi::Py_ssize_t = 0;
        let mut key = std::ptr::null_mut();
        let mut field = std::ptr::null_mut();
        while unsafe { ffi::PyDict_Next(declared, &mut position, &mut key, &mut field) } != 0 {
            if !defaulted(field)? {
                continue;
            }
            if ob_type(key) != t_str() {
                return None;
            }
            fields.push(FieldDefault::new(
                unsafe { Py::from_borrowed_ptr(py, key) },
                unsafe { Py::from_borrowed_ptr(py, field) },
            ));
        }
        if fields.is_empty() {
            return Some(None);
        }
        let defaults = ClassDefaults {
            fields,
            by_slot: OnceCell::new(),
        };
        let by_slot = names
            .map(|name| {
                defaults
                    .find(name.bind(py).as_any())
                    .map_or(u32::MAX, |found| found as u32)
            })
            .collect();
        defaults.by_slot.set(by_slot).ok()?;
        Some(Some(defaults))
    }

    fn attribute<'py>(&self, view: &FieldView<'py>, name: &str) -> PyResult<B<'py>> {
        match (&self.slots, name) {
            (Some(slots), "name") => slots.name(view),
            (Some(slots), "default") => slots.default(view),
            (Some(slots), _) => slots.factory(view),
            (None, _) => view.field().getattr(name),
        }
    }

    /// ``_Default.sample``: what ``default``'s field reads back as, its
    /// ``default`` and ``default_factory`` read now; a factory is called once,
    /// until it is replaced, and one that raises leaves the field written.
    /// ``None`` where that runs Python code and ``call`` is not set.
    pub fn sample<'py>(
        &self,
        py: Python<'py>,
        default: &FieldDefault,
        call: bool,
    ) -> PyResult<Option<Sample<'py>>> {
        let field = default.field.bind(py);
        let native = self.slots.as_ref().and_then(|slots| {
            Some((
                slots.get_native(field, slots.default)?,
                slots.get_native(field, slots.factory)?,
            ))
        });
        let (value, factory) = match native {
            Some(read) => read,
            None if !call => return Ok(None),
            None => {
                let view = FieldView::of(field.clone());
                (
                    self.attribute(&view, "default")?,
                    self.attribute(&view, "default_factory")?,
                )
            }
        };
        let missing = self.missing.bind(py);
        if !value.is(missing) {
            return Ok(Some(Sample::Value(value)));
        }
        if factory.is(missing) {
            return Ok(Some(Sample::Written));
        }
        if let Some((last, built)) = &*default.made.borrow()
            && last.is(&factory)
        {
            return Ok(Some(built.sample(py)));
        }
        let built = match builtin_factory(&factory, self.generic_alias.bind(py))? {
            Some(kind) => Built::Empty(kind),
            None if !call => return Ok(None),
            // Not borrowed while it runs: a factory may encode this class.
            None => match factory.call0() {
                Ok(built) => Built::Value(built.unbind()),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                    Built::Raised
                }
                Err(error) => return Err(error),
            },
        };
        let sample = built.sample(py);
        *default.made.borrow_mut() = Some((factory.unbind(), built));
        Ok(Some(sample))
    }

    /// ``_omittable``: whether ``value`` is of a kind ``inferred`` can match,
    /// so its default is worth building.
    pub fn omittable(&self, value: &B<'_>) -> PyResult<bool> {
        let kind = tp(value);
        if kind == t_none()
            || kind == t_bool()
            || kind == t_int()
            || kind == t_float()
            || kind == t_str()
            || kind == &raw mut ffi::PyBytes_Type
            || kind == t_tuple()
            || unsafe { ffi::PyType_IsSubtype(kind, self.enum_type.as_ptr().cast()) } != 0
        {
            return Ok(true);
        }
        Ok(empty_kind(kind) && value.len()? == 0)
    }

    /// ``_inferred``: whether reading ``sample`` back rebuilds ``value``.
    /// Runs no Python code.
    pub fn inferred(&self, value: &B<'_>, sample: &Sample<'_>) -> PyResult<bool> {
        match sample {
            Sample::Written => Ok(false),
            Sample::Empty(kind) => Ok(tp(value) == *kind && value.len()? == 0),
            Sample::Value(sample) => self.same(value, sample, false),
        }
    }

    fn same(&self, value: &B<'_>, sample: &B<'_>, nested: bool) -> PyResult<bool> {
        let kind = tp(value);
        if kind != tp(sample) {
            return Ok(false);
        }
        if kind == t_float() {
            let (number, other) = unsafe {
                (
                    ffi::PyFloat_AS_DOUBLE(value.as_ptr()),
                    ffi::PyFloat_AS_DOUBLE(sample.as_ptr()),
                )
            };
            if number.is_nan() {
                return Ok(other.is_nan());
            }
            return Ok(number == other && number.is_sign_negative() == other.is_sign_negative());
        }
        if kind == t_none() || kind == t_bool() {
            return Ok(value.is(sample));
        }
        if kind == t_int() || kind == t_str() || kind == &raw mut ffi::PyBytes_Type {
            return py_eq(value, sample);
        }
        if unsafe { ffi::PyType_IsSubtype(kind, self.enum_type.as_ptr().cast()) } != 0 {
            return Ok(value.is(sample));
        }
        if kind == t_tuple() {
            let (items, others) = unsafe { (borrowed_items(value), borrowed_items(sample)) };
            if items.len() != others.len() {
                return Ok(false);
            }
            for (&item, &other) in items.iter().zip(others) {
                let (item, other) = unsafe {
                    (
                        Bound::from_borrowed_ptr(value.py(), item),
                        Bound::from_borrowed_ptr(value.py(), other),
                    )
                };
                if !self.same(&item, &other, true)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if nested || !empty_kind(kind) {
            return Ok(false);
        }
        Ok(value.len()? == 0 && sample.len()? == 0)
    }
}

/// The builtin container ``factory`` builds an empty instance of, where it is
/// that class or a builtin generic of it (an instance of ``generic_alias``).
pub fn builtin_factory(
    factory: &B<'_>,
    generic_alias: &B<'_>,
) -> PyResult<Option<*mut ffi::PyTypeObject>> {
    let py = factory.py();
    let origin = if tp(factory) == generic_alias.as_ptr().cast() {
        factory.getattr(pyo3::intern!(py, "__origin__"))?
    } else {
        factory.clone()
    };
    let builtin = origin.as_ptr().cast::<ffi::PyTypeObject>();
    Ok([
        t_dict(),
        t_list(),
        &raw mut ffi::PySet_Type,
        t_tuple(),
        &raw mut ffi::PyFrozenSet_Type,
    ]
    .contains(&builtin)
    .then_some(builtin))
}

/// A new empty ``kind``, one of the containers ``builtin_factory`` names.
pub fn empty_of(py: Python<'_>, kind: *mut ffi::PyTypeObject) -> PyResult<B<'_>> {
    Ok(if kind == t_dict() {
        pyo3::types::PyDict::new(py).into_any()
    } else if kind == t_list() {
        pyo3::types::PyList::empty(py).into_any()
    } else if kind == t_tuple() {
        pyo3::types::PyTuple::empty(py).into_any()
    } else if kind == &raw mut ffi::PySet_Type {
        pyo3::types::PySet::empty(py)?.into_any()
    } else {
        pyo3::types::PyFrozenSet::empty(py)?.into_any()
    })
}

/// A container a default may leave out when empty.
fn empty_kind(kind: *mut ffi::PyTypeObject) -> bool {
    registered_kind(kind) || kind == &raw mut ffi::PyFrozenSet_Type
}

/// Whether the walk registers a value of ``kind`` for ``py/ref`` when written,
/// of the containers a default may leave out.
pub fn registered_kind(kind: *mut ffi::PyTypeObject) -> bool {
    kind == t_dict()
        || kind == t_list()
        || kind == &raw mut ffi::PySet_Type
        || kind == &raw mut ffi::PyDictProxy_Type
}
