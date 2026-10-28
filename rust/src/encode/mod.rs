//! ``to_plain``: Python values to plain trees, with ``treekle.serialize``'s
//! output, and ``json_dumps``: the same walk writing JSON text.
//!
//! One walk serves both outputs; ``Sink`` decides, container by container,
//! whether a node is built or written. The walk follows ``serialize``'s own
//! order: a container's parts are read when it is reached and encoded depth
//! first, each registered for ``py/ref`` where Python registers it, and a
//! reduce whose part raises ``TypeError`` restarts by protocol from where it
//! began. Type-level decisions are ``serialize``'s own helpers, asked once
//! per class or declared type and kept; nothing outside a class is watched.

mod facts;
mod freeze;
mod sink;

use std::rc::Rc;

use hashbrown::HashMap;
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::class_names::ClassNames;
use crate::json::{self, Style, Writer};
use crate::py::*;
use facts::{DictPlan, Enc, Facts, Layout, Leaf, PathFact, Pick, Slot, Slots, Table, Types};
use freeze::freeze;
use sink::{Mark, Sink};

// ----------------------------------------------------------------- globals --

/// What the walk reads from ``treekle.serialize`` once, when it is set up.
pub struct Globals {
    pending: Py<PyAny>,
    settled_leaf: Py<PyAny>,
    pure_path: Py<PyAny>,
    partial: Py<PyAny>,
    form: Py<PyAny>,
    settled_form: Py<PyAny>,
    record_implied: Py<PyAny>,
    record_built: Py<PyAny>,
    field_types_cache: Py<PyAny>,
    read_field_types: Py<PyAny>,
    import_path: Py<PyAny>,
    resolves_to: Py<PyAny>,
    verified_path: Py<PyAny>,
    datetime_text: Py<PyAny>,
    concrete_overrides: Py<PyAny>,
    immutable: Py<PyAny>,
    slot_names: Py<PyAny>,
    path: Py<PyAny>,
    concrete_path: Py<PyAny>,
    uuid: Py<PyAny>,
    datetime: Py<PyAny>,
    enum_type: Py<PyAny>,
    module_type: Py<PyAny>,
    mapping: Py<PyAny>,
    sequence: Py<PyAny>,
    abstract_set: Py<PyAny>,
    str_bytes: Py<PyAny>,
    dialects: Py<PyAny>,
    /// ``list``, ``tuple``, ``set``, ``frozenset``, ``dict``,
    /// ``MappingProxyType``: the class ``_form`` is asked about.
    containers: [Py<PyAny>; 6],
    sys_dict: Py<PyDict>,
    paths: crate::class::PathGlobals,
    hints: facts::HintGlobals,
    records: facts::RecordGlobals,
    defaults: crate::fields::DefaultGlobals,
    /// ``subtype_dict``, the getter of a plain class's ``__dict__``.
    dict_getter: Option<usize>,
}

static GLOBALS: GilCell<Option<Rc<Globals>>> = GilCell::new(None);

/// Facts about classes, kept across calls; a call takes the table and puts
/// it back, so a nested call from a hook starts with its own.
static TABLE: GilCell<Option<Box<Table>>> = GilCell::new(None);
static FORMS: GilCell<Option<Box<Forms>>> = GilCell::new(None);

pub fn setup(module: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = module.py();
    let get = |name: &str| -> PyResult<Py<PyAny>> {
        module
            .get_item(name)?
            .map(Bound::unbind)
            .ok_or_else(|| PyAttributeError::new_err(format!("treekle.serialize has no {name}")))
    };
    let builtins = py.import("builtins")?;
    let name = get("__name__")?.extract::<String>(py)?;
    let package = name.rsplit_once('.').map_or("", |(package, _)| package);
    let reflect = py.import(format!("{package}.reflect").as_str())?;
    let plain = py.import(format!("{package}.plain").as_str())?;
    let deserialize = py.import(format!("{package}.deserialize").as_str())?;
    let dataclasses = py.import("dataclasses")?;
    let dialect = get("Dialect")?;
    let dialects = get("get_args")?
        .bind(py)
        .call1((dialect.bind(py).getattr("__value__")?,))?;
    let probe = builtins.getattr("type")?.call1((
        "_DictProbe",
        PyTuple::new(py, [builtins.getattr("object")?])?,
        PyDict::new(py),
    ))?;
    let dict_getter = unsafe {
        let found =
            crate::class::type_lookup(probe.as_ptr().cast(), intern!(py, "__dict__").as_ptr());
        (!found.is_null() && ffi::Py_TYPE(found) == &raw mut ffi::PyGetSetDescr_Type)
            .then(|| (*(*found.cast::<ffi::PyGetSetDescrObject>()).d_getset).get)
            .flatten()
            .map(|f| f as usize)
    };
    let globals = Globals {
        pending: get("PENDING")?,
        settled_leaf: get("_settled_leaf")?,
        pure_path: get("PurePath")?,
        partial: get("partial")?,
        form: get("_form")?,
        settled_form: get("_settled_form")?,
        record_implied: get("_record_implied")?,
        record_built: get("_record_built")?,
        field_types_cache: deserialize.getattr("_FIELD_TYPES")?.unbind(),
        read_field_types: deserialize.getattr("_read_field_types")?.unbind(),
        import_path: get("_import_path")?,
        resolves_to: get("resolves_to")?,
        verified_path: get("_verified_path")?,
        datetime_text: get("_datetime_text")?,
        concrete_overrides: get("concrete_overrides")?,
        immutable: plain.getattr("immutable")?.unbind(),
        slot_names: reflect.getattr("slot_names")?.unbind(),
        path: get("Path")?,
        concrete_path: get("_CONCRETE_PATH")?,
        uuid: get("UUID")?,
        datetime: get("datetime")?,
        enum_type: get("Enum")?,
        module_type: get("ModuleType")?,
        mapping: get("Mapping")?,
        sequence: get("Sequence")?,
        abstract_set: get("AbstractSet")?,
        str_bytes: PyTuple::new(py, [builtins.getattr("str")?, builtins.getattr("bytes")?])?
            .into_any()
            .unbind(),
        dialects: dialects.unbind(),
        containers: [
            builtins.getattr("list")?.unbind(),
            builtins.getattr("tuple")?.unbind(),
            builtins.getattr("set")?.unbind(),
            builtins.getattr("frozenset")?.unbind(),
            builtins.getattr("dict")?.unbind(),
            get("MappingProxyType")?,
        ],
        sys_dict: py.import("sys")?.dict().unbind(),
        hints: facts::HintGlobals {
            annotations: builtins
                .getattr("type")?
                .getattr("__dict__")?
                .get_item("__annotations__")?
                .unbind(),
            union_type: py.import("types")?.getattr("UnionType")?.unbind(),
            generic_alias: py.import("types")?.getattr("GenericAlias")?.unbind(),
            field: py.import("dataclasses")?.getattr("_FIELD")?.unbind(),
            alias: facts::AliasParts::new(py)?,
        },
        records: facts::RecordGlobals {
            dispatch: deserialize.getattr("_DECODERS")?.unbind(),
            union_type: py.import("types")?.getattr("UnionType")?.unbind(),
            enum_type: get("Enum")?,
            pure_path: get("PurePath")?,
        },
        paths: crate::class::PathGlobals {
            sys_dict: py.import("sys")?.dict().unbind(),
            module_type: get("ModuleType")?,
            last: std::cell::RefCell::new(None),
        },
        defaults: crate::fields::DefaultGlobals {
            slots: crate::fields::FieldSlots::new(py, &dataclasses.getattr("fields")?)?,
            fields: dataclasses.getattr("fields")?.unbind(),
            missing: dataclasses.getattr("MISSING")?.unbind(),
            enum_type: get("Enum")?,
            generic_alias: py.import("types")?.getattr("GenericAlias")?.unbind(),
        },
        dict_getter,
    };
    // Kept facts name the old module's helpers, so they go with it.
    let stale = (
        GLOBALS.replace(py, Some(Rc::new(globals))),
        TABLE.take(py),
        FORMS.take(py),
    );
    drop(stale);
    Ok(())
}

// ------------------------------------------------------------ decisions --

/// A container's parts' declared types: ``_DeclaredParts``. ``None`` for
/// ``object``.
#[derive(Default)]
struct Parts {
    each: Option<Py<PyAny>>,
    named: Option<Py<PyAny>>,
    indexed: Option<Py<PyAny>>,
}

impl Parts {
    fn read(py: Python<'_>, parts: &B<'_>, object: &B<'_>) -> PyResult<Self> {
        let field = |name: &Bound<'_, PyString>| -> PyResult<Option<Py<PyAny>>> {
            let value = parts.getattr(name)?;
            Ok((!value.is_none() && !value.is(object)).then(|| value.unbind()))
        };
        Ok(Self {
            each: field(intern!(py, "each"))?,
            named: field(intern!(py, "named"))?,
            indexed: field(intern!(py, "indexed"))?,
        })
    }

    /// ``of(key)``: the declared type of the part at an index or a name.
    fn of_index<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Option<B<'py>>> {
        if let Some(indexed) = &self.indexed {
            let found = indexed.bind(py).get_item(index)?;
            return Ok(declared(py, found));
        }
        if self.named.is_some() {
            return Ok(None);
        }
        Ok(self.each.as_ref().map(|each| each.bind(py).clone()))
    }

    fn of_key<'py>(&self, py: Python<'py>, key: &B<'py>) -> PyResult<Option<B<'py>>> {
        if let Some(indexed) = &self.indexed {
            let found = indexed.bind(py).get_item(key)?;
            return Ok(declared(py, found));
        }
        if let Some(named) = &self.named {
            if !is_str(key) {
                return Ok(None);
            }
            let object = py.get_type::<PyAny>().into_any();
            let found = named
                .bind(py)
                .call_method1(intern!(py, "get"), (key, &object))?;
            return Ok(declared(py, found));
        }
        Ok(self.each.as_ref().map(|each| each.bind(py).clone()))
    }

    /// The type an escaped key's value is declared: ``each`` unless named.
    fn of_raw<'py>(&self, py: Python<'py>) -> Option<B<'py>> {
        if self.named.is_some() {
            return None;
        }
        self.each.as_ref().map(|each| each.bind(py).clone())
    }
}

/// ``None`` for ``object``, else the type itself.
#[inline]
fn declared<'py>(py: Python<'py>, value: B<'py>) -> Option<B<'py>> {
    (!value.is(py.get_type::<PyAny>())).then_some(value)
}

/// ``_form``'s answer: whether a container goes untagged, its keys' type,
/// and its parts' types.
struct Form {
    untagged: bool,
    keys: Option<Py<PyAny>>,
    parts: Parts,
}

impl Form {
    fn read(py: Python<'_>, answer: &B<'_>) -> PyResult<Self> {
        let object = py.get_type::<PyAny>().into_any();
        let untagged = answer.get_item(0)?.is_truthy()?;
        let keys = answer.get_item(1)?;
        Ok(Self {
            untagged,
            keys: (!keys.is(&object)).then(|| keys.unbind()),
            parts: Parts::read(py, &answer.get_item(2)?, &object)?,
        })
    }
}

/// Decisions about declared types, kept as ``serialize`` keeps them: for
/// good, each with its declared type held so its address is not reused.
/// A decision kept by the addresses of what it was made about, with what
/// it holds alive so those addresses are not reused while it stands.
type Kept<K, H, A> = Bounded<K, (H, A)>;

struct Forms {
    /// The forms of a container with no declared type: tagged, untagged.
    undeclared: [Rc<Form>; 2],
    /// ``_settled_form`` by container class index, declared type and size.
    settled: Kept<FormKey, Py<PyAny>, Option<Rc<Form>>>,
    /// ``_record_built`` by declared type.
    records: Kept<usize, Py<PyAny>, Option<Py<PyAny>>>,
    /// ``_settled_leaf`` by leaf class, text class and declared type.
    leaves: Kept<LeafKey, [Py<PyAny>; 2], bool>,
}

/// A container class's index, a declared type's address, and a tuple's
/// size or -1.
type FormKey = (usize, usize, isize);

/// A leaf class's, its text class's and a declared type's addresses.
type LeafKey = (usize, usize, usize);

const FORMS_LIMIT: usize = 1 << 14;

impl Default for Forms {
    fn default() -> Self {
        let undeclared = |untagged| {
            Rc::new(Form {
                untagged,
                keys: None,
                parts: Parts::default(),
            })
        };
        Self {
            undeclared: [undeclared(false), undeclared(true)],
            settled: Bounded::new(FORMS_LIMIT),
            records: Bounded::new(FORMS_LIMIT),
            leaves: Bounded::new(FORMS_LIMIT),
        }
    }
}

// --------------------------------------------------------------- places --

/// Where a node sits: its parent's place and its own key.
enum Step<'py> {
    Root,
    Index(usize),
    Key(B<'py>),
}

const ROOT: u32 = 0;
/// ``_IN_KEY``: where a key's parts sit; no ``py/ref`` names them.
const IN_KEY: u32 = 1;

/// A part still to encode: its value, where it goes, and its declared type.
struct Part<'py> {
    value: B<'py>,
    key: Key<'py>,
    declared: Option<B<'py>>,
}

enum Key<'py> {
    Index(usize),
    Name(B<'py>),
    /// A name a class's instances share, whose JSON text is made once.
    Known(Slots, u32),
    /// A dict key escaped when its entry is reached, with its declared type.
    Raw(B<'py>, Option<B<'py>>),
}

/// An object's member: its name and value.
struct Member<'py> {
    name: Name<'py>,
    value: B<'py>,
}

enum Name<'py> {
    /// An exact str other than ``py/object``, shared by its class's instances.
    Known(Slots, u32),
    Other(B<'py>),
}

impl<'py> Name<'py> {
    fn get(&self, py: Python<'py>) -> B<'py> {
        match self {
            Self::Known(slots, at) => slots[*at as usize].name.bind(py).clone().into_any(),
            Self::Other(name) => name.clone(),
        }
    }
}

/// What a record's slots are declared, read as its first part needing it
/// is reached.
#[derive(Default)]
struct SlotTypes {
    types: Option<Rc<Types>>,
}

/// A reduce recipe: a global's bare name, or its parts.
enum Reduced<'py> {
    Name(B<'py>),
    Recipe(Vec<B<'py>>),
}

/// Where a reduce restarts by protocol if one of its parts fails.
struct Checkpoint<'py> {
    value: B<'py>,
    alive: usize,
    sink: Mark,
    here: u32,
    declared: Option<B<'py>>,
}

/// An open container: its parts left, last first, and where they sit.
struct Frame<'py> {
    parts: Vec<Part<'py>>,
    here: u32,
    closes: u8,
    checkpoint: Option<Box<Checkpoint<'py>>>,
}

// --------------------------------------------------------------- walker --

struct Encoder<'py, 'g> {
    py: Python<'py>,
    g: &'g Globals,
    python: bool,
    hooks: Option<B<'py>>,
    sort: Flag<'py>,
    names: Option<ClassNames>,
    overrides: Option<B<'py>>,
    table: Box<Table>,
    forms: Box<Forms>,
    places: Vec<(u32, Step<'py>)>,
    seen: HashMap<usize, u32>,
    alive: Vec<B<'py>>,
    frames: Vec<Frame<'py>>,
    spare: Vec<Vec<Part<'py>>>,
    sink: Sink<'py>,
}

/// A finished walk's emptied buffers, kept for the next call's.
#[derive(Default)]
struct Buffers {
    places: Vec<(u32, Step<'static>)>,
    seen: HashMap<usize, u32>,
    alive: Vec<B<'static>>,
    frames: Vec<Frame<'static>>,
    spare: Vec<Vec<Part<'static>>>,
}

static BUFFERS: GilCell<Option<Buffers>> = GilCell::new(None);

/// Past this many entries a buffer is dropped rather than kept.
const BUFFER_LIMIT: usize = 1 << 16;

impl Buffers {
    fn take(py: Python<'_>) -> Self {
        BUFFERS.take(py).unwrap_or_default()
    }

    /// Hand the buffers to ``encoder``.
    fn bind<'py>(self, encoder: &mut Encoder<'py, '_>) {
        // SAFETY: every buffer is empty, so no value of the old lifetime is used.
        unsafe {
            encoder.places = std::mem::transmute::<Vec<(u32, Step<'static>)>, Vec<(u32, Step<'py>)>>(
                self.places,
            );
            encoder.alive = std::mem::transmute::<Vec<B<'static>>, Vec<B<'py>>>(self.alive);
            encoder.frames =
                std::mem::transmute::<Vec<Frame<'static>>, Vec<Frame<'py>>>(self.frames);
            encoder.spare =
                std::mem::transmute::<Vec<Vec<Part<'static>>>, Vec<Vec<Part<'py>>>>(self.spare);
        }
        encoder.seen = self.seen;
    }

    /// Empty ``encoder``'s buffers and keep them.
    fn keep(encoder: &mut Encoder<'_, '_>) -> Self {
        let mut places = std::mem::take(&mut encoder.places);
        let mut seen = std::mem::take(&mut encoder.seen);
        let mut alive = std::mem::take(&mut encoder.alive);
        let mut frames = std::mem::take(&mut encoder.frames);
        let mut spare = std::mem::take(&mut encoder.spare);
        places.clear();
        seen.clear();
        alive.clear();
        frames.clear();
        for parts in &mut spare {
            parts.clear();
        }
        if places.capacity() > BUFFER_LIMIT
            || alive.capacity() > BUFFER_LIMIT
            || seen.capacity() > BUFFER_LIMIT
        {
            return Self::default();
        }
        // SAFETY: every buffer is empty, so no value of the old lifetime is kept.
        unsafe {
            Self {
                places: std::mem::transmute::<Vec<(u32, Step<'_>)>, Vec<(u32, Step<'static>)>>(
                    places,
                ),
                seen,
                alive: std::mem::transmute::<Vec<B<'_>>, Vec<B<'static>>>(alive),
                frames: std::mem::transmute::<Vec<Frame<'_>>, Vec<Frame<'static>>>(frames),
                spare: std::mem::transmute::<Vec<Vec<Part<'_>>>, Vec<Vec<Part<'static>>>>(spare),
            }
        }
    }
}

impl<'py, 'g> Encoder<'py, 'g> {
    #[inline(always)]
    fn b<T>(&self, value: &'g Py<T>) -> &'g Bound<'py, T> {
        value.bind(self.py)
    }

    #[inline(always)]
    fn is_leaf(&self, value: &B<'py>) -> bool {
        let kind = tp(value);
        kind == t_str()
            || kind == t_int()
            || kind == t_bool()
            || kind == t_none()
            || kind == t_float()
                && (self.python || unsafe { ffi::PyFloat_AS_DOUBLE(value.as_ptr()) }.is_finite())
    }

    fn place(&mut self, parent: u32, step: Step<'py>) -> u32 {
        self.places.push((parent, step));
        (self.places.len() - 1) as u32
    }

    fn register(&mut self, value: &B<'py>, here: u32) {
        self.seen.insert(value.as_ptr() as usize, here);
        self.alive.push(value.clone());
    }

    /// ``_rollback``: forget every registration since ``mark``.
    fn rollback(&mut self, mark: usize) {
        for value in self.alive.drain(mark..) {
            self.seen.remove(&(value.as_ptr() as usize));
        }
    }

    /// The path ``place`` names, or ``None`` if it is inside a key.
    fn reference(&self, mut place: u32) -> Option<Vec<B<'py>>> {
        let mut steps = Vec::new();
        while place != ROOT {
            if place == IN_KEY {
                return None;
            }
            let (parent, step) = &self.places[place as usize];
            steps.push(match step {
                Step::Index(index) => index.into_pyobject(self.py).ok()?.into_any(),
                Step::Key(key) => key.clone(),
                Step::Root => unreachable!("only the root is the root"),
            });
            place = *parent;
        }
        steps.reverse();
        Some(steps)
    }

    fn encloses(&self, place: u32, mut here: u32) -> bool {
        while here != ROOT {
            if here == place {
                return true;
            }
            if here == IN_KEY {
                return false;
            }
            here = self.places[here as usize].0;
        }
        false
    }

    fn spare_parts(&mut self) -> Vec<Part<'py>> {
        self.spare.pop().unwrap_or_default()
    }

    fn push_frame(&mut self, parts: Vec<Part<'py>>, here: u32, closes: u8) {
        self.frames.push(Frame {
            parts,
            here,
            closes,
            checkpoint: None,
        });
    }

    /// Encode every part of the frames above ``base``, depth first.
    fn drain(&mut self, base: usize) -> PyResult<()> {
        while self.frames.len() > base {
            let frame = self.frames.last_mut().expect("above base");
            let Some(part) = frame.parts.pop() else {
                let frame = self.frames.pop().expect("above base");
                for _ in 0..frame.closes {
                    self.sink.close()?;
                }
                self.spare.push(frame.parts);
                continue;
            };
            let parent = frame.here;
            if let Err(error) = self.part(part, parent)
                && let Err(error) = self.recover(error, base)
            {
                return Err(error);
            }
        }
        Ok(())
    }

    #[inline]
    fn part(&mut self, part: Part<'py>, parent: u32) -> PyResult<()> {
        let step = match part.key {
            Key::Index(index) => {
                self.sink.item_at(index);
                Step::Index(index)
            }
            Key::Name(name) => {
                self.sink.key(&name);
                Step::Key(name)
            }
            Key::Known(slots, at) => {
                let slot = &slots[at as usize];
                let name = slot.name.bind(self.py).clone().into_any();
                let text = if self.sink.text() {
                    slot.text(self.py)
                } else {
                    None
                };
                self.sink.key_text(&name, text);
                Step::Key(name)
            }
            Key::Raw(key, declared) => {
                let text = self.escape_key(&key, declared.as_ref())?;
                self.sink.key(&text);
                Step::Key(text)
            }
        };
        if self.is_leaf(&part.value) {
            self.sink.leaf(&part.value)?;
            return Ok(());
        }
        let here = self.place(parent, step);
        self.node(&part.value, here, part.declared.as_ref())
    }

    /// ``_recover``: drop parts down to the innermost reduce above ``base``
    /// and encode it by protocol instead; raise ``error`` if none takes it.
    fn recover(&mut self, mut error: PyErr, base: usize) -> PyResult<()> {
        loop {
            if !error.is_instance_of::<PyTypeError>(self.py) || self.frames.len() <= base {
                return Err(error);
            }
            let frame = self.frames.pop().expect("above base");
            let Some(checkpoint) = frame.checkpoint else {
                continue;
            };
            let Checkpoint {
                value,
                alive,
                sink,
                here,
                declared,
            } = *checkpoint;
            self.rollback(alive);
            self.sink.rewind(sink);
            match self.unreduced(&value, here, declared.as_ref()) {
                Ok(()) => return Ok(()),
                Err(again) => error = again,
            }
        }
    }

    /// ``_encode`` called inside the walk (a key, a set member's sort key):
    /// a tree of its own, with its own reduces to restart.
    fn nested(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<B<'py>> {
        let _depth = Recursion::enter(self.py, c" while encoding a value")?;
        let mark = self.sink.mark();
        let base = self.frames.len();
        self.sink.open_nested();
        self.sink.item();
        let done = self
            .node(value, here, declared)
            .and_then(|()| self.drain(base));
        match done {
            Ok(()) => Ok(self.sink.take_nested()),
            Err(error) => {
                self.frames.truncate(base);
                self.sink.rewind(mark);
                Err(error)
            }
        }
    }

    // ------------------------------------------------------------ nodes --

    fn node(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let kind = tp(value);
        if kind == t_str() || kind == t_int() || kind == t_bool() || kind == t_none() {
            self.sink.leaf(value)?;
            return Ok(());
        }
        if kind == t_float() {
            return self.float(value);
        }
        if let Some(&place) = self.seen.get(&(value.as_ptr() as usize)) {
            if let Some(steps) = self.reference(place) {
                return self.write_reference(steps);
            }
            // Inside a dict key, a repeated value is written in full again;
            // a cycle through the key itself cannot be.
            if self.encloses(place, here) {
                return Err(PyTypeError::new_err(format!(
                    "Cannot serialize a dict key holding a cycle through {}.",
                    repr(value)?
                )));
            }
        }
        if let Some(hooks) = &self.hooks
            && hooked(hooks, value)?
        {
            return self.hooked(value, here);
        }
        match self.encoder(value)? {
            Enc::Float => self.float(value),
            Enc::Complex => self.complex(value),
            Enc::Bytes => self.bytes(value),
            Enc::List => self.list(value, here, declared),
            Enc::Tuple => self.tuple(value, here, declared),
            Enc::Set => self.set(value, here, declared, true),
            Enc::FrozenSet => self.set(value, here, declared, false),
            Enc::Dict => self.dict(value, here, declared),
            Enc::Proxy => self.proxy(value, here, declared),
            Enc::Function => self.function(value, here, declared),
            Enc::Type => {
                let path = self.path(value)?;
                self.tagged_leaf(intern!(self.py, "py/type"), &path)
            }
            Enc::Reduce => self.reduce(value, here, declared),
            Enc::Path => self.leaf_text(value, declared, intern!(self.py, "py/path"), false),
            Enc::Uuid => self.leaf_text(value, declared, intern!(self.py, "py/uuid"), false),
            Enc::Datetime => self.leaf_text(value, declared, intern!(self.py, "py/datetime"), true),
            Enc::Object => self.object(value, here, declared),
        }
    }

    /// ``{tag: ``, a tagged node's start; its payload is written next.
    fn open_tag(&mut self, tag: &Bound<'py, PyString>) -> PyResult<()> {
        self.sink.open(true, false)?;
        self.sink.key(tag.as_any());
        Ok(())
    }

    fn tagged_leaf(&mut self, tag: &Bound<'py, PyString>, value: &B<'py>) -> PyResult<()> {
        self.open_tag(tag)?;
        self.sink.leaf(value)?;
        self.sink.close()
    }

    fn write_reference(&mut self, steps: Vec<B<'py>>) -> PyResult<()> {
        self.open_tag(intern!(self.py, "py/ref"))?;
        self.sink.open(false, false)?;
        for step in &steps {
            self.sink.item();
            self.sink.leaf(step)?;
        }
        self.sink.close()?;
        self.sink.close()
    }

    /// An exact float, tagged where it is not finite and the dialect must.
    fn float(&mut self, value: &B<'py>) -> PyResult<()> {
        let number = unsafe { ffi::PyFloat_AS_DOUBLE(value.as_ptr()) };
        if number.is_finite() || self.python {
            self.sink.leaf(value)?;
            return Ok(());
        }
        let text = PyString::new(self.py, non_finite(number)).into_any();
        self.tagged_leaf(intern!(self.py, "py/float"), &text)
    }

    fn complex(&mut self, value: &B<'py>) -> PyResult<()> {
        let (real, imag) = unsafe {
            (
                ffi::PyComplex_RealAsDouble(value.as_ptr()),
                ffi::PyComplex_ImagAsDouble(value.as_ptr()),
            )
        };
        self.open_tag(intern!(self.py, "py/complex"))?;
        self.sink.open(false, false)?;
        for part in [real, imag] {
            self.sink.item();
            self.float(&part.into_pyobject(self.py)?.into_any())?;
        }
        self.sink.close()?;
        self.sink.close()
    }

    fn bytes(&mut self, value: &B<'py>) -> PyResult<()> {
        let data = unsafe {
            std::slice::from_raw_parts(
                ffi::PyBytes_AS_STRING(value.as_ptr()).cast::<u8>(),
                ffi::PyBytes_Size(value.as_ptr()) as usize,
            )
        };
        let text = PyString::new(self.py, &base64(data)).into_any();
        self.tagged_leaf(intern!(self.py, "py/b64"), &text)
    }

    /// A path, UUID or datetime: its text where its declared type reads that
    /// back as it, else tagged.
    fn leaf_text(
        &mut self,
        value: &B<'py>,
        declared: Option<&B<'py>>,
        tag: &Bound<'py, PyString>,
        datetime: bool,
    ) -> PyResult<()> {
        if let Some(text) = self.untagged_leaf(value, declared)? {
            self.sink.leaf(&text)?;
            return Ok(());
        }
        let text = if datetime {
            self.b(&self.g.datetime_text).call1((value,))?
        } else {
            value.str()?.into_any()
        };
        self.tagged_leaf(tag, &text)
    }

    /// ``_untagged_leaf(value, declared)``: an enum member, datetime, path or
    /// UUID's text where ``declared`` reads it back as ``value``.
    fn untagged_leaf(
        &mut self,
        value: &B<'py>,
        declared: Option<&B<'py>>,
    ) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        let Some(declared) = declared else {
            return Ok(None);
        };
        let kind = value.get_type().into_any();
        let text = match self.leaf_kind(&kind)? {
            Leaf::None => return Ok(None),
            Leaf::Enum => {
                let text = value.getattr(intern!(py, "value"))?;
                let each = tp(&text);
                let finite = each != t_float()
                    || unsafe { ffi::PyFloat_AS_DOUBLE(text.as_ptr()) }.is_finite();
                if !(each == t_str() || each == t_int() || each == t_float()) || !finite {
                    return Ok(None);
                }
                text
            }
            Leaf::Datetime => self.b(&self.g.datetime_text).call1((value,))?,
            Leaf::Text => value.str()?.into_any(),
        };
        let key = (
            kind.as_ptr() as usize,
            tp(&text) as usize,
            declared.as_ptr() as usize,
        );
        let settled = match self.forms.leaves.get(&key) {
            Some((_, settled)) => *settled,
            None => {
                let settled = self
                    .b(&self.g.settled_leaf)
                    .call1((&kind, text.get_type(), declared))?
                    .is_truthy()?;
                let entry = ([kind.clone().unbind(), declared.clone().unbind()], settled);
                drop(self.forms.leaves.insert(key, entry));
                settled
            }
        };
        Ok(settled.then_some(text))
    }

    /// Which of ``_untagged_leaf``'s classes ``kind`` is an instance of.
    fn leaf_kind(&mut self, kind: &B<'py>) -> PyResult<Leaf> {
        if let Some(leaf) = self.facts(kind).leaf {
            return Ok(leaf);
        }
        let subclass = |base: &Py<PyAny>| unsafe {
            ffi::PyType_IsSubtype(kind.as_ptr().cast(), base.as_ptr().cast()) != 0
        };
        let leaf = if subclass(&self.g.enum_type) {
            Leaf::Enum
        } else if subclass(&self.g.datetime) {
            Leaf::Datetime
        } else if subclass(&self.g.pure_path) || subclass(&self.g.uuid) {
            Leaf::Text
        } else {
            Leaf::None
        };
        self.facts(kind).leaf = Some(leaf);
        Ok(leaf)
    }

    /// ``_canonical_reduce(value)``: its reduce recipe, the arguments the
    /// same on every interpreter; ``None`` where it has none to use.
    fn canonical_reduce(&self, value: &B<'py>) -> PyResult<Option<Reduced<'py>>> {
        let py = self.py;
        let reduced = match value.call_method1(intern!(py, "__reduce_ex__"), (2,)) {
            Ok(reduced) => reduced,
            Err(error)
                if error.is_instance_of::<PyAttributeError>(py)
                    || error.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py)
                    || error.is_instance_of::<PyTypeError>(py)
                    || error.is_instance_of::<PyValueError>(py) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if is_str(&reduced) {
            return Ok(Some(Reduced::Name(reduced)));
        }
        if !is_tuple(&reduced) {
            return Ok(None);
        }
        let mut parts: Vec<B<'py>> = reduced.try_iter()?.collect::<PyResult<_>>()?;
        if parts.len() < 2 || parts.len() > 5 || !is_callable(&parts[0]) || !is_tuple(&parts[1]) {
            return Ok(None);
        }
        if value.is_instance(self.b(&self.g.pure_path))? {
            parts[1] = PyTuple::new(py, [value.str()?])?.into_any();
        }
        for part in parts.iter_mut().skip(3) {
            if !part.is_none() {
                *part =
                    PyList::new(py, part.try_iter()?.collect::<PyResult<Vec<_>>>()?)?.into_any();
            }
        }
        if parts.len() > 2 && value.is_instance(self.b(&self.g.partial))? && is_tuple(&parts[2]) {
            // A partial's state ends with its instance ``__dict__``, which
            // CPython creates, empty, the first time anything reads it.
            let state: Vec<B<'py>> = parts[2].try_iter()?.collect::<PyResult<_>>()?;
            if let Some((last, fields)) = state.split_last()
                && py_eq(last, PyDict::new(py).as_any())?
            {
                let mut kept = fields.to_vec();
                kept.push(py.None().into_bound(py));
                parts[2] = PyTuple::new(py, kept)?.into_any();
            }
        }
        Ok(Some(Reduced::Recipe(parts)))
    }

    /// A reduce recipe's bare name: the path of a global in its class's
    /// module, verified to name ``value``. Kept per value where its class
    /// holds no instance dict, so its ``__reduce_ex__`` is its class's.
    fn global_path(&mut self, value: &B<'py>, name: &B<'py>) -> PyResult<B<'py>> {
        let py = self.py;
        let module = value.get_type().getattr(intern!(py, "__module__"))?;
        let path = PyString::new(py, &format!("{}.{}", text(&module)?, text(name)?));
        let path = self.b(&self.g.verified_path).call1((path, value))?;
        let dictless = !crate::class::dict_storage(tp(value));
        if dictless
            && tp(&path) == t_str()
            && let Some(fact) = self.path_fact(&path)?
        {
            self.keep_path(value, fact);
        }
        Ok(path)
    }

    /// Keep ``fact``, the verified path of ``value``, a function or global.
    fn keep_path(&mut self, value: &B<'py>, fact: Rc<PathFact>) {
        let (key, entry) = (value.as_ptr() as usize, (value.clone().unbind(), fact));
        drop(self.table.paths.insert(key, entry));
    }

    /// The path a reduce to a global name gave ``value`` before, while it
    /// still names ``value`` and its class is unchanged.
    fn kept_global(&mut self, value: &B<'py>) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        let Some((_, fact)) = self.table.paths.get(&(value.as_ptr() as usize)) else {
            return Ok(None);
        };
        let fact = fact.clone();
        // A changed class may reduce otherwise; an immutable one cannot change.
        let kind = value.get_type().into_any();
        if crate::class::Stamp::of(&kind).is_some() && self.facts(&kind).enc != Some(Enc::Reduce) {
            return Ok(None);
        }
        Ok(self
            .verified(&fact, value)?
            .then(|| fact.path.bind(py).clone().into_any()))
    }

    /// Whether ``fact``'s path still names ``value``, as ``resolves_to`` says.
    fn verified(&self, fact: &PathFact, value: &B<'py>) -> PyResult<bool> {
        let py = self.py;
        if let Some(check) = &fact.check
            && let Some(holds) = check.holds(&self.g.paths, value)?
        {
            return Ok(holds);
        }
        self.b(&self.g.resolves_to)
            .call1((fact.path.bind(py), value))?
            .is_truthy()
    }

    /// ``_import_path(kind)`` for a heap class under ``type``, read with no
    /// Python code: ``f"{__module__}.{__qualname__}"``, proved natively to
    /// name the class. ``None`` to ask Python.
    fn native_path(&self, kind: &B<'py>) -> PyResult<Option<Rc<PathFact>>> {
        let py = self.py;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        if tp(kind) != &raw mut ffi::PyType_Type
            || unsafe { ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) } == 0
        {
            return Ok(None);
        }
        let own = unsafe { Bound::from_owned_ptr(py, ffi::PyType_GetDict(class)) };
        let module =
            unsafe { ffi::PyDict_GetItem(own.as_ptr(), intern!(py, "__module__").as_ptr()) };
        if module.is_null() {
            return Ok(None);
        }
        let module = unsafe { Bound::from_borrowed_ptr(py, module) };
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
        if qual_text.windows(8).any(|window| window == b"<locals>") {
            return Ok(None);
        }
        let path = dotted(py, module_text, qual_text)?;
        let check = if qual_text.contains(&b'.') {
            crate::class::PathCheck::new(&path, &module).map(facts::Check::Walk)
        } else {
            Some(facts::Check::Top(
                unsafe { module.cast_into_unchecked::<PyString>() }.unbind(),
                std::cell::Cell::new(u64::MAX),
            ))
        };
        let fact = Rc::new(PathFact {
            check,
            path: unsafe { path.cast_into_unchecked::<PyString>() }.unbind(),
        });
        let holds = match &fact.check {
            Some(check) => check.holds(&self.g.paths, kind)?,
            None => None,
        };
        Ok((holds == Some(true)).then_some(fact))
    }

    /// The ``PathFact`` of ``path``, an import path just verified.
    fn path_fact(&self, path: &B<'py>) -> PyResult<Option<Rc<PathFact>>> {
        if tp(path) != t_str() {
            return Ok(None);
        }
        let check = crate::class::PathCheck::loaded(&self.g.paths, path)?.map(facts::Check::Walk);
        Ok(Some(Rc::new(PathFact {
            path: unsafe { path.cast_unchecked::<PyString>() }
                .clone()
                .unbind(),
            check,
        })))
    }

    /// The node of a list, tuple or set: ``open`` and ``tag`` say how it is
    /// written; ``items`` are its parts, in order.
    /// ``items`` are borrowed: from an exact list or tuple, read with no code
    /// run until they are held, or from a list kept alive by the caller.
    fn sequence(
        &mut self,
        items: &[*mut ffi::PyObject],
        form: &Form,
        tag: &Bound<'py, PyString>,
        here: u32,
    ) -> PyResult<()> {
        if self.sink.makes_tree() {
            return self.tree_sequence(items, form, tag, here);
        }
        let mut parts = self.spare_parts();
        parts.reserve(items.len());
        for (index, &item) in items.iter().enumerate().rev() {
            let value = unsafe { Bound::from_borrowed_ptr(self.py, item) };
            let declared = if self.is_leaf(&value) {
                None
            } else {
                form.parts.of_index(self.py, index)?
            };
            parts.push(Part {
                value,
                key: Key::Index(index),
                declared,
            });
        }
        let closes = if form.untagged {
            1
        } else {
            self.open_tag(tag)?;
            2
        };
        self.open_parts(false, parts, None, here, closes)
    }

    /// ``sequence`` into a tree: the list made whole at once, each leaf in
    /// its place, every other item held by ``None`` until it is encoded.
    fn tree_sequence(
        &mut self,
        items: &[*mut ffi::PyObject],
        form: &Form,
        tag: &Bound<'py, PyString>,
        here: u32,
    ) -> PyResult<()> {
        let py = self.py;
        let closes = if form.untagged {
            1
        } else {
            self.open_tag(tag)?;
            2
        };
        let node = unsafe {
            Bound::from_owned_ptr_or_err(py, ffi::PyList_New(items.len() as ffi::Py_ssize_t))?
        };
        let mut parts = self.spare_parts();
        for (index, &item) in items.iter().enumerate() {
            let value = unsafe { Bound::from_borrowed_ptr(py, item) };
            if self.is_leaf(&value) {
                unsafe {
                    ffi::PyList_SET_ITEM(node.as_ptr(), index as ffi::Py_ssize_t, value.into_ptr())
                };
                continue;
            }
            unsafe {
                ffi::PyList_SET_ITEM(
                    node.as_ptr(),
                    index as ffi::Py_ssize_t,
                    py.None().into_ptr(),
                )
            };
            let declared = form.parts.of_index(py, index)?;
            parts.push(Part {
                value,
                key: Key::Index(index),
                declared,
            });
        }
        self.sink.open_filled(node, false);
        if parts.is_empty() {
            self.spare.push(parts);
            for _ in 0..closes {
                self.sink.close()?;
            }
            return Ok(());
        }
        parts.reverse();
        self.push_frame(parts, here, closes);
        Ok(())
    }

    /// Open a list or dict for ``parts`` (last first) and push their frame.
    /// A tree node is made whole at once: each leaf written in its place,
    /// every other part held by ``None`` until it is encoded. ``head`` is a
    /// key and leaf written first, as a ``py/object`` tag is.
    fn open_parts(
        &mut self,
        dict: bool,
        parts: Vec<Part<'py>>,
        head: Option<(&B<'py>, &B<'py>)>,
        here: u32,
        closes: u8,
    ) -> PyResult<()> {
        let named = dict
            && parts
                .iter()
                .all(|part| matches!(part.key, Key::Name(_) | Key::Known(..)));
        if !self.sink.makes_tree() || (dict && !named) {
            self.sink.open(dict, false)?;
            if let Some((key, leaf)) = head {
                self.sink.key(key);
                self.sink.leaf(leaf)?;
            }
            self.push_frame(parts, here, closes);
            return Ok(());
        }
        let py = self.py;
        let none = unsafe { ffi::Py_None() };
        let node = unsafe {
            if dict {
                Bound::from_owned_ptr_or_err(
                    py,
                    ffi::_PyDict_NewPresized((parts.len() + 1) as ffi::Py_ssize_t),
                )?
            } else {
                Bound::from_owned_ptr_or_err(py, ffi::PyList_New(parts.len() as ffi::Py_ssize_t))?
            }
        };
        if let Some((key, leaf)) = head {
            set_item(&node, key, leaf.as_ptr())?;
        }
        let mut kept = self.spare_parts();
        let mut all = parts;
        for (index, part) in all.drain(..).rev().enumerate() {
            let leaf = self.is_leaf(&part.value);
            let value = if leaf { part.value.as_ptr() } else { none };
            if dict {
                let key = match &part.key {
                    Key::Name(name) => name.clone(),
                    Key::Known(slots, at) => slots[*at as usize].name.bind(py).clone().into_any(),
                    _ => unreachable!("only named parts are filled in"),
                };
                set_item(&node, &key, value)?;
            } else {
                unsafe {
                    ffi::Py_INCREF(value);
                    ffi::PyList_SET_ITEM(node.as_ptr(), index as ffi::Py_ssize_t, value);
                }
            }
            if !leaf {
                kept.push(part);
            }
        }
        self.spare.push(all);
        kept.reverse();
        self.sink.open_filled(node, dict);
        self.push_frame(kept, here, closes);
        Ok(())
    }

    /// Open a mapping's dict for its ``parts``: as a tree where escaped keys
    /// may repeat, so one entry per key stands as in Python's dict.
    fn open_entries(
        &mut self,
        parts: Vec<Part<'py>>,
        tree: bool,
        here: u32,
        closes: u8,
    ) -> PyResult<()> {
        if tree {
            self.sink.open(true, true)?;
            self.push_frame(parts, here, closes);
            return Ok(());
        }
        self.open_parts(true, parts, None, here, closes)
    }

    /// ``list(value)``: an exact list or tuple read at once, anything else
    /// iterated.
    /// ``sequence`` over ``list(value)``: an exact list or tuple's own
    /// items, anything else's iterated into a list held for the call.
    fn sequence_of(
        &mut self,
        value: &B<'py>,
        form: &Form,
        tag: &Bound<'py, PyString>,
        here: u32,
    ) -> PyResult<()> {
        let kind = tp(value);
        if kind == t_list() || kind == t_tuple() {
            return self.sequence(unsafe { borrowed_items(value) }, form, tag, here);
        }
        let items = PyList::new(self.py, value.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        self.sequence(unsafe { borrowed_items(items.as_any()) }, form, tag, here)
    }

    fn list(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        self.register(value, here);
        let form = self.form(value, 0, declared, None)?;
        self.sequence_of(value, &form, intern!(self.py, "py/list"), here)
    }

    fn tuple(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let size = value.len()?;
        let form = self.form(value, 1, declared, Some(size))?;
        self.sequence_of(value, &form, intern!(self.py, "py/tuple"), here)
    }

    fn set(
        &mut self,
        value: &B<'py>,
        here: u32,
        declared: Option<&B<'py>>,
        mutable: bool,
    ) -> PyResult<()> {
        if mutable {
            self.register(value, here);
        }
        let form = self.form(value, if mutable { 2 } else { 3 }, declared, None)?;
        let each = form
            .parts
            .each
            .as_ref()
            .map(|each| each.bind(self.py).clone());
        let items = PyList::new(self.py, self.ordered(value, here, each.as_ref())?)?;
        let tag = if mutable {
            intern!(self.py, "py/set")
        } else {
            intern!(self.py, "py/frozenset")
        };
        self.sequence_of(items.as_any(), &form, tag, here)
    }

    /// ``_ordered``: a set's members sorted by their encoding, then ``repr``.
    fn ordered(
        &mut self,
        members: &B<'py>,
        here: u32,
        each: Option<&B<'py>>,
    ) -> PyResult<Vec<B<'py>>> {
        let members: Vec<B<'py>> = members.try_iter()?.collect::<PyResult<_>>()?;
        let mut keys = Vec::with_capacity(members.len());
        for member in &members {
            let mark = self.alive.len();
            let encoded = self.nested(member, here, each);
            self.rollback(mark);
            let encoded = encoded?;
            let style = Style {
                sort_keys: true,
                ..Style::plain()
            };
            let text = dumps(self.py, &encoded, style, true)?;
            keys.push((text, member.repr()?.into_any()));
        }
        let mut order: Vec<usize> = (0..members.len()).collect();
        let compare = |a: &B<'_>, b: &B<'_>| {
            unsafe { ffi::PyUnicode_Compare(a.as_ptr(), b.as_ptr()) }.cmp(&0)
        };
        order.sort_by(|&a, &b| {
            compare(&keys[a].0, &keys[b].0).then_with(|| compare(&keys[a].1, &keys[b].1))
        });
        Ok(order
            .into_iter()
            .map(|index| members[index].clone())
            .collect())
    }

    fn dict(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        self.register(value, here);
        let form = self.form(value, 4, declared, None)?;
        let (parts, tree) = self.entries(value, &form)?;
        let closes = if form.untagged {
            1
        } else {
            self.open_tag(intern!(self.py, "py/dict"))?;
            2
        };
        self.open_entries(parts, tree, here, closes)
    }

    fn proxy(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        self.register(value, here);
        let form = self.form(value, 5, declared, None)?;
        if form.untagged {
            let (parts, tree) = self.entries(value, &form)?;
            return self.open_entries(parts, tree, here, 1);
        }
        // Its entries sit below the dict the tag holds.
        let tag = intern!(self.py, "py/mappingproxy");
        let inner = self.place(here, Step::Key(tag.clone().into_any()));
        let (parts, tree) = self.entries(value, &form)?;
        self.open_tag(tag)?;
        self.open_entries(parts, tree, inner, 2)
    }

    /// ``_entries``: a mapping's parts, and whether its node must be built
    /// as a tree (escaped keys that could repeat).
    fn entries(&mut self, mapping: &B<'py>, form: &Form) -> PyResult<(Vec<Part<'py>>, bool)> {
        let py = self.py;
        let exact = tp(mapping) == t_dict();
        let plain = if exact {
            let mut position = 0;
            let mut key = std::ptr::null_mut();
            let mut item = std::ptr::null_mut();
            let mut plain = true;
            while unsafe { ffi::PyDict_Next(mapping.as_ptr(), &mut position, &mut key, &mut item) }
                != 0
            {
                if !is_plain_key(&unsafe { Bound::from_borrowed_ptr(py, key) })? {
                    plain = false;
                    break;
                }
            }
            plain
        } else {
            let mut plain = true;
            for key in mapping.try_iter()? {
                if !is_plain_key(&key?)? {
                    plain = false;
                    break;
                }
            }
            plain
        };
        let mut items = if exact {
            dict_entries(unsafe { mapping.cast_unchecked::<PyDict>() })
        } else {
            let mut items = Vec::new();
            let mut each = MapItems::new(mapping)?;
            while let Some(entry) = each.next_item()? {
                items.push(entry);
            }
            items
        };
        let mut parts = self.spare_parts();
        parts.reserve(items.len());
        if plain {
            if !exact {
                // ``{key: item for ...}``: one entry per key, first place,
                // last value.
                let node = PyDict::new(py);
                for (key, item) in &items {
                    node.set_item(key, item)?;
                }
                items = dict_entries(&node);
            }
            for (key, value) in items.into_iter().rev() {
                let declared = if self.is_leaf(&value) {
                    None
                } else {
                    form.parts.of_key(py, &key)?
                };
                parts.push(Part {
                    value,
                    key: Key::Name(key),
                    declared,
                });
            }
            return Ok((parts, false));
        }
        let keys = form.keys.as_ref().map(|keys| keys.bind(py).clone());
        let raw_declared = form.parts.of_raw(py);
        let mut tree = false;
        for (key, value) in items.into_iter().rev() {
            if is_plain_key(&key)? {
                let declared = form.parts.of_key(py, &key)?;
                parts.push(Part {
                    value,
                    key: Key::Name(key),
                    declared,
                });
            } else {
                tree = tree || !is_leaf_key(&key);
                parts.push(Part {
                    value,
                    key: Key::Raw(key, keys.clone()),
                    declared: raw_declared.clone(),
                });
            }
        }
        Ok((parts, tree))
    }

    /// ``_encode_key``: a key written as itself, or ``json://`` and its
    /// encoding's JSON text.
    fn escape_key(&mut self, key: &B<'py>, declared: Option<&B<'py>>) -> PyResult<B<'py>> {
        if is_plain_key(key)? {
            return Ok(key.clone());
        }
        let mark = self.alive.len();
        let encoded = self.nested(key, IN_KEY, declared);
        self.rollback(mark);
        let text = dumps(self.py, &encoded?, Style::plain(), false)?;
        Ok(PyString::new(
            self.py,
            &format!("json://{}", text.cast::<PyString>()?.to_cow()?),
        )
        .into_any())
    }

    fn function(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let owner = optional_attr(value, intern!(self.py, "__self__"))?
            .unwrap_or_else(|| self.py.None().into_bound(self.py));
        if !owner.is_none() && !owner.is_instance(self.b(&self.g.module_type))? {
            return self.reduce(value, here, declared);
        }
        let builtin = tp(value) == &raw mut ffi::PyCFunction_Type;
        if builtin && self.unpathed(value) {
            return self.reduce(value, here, declared);
        }
        match self.path(value) {
            Ok(path) => self.tagged_leaf(intern!(self.py, "py/function"), &path),
            // A static builtin method has no path of its own; its reduce
            // recipe names one that imports.
            Err(error) if error.is_instance_of::<PyTypeError>(self.py) && builtin => {
                self.note_unpathed(value)?;
                self.reduce(value, here, declared)
            }
            Err(error) => Err(error),
        }
    }

    /// Whether ``value``'s own path failed to name it before, with
    /// ``sys.modules`` unchanged since: a builtin's path names a fresh
    /// bound method on every lookup, so only an import could change that.
    fn unpathed(&self, value: &B<'py>) -> bool {
        self.table
            .unpathed
            .get(&(value.as_ptr() as usize))
            .is_some_and(|(_, at)| crate::sync::imports() == Some(*at))
    }

    /// Keep that ``_import_path`` just failed on ``value``.
    fn note_unpathed(&mut self, value: &B<'py>) -> PyResult<()> {
        let py = self.py;
        let Some(modules) = self.g.paths.modules(py)? else {
            return Ok(());
        };
        if crate::sync::MODULES.watch(&modules)
            && let Some(at) = crate::sync::imports()
        {
            let (key, entry) = (value.as_ptr() as usize, (value.clone().unbind(), at));
            drop(self.table.unpathed.insert(key, entry));
        }
        Ok(())
    }

    fn hooked(&mut self, value: &B<'py>, here: u32) -> PyResult<()> {
        let py = self.py;
        let kind = value.get_type().into_any();
        let hooks = self.hooks.clone().expect("only hooked values get here");
        let encode = hooks.get_item(&kind)?.get_item(0)?;
        self.register(value, here);
        let path = self.path(&kind)?;
        let payload = encode.call1((value,))?;
        if payload.get_type().is(&kind) {
            let name = kind.getattr(intern!(py, "__name__"))?;
            return Err(PyTypeError::new_err(format!(
                "Cannot serialize {}: its encode hook returned a {}.",
                text(&path)?,
                text(&name)?
            )));
        }
        let mut parts = self.spare_parts();
        parts.push(Part {
            value: payload,
            key: Key::Index(1),
            declared: None,
        });
        parts.push(Part {
            value: path,
            key: Key::Index(0),
            declared: None,
        });
        self.open_tag(intern!(py, "py/hook"))?;
        self.sink.open(false, false)?;
        self.push_frame(parts, here, 2);
        Ok(())
    }

    fn reduce(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let py = self.py;
        if let Some(text) = self.untagged_leaf(value, declared)? {
            self.sink.leaf(&text)?;
            return Ok(());
        }
        if let Some(path) = self.kept_global(value)? {
            return self.tagged_leaf(intern!(py, "py/type"), &path);
        }
        let mut items = match self.canonical_reduce(value)? {
            None => return self.unreduced(value, here, declared),
            Some(Reduced::Name(name)) => {
                let path = self.global_path(value, &name)?;
                return self.tagged_leaf(intern!(py, "py/type"), &path);
            }
            Some(Reduced::Recipe(items)) => items,
        };
        let alive = self.alive.len();
        if is_unhashable(value)? || items.iter().skip(2).any(|part| !part.is_none()) {
            self.register(value, here);
        }
        while items.last().is_some_and(PyAnyMethods::is_none) {
            items.pop();
        }
        let checkpoint = Checkpoint {
            value: value.clone(),
            alive,
            sink: self.sink.mark(),
            here,
            declared: declared.cloned(),
        };
        let mut parts = self.spare_parts();
        for (index, part) in items.into_iter().enumerate().rev() {
            parts.push(Part {
                value: part,
                key: Key::Index(index),
                declared: None,
            });
        }
        self.open_tag(intern!(py, "py/reduce"))?;
        self.sink.open(false, false)?;
        self.frames.push(Frame {
            parts,
            here,
            closes: 2,
            checkpoint: Some(Box::new(checkpoint)),
        });
        Ok(())
    }

    /// ``_encode_unreduced``: a value with no usable reduce recipe, by the
    /// protocols it has.
    fn unreduced(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let py = self.py;
        let kind = value.get_type();
        if kind.hasattr(intern!(py, "__dataclass_fields__"))?
            || kind
                .getattr(intern!(py, "__dict__"))?
                .contains(intern!(py, "__slots__"))?
        {
            return self.object(value, here, declared);
        }
        if kind.is_subclass(self.b(&self.g.mapping))? {
            return self.dict(value, here, declared);
        }
        if kind.is_subclass(self.b(&self.g.sequence))?
            && !kind.is_subclass(self.b(&self.g.str_bytes))?
        {
            return self.list(value, here, declared);
        }
        if kind.is_subclass(self.b(&self.g.abstract_set))? {
            return self.set(value, here, declared, true);
        }
        let name = text(&kind.getattr(intern!(py, "__name__"))?)?;
        Err(PyTypeError::new_err(format!(
            "Cannot serialize leaf of type '{name}'. Pass hooks={{{name}: (encode, decode)}} to to_plain()."
        )))
    }

    // --------------------------------------------------------- objects --

    fn object(&mut self, value: &B<'py>, here: u32, declared: Option<&B<'py>>) -> PyResult<()> {
        let py = self.py;
        let kind = value.get_type().into_any();
        self.register(value, here);
        if self.slotted_object(value, &kind, here, declared)? {
            return Ok(());
        }
        let mut members = self.members(value, &kind)?;
        let record = self.is_dataclass(&kind)?;
        if record && let Some(defaults) = self.defaults_of(&kind)? {
            let mut kept = Vec::with_capacity(members.len());
            for member in members {
                let name = member.name.get(py);
                let found = defaults.find(&name);
                if self.left_out(&defaults, found, here, &name, &member.value, true)? != Some(true)
                {
                    kept.push(member);
                }
            }
            members = kept;
        }
        let implied = record && self.record_implied(&kind, declared)?;
        let types =
            if record && (implied || members.iter().any(|member| !is_plain_leaf(&member.value))) {
                Some(self.types(&kind)?)
            } else {
                None
            };
        let untagged = match &types {
            Some(types) if implied => self.within(&members, types)?,
            _ => false,
        };
        let tag = if untagged {
            None
        } else {
            Some(match &self.names {
                Some(names) => names.tag(py, &kind).map(Bound::into_any).ok_or_else(|| {
                    PyTypeError::new_err(format!(
                        "Cannot serialize {}: it is not in classes.",
                        repr(&kind).unwrap_or_default()
                    ))
                })?,
                None => self.path(&kind)?,
            })
        };
        // A member named ``py/object`` takes the tag's place, first, as
        // ``{"py/object": tag, **members}`` places it.
        let mut first = None;
        if tag.is_some() {
            for (index, member) in members.iter().enumerate() {
                if let Name::Other(name) = &member.name
                    && is_str(name)
                    && py_eq(name, intern!(py, "py/object").as_any())?
                {
                    first = Some(index);
                    break;
                }
            }
        }
        let exact = members.iter().all(|member| match &member.name {
            Name::Known(..) => true,
            Name::Other(name) => tp(name) == t_str(),
        });
        let mut parts = self.spare_parts();
        parts.reserve(members.len());
        let mut moved = None;
        for (index, member) in members.into_iter().enumerate().rev() {
            let declared = match &types {
                Some(types) if !self.is_leaf(&member.value) => {
                    self.declared_member(types, &member.name)?
                }
                _ => None,
            };
            let part = Part {
                value: member.value,
                key: match member.name {
                    Name::Known(slots, at) => Key::Known(slots, at),
                    Name::Other(name) => Key::Name(name),
                },
                declared,
            };
            if Some(index) == first {
                moved = Some(part);
            } else {
                parts.push(part);
            }
        }
        parts.extend(moved);
        let head = intern!(py, "py/object").as_any();
        let head = tag
            .as_ref()
            .filter(|_| first.is_none())
            .map(|tag| (head, tag));
        if !exact {
            self.sink.open(true, true)?;
            if let Some((key, leaf)) = head {
                self.sink.key(key);
                self.sink.leaf(leaf)?;
            }
            self.push_frame(parts, here, 1);
            return Ok(());
        }
        self.open_parts(true, parts, head, here, 1)
    }

    /// ``object`` for an instance of a dataclass whose state is its slots,
    /// each an exact str read at its offset, and whose field types name
    /// them all: one read of the class's facts, no member list. ``false``
    /// where ``object`` must read it the long way.
    fn slotted_object(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        here: u32,
        declared: Option<&B<'py>>,
    ) -> PyResult<bool> {
        let py = self.py;
        // One read of the class's facts where it was met before.
        let (dataclass, layout, path, ready, defaults) = {
            let facts = self.facts(kind);
            let ready = facts
                .layout
                .as_ref()
                .is_some_and(|layout| layout.hints.get().is_some())
                && facts
                    .types
                    .as_ref()
                    .is_some_and(|(_, guard)| guard.is_empty());
            (
                facts.dataclass,
                facts.layout.clone(),
                facts.path.clone(),
                ready,
                facts.defaults.clone(),
            )
        };
        let dataclass = match dataclass {
            Some(known) => known,
            None => self.is_dataclass(kind)?,
        };
        if !dataclass {
            return Ok(false);
        }
        let layout = match layout {
            Some(layout) => layout,
            None => self.layout_of(kind)?,
        };
        if !(layout.plain && layout.plain_slots) {
            return Ok(false);
        }
        let sort = self.sort.get()?;
        let implied = self.record_implied(kind, declared)?;
        let tag = if implied {
            None
        } else {
            Some(match &self.names {
                Some(names) => match names.tag(py, kind) {
                    Some(tag) => tag.into_any(),
                    None => return Ok(false),
                },
                None => match path {
                    Some(fact) if self.verified(&fact, kind)? => {
                        fact.path.bind(py).clone().into_any()
                    }
                    _ => self.path(kind)?,
                },
            })
        };
        let defaults = match defaults {
            Some(known) => known,
            None => match self
                .g
                .defaults
                .defaulted_slots(kind, layout.slots.iter().map(|slot| &slot.name))
            {
                Some(found) => {
                    let found = found.map(Rc::new);
                    self.facts(kind).defaults = Some(found.clone());
                    found
                }
                None => self.defaults_of(kind)?,
            },
        };
        let by_slot = defaults.as_ref().map(|defaults| {
            defaults.by_slot.get_or_init(|| {
                layout
                    .slots
                    .iter()
                    .map(|slot| {
                        defaults
                            .find(slot.name.bind(py).as_any())
                            .map_or(u32::MAX, |found| found as u32)
                    })
                    .collect()
            })
        });
        let alive = self.alive.len();
        let mut read = SlotTypes::default();
        // A tree node is made whole at once: each member set in its place,
        // one that is no leaf replaced by its encoding once that is made.
        if !layout.dictless && !ready {
            // Read before the instance dict: reading them may run Python
            // code, which may change it. Kept, they run none.
            read.types = Some(self.types(kind)?);
            self.slot_hints(kind, &layout)?;
        }
        // An instance dict's entries, as a kept ``DictPlan`` names them.
        let mut values = Vec::new();
        let plan = if layout.dictless {
            None
        } else {
            let state = unsafe {
                Bound::from_owned_ptr_or_err(
                    py,
                    ffi::PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()),
                )?
            };
            if tp(&state) != t_dict() {
                return Ok(false);
            }
            let dict = unsafe { state.cast_unchecked::<PyDict>() };
            if dict.is_empty() {
                None
            } else {
                let Some(plan) = self.dict_plan(kind, &layout, dict, sort, &mut values) else {
                    return Ok(false);
                };
                Some((plan, state))
            }
        };
        // A record goes untagged only where every member is a field:
        // ``_field_types`` names every slot, so only dict keys are asked. The
        // types are kept by now, so asking runs no code that could change the
        // dict read above; a key that is no field leaves it to ``object``.
        if implied && let Some((plan, _)) = &plan {
            let types = match &read.types {
                Some(types) => types.clone(),
                None => self.types(kind)?,
            };
            for key in plan.keys.iter() {
                if types.get(py, key.name.bind(py).as_any()).is_none() {
                    return Ok(false);
                }
            }
        }
        // Text is written by the frame, every member a part.
        let tree = self.sink.makes_tree();
        let count = layout.slots.len() + values.len();
        let keys = plan.as_ref().map(|(plan, _)| plan.keys.clone());
        let node = if tree {
            let node = unsafe {
                Bound::from_owned_ptr_or_err(
                    py,
                    ffi::_PyDict_NewPresized((count + 1) as ffi::Py_ssize_t),
                )?
            };
            if let Some(tag) = &tag {
                set_item(&node, intern!(py, "py/object").as_any(), tag.as_ptr())?;
            }
            Some(node)
        } else {
            None
        };
        // Text: leaves are written at once until the first member that is
        // none, which with every member after it is left to the frame.
        let mark = self.sink.mark();
        if !tree {
            self.sink.open(true, false)?;
            if let Some(tag) = &tag {
                self.sink.key(intern!(py, "py/object").as_any());
                self.sink.leaf(tag)?;
            }
        }
        let pending = self.g.pending.as_ptr();
        let mut parts = self.spare_parts();
        let order = layout.order(py, sort);
        let picks = match &plan {
            Some((plan, _)) => plan.order.len(),
            None => layout.slots.len(),
        };
        for index in 0..picks {
            let pick = match &plan {
                Some((plan, _)) => plan.order[index],
                None => Pick::Slot(order.get(index).map_or(index as u32, |&at| at)),
            };
            let (owner, slot, member) = match pick {
                Pick::Slot(at) => {
                    let offset = layout.slots[at as usize]
                        .offset
                        .expect("plain slots have offsets");
                    let member = unsafe {
                        *value
                            .as_ptr()
                            .cast::<u8>()
                            .offset(offset)
                            .cast::<*mut ffi::PyObject>()
                    };
                    (&layout.slots, at, member)
                }
                Pick::Key(at) => (
                    keys.as_ref().expect("a key is picked from a plan"),
                    at,
                    values[at as usize],
                ),
            };
            if member.is_null() || member == pending {
                continue;
            }
            let name = owner[slot as usize].name.bind(py).clone().into_any();
            if let (Some(defaults), Some(by_slot)) = (&defaults, by_slot) {
                let found = match pick {
                    Pick::Slot(at) => Some(by_slot[at as usize] as usize)
                        .filter(|&found| found != u32::MAX as usize),
                    Pick::Key(_) => defaults.find(&name),
                };
                let held = unsafe { Bound::from_borrowed_ptr(py, member) };
                match self.left_out(defaults, found, here, &name, &held, false)? {
                    Some(true) => continue,
                    Some(false) => {}
                    // A factory runs Python code, which could free a member
                    // borrowed here; ``object`` holds each one it reads.
                    None => {
                        self.rollback(alive);
                        self.sink.rewind(mark);
                        return Ok(false);
                    }
                }
            }
            if let Some(node) = &node
                && unsafe { ffi::PyDict_SetItem(node.as_ptr(), name.as_ptr(), member) } < 0
            {
                return Err(PyErr::fetch(py));
            }
            let member = unsafe { Bound::from_borrowed_ptr(py, member) };
            if self.is_leaf(&member) {
                if !tree {
                    if parts.is_empty() {
                        let text = owner[slot as usize].text(py);
                        self.sink.key_text(&name, text);
                        self.sink.leaf(&member)?;
                    } else {
                        parts.push(Part {
                            value: member,
                            key: Key::Known(owner.clone(), slot),
                            declared: None,
                        });
                    }
                }
                continue;
            }
            let declared = match pick {
                Pick::Slot(at) => match self.slot_hint(kind, &layout, &mut read, at)? {
                    Some(declared) => declared,
                    None => {
                        // ``object`` reads the record again, defaults included.
                        self.rollback(alive);
                        self.sink.rewind(mark);
                        return Ok(false);
                    }
                },
                Pick::Key(_) => {
                    if read.types.is_none() {
                        read.types = Some(self.types(kind)?);
                    }
                    let types = read.types.as_ref().expect("just read");
                    types.get(py, &name).flatten()
                }
            };
            parts.push(Part {
                value: member,
                key: Key::Known(owner.clone(), slot),
                declared,
            });
        }
        parts.reverse();
        if let Some(node) = node {
            self.sink.open_filled(node, true);
        }
        if parts.is_empty() {
            self.spare.push(parts);
            self.sink.close()?;
            return Ok(true);
        }
        self.push_frame(parts, here, 1);
        Ok(true)
    }

    /// ``kind``'s ``SlotHints``, read once into ``layout``; ``None`` where
    /// only ``_field_types`` can say.
    fn slot_hints<'l>(
        &self,
        kind: &B<'py>,
        layout: &'l Layout,
    ) -> PyResult<Option<&'l facts::SlotHints>> {
        if let Some(hints) = layout.hints.get() {
            return Ok(hints.as_ref());
        }
        let hints = facts::SlotHints::read(self.py, &self.g.hints, kind)?;
        Ok(layout.hints.get_or_init(|| hints).as_ref())
    }

    /// The declared type of the slot at ``at``, ``None`` inside for
    /// ``object``: read from the class's ``SlotHints``, or from
    /// ``_field_types`` (kept in ``types``) where a hint is not plain.
    /// ``None`` where neither tells.
    fn slot_hint(
        &mut self,
        kind: &B<'py>,
        layout: &Layout,
        read: &mut SlotTypes,
        at: u32,
    ) -> PyResult<Option<Option<B<'py>>>> {
        let py = self.py;
        if let Some(hints) = self.slot_hints(kind, layout)? {
            match hints.get(py, &self.g.hints, &layout.slots[at as usize])? {
                facts::Hint::Object => return Ok(Some(None)),
                facts::Hint::Type(hint) => return Ok(Some(Some(hint.into_bound(py)))),
                facts::Hint::Ask => {}
            }
        }
        if read.types.is_none() {
            read.types = Some(self.types(kind)?);
        }
        let types = read.types.as_ref().expect("just read");
        Ok(types
            .slots
            .get(at as usize)
            .map(|declared| declared.as_ref().map(|each| each.bind(py).clone())))
    }

    /// ``types.get(name, object)``, ``None`` for ``object``.
    fn declared_member(&self, types: &Types, name: &Name<'py>) -> PyResult<Option<B<'py>>> {
        let py = self.py;
        let name = name.get(py);
        if tp(&name) == t_str() {
            return Ok(types.get(py, &name).flatten());
        }
        if !is_str(&name) {
            return Ok(None);
        }
        let object = py.get_type::<PyAny>().into_any();
        Ok(declared(
            py,
            types
                .mapping
                .bind(py)
                .call_method1(intern!(py, "get"), (name, object))?,
        ))
    }

    /// ``members.keys() <= types.keys()``.
    fn within(&self, members: &[Member<'py>], types: &Types) -> PyResult<bool> {
        let py = self.py;
        for member in members {
            let name = member.name.get(py);
            let found = if tp(&name) == t_str() {
                types.get(py, &name).is_some()
            } else {
                types.mapping.bind(py).contains(name)?
            };
            if !found {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[inline]
    fn facts(&mut self, kind: &B<'py>) -> &mut Facts {
        self.table.of(kind)
    }

    /// Whether ``member`` is written as other than itself: a ``py/ref`` to
    /// where it was met before, or by its hook.
    fn referred(&self, member: &B<'py>) -> bool {
        self.seen.contains_key(&(member.as_ptr() as usize))
            || self
                .hooks
                .as_ref()
                .is_some_and(|hooks| hooked(hooks, member).unwrap_or(true))
    }

    /// ``kind``'s fields with a default, read once per class; ``None`` where
    /// it has none.
    fn defaults_of(&mut self, kind: &B<'py>) -> PyResult<Option<facts::Defaulted>> {
        if let Some(found) = &self.facts(kind).defaults {
            return Ok(found.clone());
        }
        let found = self.g.defaults.defaulted(kind)?.map(Rc::new);
        self.facts(kind).defaults = Some(found.clone());
        Ok(found)
    }

    /// ``_leave_out_defaults`` for one member, ``defaults``' field ``found``:
    /// whether ``member``, the record at ``here``'s ``name``, is what its
    /// default rebuilds, and so left out; registered where the walk would
    /// register it written. ``None`` where that needs Python code run and
    /// ``call`` is not set.
    fn left_out(
        &mut self,
        defaults: &crate::fields::ClassDefaults,
        found: Option<usize>,
        here: u32,
        name: &B<'py>,
        member: &B<'py>,
        call: bool,
    ) -> PyResult<Option<bool>> {
        let Some(found) = found else {
            return Ok(Some(false));
        };
        if !self.g.defaults.omittable(member)? {
            return Ok(Some(false));
        }
        let default = &defaults.fields[found];
        let sample = match self.g.defaults.sample(self.py, default, false)? {
            Some(sample) => sample,
            // Python builds the default only for a member it may leave out.
            None if self.referred(member) => return Ok(Some(false)),
            None if !call => return Ok(None),
            None => self
                .g
                .defaults
                .sample(self.py, default, true)?
                .expect("a call reads the sample"),
        };
        if !self.g.defaults.inferred(member, &sample)? || self.referred(member) {
            return Ok(Some(false));
        }
        if crate::fields::registered_kind(tp(member)) {
            let child = self.place(here, Step::Key(name.clone()));
            self.register(member, child);
        }
        Ok(Some(true))
    }

    /// ``_encoder_for(type(value))``.
    #[inline]
    fn encoder(&mut self, value: &B<'py>) -> PyResult<Enc> {
        let kind = tp(value);
        if let Some(enc) = facts::exact_encoder(kind) {
            return Ok(enc);
        }
        let kind = value.get_type().into_any();
        if let Some(enc) = self.facts(&kind).enc {
            return Ok(enc);
        }
        let enc = self.read_encoder(&kind)?;
        self.facts(&kind).enc = Some(enc);
        Ok(enc)
    }

    fn read_encoder(&mut self, kind: &B<'py>) -> PyResult<Enc> {
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let bases = unsafe { Bound::from_borrowed_ptr(self.py, (*class).tp_mro.cast()) };
        let items = unsafe { borrowed_items(&bases) };
        // ``base in __mro__``: identity, where every class's ``==`` is ``type``'s.
        let plain = items
            .iter()
            .all(|&each| ob_type(each) == &raw mut ffi::PyType_Type);
        let has = |base: &B<'py>| -> PyResult<bool> {
            if plain {
                return Ok(items.contains(&base.as_ptr()));
            }
            bases.contains(base)
        };
        if has(self.py.get_type::<pyo3::types::PyType>().as_any())? {
            return Ok(Enc::Type);
        }
        if has(self.b(&self.g.enum_type))? {
            return Ok(Enc::Reduce);
        }
        if kind.is(self.b(&self.g.path)) || kind.is(self.b(&self.g.concrete_path)) {
            return Ok(Enc::Path);
        }
        if kind.is(self.b(&self.g.uuid)) {
            return Ok(Enc::Uuid);
        }
        if kind.is(self.b(&self.g.datetime)) {
            return Ok(Enc::Datetime);
        }
        if self.is_dataclass(kind)? {
            return Ok(Enc::Object);
        }
        Ok(Enc::Reduce)
    }

    /// ``is_dataclass(kind)``: whether the class has ``__dataclass_fields__``.
    fn is_dataclass(&mut self, kind: &B<'py>) -> PyResult<bool> {
        if let Some(known) = self.facts(kind).dataclass {
            return Ok(known);
        }
        // A dataclass's own namespace holds its fields dict, which ``getattr``
        // gives as it is: no descriptor, and the metaclass ``type``.
        let own = unsafe { (*kind.as_ptr().cast::<ffi::PyTypeObject>()).tp_dict };
        let found = if tp(kind) == &raw mut ffi::PyType_Type && !own.is_null() {
            unsafe { ffi::PyDict_GetItem(own, intern!(self.py, "__dataclass_fields__").as_ptr()) }
        } else {
            std::ptr::null_mut()
        };
        if !found.is_null() && unsafe { (*ob_type(found)).tp_descr_get.is_none() } {
            self.facts(kind).dataclass = Some(true);
            return Ok(true);
        }
        let name = intern!(self.py, "__dataclass_fields__");
        let known = match unsafe { crate::class::class_attr(kind.as_ptr(), name.as_ptr()) } {
            crate::class::ClassAttr::Found(..) => true,
            crate::class::ClassAttr::Missing => false,
            crate::class::ClassAttr::Ask => return kind.hasattr(name),
        };
        self.facts(kind).dataclass = Some(known);
        Ok(known)
    }

    fn layout_of(&mut self, kind: &B<'py>) -> PyResult<Rc<Layout>> {
        if let Some(layout) = &self.facts(kind).layout {
            return Ok(layout.clone());
        }
        if let Some(layout) = facts::own_layout(self.py, kind) {
            let layout = Rc::new(layout);
            self.facts(kind).layout = Some(layout.clone());
            return Ok(layout);
        }
        let names = match facts::slot_names(self.py, kind) {
            Some(names) => names,
            None => self
                .b(&self.g.slot_names)
                .call1((kind,))?
                .try_iter()?
                .map(|name| Ok(name?.cast_into::<PyString>()?.unbind()))
                .collect::<PyResult<_>>()?,
        };
        let layout = facts::layout(self.py, kind, names, self.g.dict_getter);
        let layout = Rc::new(layout);
        self.facts(kind).layout = Some(layout.clone());
        Ok(layout)
    }

    /// ``attribute_names`` with each ``getattr`` that finds a member: the
    /// members in order, ``PENDING`` and unset slots left out.
    fn members(&mut self, value: &B<'py>, kind: &B<'py>) -> PyResult<Vec<Member<'py>>> {
        let layout = self.layout_of(kind)?;
        let sort = self.sort.get()?;
        if layout.plain && layout.plain_slots {
            if layout.dictless {
                return Ok(self.slot_members(value, &layout, sort));
            }
            if let Some(members) = self.planned_members(value, kind, &layout, sort)? {
                return Ok(members);
            }
        }
        self.read_members(value, kind, &layout, sort)
    }

    /// The slots of ``value``, each read at its offset.
    fn slot_members(&self, value: &B<'py>, layout: &Rc<Layout>, sort: bool) -> Vec<Member<'py>> {
        let count = layout.slots.len();
        let mut members = Vec::with_capacity(count);
        let order = layout.order(self.py, sort);
        for index in 0..count {
            let slot = order.get(index).map_or(index as u32, |&at| at);
            if let Some(member) = layout.slots[slot as usize]
                .offset
                .and_then(|offset| self.read_slot(value, offset))
            {
                members.push(Member {
                    name: Name::Known(layout.slots.clone(), slot),
                    value: member,
                });
            }
        }
        members
    }

    /// The object held at ``offset`` in ``value``, unless unset or ``PENDING``.
    #[inline]
    fn read_slot(&self, value: &B<'py>, offset: isize) -> Option<B<'py>> {
        let held = unsafe {
            *value
                .as_ptr()
                .cast::<u8>()
                .offset(offset)
                .cast::<*mut ffi::PyObject>()
        };
        (!held.is_null() && held != self.g.pending.as_ptr())
            .then(|| unsafe { Bound::from_borrowed_ptr(self.py, held) })
    }

    /// The members of an instance whose dict ``layout`` has a ``DictPlan``
    /// for, or can make one; ``None`` to read them by ``getattr``.
    fn planned_members(
        &mut self,
        value: &B<'py>,
        kind: &B<'py>,
        layout: &Rc<Layout>,
        sort: bool,
    ) -> PyResult<Option<Vec<Member<'py>>>> {
        let py = self.py;
        let state = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                ffi::PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()),
            )?
        };
        if tp(&state) != t_dict() {
            return Ok(None);
        }
        let dict = unsafe { state.cast_unchecked::<PyDict>() };
        if dict.is_empty() {
            return Ok(Some(self.slot_members(value, layout, sort)));
        }
        let mut values = Vec::with_capacity(dict.len());
        let Some(plan) = self.dict_plan(kind, layout, dict, sort, &mut values) else {
            return Ok(None);
        };
        let mut members = Vec::with_capacity(plan.order.len());
        for pick in &plan.order {
            match *pick {
                Pick::Slot(at) => {
                    if let Some(member) = layout.slots[at as usize]
                        .offset
                        .and_then(|offset| self.read_slot(value, offset))
                    {
                        members.push(Member {
                            name: Name::Known(layout.slots.clone(), at),
                            value: member,
                        });
                    }
                }
                Pick::Key(at) => {
                    let member = values[at as usize];
                    if member != self.g.pending.as_ptr() {
                        members.push(Member {
                            name: Name::Known(plan.keys.clone(), at),
                            value: unsafe { Bound::from_borrowed_ptr(py, member) },
                        });
                    }
                }
            }
        }
        Ok(Some(members))
    }

    /// The ``DictPlan`` for instance dict ``dict``, kept or made, with its
    /// values pushed to ``values`` in its keys' order; ``None`` where some
    /// key must be read by ``getattr``.
    fn dict_plan(
        &mut self,
        kind: &B<'py>,
        layout: &Layout,
        dict: &Bound<'py, PyDict>,
        sort: bool,
        values: &mut Vec<*mut ffi::PyObject>,
    ) -> Option<Rc<DictPlan>> {
        let held = &layout.plans[usize::from(!sort)];
        let kept = held.borrow().clone();
        if let Some(plan) = kept {
            if dict_matches(dict, &plan.keys, values) {
                return Some(plan);
            }
            values.clear();
        }
        let plan = Rc::new(self.plan_dict(kind, layout, dict, sort)?);
        dict_matches(dict, &plan.keys, values);
        let stale = held.replace(Some(plan.clone()));
        drop(stale);
        Some(plan)
    }

    /// A ``DictPlan`` for instance dicts keyed as ``dict`` is, or ``None``
    /// where some key must be read by ``getattr``.
    fn plan_dict(
        &mut self,
        kind: &B<'py>,
        layout: &Layout,
        dict: &Bound<'py, PyDict>,
        sort: bool,
    ) -> Option<DictPlan> {
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let slots = &layout.slots;
        let mut keys = Vec::with_capacity(dict.len());
        for (key, _) in dict_entries(dict) {
            let text = (tp(&key) == t_str()).then(|| utf8(&key)).flatten()?;
            let shadowed = unsafe {
                let found = crate::class::type_lookup(class, key.as_ptr());
                !found.is_null() && (*ffi::Py_TYPE(found)).tp_descr_set.is_some()
            };
            let slot = slots.iter().any(
                |each| unsafe { ffi::PyUnicode_Compare(each.name.as_ptr(), key.as_ptr()) } == 0,
            );
            if text == b"py/object" || shadowed || slot {
                return None;
            }
            keys.push(unsafe { key.cast_into_unchecked::<PyString>() }.unbind());
        }
        let keys = facts::slots(keys.into_iter().map(|key| (key, None)));
        let mut order: Vec<Pick> = layout
            .positions(self.py, sort)
            .into_iter()
            .map(Pick::Slot)
            .collect();
        order.extend((0..keys.len() as u32).map(Pick::Key));
        if sort {
            let name = |pick: &Pick| match *pick {
                Pick::Slot(at) => slots[at as usize].name.as_ptr(),
                Pick::Key(at) => keys[at as usize].name.as_ptr(),
            };
            order.sort_by(|a, b| unsafe { ffi::PyUnicode_Compare(name(a), name(b)) }.cmp(&0));
        }
        Some(DictPlan { keys, order })
    }

    /// ``attribute_names`` and ``getattr``, as Python reads them.
    fn read_members(
        &self,
        value: &B<'py>,
        kind: &B<'py>,
        layout: &Rc<Layout>,
        sort: bool,
    ) -> PyResult<Vec<Member<'py>>> {
        let py = self.py;
        let slots = &layout.slots;
        let order = layout.positions(py, sort);
        // ``getattr(value, "__dict__", None)``
        let state = if layout.dictless {
            None
        } else if layout.plain {
            let dict =
                unsafe { ffi::PyObject_GenericGetDict(value.as_ptr(), std::ptr::null_mut()) };
            Some(unsafe { Bound::from_owned_ptr_or_err(py, dict)? })
        } else {
            match value.getattr(intern!(py, "__dict__")) {
                Ok(state) => Some(state),
                Err(error) if error.is_instance_of::<PyAttributeError>(py) => None,
                Err(error) => return Err(error),
            }
        };
        let state = match state {
            Some(state) if state.is_truthy()? => Some(state),
            _ => None,
        };
        let mut names: Vec<(B<'py>, Option<u32>)> = order
            .iter()
            .map(|&slot| {
                (
                    slots[slot as usize].name.bind(py).clone().into_any(),
                    Some(slot),
                )
            })
            .collect();
        if let Some(state) = &state {
            let slot_names = PyTuple::new(py, names.iter().map(|(name, _)| name))?;
            let mut extra = Vec::new();
            for key in state.try_iter()? {
                let key = key?;
                if !slot_names.contains(&key)? {
                    extra.push((key, None));
                }
            }
            if !extra.is_empty() {
                names.extend(extra);
                if sort {
                    let all = PyList::new(py, names.iter().map(|(name, _)| name))?;
                    all.sort()?;
                    let mut ordered = Vec::with_capacity(names.len());
                    for name in all.iter() {
                        let at = names
                            .iter()
                            .position(|(each, _)| each.is(&name))
                            .expect("sorted from these names");
                        ordered.push(names[at].clone());
                    }
                    names = ordered;
                }
            }
        }
        let instance_dict = layout
            .plain
            .then(|| state.as_ref().filter(|state| tp(state) == t_dict()))
            .flatten();
        let mut members = Vec::with_capacity(names.len());
        for (name, slot) in names {
            let offset = slot.and_then(|slot| slots[slot as usize].offset);
            let member = match offset {
                Some(offset) => self.read_slot(value, offset),
                None => self.member(value, kind, &name, instance_dict)?,
            };
            if let Some(member) = member {
                let name = match slot {
                    Some(at) if layout.plain_slots => Name::Known(layout.slots.clone(), at),
                    _ => Name::Other(name),
                };
                members.push(Member {
                    name,
                    value: member,
                });
            }
        }
        Ok(members)
    }

    /// ``getattr(value, name, PENDING)``, ``None`` for ``PENDING``; read from
    /// the instance dict where nothing on the class shadows it.
    fn member(
        &self,
        value: &B<'py>,
        kind: &B<'py>,
        name: &B<'py>,
        instance_dict: Option<&B<'py>>,
    ) -> PyResult<Option<B<'py>>> {
        let pending = self.g.pending.as_ptr();
        if let Some(dict) = instance_dict
            && tp(name) == t_str()
        {
            let shadowed = unsafe {
                let found = crate::class::type_lookup(kind.as_ptr().cast(), name.as_ptr());
                !found.is_null() && (*ffi::Py_TYPE(found)).tp_descr_set.is_some()
            };
            if !shadowed
                && let Some(found) = unsafe { dict.cast_unchecked::<PyDict>() }.get_item(name)?
            {
                return Ok((found.as_ptr() != pending).then_some(found));
            }
        }
        let name = unsafe { name.cast_unchecked::<PyString>() };
        Ok(optional_attr(value, name)?.filter(|found| found.as_ptr() != pending))
    }

    /// ``_field_types(kind)``, kept while the class is unchanged: for good
    /// where every hint resolved, else while no module the hints are read
    /// in gains a name, which is all that could resolve one.
    fn types(&mut self, kind: &B<'py>) -> PyResult<Rc<Types>> {
        let py = self.py;
        if let Some((types, guard)) = &self.facts(kind).types
            && guard
                .iter()
                .all(|(namespace, size)| dict_size(namespace.bind(py)) == *size)
        {
            return Ok(types.clone());
        }
        let layout = self.layout_of(kind)?;
        let native = facts::plain_field_types(py, &self.g.hints, kind, &layout.slots)?;
        let cached = match native {
            Some(native) => native.into_any(),
            None => self
                .b(&self.g.field_types_cache)
                .call_method1(intern!(py, "get"), (kind,))?,
        };
        let (mapping, resolved) = if cached.is_none() {
            let read = self.b(&self.g.read_field_types).call1((kind,))?;
            (read.get_item(0)?, read.get_item(1)?.is_truthy()?)
        } else {
            (cached, true)
        };
        let guard = if resolved {
            Vec::new()
        } else {
            self.namespaces(kind)?
        };
        let by_name = match mapping.cast::<PyDict>() {
            Ok(dict) if tp(&mapping) == t_dict() && str_keys(dict) => dict.clone(),
            _ => {
                let by_name = PyDict::new(py);
                for item in mapping.call_method0(intern!(py, "items"))?.try_iter()? {
                    let item = item?;
                    let name = item.get_item(0)?;
                    if tp(&name) == t_str() {
                        by_name.set_item(name, item.get_item(1)?)?;
                    }
                }
                by_name
            }
        };
        let mut types = Types {
            mapping: mapping.unbind(),
            by_name: by_name.unbind(),
            slots: Vec::new(),
        };
        let slots: Option<Vec<_>> = layout
            .slots
            .iter()
            .map(|slot| {
                types
                    .get(py, slot.name.bind(py).as_any())
                    .map(|found| found.map(Bound::unbind))
            })
            .collect();
        types.slots = slots.unwrap_or_default();
        let types = Rc::new(types);
        self.facts(kind).types = Some((types.clone(), guard));
        Ok(types)
    }

    /// The dicts a class's hints read names from: builtins, and the module
    /// of each class in its MRO, each with its size now.
    fn namespaces(&self, kind: &B<'py>) -> PyResult<Vec<(Py<PyDict>, isize)>> {
        let py = self.py;
        let builtins = unsafe { Bound::from_borrowed_ptr(py, ffi::PyEval_GetBuiltins()) }
            .cast_into::<PyDict>()?;
        let mut found = vec![(builtins.clone().unbind(), dict_size(&builtins))];
        let modules = self.g.sys_dict.bind(py).get_item(intern!(py, "modules"))?;
        let class = kind.as_ptr().cast::<ffi::PyTypeObject>();
        let mro = unsafe { Bound::from_borrowed_ptr(py, (*class).tp_mro) };
        for base in mro.try_iter()? {
            let name = base?.getattr(intern!(py, "__module__"))?;
            let module = match &modules {
                Some(modules) => modules.get_item(&name).ok(),
                None => None,
            };
            if let Some(namespace) = module
                .and_then(|module| module.getattr(intern!(py, "__dict__")).ok())
                .and_then(|namespace| namespace.cast_into::<PyDict>().ok())
                && !found
                    .iter()
                    .any(|(each, _)| each.as_ptr() == namespace.as_ptr())
            {
                let size = dict_size(&namespace);
                found.push((namespace.unbind(), size));
            }
        }
        Ok(found)
    }

    fn record_implied(&mut self, kind: &B<'py>, declared: Option<&B<'py>>) -> PyResult<bool> {
        let Some(declared) = declared else {
            return Ok(false);
        };
        if let Some(overrides) = &self.overrides {
            return self
                .b(&self.g.record_implied)
                .call1((kind, declared, overrides))?
                .is_truthy();
        }
        let address = declared.as_ptr() as usize;
        if let Some((_, built)) = self.forms.records.get(&address) {
            return Ok(built
                .as_ref()
                .is_some_and(|built| built.as_ptr() == kind.as_ptr()));
        }
        let built =
            match facts::record_built(self.py, &self.g.records, &self.g.hints.alias, declared)? {
                Some(built) => built.map(Bound::unbind),
                None => {
                    let built = self
                        .b(&self.g.record_built)
                        .call1((declared, PyDict::new(self.py)))?;
                    (!built.is_none()).then(|| built.unbind())
                }
            };
        let implied = built
            .as_ref()
            .is_some_and(|built| built.as_ptr() == kind.as_ptr());
        let entry = (declared.clone().unbind(), built);
        drop(self.forms.records.insert(address, entry));
        Ok(implied)
    }

    /// ``_form(value, kind, declared, ...)``, ``kind`` by its index in
    /// ``containers``; kept per class where it holds for every value.
    fn form(
        &mut self,
        value: &B<'py>,
        kind: usize,
        declared: Option<&B<'py>>,
        size: Option<usize>,
    ) -> PyResult<Rc<Form>> {
        let py = self.py;
        let Some(declared) = declared else {
            return Ok(self.forms.undeclared[usize::from(kind == 0 || kind == 4)].clone());
        };
        let class = self.b(&self.g.containers[kind]);
        if let Some(overrides) = &self.overrides {
            let answer = self
                .b(&self.g.form)
                .call1((value, class, declared, overrides))?;
            return Ok(Rc::new(Form::read(py, &answer)?));
        }
        let key = (
            kind,
            declared.as_ptr() as usize,
            size.map_or(-1, |size| size as isize),
        );
        let known = self
            .forms
            .settled
            .get(&key)
            .map(|(_, settled)| settled.clone());
        let settled = match known {
            Some(settled) => settled,
            None => match self.native_form(kind, declared, size)? {
                Some(form) => {
                    let form = Rc::new(form);
                    let entry = (declared.clone().unbind(), Some(form.clone()));
                    drop(self.forms.settled.insert(key, entry));
                    Some(form)
                }
                None => {
                    let size = size.map(|size| size.into_pyobject(py)).transpose()?;
                    let answer = self.b(&self.g.settled_form).call1((
                        class,
                        declared,
                        PyDict::new(py),
                        size,
                    ))?;
                    let settled = if answer.is_none() {
                        None
                    } else {
                        Some(Rc::new(Form::read(py, &answer)?))
                    };
                    let entry = (declared.clone().unbind(), settled.clone());
                    drop(self.forms.settled.insert(key, entry));
                    settled
                }
            },
        };
        match settled {
            Some(form) => Ok(form),
            None => {
                let answer =
                    self.b(&self.g.form)
                        .call1((value, class, declared, PyDict::new(py)))?;
                Ok(Rc::new(Form::read(py, &answer)?))
            }
        }
    }

    /// ``_settled_form`` where ``declared`` is a builtin generic of the
    /// container's own class, with no ``default_concrete_types``: that class
    /// reads untagged data as itself, so the container goes untagged, its
    /// parts declared by the generic's arguments as ``_part_types`` reads
    /// them. ``None`` for anything else, which Python decides.
    fn native_form(
        &self,
        kind: usize,
        declared: &B<'py>,
        size: Option<usize>,
    ) -> PyResult<Option<Form>> {
        let py = self.py;
        if !declared.get_type().is(self.b(&self.g.hints.generic_alias)) {
            return Ok(None);
        }
        let origin = self.g.hints.alias.origin(declared)?;
        if !origin.is(self.b(&self.g.containers[kind]))
            || !self.b(&self.g.records.dispatch).contains(&origin)?
        {
            return Ok(None);
        }
        let args = self.g.hints.alias.args(declared)?;
        if tp(&args) != t_tuple() {
            return Ok(None);
        }
        let args: Vec<B<'py>> = unsafe { borrowed_items(&args) }
            .iter()
            .map(|&arg| unsafe { Bound::from_borrowed_ptr(py, arg) })
            .collect();
        let object = py.get_type::<PyAny>().into_any();
        let each = |item: Option<&B<'py>>| {
            item.filter(|item| !item.is(&object))
                .map(|item| item.clone().unbind())
        };
        let form = match kind {
            // dict, MappingProxyType: ``_entry_types``.
            4 | 5 => Form {
                untagged: true,
                keys: (args.len() == 2).then(|| each(args.first())).flatten(),
                parts: Parts {
                    each: (args.len() == 2).then(|| each(args.get(1))).flatten(),
                    ..Parts::default()
                },
            },
            // tuple: ``_tuple_types``, fixed only at its own arity.
            1 if args.len() == 2 && args[1].is(py.Ellipsis()) => Form {
                untagged: true,
                keys: None,
                parts: Parts {
                    each: each(args.first()),
                    ..Parts::default()
                },
            },
            1 => {
                if size != Some(args.len()) {
                    return Ok(None);
                }
                Form {
                    untagged: true,
                    keys: None,
                    parts: Parts {
                        indexed: Some(PyTuple::new(py, &args)?.into_any().unbind()),
                        ..Parts::default()
                    },
                }
            }
            // list, set, frozenset: ``_element_type``.
            _ => Form {
                untagged: true,
                keys: None,
                parts: Parts {
                    each: (args.len() == 1).then(|| each(args.first())).flatten(),
                    ..Parts::default()
                },
            },
        };
        Ok(Some(form))
    }

    // ----------------------------------------------------------- paths --

    /// ``_import_path(value)``: kept per class or object, and proved to
    /// name it still at every use, as Python proves it.
    fn path(&mut self, value: &B<'py>) -> PyResult<B<'py>> {
        let py = self.py;
        let class = is_class(value);
        let kept = if class {
            self.facts(value).path.clone()
        } else {
            self.table
                .paths
                .get(&(value.as_ptr() as usize))
                .map(|(_, fact)| fact.clone())
        };
        if let Some(fact) = kept
            && self.verified(&fact, value)?
        {
            return Ok(fact.path.bind(py).clone().into_any());
        }
        if class && let Some(fact) = self.native_path(value)? {
            let path = fact.path.bind(py).clone().into_any();
            self.facts(value).path = Some(fact);
            return Ok(path);
        }
        let path = self.b(&self.g.import_path).call1((value,))?;
        if let Some(fact) = self.path_fact(&path)? {
            if class {
                self.facts(value).path = Some(fact);
            } else {
                self.keep_path(value, fact);
            }
        }
        Ok(path)
    }
}

/// ``type(value) in hooks``, at once where ``hooks`` is an empty dict.
#[inline]
fn hooked(hooks: &B<'_>, value: &B<'_>) -> PyResult<bool> {
    if tp(hooks) == t_dict() && unsafe { ffi::PyDict_Size(hooks.as_ptr()) } == 0 {
        return Ok(false);
    }
    hooks.contains(value.get_type())
}

/// ``f"{module}.{qualname}"`` from two strs' UTF-8: written straight into
/// a new compact str where both are ASCII.
fn dotted<'py>(py: Python<'py>, module: &[u8], qualname: &[u8]) -> PyResult<B<'py>> {
    let size = module.len() + 1 + qualname.len();
    if !(module.is_ascii() && qualname.is_ascii()) {
        let mut text = Vec::with_capacity(size);
        text.extend_from_slice(module);
        text.push(b'.');
        text.extend_from_slice(qualname);
        // Both are a str's own UTF-8, so their join is too.
        return Ok(PyString::new(py, unsafe { std::str::from_utf8_unchecked(&text) }).into_any());
    }
    unsafe {
        let made =
            Bound::from_owned_ptr_or_err(py, ffi::PyUnicode_New(size as ffi::Py_ssize_t, 127))?;
        let data = ffi::PyUnicode_DATA(made.as_ptr()).cast::<u8>();
        std::ptr::copy_nonoverlapping(module.as_ptr(), data, module.len());
        *data.add(module.len()) = b'.';
        std::ptr::copy_nonoverlapping(
            qualname.as_ptr(),
            data.add(module.len() + 1),
            qualname.len(),
        );
        Ok(made)
    }
}

/// Whether every key of ``dict`` is an exact str.
fn str_keys(dict: &Bound<'_, PyDict>) -> bool {
    let mut position = 0;
    let mut key = std::ptr::null_mut();
    let mut value = std::ptr::null_mut();
    while unsafe { ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) } != 0 {
        if ob_type(key) != t_str() {
            return false;
        }
    }
    true
}

/// ``node[key] = value`` for a dict ``node``.
#[inline]
fn set_item(node: &B<'_>, key: &B<'_>, value: *mut ffi::PyObject) -> PyResult<()> {
    if unsafe { ffi::PyDict_SetItem(node.as_ptr(), key.as_ptr(), value) } < 0 {
        return Err(PyErr::fetch(node.py()));
    }
    Ok(())
}

/// ``_is_plain_key``: an exact str that does not read as a tag or an escape.
#[inline]
fn is_plain_key(key: &B<'_>) -> PyResult<bool> {
    if tp(key) != t_str() {
        return Ok(false);
    }
    Ok(!str_starts_any(key, &["py/", "json://"])?)
}

/// A key whose escape is its own JSON text, distinct from every other
/// key's: an exact int, bool, ``None``, or finite float.
#[inline]
fn is_leaf_key(key: &B<'_>) -> bool {
    let kind = tp(key);
    kind == t_int()
        || kind == t_bool()
        || kind == t_none()
        || kind == t_float() && unsafe { ffi::PyFloat_AS_DOUBLE(key.as_ptr()) }.is_finite()
}

fn non_finite(number: f64) -> &'static str {
    if number.is_nan() {
        "nan"
    } else if number > 0.0 {
        "inf"
    } else {
        "-inf"
    }
}

/// ``json.dumps(tree)`` in ``style``, by ``json`` itself where the writer
/// does not write a tree exactly.
fn dumps<'py>(py: Python<'py>, tree: &B<'py>, style: Style, sort_keys: bool) -> PyResult<B<'py>> {
    let (dumps, own) = json::current_dumps(py)?;
    if own {
        let mut writer = Writer::new(style);
        if writer.plain(tree, 0).is_ok() {
            return writer.finish(py);
        }
    }
    let options = PyDict::new(py);
    if sort_keys {
        options.set_item("sort_keys", true)?;
    }
    dumps.call((tree,), Some(&options))
}

/// ``base64.b64encode(data).decode()``.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let bytes = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let triple = u32::from(bytes[0]) << 16 | u32::from(bytes[1]) << 8 | u32::from(bytes[2]);
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[(triple >> (18 - 6 * index) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

// ---------------------------------------------------------- entry points --

struct Call<'py> {
    python: bool,
    names: Option<ClassNames>,
    overrides: Option<B<'py>>,
    hooks: Option<B<'py>>,
}

/// ``to_plain``'s checks of its options, in its order.
fn options<'py>(
    py: Python<'py>,
    g: &Globals,
    dialect: Option<B<'py>>,
    hooks: Option<B<'py>>,
    classes: Option<B<'py>>,
    concrete: Option<B<'py>>,
) -> PyResult<Call<'py>> {
    let python = match dialect {
        None => false,
        Some(dialect) => {
            if !g.dialects.bind(py).contains(&dialect)? {
                return Err(PyValueError::new_err(format!(
                    "unknown dialect {}; expected 'json' or 'python'",
                    repr(&dialect)?
                )));
            }
            py_eq(&dialect, intern!(py, "python").as_any())?
        }
    };
    let names = ClassNames::read(py, classes.as_ref())?;
    let overrides = match concrete {
        Some(concrete) => {
            let overrides = g.concrete_overrides.bind(py).call1((concrete,))?;
            (overrides.len()? > 0).then_some(overrides)
        }
        None => None,
    };
    Ok(Call {
        python,
        names,
        overrides,
        hooks,
    })
}

/// Run the walk over ``obj`` into ``sink``.
fn walk<'py>(
    py: Python<'py>,
    g: &Globals,
    call: Call<'py>,
    sort: Option<B<'py>>,
    obj: &B<'py>,
    declared: Option<B<'py>>,
    sink: Sink<'py>,
) -> PyResult<Sink<'py>> {
    let mut encoder = Encoder {
        py,
        g,
        python: call.python,
        hooks: call.hooks,
        sort: Flag::new(sort, true),
        names: call.names,
        overrides: call.overrides,
        table: TABLE.take(py).unwrap_or_default(),
        forms: FORMS.take(py).unwrap_or_default(),
        places: Vec::new(),
        seen: HashMap::new(),
        alive: Vec::new(),
        frames: Vec::new(),
        spare: Vec::new(),
        sink,
    };
    Buffers::take(py).bind(&mut encoder);
    encoder
        .places
        .extend([(ROOT, Step::Root), (ROOT, Step::Root)]);
    encoder.sink.item();
    let done = encoder
        .node(obj, ROOT, declared.as_ref())
        .and_then(|()| encoder.drain(0));
    let buffers = Buffers::keep(&mut encoder);
    let Encoder {
        mut table,
        forms,
        sink,
        ..
    } = encoder;
    table.settle();
    let stale = (
        TABLE.replace(py, Some(table)),
        FORMS.replace(py, Some(forms)),
        BUFFERS.replace(py, Some(buffers)),
    );
    drop(stale);
    done.map(|()| sink)
}

fn globals(py: Python<'_>) -> PyResult<Rc<Globals>> {
    GLOBALS
        .cloned(py)
        .ok_or_else(|| PyTypeError::new_err("treekle._rust.setup_encode was not called"))
}

#[allow(clippy::too_many_arguments)]
pub fn to_plain<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Option<B<'py>>,
    mutable: Option<B<'py>>,
    hooks: Option<B<'py>>,
    classes: Option<B<'py>>,
    sort: Option<B<'py>>,
    target: Option<B<'py>>,
    concrete: Option<B<'py>>,
) -> PyResult<B<'py>> {
    let g = globals(py)?;
    let call = options(py, &g, dialect, hooks, classes, concrete)?;
    let declared = target.and_then(|target| declared(py, target));
    let tree = walk(py, &g, call, sort, obj, declared, Sink::new(py, None))?.take_tree();
    let mutable = match mutable {
        Some(mutable) => mutable.is_truthy()?,
        None => true,
    };
    if mutable {
        return Ok(tree);
    }
    if let Some(frozen) = freeze(py, &tree)? {
        return Ok(frozen);
    }
    g.immutable.bind(py).call1((tree,))
}

/// ``json.dumps(to_plain(obj, ...), ...)`` written in one walk. ``style`` is
/// what ``options``, ``json.dumps``' own keyword arguments, ask for: one line,
/// keys in tree order. The caller has checked that ``json.dumps`` is
/// ``json``'s own.
#[allow(clippy::too_many_arguments)]
pub fn to_json<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Option<B<'py>>,
    hooks: Option<B<'py>>,
    classes: Option<B<'py>>,
    sort: Option<B<'py>>,
    concrete: Option<B<'py>>,
    style: Style,
    options_given: [Option<&B<'py>>; 4],
) -> PyResult<B<'py>> {
    let g = globals(py)?;
    let call = options(py, &g, dialect, hooks, classes, concrete)?;
    walk(py, &g, call, sort, obj, None, Sink::new(py, Some(style)))?.take_text(options_given)
}

/// ``len(dict)``.
#[inline]
fn dict_size(dict: &Bound<'_, PyDict>) -> isize {
    unsafe { (*dict.as_ptr().cast::<ffi::PyDictObject>()).ma_used }
}

/// A leaf every declared type writes as it is: an exact str, int, bool,
/// float or ``None``.
#[inline]
fn is_plain_leaf(value: &B<'_>) -> bool {
    is_leaf_kind(tp(value))
}

/// Whether ``dict``'s keys are ``keys``, the same objects in order; its
/// values, borrowed, are pushed to ``values`` as they are read.
fn dict_matches(
    dict: &Bound<'_, PyDict>,
    keys: &[Slot],
    values: &mut Vec<*mut ffi::PyObject>,
) -> bool {
    if dict_size(dict) as usize != keys.len() {
        return false;
    }
    let mut position = 0;
    let mut key = std::ptr::null_mut();
    let mut value = std::ptr::null_mut();
    for expected in keys {
        if unsafe { ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) } == 0
            || key != expected.name.as_ptr()
        {
            return false;
        }
        values.push(value);
    }
    true
}
