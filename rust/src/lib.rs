//! Treekle's Rust core: ``to_plain`` and ``from_plain`` with the semantics of
//! ``treekle.serialize`` and ``treekle.deserialize``, which install them.

// Perf notes. Code written for speed carries a comment "Perf: <workload>, <method>: <gain>". The
// gain is old time / new time against the build just before that change; gains overlap, so they do
// not multiply.
// Workloads:
//   exp000: the 10 local priml exp000 configs.
//   bench_configs: scripts/bench_configs.py's 28 priml configs; decode reads their generated
//     slots-dataclass twins.
//   rows: 1k-10k dataclass rows (flat, slots, frozen, nested), msgspec's benchmark shape.
//   tiny: one small call, ns per call.
// Methods:
//   lockstep: two builds in separate processes taking turns; 50 passes, median, best of 3, one
//     pinned core.
//   paired: two builds in one process taking turns; min or median of rounds.
// Timed on CPython 3.14, Linux (WSL), a shared machine: under ~3% is noise.

// GilCell and the per-interpreter Rc caches require a real GIL. Refuse
// free-threaded builds, including configurations that keep the GIL enabled
// today but could later disable it with PYTHON_GIL=0 or -Xgil=0. Those
// interpreters can install the universal Python wheel instead.
#[cfg(Py_GIL_DISABLED)]
compile_error!(
    "treekle's Rust core requires a GIL; use the pure-Python wheel on free-threaded Python"
);

mod attrs;
mod class;
mod classes;
mod decode;
mod encode;
mod errors;
mod freeze;
mod json;
mod places;
mod py;
mod sync;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py::{B, Flag, Maybe, t_tuple, tp};

/// Read the decoder's helpers and sentinels from ``treekle.deserialize``'s globals.
#[pyfunction]
fn setup_decode(module: &Bound<'_, PyDict>) -> PyResult<()> {
    decode::setup(module)
}

// The public functions carry ``treekle.deserialize`` and ``treekle.serialize``'s
// docstrings verbatim, since they replace those functions; ``rust_parity_test``
// checks that the two stay alike. A text signature can only show literal
// defaults, so the sentinel ones read ``...``.

/// Convert plain data to ``target``; a ``py/*`` tag decides where present.
///
/// Read one field as ``from_plain(row.get("name"), str, default="")``.
///
/// Args:
///   data: A plain tree, mutable or not, as any reader or ``to_plain`` made it.
///     A non-plain leaf already of its target type, such as a ``datetime``
///     from ``tomllib``, passes through.
///   target: The type to produce; ``object`` reads by tags alone.
///   default: Returned when ``data`` is ``None`` (a missing or null field);
///     omit to read ``None`` like any other value.
///   hooks: The codecs ``to_plain`` used.
///   classes: The short names ``to_plain`` wrote in ``py/object``; when given,
///     a ``py/object`` tag builds only these classes, and imports nothing.
///   allow_imports: Whether tags may import modules and call code they name.
///     A ``py/object`` tag naming the target dataclass itself needs none.
///   strict: When false, also read numeric and boolean text (``"3"`` as
///     ``3``) and integral numbers across ``int``, ``float``, and ``bool``.
///
/// Returns:
///   value: ``data`` as a ``target``, or ``default``.
///
/// Raises:
///   ReadError: ``data`` does not fit ``target``, a tag needs imports while
///     ``allow_imports`` is false, or a ``py/object`` tag is not in a given
///     ``classes``.
///   TypeError: ``classes`` has a key that is not a ``str``, or a value that
///     is not a class.
///   ValueError: ``classes`` names one class twice.
#[pyfunction]
#[pyo3(
    signature = (data, target, *, default = Maybe(None), hooks = Maybe(None), classes = Maybe(None), allow_imports = Maybe(None), strict = Maybe(None)),
    text_signature = "(data, target, *, default=..., hooks=..., classes=..., allow_imports=False, strict=True)"
)]
#[allow(clippy::too_many_arguments)]
fn from_plain<'py>(
    py: Python<'py>,
    data: &B<'py>,
    target: &B<'py>,
    default: Maybe<'py>,
    hooks: Maybe<'py>,
    classes: Maybe<'py>,
    allow_imports: Maybe<'py>,
    strict: Maybe<'py>,
) -> PyResult<B<'py>> {
    let allow_imports = Flag::new(allow_imports.0, false);
    let strict = Flag::new(strict.0, true);
    decode::from_plain(
        py,
        data,
        target,
        default.0,
        hooks.0,
        classes.0,
        allow_imports,
        strict,
    )
}

/// Read the encoder's helpers and sentinels from ``treekle.serialize``'s globals.
#[pyfunction]
fn setup_encode(module: &Bound<'_, PyDict>) -> PyResult<()> {
    encode::setup(module)
}

/// Convert a Python value to plain data, tagging what ``dialect`` lacks.
///
/// Args:
///   obj: The value to convert; any object graph, shared references and cycles
///     included, except a cycle inside a dict key. An object repeated inside a
///     dict key reads back as an equal copy, not the same object.
///   dialect: The target format, which decides which values need a tag.
///   mutable: Build dicts and lists when true; ``MappingProxyType`` and tuples
///     when false.
///   hooks: Codecs for leaf types with no built-in conversion, such as tensors.
///   classes: Short names to write in ``py/object`` in place of import paths;
///     when given, no other class is written as ``py/object``.
///   sort: Order each object's attributes by name; when false, keep slots in
///     MRO order then ``__dict__`` keys in insertion order. A stored form that
///     must stay byte-stable sets this false.
///
/// Returns:
///   tree: Fresh plain data that ``from_plain`` turns back into ``obj``.
///
/// Raises:
///   ValueError: ``dialect`` names no known format, or ``classes`` names one
///     class twice.
///   TypeError: A leaf has no conversion and no hook, a hook returns the type
///     it encodes, a dict key holds a cycle, or ``classes`` is given and lacks
///     a ``py/object`` class.
#[pyfunction]
#[pyo3(
    signature = (obj, *, dialect = Maybe(None), mutable = Maybe(None), hooks = Maybe(None), classes = Maybe(None), sort = Maybe(None)),
    text_signature = "(obj, *, dialect='json', mutable=True, hooks=..., classes=..., sort=True)"
)]
fn to_plain<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Maybe<'py>,
    mutable: Maybe<'py>,
    hooks: Maybe<'py>,
    classes: Maybe<'py>,
    sort: Maybe<'py>,
) -> PyResult<B<'py>> {
    encode::to_plain(py, obj, dialect.0, mutable.0, hooks.0, classes.0, sort.0)
}

/// ``json.dumps(tree, ...)`` for a plain tree, with the same text.
#[pyfunction]
#[pyo3(
    signature = (tree, *, ensure_ascii = Maybe(None), separators = Maybe(None), indent = Maybe(None), sort_keys = Maybe(None)),
    text_signature = "(tree, *, ensure_ascii=True, separators=None, indent=None, sort_keys=False)"
)]
fn json_dumps_plain<'py>(
    py: Python<'py>,
    tree: &B<'py>,
    ensure_ascii: Maybe<'py>,
    separators: Maybe<'py>,
    indent: Maybe<'py>,
    sort_keys: Maybe<'py>,
) -> PyResult<B<'py>> {
    json::dumps(
        py,
        tree,
        ensure_ascii.0.as_ref(),
        separators.0.as_ref(),
        indent.0.as_ref(),
        sort_keys.0.as_ref(),
    )
}

/// ``json.loads(text)``, with the same value or error.
#[pyfunction]
fn json_loads_plain<'py>(py: Python<'py>, text: &B<'py>) -> PyResult<B<'py>> {
    json::loads(py, text)
}

/// ``from_plain(json.loads(text), target, ...)``, with the same value or error.
#[pyfunction]
#[pyo3(
    signature = (text, target, *, hooks = Maybe(None), classes = Maybe(None), allow_imports = Maybe(None), strict = Maybe(None)),
    text_signature = "(text, target, *, hooks=..., classes=..., allow_imports=False, strict=True)"
)]
fn json_loads_typed<'py>(
    py: Python<'py>,
    text: &B<'py>,
    target: &B<'py>,
    hooks: Maybe<'py>,
    classes: Maybe<'py>,
    allow_imports: Maybe<'py>,
    strict: Maybe<'py>,
) -> PyResult<B<'py>> {
    decode::from_json(
        py,
        text,
        target,
        hooks.0,
        classes.0,
        allow_imports.0,
        strict.0,
    )
}

/// ``json.dumps(to_plain(obj, ...), ...)`` in one walk, and ``json_loads``'s
/// other half: one call from object to text or back.
#[pyfunction]
#[pyo3(
    signature = (obj, *, dialect = Maybe(None), hooks = Maybe(None), classes = Maybe(None), sort = Maybe(None), ensure_ascii = Maybe(None), separators = Maybe(None), indent = Maybe(None), sort_keys = Maybe(None)),
    text_signature = "(obj, *, dialect='json', hooks=..., classes=..., sort=True, ensure_ascii=True, separators=None, indent=None, sort_keys=False)"
)]
#[allow(clippy::too_many_arguments)]
fn json_dumps<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Maybe<'py>,
    hooks: Maybe<'py>,
    classes: Maybe<'py>,
    sort: Maybe<'py>,
    ensure_ascii: Maybe<'py>,
    separators: Maybe<'py>,
    indent: Maybe<'py>,
    sort_keys: Maybe<'py>,
) -> PyResult<B<'py>> {
    let options = [
        ensure_ascii.0.as_ref(),
        separators.0.as_ref(),
        indent.0.as_ref(),
        sort_keys.0.as_ref(),
    ];
    // Read only where reading runs no Python code, so no option is read
    // before ``to_plain``'s user code runs, as ``json.dumps`` reads them.
    let style = json::Style::from_options(options[0], options[1], options[2], options[3])?.ok();
    // ``json.dumps`` reads ``separators`` after that code ran, which may
    // change a list; a tuple stays as the walk read it.
    let fixed = options[1].is_none_or(|seps| seps.is_none() || tp(seps) == t_tuple());
    // The text walk writes one line in tree order only; layout options go
    // through the tree and the plain writer. A ``json.dumps`` replaced since
    // import gets the whole tree, once, as in Python.
    if let Some(style) = style.filter(|style| style.indent.is_none() && !style.sort_keys)
        && fixed
        && json::current_dumps(py)?.1
        && let Some(text) = encode::to_json(
            py,
            obj,
            dialect.0.clone(),
            hooks.0.clone(),
            classes.0.clone(),
            sort.0.clone(),
            style,
            options,
        )?
    {
        return Ok(text);
    }
    // The text walk declined before running any user code; the tree walk's
    // answer goes through ``json.dumps`` as it is once that code has run.
    let tree = encode::to_plain(py, obj, dialect.0, None, hooks.0, classes.0, sort.0)?;
    json::dumps(
        py,
        &tree,
        ensure_ascii.0.as_ref(),
        separators.0.as_ref(),
        indent.0.as_ref(),
        sort_keys.0.as_ref(),
    )
}

#[pymodule(gil_used = true)]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    json::setup(module.py())?;
    module.add_function(wrap_pyfunction!(setup_decode, module)?)?;
    module.add_function(wrap_pyfunction!(from_plain, module)?)?;
    module.add_function(wrap_pyfunction!(setup_encode, module)?)?;
    module.add_function(wrap_pyfunction!(to_plain, module)?)?;
    module.add_function(wrap_pyfunction!(json_dumps_plain, module)?)?;
    module.add_function(wrap_pyfunction!(json_loads_plain, module)?)?;
    module.add_function(wrap_pyfunction!(json_loads_typed, module)?)?;
    module.add_function(wrap_pyfunction!(json_dumps, module)?)?;
    Ok(())
}
