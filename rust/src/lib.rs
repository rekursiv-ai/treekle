//! Treekle's Rust core: ``to_plain`` and ``from_plain`` with the semantics of
//! ``treekle.serialize`` and ``treekle.deserialize``, which install them.

// GilCell and the per-interpreter Rc caches require a real GIL. Refuse
// free-threaded builds, including configurations that keep the GIL enabled
// today but could later disable it with PYTHON_GIL=0 or -Xgil=0. Those
// interpreters can install the universal Python wheel instead.
#[cfg(Py_GIL_DISABLED)]
compile_error!(
    "treekle's Rust core requires a GIL; use the pure-Python wheel on free-threaded Python"
);

mod class;
mod decode;
mod encode;
mod errors;
mod freeze;
mod places;
mod py;
mod sync;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py::{B, Flag, Maybe};

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
///   allow_imports: Whether tags may import modules and call code they name.
///     A ``py/object`` tag naming the target dataclass itself needs none.
///   strict: When false, also read numeric and boolean text (``"3"`` as
///     ``3``) and integral numbers across ``int``, ``float``, and ``bool``.
///
/// Returns:
///   value: ``data`` as a ``target``, or ``default``.
///
/// Raises:
///   ReadError: ``data`` does not fit ``target``, or a tag needs imports while
///     ``allow_imports`` is false.
#[pyfunction]
#[pyo3(
    signature = (data, target, *, default = Maybe(None), hooks = Maybe(None), allow_imports = Maybe(None), strict = Maybe(None)),
    text_signature = "(data, target, *, default=..., hooks=..., allow_imports=False, strict=True)"
)]
fn from_plain<'py>(
    py: Python<'py>,
    data: &B<'py>,
    target: &B<'py>,
    default: Maybe<'py>,
    hooks: Maybe<'py>,
    allow_imports: Maybe<'py>,
    strict: Maybe<'py>,
) -> PyResult<B<'py>> {
    let allow_imports = Flag::new(allow_imports.0, false);
    let strict = Flag::new(strict.0, true);
    decode::from_plain(py, data, target, default.0, hooks.0, allow_imports, strict)
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
///   sort: Order each object's attributes by name; when false, keep slots in
///     MRO order then ``__dict__`` keys in insertion order. A stored form that
///     must stay byte-stable sets this false.
///
/// Returns:
///   tree: Fresh plain data that ``from_plain`` turns back into ``obj``.
///
/// Raises:
///   ValueError: ``dialect`` names no known format.
///   TypeError: A leaf has no conversion and no hook, or a dict key holds a
///     cycle.
#[pyfunction]
#[pyo3(
    signature = (obj, *, dialect = Maybe(None), mutable = Maybe(None), hooks = Maybe(None), sort = Maybe(None)),
    text_signature = "(obj, *, dialect='json', mutable=True, hooks=..., sort=True)"
)]
fn to_plain<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Maybe<'py>,
    mutable: Maybe<'py>,
    hooks: Maybe<'py>,
    sort: Maybe<'py>,
) -> PyResult<B<'py>> {
    encode::to_plain(py, obj, dialect.0, mutable.0, hooks.0, sort.0)
}

#[pymodule(gil_used = true)]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(setup_decode, module)?)?;
    module.add_function(wrap_pyfunction!(from_plain, module)?)?;
    module.add_function(wrap_pyfunction!(setup_encode, module)?)?;
    module.add_function(wrap_pyfunction!(to_plain, module)?)?;
    Ok(())
}
