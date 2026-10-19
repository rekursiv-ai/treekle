//! How a read fails: the error type the decoder threads through every step.

use pyo3::ffi;
use pyo3::prelude::*;

/// A decode failure: a read error (its message, and what caused it), or
/// any other Python error, raised as is.
pub enum DErr {
    Read(String, Option<PyErr>),
    Py(PyErr),
}

impl From<PyErr> for DErr {
    fn from(error: PyErr) -> Self {
        Self::Py(error)
    }
}

pub type DR<T> = Result<T, DErr>;

pub fn fail<T>(message: String) -> DR<T> {
    Err(DErr::Read(message, None))
}

pub fn fail_from<T>(message: String, cause: PyErr) -> DR<T> {
    Err(DErr::Read(message, Some(cause)))
}

/// Link ``cause`` to ``error`` as Python's ``raise error from cause`` does
/// inside ``except``: as both its ``__cause__`` and its ``__context__``.
pub fn chain(py: Python<'_>, error: &PyErr, cause: PyErr) {
    let cause = cause.into_value(py);
    unsafe {
        ffi::PyException_SetContext(error.value(py).as_ptr(), cause.clone_ref(py).into_ptr());
        ffi::PyException_SetCause(error.value(py).as_ptr(), cause.into_ptr());
    }
}
