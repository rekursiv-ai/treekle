//! ``classes``: short names ``py/object`` carries in place of import paths,
//! and, when given, the only classes a call writes or reads by that tag.

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::py::{B, MapItems, is_class, repr, t_str, tp};

pub struct ClassNames {
    /// Each name, an exact str, to its class; it also keeps every class alive.
    pub names: Py<PyDict>,
    /// Each class, by address, to its name.
    pub tags: HashMap<usize, Py<PyString>>,
}

impl ClassNames {
    /// ``reflect.class_tags(classes)``'s checks, then the mapping both ways;
    /// ``None`` when ``classes`` is absent or empty.
    pub fn read<'py>(py: Python<'py>, classes: Option<&B<'py>>) -> PyResult<Option<Self>> {
        let Some(classes) = classes else {
            return Ok(None);
        };
        let names = PyDict::new(py);
        let mut tags: HashMap<usize, Py<PyString>> = HashMap::new();
        let mut items = MapItems::new(classes)?;
        while let Some((name, kind)) = items.next_item()? {
            if tp(&name) != t_str() {
                let kind_name = name.get_type().name()?;
                return Err(PyTypeError::new_err(format!(
                    "classes keys must be str, not {kind_name}"
                )));
            }
            if !is_class(&kind) {
                return Err(PyTypeError::new_err(format!(
                    "classes[{}] is not a class",
                    repr(&name)?
                )));
            }
            let address = kind.as_ptr() as usize;
            if let Some(first) = tags.get(&address) {
                return Err(PyValueError::new_err(format!(
                    "classes names one class twice: {} and {}",
                    repr(first.bind(py).as_any())?,
                    repr(&name)?
                )));
            }
            tags.insert(
                address,
                unsafe { name.cast_unchecked::<PyString>() }
                    .clone()
                    .unbind(),
            );
            names.set_item(&name, &kind)?;
        }
        if tags.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            names: names.unbind(),
            tags,
        }))
    }

    /// The name ``class`` is written under, if it has one.
    pub fn tag<'py>(&self, py: Python<'py>, class: &B<'py>) -> Option<Bound<'py, PyString>> {
        self.tags
            .get(&(class.as_ptr() as usize))
            .map(|name| name.bind(py).clone())
    }

    /// Each class to its name, as Python's ``_Encoding.tags`` holds them.
    pub fn tags_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let tags = PyDict::new(py);
        for (name, kind) in self.names.bind(py).iter() {
            tags.set_item(kind, name)?;
        }
        Ok(tags)
    }

    /// The class ``name`` names, if any.
    pub fn class<'py>(&self, py: Python<'py>, name: &B<'py>) -> PyResult<Option<B<'py>>> {
        self.names.bind(py).get_item(name)
    }
}
