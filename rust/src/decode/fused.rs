//! ``json_loads(text, target)`` read straight from the text by the target's
//! compiled program: each array, object and record the program reads is read
//! from its bytes, without first making the list or dict ``json.loads`` makes
//! for it. Any other node (a leaf, a union, a tuple, an ``object`` part) is
//! parsed and read as the tree program reads it.
//!
//! A read either finishes, giving what the program gives on the parsed
//! tree, or gives up, and the text is then parsed and read by ``from_plain``;
//! so a document reads the same either way, and every error comes from the
//! tree path.

use super::*;
use crate::json::{self, Reader};

// Perf: rows json_loads, lockstep: with json.rs exact_float, 1.63-1.77x; exp000 1.00x (its records
// have their own __new__, so this bails).
/// ``json_loads(text, target)``: ``from_plain(json.loads(text), target)``.
pub fn from_json<'py>(
    py: Python<'py>,
    text: &B<'py>,
    target: &B<'py>,
    hooks: Option<B<'py>>,
    classes: Option<B<'py>>,
    allow_imports: Option<B<'py>>,
    strict: Option<B<'py>>,
) -> PyResult<B<'py>> {
    // A ``classes`` that does not read is left to ``from_plain``, which raises
    // only once ``json.loads`` has read the text, as Python does.
    if let Some(globals) = DEC.cloned(py)
        && let Ok(class_names) = ClassNames::read(py, classes.as_ref())
        && json::current_loads(py)?.1
        && let Ok(bytes) = json::document(text)
        && let Some(&(b'{' | b'[')) = bytes.iter().find(|byte| !byte.is_ascii_whitespace())
    {
        let mut decoder = Decoder::new(
            py,
            &globals,
            hooks.clone(),
            Flag::new(allow_imports.clone(), false),
            Flag::new(strict.clone(), true),
            class_names,
        );
        let value = decoder.json_root(bytes, target);
        decoder.keep_caches(&globals);
        drop(decoder);
        if let Some(value) = value? {
            return Ok(value);
        }
    }
    let tree = json::loads(py, text)?;
    from_plain(
        py,
        &tree,
        target,
        None,
        hooks,
        classes,
        Flag::new(allow_imports, false),
        Flag::new(strict, true),
    )
}

impl<'py> Decoder<'py, '_> {
    /// ``quick_root`` on ``json.loads(text)``, read from the text: ``None``
    /// where the program would not read the whole tree, or the text is not
    /// one JSON document.
    fn json_root(&mut self, bytes: &[u8], target: &B<'py>) -> PyResult<Option<B<'py>>> {
        let persistent = self.persistent_root(target);
        let Some(plan) = self.plan_cached(target, persistent) else {
            return Ok(None);
        };
        if !matches!(&*plan.program.borrow(), Compiled::Yes(_)) {
            return Ok(None);
        }
        let Some(program) = self.program(target, &plan) else {
            return Ok(None);
        };
        let mut reader = Reader::open(self.py, bytes)?;
        let mut exec = Exec::new(self.py, &program, self.class_names.as_ref(), None);
        reader.space();
        let read = exec.json(&mut reader, &program.root, 0);
        reader.space();
        let whole = reader.at == bytes.len();
        reader.close();
        Ok(Self::ended(&plan, read).filter(|_| whole))
    }
}

/// Release the items a list read left on the reader's stack past ``base``.
fn release(reader: &mut Reader<'_, '_>, base: usize) {
    for left in reader.stack.drain(base..) {
        unsafe { ffi::Py_DECREF(left) };
    }
}

/// Step past ``byte`` and the space after it, or stop.
#[inline(always)]
fn expect(reader: &mut Reader<'_, '_>, byte: u8) -> Result<(), Stop> {
    if reader.peek() != byte {
        return Err(Stop::Abort);
    }
    reader.at += 1;
    reader.space();
    Ok(())
}

/// Step past a string's opening quote, or stop.
#[inline(always)]
fn open_string(reader: &mut Reader<'_, '_>) -> Result<(), Stop> {
    if reader.peek() != b'"' {
        return Err(Stop::Abort);
    }
    reader.at += 1;
    Ok(())
}

/// After a member: past its ``,``, ``false``, or past the closing ``end``,
/// ``true``.
#[inline(always)]
fn next_member(reader: &mut Reader<'_, '_>, end: u8) -> Result<bool, Stop> {
    reader.space();
    let byte = reader.peek();
    reader.at += 1;
    if byte == end {
        return Ok(true);
    }
    if byte != b',' {
        return Err(Stop::Abort);
    }
    reader.space();
    Ok(false)
}

impl<'py> Exec<'_, 'py, false> {
    /// ``read`` on the next JSON value, at a value's first byte.
    fn json(&mut self, reader: &mut Reader<'_, 'py>, node: &Node, depth: usize) -> FR<'py> {
        if depth >= FAST_DEPTH {
            return Err(Stop::Abort);
        }
        match (reader.peek(), node) {
            (b'[', Node::List(item)) => self.json_list(reader, item, depth + 1),
            (b'{', Node::Dict(keys, value)) => self.json_dict(reader, *keys, value, depth + 1),
            (b'{', Node::Record(index)) => self.json_record(reader, *index, depth + 1),
            _ => {
                let raw = reader.value(depth as u32).map_err(|_| Stop::Abort)?;
                self.read(&raw, node, Loc::At(ROOT))
            }
        }
    }

    fn json_list(&mut self, reader: &mut Reader<'_, 'py>, item: &Node, depth: usize) -> FR<'py> {
        expect(reader, b'[')?;
        let base = reader.stack.len();
        if reader.peek() == b']' {
            reader.at += 1;
        } else {
            loop {
                let value = self.json(reader, item, depth).and_then(|value| {
                    reader.stack.push(value.into_ptr());
                    next_member(reader, b']')
                });
                match value {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(stop) => {
                        release(reader, base);
                        return Err(stop);
                    }
                }
            }
        }
        unsafe {
            let list = ffi::PyList_New((reader.stack.len() - base) as ffi::Py_ssize_t);
            if list.is_null() {
                ffi::PyErr_Clear();
                release(reader, base);
                return Err(Stop::Abort);
            }
            for (index, item) in reader.stack.drain(base..).enumerate() {
                ffi::PyList_SET_ITEM(list, index as ffi::Py_ssize_t, item);
            }
            Ok(Bound::from_owned_ptr(self.py, list))
        }
    }

    /// A JSON object as ``dict``. A repeated key keeps its first place and
    /// takes its last value, as in ``json.loads``'s dict.
    fn json_dict(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        keys: Keys,
        value: &Node,
        depth: usize,
    ) -> FR<'py> {
        let py = self.py;
        let out = PyDict::new(py);
        expect(reader, b'{')?;
        if reader.peek() == b'}' {
            reader.at += 1;
            return Ok(out.into_any());
        }
        loop {
            open_string(reader)?;
            let name = reader.string(true).map_err(|_| Stop::Abort)?;
            let Some(text) = utf8(&name) else {
                return Err(Stop::Abort);
            };
            let key = match keys {
                Keys::Str if text.starts_with(b"py/") || text.starts_with(b"json://") => {
                    return Err(Stop::Abort);
                }
                Keys::Str => name.clone(),
                Keys::Int => int_key(py, text).ok_or(Stop::Abort)?,
            };
            reader.space();
            expect(reader, b':')?;
            let member = self.json(reader, value, depth)?;
            out.set_item(key, member).map_err(|_| Stop::Abort)?;
            if next_member(reader, b'}')? {
                return Ok(out.into_any());
            }
        }
    }

    /// A JSON object as a record of ``index``, its keys matched to fields by
    /// their bytes, each field stored in its slot as ``fill`` stores it.
    fn json_record(&mut self, reader: &mut Reader<'_, 'py>, index: usize, depth: usize) -> FR<'py> {
        let py = self.py;
        let prog = self.prog;
        let rec = &prog.recs[index];
        let class = unsafe { Borrowed::from_ptr(py, rec.class) };
        if !rec.stamp.holds(&class) {
            return Err(Stop::Stale);
        }
        let made = unsafe {
            (rec.new)(
                class.as_ptr().cast(),
                PyTuple::empty(py).as_ptr(),
                std::ptr::null_mut(),
            )
        };
        let Some(value) = (unsafe { Bound::from_owned_ptr_or_opt(py, made) }) else {
            drop(PyErr::fetch(py));
            return Err(Stop::Abort);
        };
        let mut tagged = false;
        let mut count = 0;
        let mut cursor = 0;
        let mut seen: u64 = 0;
        expect(reader, b'{')?;
        if reader.peek() == b'}' {
            reader.at += 1;
        } else {
            loop {
                open_string(reader)?;
                let spelled;
                let key = match reader.plain_ascii() {
                    Some(key) => key,
                    None => {
                        spelled = reader.string(true).map_err(|_| Stop::Abort)?;
                        utf8(&spelled).ok_or(Stop::Abort)?
                    }
                };
                reader.space();
                expect(reader, b':')?;
                if key == b"py/object" {
                    if tagged {
                        return Err(Stop::Abort);
                    }
                    open_string(reader)?;
                    let path = reader.string(false).map_err(|_| Stop::Abort)?;
                    if !self.names_rec(&path, rec) {
                        return Err(Stop::Abort);
                    }
                    tagged = true;
                } else {
                    let slot = rec.info.slot_of(key, &mut cursor).ok_or(Stop::Abort)?;
                    // A field named twice: the record is not one the program
                    // reads, as ``json.loads`` would keep the last value.
                    let bit = 1u64.checked_shl(slot as u32).ok_or(Stop::Abort)?;
                    if seen & bit != 0 {
                        return Err(Stop::Abort);
                    }
                    seen |= bit;
                    let field = &rec.fields[slot];
                    let member = self.json(reader, &field.node, depth)?;
                    if !field.set(&value, member) {
                        return Err(Stop::Abort);
                    }
                    count += 1;
                }
                if next_member(reader, b'}')? {
                    break;
                }
            }
        }
        if count != rec.fields.len() {
            return Err(Stop::Abort);
        }
        Ok(value)
    }
}
