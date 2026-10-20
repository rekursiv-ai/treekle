//! ``from_plain(json.loads(text), target)`` read straight from the text, for
//! documents the quick loops read whole: each container is read as the quick
//! loops would read the dict or list ``json.loads`` makes for it, without
//! making that dict or list, and record keys match field names by their bytes.
//!
//! A read either finishes, giving what the quick loops give on the tree, or
//! gives up, and the tree is then built and read as before; so a document
//! reads the same either way, and every error comes from the tree path.

use super::*;
use crate::json::{self, Reader};

/// Why a fused read stopped. ``Retry``: the quick loops might still read
/// this value from its tree, which is built and read in its place. ``Stop``:
/// they cannot, or the text is not JSON; the whole document goes back to
/// the tree path.
enum Miss {
    Retry,
    Stop,
}

type FR<T> = Result<T, Miss>;

const TAG: &[u8] = b"py/object";

// Perf: rows json_loads, lockstep: with json.rs exact_float, 1.63-1.77x; exp000 1.00x (its records
// have their own __new__, so this bails).
/// ``json_loads(text, target)``: ``from_plain(json.loads(text), target)``.
pub fn from_json<'py>(
    py: Python<'py>,
    text: &B<'py>,
    target: &B<'py>,
    hooks: Option<B<'py>>,
    allow_imports: Option<B<'py>>,
    strict: Option<B<'py>>,
) -> PyResult<B<'py>> {
    if let Some(globals) = DEC.cloned(py)
        && json::current_loads(py)?.1
        && let Ok(bytes) = json::document(text)
        && let Some(&(b'{' | b'[')) = bytes.iter().find(|byte| !byte.is_ascii_whitespace())
    {
        let g = &*globals;
        let caches = CACHES.take(py).unwrap_or_else(|| new_caches(py));
        let mut decoder = Decoder::new(
            py,
            g,
            hooks.clone(),
            Flag::new(allow_imports.clone(), false),
            Flag::new(strict.clone(), true),
            false,
            caches,
        );
        let mut reader = Reader::open(py, bytes)?;
        let value = decoder.fused_root(&mut reader, target);
        reader.close();
        let kept = std::mem::replace(
            &mut decoder.caches,
            Caches::around(decoder.object_plan.clone()),
        );
        if DEC
            .cloned(py)
            .is_some_and(|current| Rc::ptr_eq(&current, &globals))
        {
            let stale = CACHES.replace(py, Some(kept));
            drop(stale);
        }
        drop(decoder);
        if let Some(value) = value {
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
        Flag::new(allow_imports, false),
        Flag::new(strict, true),
    )
}

/// Release the items a list read left on the reader's stack past ``base``.
fn release(reader: &mut Reader<'_, '_>, base: usize) {
    for left in reader.stack.drain(base..) {
        unsafe { ffi::Py_DECREF(left) };
    }
}

impl<'py> Decoder<'py, '_> {
    /// ``quick_root`` on ``json.loads(text)``, read from the text: ``None``
    /// where the quick loops would not read the whole tree, or the text is
    /// not one JSON document.
    fn fused_root(&mut self, reader: &mut Reader<'_, 'py>, target: &B<'py>) -> Option<B<'py>> {
        let persistent = self.persistent_root(target);
        let plan = self.plan_cached(target, persistent)?;
        if !matches!(plan.kind, Kind::List | Kind::Object) {
            return None;
        }
        self.epoch += 1;
        self.untagged = false;
        self.quick_place = false;
        reader.space();
        let value = self.fused_value(reader, target, &plan, 0).ok()?;
        reader.space();
        (reader.at == reader.text.len()).then_some(value)
    }

    /// ``quick_node`` on the next JSON value, as ``target``.
    fn fused_value(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        let start = reader.at;
        if !matches!(reader.peek(), b'{' | b'[') {
            let leaf = reader.value(depth).map_err(|_| Miss::Stop)?;
            // ``quick_part``'s leaf rules, ahead of the call: every JSON leaf
            // is a plain leaf.
            let wanted = target.as_ptr();
            if wanted == tp(&leaf).cast() || wanted == self.g.object.as_ptr() {
                return Ok(leaf);
            }
            return self.quick_tree(&leaf, target, plan, depth);
        }
        let mark = self.mark();
        match self.fused_inner(reader, target, plan, depth) {
            Err(Miss::Retry) => {
                self.rollback(mark).map_err(|_| Miss::Stop)?;
                reader.at = start;
                let tree = reader.value(depth).map_err(|_| Miss::Stop)?;
                self.quick_tree(&tree, target, plan, depth)
            }
            done => done,
        }
    }

    #[inline]
    fn quick_tree(
        &mut self,
        data: &B<'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        match self.quick_node(data, target, plan, Loc::At(ROOT), depth) {
            Ok(Some(value)) => Ok(value),
            _ => Err(Miss::Stop),
        }
    }

    /// ``quick_node``'s container cases, at a ``{`` or ``[``.
    fn fused_inner(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        if depth >= 100 {
            return Err(Miss::Stop);
        }
        match (reader.peek(), plan.kind) {
            (b'[', Kind::List) => {
                let args = plan.args.bind(self.py);
                let (item, item_plan) = match args.len() {
                    1 => {
                        let item = args.get_item(0).map_err(|_| Miss::Stop)?;
                        let item_plan =
                            self.plan_cached(&item, plan.persistent).ok_or(Miss::Stop)?;
                        (item, item_plan)
                    }
                    _ => (self.b(&self.g.object).clone(), self.object_plan.clone()),
                };
                self.fused_items(reader, &item, &item_plan, depth + 1)
            }
            (b'{', Kind::Dict) => {
                let args = plan.args.bind(self.py);
                let object = self.b(&self.g.object).clone();
                let (key, value) = match args.len() {
                    2 => (
                        args.get_item(0).map_err(|_| Miss::Stop)?,
                        args.get_item(1).map_err(|_| Miss::Stop)?,
                    ),
                    _ => (object.clone(), object),
                };
                if !(key.is(self.b(&self.g.object)) || key.is(self.b(&self.g.str_type))) {
                    return Err(Miss::Stop);
                }
                let value_plan = self
                    .plan_cached(&value, plan.persistent)
                    .ok_or(Miss::Stop)?;
                self.fused_entries(reader, &value, &value_plan, depth + 1)
            }
            (b'{', Kind::Object) => self.fused_object(reader, target, plan, depth + 1),
            (b'{', Kind::Union) => self.fused_union(reader, plan, depth),
            _ => Err(Miss::Stop),
        }
    }

    /// ``quick_items`` on a JSON array.
    fn fused_items(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        item: &B<'py>,
        item_plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        reader.at += 1;
        reader.space();
        let base = reader.stack.len();
        if reader.peek() == b']' {
            reader.at += 1;
        } else {
            loop {
                match self.fused_value(reader, item, item_plan, depth) {
                    Ok(value) => reader.stack.push(value.into_ptr()),
                    Err(miss) => {
                        release(reader, base);
                        return Err(miss);
                    }
                }
                reader.space();
                match reader.peek() {
                    b',' => {
                        reader.at += 1;
                        reader.space();
                    }
                    b']' => {
                        reader.at += 1;
                        break;
                    }
                    _ => {
                        release(reader, base);
                        return Err(Miss::Stop);
                    }
                }
            }
        }
        unsafe {
            let list = ffi::PyList_New((reader.stack.len() - base) as ffi::Py_ssize_t);
            if list.is_null() {
                ffi::PyErr_Clear();
                release(reader, base);
                return Err(Miss::Stop);
            }
            for (index, item) in reader.stack.drain(base..).enumerate() {
                ffi::PyList_SET_ITEM(list, index as ffi::Py_ssize_t, item);
            }
            Ok(Bound::from_owned_ptr(self.py, list))
        }
    }

    /// ``quick_entries`` on a JSON object. A repeated key keeps its first
    /// place and takes its last value, as in ``json.loads``'s dict.
    fn fused_entries(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        value: &B<'py>,
        value_plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        let result = PyDict::new(self.py);
        reader.at += 1;
        reader.space();
        if reader.peek() == b'}' {
            reader.at += 1;
            return Ok(result.into_any());
        }
        loop {
            if reader.peek() != b'"' {
                return Err(Miss::Stop);
            }
            reader.at += 1;
            let name = reader.string(true).map_err(|_| Miss::Stop)?;
            // A ``py/`` key makes the object a tag (``py/id``, or an unknown
            // tag's error), which only the walk reads.
            match utf8(&name) {
                Some(text) if !text.starts_with(b"json://") && !text.starts_with(b"py/") => {}
                _ => return Err(Miss::Stop),
            }
            reader.space();
            if reader.peek() != b':' {
                return Err(Miss::Stop);
            }
            reader.at += 1;
            reader.space();
            let member = self.fused_value(reader, value, value_plan, depth)?;
            result.set_item(name, member).map_err(|_| Miss::Stop)?;
            reader.space();
            match reader.peek() {
                b',' => {
                    reader.at += 1;
                    reader.space();
                }
                b'}' => {
                    reader.at += 1;
                    return Ok(result.into_any());
                }
                _ => return Err(Miss::Stop),
            }
        }
    }

    /// ``quick_union_record`` on a JSON object whose first key is its
    /// ``py/object`` tag; a tag further in is left to the tree.
    fn fused_union(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        let py = self.py;
        if !self.unread.is_empty() {
            return Err(Miss::Retry);
        }
        let Some(info) = plan.union.as_ref() else {
            return Err(Miss::Stop);
        };
        let start = reader.at;
        reader.at += 1;
        reader.space();
        if reader.peek() != b'"' {
            return Err(Miss::Retry);
        }
        reader.at += 1;
        if reader.plain_ascii() != Some(TAG) {
            return Err(Miss::Retry);
        }
        reader.space();
        if reader.peek() != b':' {
            return Err(Miss::Stop);
        }
        reader.at += 1;
        reader.space();
        // A later repeat of the tag would decide the tree's reading; the
        // record read below stops at one.
        if reader.peek() != b'"' {
            return Err(Miss::Stop);
        }
        reader.at += 1;
        let tag = reader.string(false).map_err(|_| Miss::Stop)?;
        let first = info
            .import_names
            .iter()
            .position(|name| {
                name.as_ref().is_some_and(|name| {
                    ob_type(name.as_ptr()) == t_str()
                        && unsafe { same_text(name.as_ptr(), tag.as_ptr()) }
                })
            })
            .ok_or(Miss::Stop)?;
        let member = plan.args.bind(py).get_item(first).map_err(|_| Miss::Stop)?;
        let member_plan = self
            .plan_cached(&member, plan.persistent)
            .ok_or(Miss::Stop)?;
        reader.at = start;
        self.fused_inner(reader, &member, &member_plan, depth)
    }

    /// The slot a plain ASCII record key names, as ``quick_slot`` finds it.
    #[inline]
    fn fused_slot(&self, facts: &RowFacts, position: usize, key: &[u8]) -> Option<u32> {
        let text = |slot: usize| {
            let (data, len) = facts.texts[slot];
            if len == usize::MAX {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(data, len) })
            }
        };
        let guess = facts.order[position].get();
        if guess == TAG_SLOT {
            if key == TAG {
                return Some(TAG_SLOT);
            }
        } else if guess != NO_GUESS && text(guess as usize) == Some(key) {
            return Some(guess);
        }
        let found = (0..facts.texts.len())
            .position(|slot| text(slot) == Some(key))
            .map(|slot| slot as u32)
            .or_else(|| (key == TAG).then_some(TAG_SLOT))?;
        facts.order[position].set(found);
        Some(found)
    }

    /// ``quick_object`` on a JSON object.
    fn fused_object(
        &mut self,
        reader: &mut Reader<'_, 'py>,
        target: &B<'py>,
        plan: &Rc<Plan>,
        depth: u32,
    ) -> FR<B<'py>> {
        let py = self.py;
        let Flag::Known(imports) = self.allow_imports else {
            return Err(Miss::Stop);
        };
        if !plan.is_dataclass
            || !is_class(target)
            || !matches!(plan.fits, Fits::IsInstance(None))
            || !plan.args.bind(py).is_empty()
        {
            return Err(Miss::Stop);
        }
        let facts = self.row_facts(target, plan).ok_or(Miss::Stop)?;
        let width = facts.info.targets.len();
        let kind = target.as_ptr().cast::<ffi::PyTypeObject>();
        let value = unsafe {
            let alloc = (*kind).tp_alloc.expect("checked by row_facts");
            Bound::from_owned_ptr_or_err(py, alloc(kind, 0)).map_err(|_| Miss::Stop)?
        };
        let object = self.b(&self.g.object).as_ptr();
        let mut seen: u64 = 0;
        let mut tagged = false;
        let mut count = 0;
        reader.at += 1;
        reader.space();
        if reader.peek() == b'}' {
            reader.at += 1;
        } else {
            loop {
                if reader.peek() != b'"' || count == QUICK_WIDTH {
                    return Err(Miss::Stop);
                }
                reader.at += 1;
                let slot = match reader.plain_ascii() {
                    Some(key) => self.fused_slot(&facts, count, key),
                    None => {
                        let key = reader.string(true).map_err(|_| Miss::Stop)?;
                        self.quick_slot(&facts, count, key.as_ptr())
                    }
                }
                .ok_or(Miss::Stop)?;
                count += 1;
                reader.space();
                if reader.peek() != b':' {
                    return Err(Miss::Stop);
                }
                reader.at += 1;
                reader.space();
                if slot == TAG_SLOT {
                    if tagged {
                        return Err(Miss::Stop);
                    }
                    let raw = reader.value(depth).map_err(|_| Miss::Stop)?;
                    if imports {
                        let own = plan
                            .import_name
                            .as_ref()
                            .expect("classes have import names");
                        if tp(&raw) != t_str() || !unsafe { same_text(raw.as_ptr(), own.as_ptr()) }
                        {
                            return Err(Miss::Stop);
                        }
                    }
                    tagged = true;
                } else {
                    if seen & (1 << slot) != 0 {
                        return Err(Miss::Stop);
                    }
                    seen |= 1 << slot;
                    let slot = slot as usize;
                    let field = facts.info.targets[slot].bind(py);
                    let member =
                        if field.as_ptr() == object && !matches!(reader.peek(), b'{' | b'[') {
                            reader.value(depth).map_err(|_| Miss::Stop)?
                        } else {
                            let field_plan = facts.plans[slot].as_ref().ok_or(Miss::Stop)?;
                            self.fused_value(reader, field, field_plan, depth)?
                        };
                    let field_name = facts.info.names[slot].as_ptr();
                    if !self.quick_set(&value, field_name, &member, facts.setters[slot]) {
                        return Err(Miss::Stop);
                    }
                }
                reader.space();
                match reader.peek() {
                    b',' => {
                        reader.at += 1;
                        reader.space();
                    }
                    b'}' => {
                        reader.at += 1;
                        break;
                    }
                    _ => return Err(Miss::Stop),
                }
            }
        }
        if !tagged {
            return Err(Miss::Stop);
        }
        if seen.count_ones() as usize != width
            && !self
                .quick_defaults(&value, target, &facts, seen)
                .map_err(|_| Miss::Stop)?
        {
            return Err(Miss::Stop);
        }
        Ok(value)
    }
}
