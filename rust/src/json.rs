//! JSON text with ``json.dumps``' and ``json.loads``' exact output.
//!
//! Each fast path gives up with ``Bail`` on anything it would not write or
//! read exactly as ``json`` does: a type it does not know, a cycle, deep
//! nesting, a malformed document. The caller then reruns the whole call
//! through ``json``, which gives its own answer or raises its own error.
//! Where ``json.dumps`` or ``json.loads`` has been replaced since import, no
//! fast path runs: the replacement answers, as it would in Python.

use std::rc::Rc;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py::*;

/// A fast path gave up; ``json`` itself must answer.
pub struct Bail;

pub type Done<T> = Result<T, Bail>;

/// Past this depth the fast paths leave nesting, cycles and recursion errors
/// to ``json``.
const MAX_DEPTH: u32 = 200;

/// For each Latin-1 character, the letter after the backslash that escapes
/// it, or zero when it is written as itself; ``u`` means ``\u00XX``, and
/// ``x`` a character ``ensure_ascii`` escapes: DEL or any non-ASCII one.
const ESCAPES: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut byte = 0;
    while byte < 0x20 {
        table[byte] = b'u';
        byte += 1;
    }
    let mut byte = 0x7f;
    while byte < 0x100 {
        table[byte] = b'x';
        byte += 1;
    }
    table[0x08] = b'b';
    table[0x09] = b't';
    table[0x0a] = b'n';
    table[0x0c] = b'f';
    table[0x0d] = b'r';
    table[b'"' as usize] = b'"';
    table[b'\\' as usize] = b'\\';
    table
};

const HEX: &[u8; 16] = b"0123456789abcdef";

// ---------------------------------------------------------------- writer --

/// How ``json.dumps`` was asked to lay text out.
#[derive(Clone)]
pub struct Style {
    pub ensure_ascii: bool,
    pub item_sep: Vec<u8>,
    pub key_sep: Vec<u8>,
    pub indent: Option<Vec<u8>>,
    pub sort_keys: bool,
}

impl Style {
    /// The style ``json.dumps`` reads from its keyword arguments, or ``Bail``
    /// for an argument only ``json`` reads exactly. Reads only arguments
    /// whose reading runs no Python code: flags that are exact bools, ints
    /// or ``None``, separators an exact tuple or list of exact strs, indent
    /// an exact int or str; ``json`` reads any other, when and as often as
    /// it does.
    pub fn from_options(
        ensure_ascii: Option<&B<'_>>,
        separators: Option<&B<'_>>,
        indent: Option<&B<'_>>,
        sort_keys: Option<&B<'_>>,
    ) -> PyResult<Done<Self>> {
        let truth = |flag: Option<&B<'_>>, default: bool| match flag {
            None => Some(default),
            Some(flag) if tp(flag) == t_bool() || tp(flag) == t_int() || flag.is_none() => {
                flag.is_truthy().ok()
            }
            Some(_) => None,
        };
        let (Some(ensure_ascii), Some(sort_keys)) =
            (truth(ensure_ascii, true), truth(sort_keys, false))
        else {
            return Ok(Err(Bail));
        };
        let indent = match indent.filter(|indent| !indent.is_none()) {
            None => None,
            Some(indent) if tp(indent) == t_int() => {
                let Ok(width) = indent.extract::<i64>() else {
                    return Ok(Err(Bail));
                };
                Some(vec![b' '; width.max(0) as usize])
            }
            Some(indent) if tp(indent) == t_str() => match utf8(indent) {
                Some(text) => Some(text.to_vec()),
                None => return Ok(Err(Bail)),
            },
            Some(_) => return Ok(Err(Bail)),
        };
        let (item_sep, key_sep) = match separators.filter(|seps| !seps.is_none()) {
            None if indent.is_some() => (b",".to_vec(), b": ".to_vec()),
            None => (b", ".to_vec(), b": ".to_vec()),
            Some(seps) => {
                if !(tp(seps) == t_tuple() || tp(seps) == t_list()) || seps.len()? != 2 {
                    return Ok(Err(Bail));
                }
                let (item, key) = (seps.get_item(0)?, seps.get_item(1)?);
                if tp(&item) != t_str() || tp(&key) != t_str() {
                    return Ok(Err(Bail));
                }
                match (utf8(&item), utf8(&key)) {
                    (Some(item), Some(key)) => (item.to_vec(), key.to_vec()),
                    _ => return Ok(Err(Bail)),
                }
            }
        };
        Ok(Ok(Self {
            ensure_ascii,
            item_sep,
            key_sep,
            indent,
            sort_keys,
        }))
    }
}

// Perf: rows json_dumps, lockstep: ryu over core::fmt's {:e}, 1.10-1.14x on float-heavy rows.
/// The shortest digits of a finite positive float that round-trip, closest
/// to it, as David Gay's ``repr`` mode gives them: the digit count written to
/// ``digits``, and where the decimal point falls counted from the first.
///
/// Ryu finds the same digits as ``repr``; only its layout differs, so the
/// digits and point are read back out of its text.
fn shortest_digits(number: f64, digits: &mut [u8; 24]) -> (usize, i32) {
    let mut buffer = ryu::Buffer::new();
    let shown = buffer.format_finite(number).as_bytes();
    let (mantissa, mut point) = match shown.iter().position(|&b| b == b'e') {
        Some(at) => {
            let (negative, exponent) = match &shown[at + 1..] {
                [b'-', rest @ ..] => (true, rest),
                rest => (false, rest),
            };
            let mut exp: i32 = 0;
            for &digit in exponent {
                exp = exp * 10 + i32::from(digit - b'0');
            }
            (&shown[..at], if negative { -exp } else { exp })
        }
        None => (shown, 0),
    };
    point += mantissa
        .iter()
        .position(|&b| b == b'.')
        .unwrap_or(mantissa.len()) as i32;
    let mut count = 0;
    for &digit in mantissa {
        if digit == b'.' {
            continue;
        }
        if count == 0 && digit == b'0' {
            point -= 1;
            continue;
        }
        digits[count] = digit;
        count += 1;
    }
    while count > 1 && digits[count - 1] == b'0' {
        count -= 1;
    }
    (count, point)
}

/// ``float.__repr__`` of a finite float.
pub fn float_repr(out: &mut Vec<u8>, number: f64) -> Done<()> {
    if number == 0.0 {
        out.extend_from_slice(if number.is_sign_negative() {
            b"-0.0"
        } else {
            b"0.0"
        });
        return Ok(());
    }
    let mut digits = [0u8; 24];
    // Where the decimal point falls, counted from the first digit.
    let (count, point) = shortest_digits(number.abs(), &mut digits);
    let digits = &digits[..count];
    // A tie between two shortest candidates is ``repr``'s to break, to the
    // even one; leave any such case to it.
    if (digits[count - 1] - b'0') % 2 == 1 && tied(number, digits, point) {
        return python_float_repr(out, number);
    }
    if number < 0.0 {
        out.push(b'-');
    }
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.extend_from_slice(b"0.");
            out.resize(out.len() + (-point) as usize, b'0');
            out.extend_from_slice(digits);
        } else if point as usize >= count {
            out.extend_from_slice(digits);
            out.resize(out.len() + point as usize - count, b'0');
            out.extend_from_slice(b".0");
        } else {
            out.extend_from_slice(&digits[..point as usize]);
            out.push(b'.');
            out.extend_from_slice(&digits[point as usize..]);
        }
        return Ok(());
    }
    out.push(digits[0]);
    if count > 1 {
        out.push(b'.');
        out.extend_from_slice(&digits[1..]);
    }
    out.push(b'e');
    let exp = point - 1;
    out.push(if exp < 0 { b'-' } else { b'+' });
    let exp = exp.unsigned_abs();
    if exp < 10 {
        out.push(b'0');
    }
    push_u64(out, u64::from(exp));
    Ok(())
}

/// Whether ``number`` lies exactly halfway between ``digits`` and one of its
/// neighbours in the last place, ``point`` digits before the decimal point.
fn tied(number: f64, digits: &[u8], point: i32) -> bool {
    let mut shown: u128 = 0;
    for &digit in digits {
        shown = shown * 10 + u128::from(digit - b'0');
    }
    let k = point - digits.len() as i32;
    let bits = number.abs().to_bits();
    let biased = (bits >> 52) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let (mut mantissa, mut exp) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | 1 << 52, biased - 1075)
    };
    let zeros = mantissa.trailing_zeros();
    mantissa >>= zeros;
    exp += zeros as i32;
    // A midpoint is t * 10^k / 2 for odd t: t * 5^k * 2^(k-1), whose odd
    // part is the mantissa's.
    if exp != k - 1 || k.abs() > 27 {
        return false;
    }
    let five = 5u128.pow(k.unsigned_abs());
    let mantissa = u128::from(mantissa);
    [2 * shown - 1, 2 * shown + 1].into_iter().any(|t| {
        if k >= 0 {
            t * five == mantissa
        } else {
            t == mantissa * five
        }
    })
}

/// ``float.__repr__``'s own digits, from CPython's ``dtoa``.
pub fn python_float_repr(out: &mut Vec<u8>, number: f64) -> Done<()> {
    unsafe {
        let text = ffi::PyOS_double_to_string(
            number,
            b'r' as std::ffi::c_char,
            0,
            ffi::Py_DTSF_ADD_DOT_0,
            std::ptr::null_mut(),
        );
        if text.is_null() {
            ffi::PyErr_Clear();
            return Err(Bail);
        }
        out.extend_from_slice(std::ffi::CStr::from_ptr(text).to_bytes());
        ffi::PyMem_Free(text.cast());
    }
    Ok(())
}

// Perf: exp000 json_dumps, lockstep: with inline str/int reads and small copies, 1.17x.
/// Whether ASCII ``text`` holds a byte ``json`` escapes: a control
/// character, ``"``, ``\`` or DEL (which ``ensure_ascii`` escapes), read
/// eight bytes at a time.
#[inline(always)]
fn ascii_needs_escape(text: &[u8]) -> bool {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    #[inline(always)]
    fn zero_byte(x: u64) -> u64 {
        x.wrapping_sub(ONES) & !x & HIGH
    }
    let (chunks, rest) = text.as_chunks::<8>();
    for chunk in chunks {
        let x = u64::from_le_bytes(*chunk);
        // Every byte is below 0x80, so these exact tests cannot carry over.
        let hit = x.wrapping_sub(ONES * 0x20) & !x & HIGH
            | zero_byte(x ^ (ONES * u64::from(b'"')))
            | zero_byte(x ^ (ONES * u64::from(b'\\')))
            | x.wrapping_add(ONES) & HIGH;
        if hit != 0 {
            return true;
        }
    }
    rest.iter().any(|&byte| ESCAPES[byte as usize] != 0)
}

/// An exact str's JSON text, quoted, when it is compact ASCII needing no
/// escape, so the same under every ``Style``.
pub fn plain_key_text(key: &B<'_>) -> Option<Box<[u8]>> {
    let (_, data, len, ascii) = unsafe { str_view(key.as_ptr()) }?;
    let text = unsafe { std::slice::from_raw_parts(data, len) };
    if !ascii || ascii_needs_escape(text) {
        return None;
    }
    let mut out = Vec::with_capacity(len + 2);
    out.push(b'"');
    out.extend_from_slice(text);
    out.push(b'"');
    Some(out.into_boxed_slice())
}

/// ``copy_nonoverlapping`` without a ``memcpy`` call for the short texts
/// that are most of a document.
#[inline(always)]
unsafe fn copy_small(from: *const u8, to: *mut u8, len: usize) {
    unsafe {
        if len <= 16 {
            for at in 0..len {
                *to.add(at) = *from.add(at);
            }
        } else {
            std::ptr::copy_nonoverlapping(from, to, len);
        }
    }
}

/// A separator, one or two bytes written without a ``memcpy`` call.
#[inline(always)]
fn put_sep(buf: &mut Vec<u8>, sep: &[u8]) {
    match *sep {
        [one] => buf.push(one),
        [one, two] => {
            buf.reserve(2);
            unsafe {
                let at = buf.as_mut_ptr().add(buf.len());
                *at = one;
                *at.add(1) = two;
                buf.set_len(buf.len() + 2);
            }
        }
        _ => buf.extend_from_slice(sep),
    }
}

#[inline]
fn push_u64(out: &mut Vec<u8>, mut value: u64) {
    let mut tmp = [0u8; 20];
    let mut at = tmp.len();
    loop {
        at -= 1;
        tmp[at] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    let len = tmp.len() - at;
    out.reserve(len);
    unsafe {
        copy_small(tmp.as_ptr().add(at), out.as_mut_ptr().add(out.len()), len);
        out.set_len(out.len() + len);
    }
}

pub struct Writer {
    pub buf: Vec<u8>,
    /// Whether ``buf`` holds only ASCII so far.
    ascii: bool,
    pub style: Style,
}

impl Writer {
    pub fn new(style: Style) -> Self {
        let ascii = style.ensure_ascii
            && style.item_sep.is_ascii()
            && style.key_sep.is_ascii()
            && style.indent.as_ref().is_none_or(|indent| indent.is_ascii());
        Self {
            buf: Vec::with_capacity(4096),
            ascii,
            style,
        }
    }

    /// A writer that is never written to, which allocates nothing.
    pub fn idle() -> Self {
        Self {
            buf: Vec::new(),
            ascii: true,
            style: Style {
                ensure_ascii: true,
                item_sep: Vec::new(),
                key_sep: Vec::new(),
                indent: None,
                sort_keys: false,
            },
        }
    }

    /// An unsigned number.
    #[inline]
    pub fn uint(&mut self, value: u64) {
        push_u64(&mut self.buf, value);
    }

    /// ``{"tag": ``, the start of a tagged node.
    #[inline]
    pub fn open_tag(&mut self, tag: &[u8]) {
        self.buf.extend_from_slice(b"{\"");
        self.buf.extend_from_slice(tag);
        self.buf.push(b'"');
        self.key_sep();
    }

    #[inline(always)]
    pub fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    #[inline(always)]
    pub fn item_sep(&mut self) {
        let Self { buf, style, .. } = self;
        put_sep(buf, &style.item_sep);
    }

    #[inline(always)]
    pub fn key_sep(&mut self) {
        let Self { buf, style, .. } = self;
        put_sep(buf, &style.key_sep);
    }

    /// The text as a str, built as ``json`` builds its own.
    pub fn finish<'py>(&self, py: Python<'py>) -> PyResult<B<'py>> {
        let len = self.buf.len();
        unsafe {
            if self.ascii || self.buf.is_ascii() {
                let out = ffi::PyUnicode_New(len as ffi::Py_ssize_t, 127);
                if out.is_null() {
                    return Err(PyErr::fetch(py));
                }
                std::ptr::copy_nonoverlapping(
                    self.buf.as_ptr(),
                    ffi::PyUnicode_DATA(out).cast::<u8>(),
                    len,
                );
                return Ok(Bound::from_owned_ptr(py, out));
            }
            // Lone surrogates were written as their three UTF-8-like bytes.
            Bound::from_owned_ptr_or_err(
                py,
                ffi::PyUnicode_DecodeUTF8(
                    self.buf.as_ptr().cast(),
                    len as ffi::Py_ssize_t,
                    c"surrogatepass".as_ptr(),
                ),
            )
        }
    }

    /// Insert each text at its offset into what was written, the offsets
    /// rising and read against the text before any insertion.
    pub fn splice(&mut self, texts: Vec<(usize, Vec<u8>)>) {
        let extra: usize = texts.iter().map(|(_, text)| text.len()).sum();
        let mut out = Vec::with_capacity(self.buf.len() + extra);
        let mut from = 0;
        for (at, text) in texts {
            out.extend_from_slice(&self.buf[from..at]);
            out.extend_from_slice(&text);
            self.ascii = self.ascii && text.is_ascii();
            from = at;
        }
        out.extend_from_slice(&self.buf[from..]);
        self.buf = out;
    }

    #[inline]
    fn unicode_escape(&mut self, unit: u32) {
        self.buf.extend_from_slice(&[
            b'\\',
            b'u',
            HEX[(unit >> 12 & 15) as usize],
            HEX[(unit >> 8 & 15) as usize],
            HEX[(unit >> 4 & 15) as usize],
            HEX[(unit & 15) as usize],
        ]);
    }

    #[inline]
    fn escape(&mut self, byte: u8, escape: u8) {
        match escape {
            b'u' => self.unicode_escape(u32::from(byte)),
            b'x' => self.non_ascii(u32::from(byte)),
            _ => self.buf.extend_from_slice(&[b'\\', escape]),
        }
    }

    /// One-byte text, ASCII or Latin-1, escaped in runs.
    #[inline]
    fn latin1_text(&mut self, text: &[u8]) {
        let mut start = 0;
        for (at, &byte) in text.iter().enumerate() {
            let escape = ESCAPES[byte as usize];
            if escape == 0 {
                continue;
            }
            self.buf.extend_from_slice(&text[start..at]);
            self.escape(byte, escape);
            start = at + 1;
        }
        self.buf.extend_from_slice(&text[start..]);
    }

    fn wide(&mut self, chars: impl Iterator<Item = u32>) {
        for c in chars {
            match ESCAPES.get(c as usize) {
                Some(0) => self.buf.push(c as u8),
                Some(&escape) => self.escape(c as u8, escape),
                None => self.non_ascii(c),
            }
        }
    }

    #[inline]
    fn non_ascii(&mut self, c: u32) {
        if c == 0x7f && !self.style.ensure_ascii {
            self.buf.push(0x7f);
        } else if self.style.ensure_ascii {
            if c >= 0x10000 {
                let v = c - 0x10000;
                self.unicode_escape(0xd800 | (v >> 10));
                self.unicode_escape(0xdc00 | (v & 0x3ff));
            } else {
                self.unicode_escape(c);
            }
        } else {
            self.ascii = false;
            push_utf8(&mut self.buf, c);
        }
    }

    /// An exact str, quoted and escaped.
    #[inline]
    pub fn str(&mut self, obj: &B<'_>) {
        let op = obj.as_ptr();
        if let Some((_, data, len, true)) = unsafe { str_view(op) } {
            let text = unsafe { std::slice::from_raw_parts(data, len) };
            if !ascii_needs_escape(text) {
                self.quoted(text);
                return;
            }
        }
        self.buf.push(b'"');
        unsafe {
            let len = ffi::PyUnicode_GET_LENGTH(op) as usize;
            let data = ffi::PyUnicode_DATA(op);
            match ffi::PyUnicode_KIND(op) {
                ffi::PyUnicode_1BYTE_KIND => {
                    self.latin1_text(std::slice::from_raw_parts(data.cast::<u8>(), len));
                }
                ffi::PyUnicode_2BYTE_KIND => self.wide(
                    std::slice::from_raw_parts(data.cast::<u16>(), len)
                        .iter()
                        .map(|&c| u32::from(c)),
                ),
                _ => self.wide(
                    std::slice::from_raw_parts(data.cast::<u32>(), len)
                        .iter()
                        .copied(),
                ),
            }
        }
        self.buf.push(b'"');
    }

    /// ``text``, which needs no escape, between quotes.
    #[inline(always)]
    pub fn quoted(&mut self, text: &[u8]) {
        let len = text.len();
        self.buf.reserve(len + 2);
        unsafe {
            let at = self.buf.as_mut_ptr().add(self.buf.len());
            *at = b'"';
            copy_small(text.as_ptr(), at.add(1), len);
            *at.add(len + 1) = b'"';
            self.buf.set_len(self.buf.len() + len + 2);
        }
    }

    /// An exact int, as ``int.__repr__`` writes it.
    #[inline]
    pub fn int(&mut self, obj: &B<'_>) -> Done<()> {
        let op = obj.as_ptr();
        if let Some(value) = unsafe { compact_int(op) } {
            if value < 0 {
                self.buf.push(b'-');
            }
            push_u64(&mut self.buf, value.unsigned_abs());
            return Ok(());
        }
        let mut overflow = 0;
        let value = unsafe { ffi::PyLong_AsLongLongAndOverflow(op, &mut overflow) };
        if overflow == 0 {
            if value == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
                unsafe { ffi::PyErr_Clear() };
                return Err(Bail);
            }
            if value < 0 {
                self.buf.push(b'-');
            }
            push_u64(&mut self.buf, value.unsigned_abs());
            return Ok(());
        }
        // Past 64 bits, and past ``int``'s digit limit, which raises.
        let Ok(text) = obj.repr() else {
            return Err(Bail);
        };
        match utf8(text.as_any()) {
            Some(text) => self.buf.extend_from_slice(text),
            None => return Err(Bail),
        }
        Ok(())
    }

    /// A float as ``json`` writes it, with ``allow_nan``.
    #[inline]
    pub fn float(&mut self, number: f64) -> Done<()> {
        if number.is_finite() {
            return float_repr(&mut self.buf, number);
        }
        self.buf.extend_from_slice(if number.is_nan() {
            b"NaN"
        } else if number > 0.0 {
            b"Infinity"
        } else {
            b"-Infinity"
        });
        Ok(())
    }

    /// A str, int, float, bool or ``None`` written; ``false`` for any other
    /// value, written or not.
    #[inline(always)]
    pub fn leaf(&mut self, obj: &B<'_>) -> Done<bool> {
        let kind = tp(obj);
        if kind == t_str() {
            self.str(obj);
        } else if kind == t_int() {
            self.int(obj)?;
        } else if kind == t_float() {
            self.float(unsafe { ffi::PyFloat_AS_DOUBLE(obj.as_ptr()) })?;
        } else if kind == t_bool() {
            self.raw(if obj.as_ptr() == unsafe { ffi::Py_True() } {
                b"true"
            } else {
                b"false"
            });
        } else if kind == t_none() {
            self.raw(b"null");
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    fn newline(&mut self, depth: u32) {
        if let Some(indent) = &self.style.indent {
            self.buf.push(b'\n');
            for _ in 0..depth {
                self.buf.extend_from_slice(indent);
            }
        }
    }

    /// A plain tree of exact dicts with str keys, lists, tuples and leaves.
    pub fn plain(&mut self, obj: &B<'_>, depth: u32) -> Done<()> {
        if self.leaf(obj)? {
            return Ok(());
        }
        if depth >= MAX_DEPTH {
            return Err(Bail);
        }
        let kind = tp(obj);
        let op = obj.as_ptr();
        let py = obj.py();
        if kind == t_list() || kind == t_tuple() {
            let len = unsafe {
                if kind == t_list() {
                    ffi::PyList_GET_SIZE(op)
                } else {
                    ffi::PyTuple_GET_SIZE(op)
                }
            };
            if len == 0 {
                self.raw(b"[]");
                return Ok(());
            }
            self.buf.push(b'[');
            for index in 0..len {
                if index > 0 {
                    self.item_sep();
                }
                self.newline(depth + 1);
                // No Python code runs while a tree is written, so a borrowed
                // item stays alive.
                let item = unsafe {
                    let item = if kind == t_list() {
                        ffi::PyList_GET_ITEM(op, index)
                    } else {
                        ffi::PyTuple_GET_ITEM(op, index)
                    };
                    Bound::from_borrowed_ptr(py, item)
                };
                self.plain(&item, depth + 1)?;
            }
            self.newline(depth);
            self.buf.push(b']');
            return Ok(());
        }
        if kind != t_dict() {
            return Err(Bail);
        }
        let dict = unsafe { obj.cast_unchecked::<PyDict>() };
        if dict.is_empty() {
            self.raw(b"{}");
            return Ok(());
        }
        let mut entries = dict_entries(dict);
        if entries.iter().any(|(key, _)| tp(key) != t_str()) {
            return Err(Bail);
        }
        if self.style.sort_keys {
            if entries.iter().any(|(key, _)| utf8(key).is_none()) {
                return Err(Bail);
            }
            // UTF-8 byte order is code point order, which ``sorted`` uses.
            entries.sort_by(|a, b| utf8(&a.0).cmp(&utf8(&b.0)));
        }
        self.buf.push(b'{');
        for (index, (key, value)) in entries.iter().enumerate() {
            if index > 0 {
                self.item_sep();
            }
            self.newline(depth + 1);
            self.str(key);
            self.key_sep();
            self.plain(value, depth + 1)?;
        }
        self.newline(depth);
        self.buf.push(b'}');
        Ok(())
    }
}

/// A code point as UTF-8; a lone surrogate as the three bytes
/// ``surrogatepass`` reads back.
#[inline]
fn push_utf8(out: &mut Vec<u8>, c: u32) {
    if c < 0x80 {
        out.push(c as u8);
    } else if c < 0x800 {
        out.extend_from_slice(&[0xc0 | (c >> 6) as u8, 0x80 | (c & 0x3f) as u8]);
    } else if c < 0x10000 {
        out.extend_from_slice(&[
            0xe0 | (c >> 12) as u8,
            0x80 | (c >> 6 & 0x3f) as u8,
            0x80 | (c & 0x3f) as u8,
        ]);
    } else {
        out.extend_from_slice(&[
            0xf0 | (c >> 18) as u8,
            0x80 | (c >> 12 & 0x3f) as u8,
            0x80 | (c >> 6 & 0x3f) as u8,
            0x80 | (c & 0x3f) as u8,
        ]);
    }
}

// -------------------------------------------------------------- json's own --

/// The ``json`` module and the ``dumps`` and ``loads`` it held when the core
/// was imported. Python's path looks ``json.dumps`` and ``json.loads`` up at
/// each call. While they are still these, the core writes and reads in their
/// place, giving what they give; once one is replaced, the core calls the
/// replacement as Python's path does: once, with the whole tree or text.
struct Own {
    module: Py<PyAny>,
    /// The module's ``__dict__``, which it keeps for life.
    names: Py<PyDict>,
    dumps: Py<PyAny>,
    loads: Py<PyAny>,
}

static OWN: GilCell<Option<Rc<Own>>> = GilCell::new(None);

/// Remember ``json``'s own functions, as the core is imported.
pub fn setup(py: Python<'_>) -> PyResult<()> {
    let module = py.import("json")?;
    let own = Rc::new(Own {
        dumps: module.getattr("dumps")?.unbind(),
        loads: module.getattr("loads")?.unbind(),
        names: module.dict().unbind(),
        module: module.into_any().unbind(),
    });
    let stale = OWN.replace(py, Some(own));
    drop(stale);
    Ok(())
}

fn own(py: Python<'_>) -> PyResult<Rc<Own>> {
    OWN.cloned(py)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("treekle._rust is not set up"))
}

// Perf: to_plain of a dict of 100 int keys, which checks once per key: 1.11x the time of no
// check, 1.15x with getattr; exp000 and rows unchanged.
/// ``json.<name>`` as Python reads it: from the module's dict, where a plain
/// module's attributes are, or by ``getattr`` for a module of another class
/// or a name the dict lacks.
fn current<'py>(own: &Own, name: &Bound<'py, pyo3::types::PyString>) -> PyResult<B<'py>> {
    let py = name.py();
    let module = own.module.bind(py);
    let plain = unsafe { ffi::Py_TYPE(module.as_ptr()) == &raw mut ffi::PyModule_Type };
    if plain && let Some(found) = own.names.bind(py).get_item(name)? {
        return Ok(found);
    }
    module.getattr(name)
}

/// What ``json.dumps`` is now, and whether it is ``json``'s own.
pub fn current_dumps(py: Python<'_>) -> PyResult<(B<'_>, bool)> {
    let own = own(py)?;
    let now = current(&own, pyo3::intern!(py, "dumps"))?;
    let is_own = now.is(own.dumps.bind(py));
    Ok((now, is_own))
}

/// What ``json.loads`` is now, and whether it is ``json``'s own.
pub fn current_loads(py: Python<'_>) -> PyResult<(B<'_>, bool)> {
    let own = own(py)?;
    let now = current(&own, pyo3::intern!(py, "loads"))?;
    let is_own = now.is(own.loads.bind(py));
    Ok((now, is_own))
}

/// ``json``'s own ``dumps``, whatever ``json.dumps`` is now.
pub fn own_dumps(py: Python<'_>) -> PyResult<B<'_>> {
    Ok(own(py)?.dumps.bind(py).clone())
}

/// ``dumps(obj, ...)``, passing the options given.
pub fn call_dumps<'py>(
    dumps: &B<'py>,
    obj: &B<'py>,
    ensure_ascii: Option<&B<'py>>,
    separators: Option<&B<'py>>,
    indent: Option<&B<'py>>,
    sort_keys: Option<&B<'py>>,
) -> PyResult<B<'py>> {
    let kwargs = PyDict::new(dumps.py());
    for (name, value) in [
        ("ensure_ascii", ensure_ascii),
        ("separators", separators),
        ("indent", indent),
        ("sort_keys", sort_keys),
    ] {
        if let Some(value) = value {
            kwargs.set_item(name, value)?;
        }
    }
    dumps.call((obj,), Some(&kwargs))
}

// Perf: exp000, min per call: to_plain plus this writer 1,157 us vs 1,938 with json.dumps (1.7x).
/// ``json.dumps`` of a plain tree.
pub fn dumps<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    ensure_ascii: Option<&B<'py>>,
    separators: Option<&B<'py>>,
    indent: Option<&B<'py>>,
    sort_keys: Option<&B<'py>>,
) -> PyResult<B<'py>> {
    let (dumps, is_own) = current_dumps(py)?;
    if is_own && let Ok(style) = Style::from_options(ensure_ascii, separators, indent, sort_keys)? {
        let mut writer = Writer::new(style);
        if writer.plain(obj, 0).is_ok() {
            return writer.finish(py);
        }
    }
    call_dumps(&dumps, obj, ensure_ascii, separators, indent, sort_keys)
}

// ---------------------------------------------------------------- reader --

/// ``json.decoder``'s ``NaN``, ``PosInf`` and ``NegInf``: ``json.loads``
/// returns these very objects for ``NaN``, ``Infinity`` and ``-Infinity``.
struct Constants {
    nan: Py<PyAny>,
    inf: Py<PyAny>,
    neg_inf: Py<PyAny>,
}

static CONSTANTS: GilCell<Option<Rc<Constants>>> = GilCell::new(None);

fn constants(py: Python<'_>) -> PyResult<Rc<Constants>> {
    if let Some(found) = CONSTANTS.cloned(py) {
        return Ok(found);
    }
    let decoder = py.import("json.decoder")?;
    let found = Rc::new(Constants {
        nan: decoder.getattr("NaN")?.unbind(),
        inf: decoder.getattr("PosInf")?.unbind(),
        neg_inf: decoder.getattr("NegInf")?.unbind(),
    });
    let stale = CONSTANTS.replace(py, Some(found.clone()));
    drop(stale);
    Ok(found)
}

const KEY_SLOTS: usize = 4096;
const KEY_LIMIT: usize = 64;
/// Slots a key may sit past its home slot. A document set with more distinct
/// keys than a direct-mapped cache holds would otherwise evict on every
/// collision: exp000 has 504 keys, and 512 direct slots missed on half of
/// its 7,257 key reads.
const KEY_PROBES: usize = 8;
const VALUE_SLOTS: usize = 2048;
const VALUE_LIMIT: usize = 64;
const VALUE_PROBES: usize = 2;
/// Misses a document may take beyond twice its hits before its values stop
/// going through the cache.
const VALUE_MISSES_ALLOWED: u32 = 256;

// Perf: exp000 json_loads, lockstep: 4,096 slots probed 8 ways, 1.07-1.09x over 512 direct-mapped
// (which evicted half of exp000's 504 keys).
/// Short ASCII keys, reused across documents as msgspec reuses them.
static KEYS: GilCell<Vec<Option<Py<PyAny>>>> = GilCell::new(Vec::new());
// Perf: exp000 parse only, lockstep: 1.07-1.11x.
/// Short ASCII string values, kept apart so that values never evict keys.
static VALUES: GilCell<Vec<Option<Py<PyAny>>>> = GilCell::new(Vec::new());

/// The cached str for ``bytes`` and whether it was found, probing up to
/// ``probes`` slots of ``table`` (a power-of-two length) and filling an empty
/// one, or the home slot, on a miss.
#[inline(always)]
fn cached<'py>(
    py: Python<'py>,
    table: &mut [Option<Py<PyAny>>],
    probes: usize,
    bytes: &[u8],
) -> Done<(B<'py>, bool)> {
    let mask = table.len() - 1;
    let home = key_hash(bytes) as usize;
    let mut free = None;
    for probe in 0..probes {
        let slot = (home + probe) & mask;
        let Some(found) = &table[slot] else {
            free = Some(slot);
            break;
        };
        let op = found.as_ptr();
        // Every cached str was made by ``ascii_str``.
        let same = unsafe {
            ffi::PyUnicode_GET_LENGTH(op) as usize == bytes.len()
                && std::slice::from_raw_parts(ascii_data(op), bytes.len()) == bytes
        };
        if same {
            return Ok((found.bind(py).clone(), true));
        }
    }
    let made = ascii_str(py, bytes)?;
    table[free.unwrap_or(home & mask)] = Some(made.clone().unbind());
    Ok((made, false))
}

/// Bytes that end a run of plain string text: a quote, a backslash, or a
/// control character, which ``strict`` mode refuses.
const STOPS: [bool; 256] = {
    let mut table = [false; 256];
    let mut byte = 0;
    while byte < 0x20 {
        table[byte] = true;
        byte += 1;
    }
    table[b'"' as usize] = true;
    table[b'\\' as usize] = true;
    table
};

/// The characters of a compact ASCII str, which ``PyUnicode_New(n, 127)``
/// makes: they follow the header. On 3.14 ``PyUnicode_DATA`` is an exported
/// call rather than this inline arithmetic.
#[inline(always)]
unsafe fn ascii_data(op: *mut ffi::PyObject) -> *mut u8 {
    unsafe {
        op.cast::<u8>()
            .add(std::mem::size_of::<ffi::PyASCIIObject>())
    }
}

/// A str of ASCII bytes, made directly.
#[inline]
fn ascii_str<'py>(py: Python<'py>, bytes: &[u8]) -> Done<B<'py>> {
    unsafe {
        let out = ffi::PyUnicode_New(bytes.len() as ffi::Py_ssize_t, 127);
        if out.is_null() {
            ffi::PyErr_Clear();
            return Err(Bail);
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ascii_data(out), bytes.len());
        Ok(Bound::from_owned_ptr(py, out))
    }
}

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGHS: u64 = 0x8080_8080_8080_8080;

/// Whether any of the eight bytes in ``word`` is a quote, a backslash or a
/// control character. Each test is exact about presence; only the positions
/// past the first hit may be wrong, and none is read.
#[inline(always)]
fn word_stops(word: u64) -> bool {
    let zero = |x: u64| x.wrapping_sub(ONES) & !x & HIGHS;
    let below_space = word.wrapping_sub(ONES * 0x20) & !word & HIGHS;
    (zero(word ^ (ONES * u64::from(b'"'))) | zero(word ^ (ONES * u64::from(b'\\'))) | below_space)
        != 0
}

// Perf: exp000 parse only, lockstep: word-at-a-time scan and hash, one shared item stack,
// 1.14-1.20x.
/// A cheap hash of a short key, read eight bytes at a time.
#[inline(always)]
fn key_hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let len = bytes.len();
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let mut hash = (len as u64).wrapping_mul(K);
    if len >= 8 {
        let mut at = 0;
        while at + 8 < len {
            hash = (hash ^ word(at)).wrapping_mul(K).rotate_left(29);
            at += 8;
        }
        hash = (hash ^ word(len - 8)).wrapping_mul(K);
    } else {
        let mut tail = 0u64;
        for (index, &byte) in bytes.iter().enumerate() {
            tail |= u64::from(byte) << (8 * index);
        }
        hash = (hash ^ tail).wrapping_mul(K);
    }
    hash ^ (hash >> 32)
}

/// A str of UTF-8 bytes, decoded as ``json.loads`` decodes bytes.
#[inline]
fn utf8_str<'py>(py: Python<'py>, bytes: &[u8]) -> Done<B<'py>> {
    unsafe {
        let out = ffi::PyUnicode_DecodeUTF8(
            bytes.as_ptr().cast(),
            bytes.len() as ffi::Py_ssize_t,
            c"surrogatepass".as_ptr(),
        );
        if out.is_null() {
            ffi::PyErr_Clear();
            return Err(Bail);
        }
        Ok(Bound::from_owned_ptr(py, out))
    }
}

/// Powers of ten a double holds exactly.
const EXACT_TENS: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

// Perf: rows json_loads, lockstep: ~1.03x.
/// A JSON number's value when its digits fit 2**53 and its decimal exponent
/// is within 22: then both the digits and the power of ten are exact
/// doubles, and one multiply or divide rounds correctly (Clinger's fast
/// path), as the full parser would. ``None`` for every other number.
#[inline]
fn exact_float(shown: &[u8]) -> Option<f64> {
    let len = shown.len();
    let mut at = usize::from(shown.first() == Some(&b'-'));
    let mut digits: u64 = 0;
    let mut count = 0;
    let mut scale: i32 = 0;
    while at < len && shown[at].is_ascii_digit() {
        if count == 19 {
            return None;
        }
        digits = digits * 10 + u64::from(shown[at] - b'0');
        count += 1;
        at += 1;
    }
    if at < len && shown[at] == b'.' {
        at += 1;
        while at < len && shown[at].is_ascii_digit() {
            if count == 19 {
                return None;
            }
            digits = digits * 10 + u64::from(shown[at] - b'0');
            count += 1;
            scale -= 1;
            at += 1;
        }
    }
    if at < len {
        // ``e`` or ``E``, an optional sign, and digits, as ``number`` read.
        at += 1;
        let negative = shown[at] == b'-';
        if matches!(shown[at], b'+' | b'-') {
            at += 1;
        }
        if len - at > 4 {
            return None;
        }
        let mut power: i32 = 0;
        while at < len {
            power = power * 10 + i32::from(shown[at] - b'0');
            at += 1;
        }
        scale += if negative { -power } else { power };
    }
    if digits > 1 << 53 || !(-22..=22).contains(&scale) {
        return None;
    }
    let value = digits as f64;
    let value = if scale >= 0 {
        value * EXACT_TENS[scale as usize]
    } else {
        value / EXACT_TENS[(-scale) as usize]
    };
    Some(if shown[0] == b'-' { -value } else { value })
}

#[inline]
fn hex4(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }
    let mut value = 0;
    for &byte in &bytes[..4] {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return None,
        };
        value = value * 16 + u32::from(digit);
    }
    Some(value)
}

pub struct Reader<'a, 'py> {
    py: Python<'py>,
    pub text: &'a [u8],
    pub at: usize,
    scratch: Vec<u8>,
    keys: Vec<Option<Py<PyAny>>>,
    values: Vec<Option<Py<PyAny>>>,
    value_hits: u32,
    value_misses: u32,
    constants: Rc<Constants>,
    /// Owned items of the open arrays and objects, innermost last. A bail
    /// leaves some behind; ``close`` releases them.
    pub stack: Vec<*mut ffi::PyObject>,
}

impl<'a, 'py> Reader<'a, 'py> {
    /// A reader of ``bytes``, holding the shared key and value caches until
    /// ``close`` returns them.
    pub fn open(py: Python<'py>, bytes: &'a [u8]) -> PyResult<Self> {
        let mut keys = KEYS.take(py);
        keys.resize_with(KEY_SLOTS, || None);
        let mut values = VALUES.take(py);
        values.resize_with(VALUE_SLOTS, || None);
        Ok(Reader {
            py,
            text: bytes,
            at: 0,
            scratch: Vec::new(),
            keys,
            values,
            value_hits: 0,
            value_misses: 0,
            constants: constants(py)?,
            stack: Vec::with_capacity(256),
        })
    }

    /// Release what a bail left on the stack and return the caches.
    pub fn close(mut self) {
        for left in self.stack.drain(..) {
            unsafe { ffi::Py_DECREF(left) };
        }
        let keys = std::mem::take(&mut self.keys);
        let stale = KEYS.replace(self.py, keys);
        drop(stale);
        let values = std::mem::take(&mut self.values);
        let stale = VALUES.replace(self.py, values);
        drop(stale);
    }

    #[inline(always)]
    pub fn peek(&self) -> u8 {
        self.text.get(self.at).copied().unwrap_or(0)
    }

    #[inline(always)]
    pub fn space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.text.get(self.at) {
            self.at += 1;
        }
    }

    /// The bytes of a string whose opening quote was read, past its closing
    /// quote, when it is plain ASCII with no escape: the same text the str
    /// ``string`` would make. ``None`` reads nothing.
    #[inline(always)]
    pub fn plain_ascii(&mut self) -> Option<&'a [u8]> {
        let text = self.text;
        let start = self.at;
        let mut at = start;
        let mut words = 0u64;
        while at + 8 <= text.len() {
            let word = u64::from_le_bytes(text[at..at + 8].try_into().unwrap());
            if word_stops(word) {
                break;
            }
            words |= word;
            at += 8;
        }
        let mut high = if words & HIGHS != 0 { 0x80u8 } else { 0 };
        while at < text.len() && !STOPS[text[at] as usize] {
            high |= text[at];
            at += 1;
        }
        if high >= 0x80 || text.get(at) != Some(&b'"') {
            return None;
        }
        self.at = at + 1;
        Some(&text[start..at])
    }

    pub fn value(&mut self, depth: u32) -> Done<B<'py>> {
        match self.peek() {
            b'"' => {
                self.at += 1;
                self.string(false)
            }
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b't' => self.word(b"true", unsafe { ffi::Py_True() }),
            b'f' => self.word(b"false", unsafe { ffi::Py_False() }),
            b'n' => self.word(b"null", unsafe { ffi::Py_None() }),
            b'N' => self.word(b"NaN", self.constants.nan.as_ptr()),
            b'I' => self.word(b"Infinity", self.constants.inf.as_ptr()),
            b'-' if self.text[self.at..].starts_with(b"-Infinity") => {
                self.word(b"-Infinity", self.constants.neg_inf.as_ptr())
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(Bail),
        }
    }

    #[inline]
    fn word(&mut self, word: &[u8], value: *mut ffi::PyObject) -> Done<B<'py>> {
        if !self.text[self.at..].starts_with(word) {
            return Err(Bail);
        }
        self.at += word.len();
        Ok(unsafe { Bound::from_borrowed_ptr(self.py, value) })
    }

    fn number(&mut self) -> Done<B<'py>> {
        let text = self.text;
        let start = self.at;
        let mut at = start;
        if text.get(at) == Some(&b'-') {
            at += 1;
        }
        let digits_start = at;
        match text.get(at) {
            Some(b'0') => at += 1,
            Some(b'1'..=b'9') => {
                while let Some(b'0'..=b'9') = text.get(at) {
                    at += 1;
                }
            }
            _ => return Err(Bail),
        }
        let int_end = at;
        let mut is_float = false;
        if text.get(at) == Some(&b'.') && matches!(text.get(at + 1), Some(b'0'..=b'9')) {
            at += 2;
            while let Some(b'0'..=b'9') = text.get(at) {
                at += 1;
            }
            is_float = true;
        }
        if let Some(b'e' | b'E') = text.get(at) {
            let mut end = at + 1;
            if let Some(b'+' | b'-') = text.get(end) {
                end += 1;
            }
            if matches!(text.get(end), Some(b'0'..=b'9')) {
                while let Some(b'0'..=b'9') = text.get(end) {
                    end += 1;
                }
                at = end;
                is_float = true;
            }
        }
        self.at = at;
        let shown = &text[start..at];
        if is_float {
            // Both parsers round correctly, so they agree on every input.
            let number: f64 = match exact_float(shown) {
                Some(number) => number,
                None => unsafe { std::str::from_utf8_unchecked(shown) }
                    .parse()
                    .map_err(|_| Bail)?,
            };
            return Ok(unsafe { Bound::from_owned_ptr(self.py, ffi::PyFloat_FromDouble(number)) });
        }
        if int_end - digits_start <= 18 {
            let mut value: i64 = 0;
            for &digit in &text[digits_start..int_end] {
                value = value * 10 + i64::from(digit - b'0');
            }
            if start != digits_start {
                value = -value;
            }
            return Ok(unsafe { Bound::from_owned_ptr(self.py, ffi::PyLong_FromLongLong(value)) });
        }
        // ``int``'s own parser, with its digit limit.
        self.scratch.clear();
        self.scratch.extend_from_slice(shown);
        self.scratch.push(0);
        unsafe {
            let out =
                ffi::PyLong_FromString(self.scratch.as_ptr().cast(), std::ptr::null_mut(), 10);
            if out.is_null() {
                ffi::PyErr_Clear();
                return Err(Bail);
            }
            Ok(Bound::from_owned_ptr(self.py, out))
        }
    }

    /// The rest of a string whose opening quote was read.
    pub fn string(&mut self, key: bool) -> Done<B<'py>> {
        let text = self.text;
        let start = self.at;
        let mut at = start;
        let mut words = 0u64;
        while at + 8 <= text.len() {
            let word = u64::from_le_bytes(text[at..at + 8].try_into().unwrap());
            if word_stops(word) {
                break;
            }
            words |= word;
            at += 8;
        }
        let mut high = if words & HIGHS != 0 { 0x80u8 } else { 0 };
        while at < text.len() && !STOPS[text[at] as usize] {
            high |= text[at];
            at += 1;
        }
        match text.get(at) {
            Some(b'"') => {
                self.at = at + 1;
                let bytes = &text[start..at];
                if high < 0x80 {
                    if key {
                        if bytes.len() <= KEY_LIMIT {
                            return self.cached_key(bytes);
                        }
                    } else if bytes.len() <= VALUE_LIMIT {
                        return self.cached_value(bytes);
                    }
                    return ascii_str(self.py, bytes);
                }
                utf8_str(self.py, bytes)
            }
            Some(b'\\') => self.escaped(start, at),
            _ => Err(Bail),
        }
    }

    /// A string with escapes, from ``start``; ``at`` is its first backslash.
    fn escaped(&mut self, start: usize, mut at: usize) -> Done<B<'py>> {
        let text = self.text;
        let mut out = std::mem::take(&mut self.scratch);
        out.clear();
        out.extend_from_slice(&text[start..at]);
        loop {
            match text.get(at) {
                Some(b'"') => {
                    at += 1;
                    break;
                }
                Some(b'\\') => {
                    let Some(&letter) = text.get(at + 1) else {
                        return Err(Bail);
                    };
                    at += 2;
                    let byte = match letter {
                        b'"' | b'\\' | b'/' => letter,
                        b'b' => 0x08,
                        b'f' => 0x0c,
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'u' => {
                            let mut c = hex4(&text[at..]).ok_or(Bail)?;
                            at += 4;
                            if (0xd800..0xdc00).contains(&c) && text[at..].starts_with(b"\\u") {
                                // As ``json``: a high surrogate takes a following
                                // ``\u`` escape as its low half when it is one.
                                let low = hex4(&text[at + 2..]).ok_or(Bail)?;
                                if (0xdc00..0xe000).contains(&low) {
                                    c = 0x10000 + ((c - 0xd800) << 10) + (low - 0xdc00);
                                    at += 6;
                                }
                            }
                            push_utf8(&mut out, c);
                            continue;
                        }
                        _ => return Err(Bail),
                    };
                    out.push(byte);
                }
                Some(&byte) if byte >= 0x20 => {
                    let run = at;
                    while at < text.len() && !STOPS[text[at] as usize] {
                        at += 1;
                    }
                    out.extend_from_slice(&text[run..at]);
                }
                _ => return Err(Bail),
            }
        }
        self.at = at;
        let made = if out.is_ascii() {
            ascii_str(self.py, &out)
        } else {
            utf8_str(self.py, &out)
        };
        self.scratch = out;
        made
    }

    #[inline]
    fn cached_key(&mut self, bytes: &[u8]) -> Done<B<'py>> {
        cached(self.py, &mut self.keys, KEY_PROBES, bytes).map(|(made, _)| made)
    }

    /// A short ASCII value, shared as keys are while the document's values
    /// repeat. Values are often all distinct (ids, names, numbers as text),
    /// where probing would only cost, so a document that keeps missing stops
    /// using the cache.
    #[inline]
    fn cached_value(&mut self, bytes: &[u8]) -> Done<B<'py>> {
        if self.value_misses > VALUE_MISSES_ALLOWED + 2 * self.value_hits {
            return ascii_str(self.py, bytes);
        }
        let (made, hit) = cached(self.py, &mut self.values, VALUE_PROBES, bytes)?;
        if hit {
            self.value_hits += 1;
        } else {
            self.value_misses += 1;
        }
        Ok(made)
    }

    fn object(&mut self, depth: u32) -> Done<B<'py>> {
        if depth >= MAX_DEPTH {
            return Err(Bail);
        }
        self.at += 1;
        self.space();
        if self.peek() == b'}' {
            self.at += 1;
            return Ok(unsafe { Bound::from_owned_ptr(self.py, ffi::PyDict_New()) });
        }
        // Pairs wait on the shared stack until the size is known, so the dict
        // is allocated once at its final size instead of growing.
        let base = self.stack.len();
        loop {
            if self.peek() != b'"' {
                return Err(Bail);
            }
            self.at += 1;
            let key = self.string(true)?;
            self.stack.push(key.into_ptr());
            self.space();
            if self.peek() != b':' {
                return Err(Bail);
            }
            self.at += 1;
            self.space();
            let value = self.value(depth + 1)?;
            self.stack.push(value.into_ptr());
            self.space();
            match self.peek() {
                b',' => {
                    self.at += 1;
                    self.space();
                }
                b'}' => {
                    self.at += 1;
                    break;
                }
                _ => return Err(Bail),
            }
        }
        let py = self.py;
        let pairs = self.stack.drain(base..);
        unsafe {
            let dict = ffi::_PyDict_NewPresized((pairs.len() / 2) as ffi::Py_ssize_t);
            let mut failed = dict.is_null();
            let mut pairs = pairs.map(|owned| Bound::from_owned_ptr(py, owned));
            while let (Some(key), Some(value)) = (pairs.next(), pairs.next()) {
                if !failed && ffi::PyDict_SetItem(dict, key.as_ptr(), value.as_ptr()) < 0 {
                    failed = true;
                }
            }
            if failed {
                ffi::PyErr_Clear();
                if !dict.is_null() {
                    ffi::Py_DECREF(dict);
                }
                return Err(Bail);
            }
            Ok(Bound::from_owned_ptr(py, dict))
        }
    }

    fn array(&mut self, depth: u32) -> Done<B<'py>> {
        if depth >= MAX_DEPTH {
            return Err(Bail);
        }
        self.at += 1;
        self.space();
        let base = self.stack.len();
        if self.peek() == b']' {
            self.at += 1;
        } else {
            loop {
                let item = self.value(depth + 1)?;
                self.stack.push(item.into_ptr());
                self.space();
                match self.peek() {
                    b',' => {
                        self.at += 1;
                        self.space();
                    }
                    b']' => {
                        self.at += 1;
                        break;
                    }
                    _ => return Err(Bail),
                }
            }
        }
        unsafe {
            let list = ffi::PyList_New((self.stack.len() - base) as ffi::Py_ssize_t);
            if list.is_null() {
                ffi::PyErr_Clear();
                return Err(Bail);
            }
            for (index, item) in self.stack.drain(base..).enumerate() {
                ffi::PyList_SET_ITEM(list, index as ffi::Py_ssize_t, item);
            }
            Ok(Bound::from_owned_ptr(self.py, list))
        }
    }
}

/// The document's bytes as ``json.loads`` would decode them, or ``Bail``
/// where only it can say: another encoding, a BOM, lone surrogates.
pub fn document<'a>(text: &'a B<'_>) -> Done<&'a [u8]> {
    if tp(text) == t_str() {
        return utf8(text).ok_or(Bail);
    }
    if unsafe { ffi::PyBytes_CheckExact(text.as_ptr()) } == 0 {
        return Err(Bail);
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(
            ffi::PyBytes_AS_STRING(text.as_ptr()).cast::<u8>(),
            ffi::PyBytes_Size(text.as_ptr()) as usize,
        )
    };
    // ``json.detect_encoding`` reads UTF-16 and UTF-32 from zero bytes.
    if bytes.first() == Some(&0) || (bytes.len() >= 2 && bytes[1] == 0) {
        return Err(Bail);
    }
    Ok(bytes)
}

/// ``json.loads(text)`` read natively, or ``None`` where ``json`` must.
pub fn read<'py>(py: Python<'py>, text: &B<'py>) -> PyResult<Option<B<'py>>> {
    let Ok(bytes) = document(text) else {
        return Ok(None);
    };
    let mut reader = Reader::open(py, bytes)?;
    reader.space();
    let mut value = reader.value(0);
    reader.space();
    if reader.at != bytes.len() {
        value = Err(Bail);
    }
    reader.close();
    Ok(value.ok())
}

// Perf: exp000, min per call: this parser plus from_plain 1,573-1,692 us vs 2,404 with json.loads
// (1.4-1.5x).
/// ``json.loads(text)``.
pub fn loads<'py>(py: Python<'py>, text: &B<'py>) -> PyResult<B<'py>> {
    let (loads, is_own) = current_loads(py)?;
    if is_own && let Some(value) = read(py, text)? {
        return Ok(value);
    }
    loads.call1((text,))
}
