//! Where a node sits, as parent-pointer frames, and the values placed by path.
//!
//! A path tuple is built only when Python needs one (a bad part); placing a
//! node costs one frame and one hash-table insert.

use hashbrown::HashTable;
use pyo3::ffi;
use pyo3::prelude::*;

use crate::py::{B, py_eq, t_int, t_str, tp};

pub const ROOT: u32 = 0;
const SEED: u64 = 0x243f_6a88_85a3_08d3;

#[inline(always)]
fn mix(hash: u64, step: u64) -> u64 {
    (hash.rotate_left(5) ^ step).wrapping_mul(0x517c_c1b7_2722_0a95)
}

pub enum Step<'py> {
    Index(usize),
    Key(B<'py>),
}

#[derive(Clone, Copy)]
pub enum StepRef<'a, 'py> {
    Index(usize),
    Key(&'a B<'py>),
}

/// A node's location: a frame already made, or one step below a frame.
#[derive(Clone, Copy)]
pub enum Loc<'a, 'py> {
    At(u32),
    Child(u32, StepRef<'a, 'py>),
}

struct Frame<'py> {
    parent: u32,
    depth: u32,
    hash: u64,
    step: Step<'py>,
}

pub struct Paths<'py> {
    frames: Vec<Frame<'py>>,
}

// Perf: exp000 from_plain, paired: part of decode.rs Reading's 1.02x.
// Python keeps paths as tuples in dicts, so a key's ``__hash__`` and ``__eq__``
// run, and raise, wherever that dict hashes or compares the path.
fn key_hash(key: &B<'_>) -> PyResult<u64> {
    // A str caches its hash at ``PyASCIIObject.hash``; reading it skips the
    // exported ``PyObject_Hash`` call for the common string step.
    if tp(key) == t_str() {
        let hash = unsafe { (*key.as_ptr().cast::<ffi::PyASCIIObject>()).hash };
        if hash != -1 {
            return Ok(hash as u64);
        }
    }
    let hash = unsafe { ffi::PyObject_Hash(key.as_ptr()) };
    if hash == -1 {
        return Err(PyErr::fetch(key.py()));
    }
    Ok(hash as u64)
}

fn index_matches(index: usize, other: &B<'_>) -> PyResult<bool> {
    let kind = tp(other);
    if kind == t_int() {
        return Ok(other.extract::<usize>().is_ok_and(|value| value == index));
    }
    if kind == t_str() {
        return Ok(false);
    }
    py_eq(index.into_pyobject(other.py())?.as_any(), other)
}

// Perf: exp000 from_plain, lockstep: part of decode.rs SPARE's ~1.04x.
/// Emptied walk tables kept between calls, so a call reuses the last one's
/// allocations instead of growing its tables from nothing.
pub struct Spare {
    frames: Vec<Frame<'static>>,
    table: HashTable<Placed<'static>>,
    log: Vec<(u64, u32)>,
    built: Vec<B<'static>>,
}

/// Past this many entries a table is freed rather than kept.
const SPARE_LIMIT: usize = 1 << 16;

impl Spare {
    /// The tables, for a walk that runs while ``'py`` holds.
    pub fn take<'py>(self) -> (Paths<'py>, PlacedTable<'py>, Vec<B<'py>>) {
        // SAFETY: every table is empty (``park`` clears it), so no value of
        // the old lifetime remains; the types differ only by lifetime.
        let (frames, table, built) = unsafe {
            (
                std::mem::transmute::<Vec<Frame<'static>>, Vec<Frame<'py>>>(self.frames),
                std::mem::transmute::<HashTable<Placed<'static>>, HashTable<Placed<'py>>>(
                    self.table,
                ),
                std::mem::transmute::<Vec<B<'static>>, Vec<B<'py>>>(self.built),
            )
        };
        (
            Paths::from_frames(frames),
            PlacedTable {
                table,
                log: self.log,
            },
            built,
        )
    }

    /// Empty the tables and keep their allocations. Dropping the values may
    /// run Python code, so the tables are emptied before they are erased.
    pub fn park<'py>(
        paths: Paths<'py>,
        placed: PlacedTable<'py>,
        built: Vec<B<'py>>,
    ) -> Option<Self> {
        let (mut frames, mut table, mut log, mut built) =
            (paths.frames, placed.table, placed.log, built);
        frames.clear();
        table.clear();
        log.clear();
        built.clear();
        if frames.capacity() > SPARE_LIMIT
            || table.capacity() > SPARE_LIMIT
            || built.capacity() > SPARE_LIMIT
        {
            return None;
        }
        // SAFETY: as in ``take``; each table is empty.
        unsafe {
            Some(Self {
                frames: std::mem::transmute::<Vec<Frame<'py>>, Vec<Frame<'static>>>(frames),
                table: std::mem::transmute::<HashTable<Placed<'py>>, HashTable<Placed<'static>>>(
                    table,
                ),
                log,
                built: std::mem::transmute::<Vec<B<'py>>, Vec<B<'static>>>(built),
            })
        }
    }
}

impl<'py> Paths<'py> {
    pub fn new(py: Python<'py>) -> Self {
        let _ = py;
        Self::from_frames(Vec::with_capacity(4))
    }

    /// No frames at all, not even the root; only a placeholder to move out of.
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    fn from_frames(mut frames: Vec<Frame<'py>>) -> Self {
        debug_assert!(frames.is_empty());
        frames.push(Frame {
            parent: ROOT,
            depth: 0,
            hash: SEED,
            step: Step::Index(0),
        });
        Self { frames }
    }

    #[inline]
    pub fn frame(&mut self, loc: Loc<'_, 'py>) -> PyResult<u32> {
        match loc {
            Loc::At(frame) => Ok(frame),
            Loc::Child(parent, step) => self.push(parent, step),
        }
    }

    fn push(&mut self, parent: u32, step: StepRef<'_, 'py>) -> PyResult<u32> {
        let (step, step_hash) = match step {
            StepRef::Index(index) => (Step::Index(index), index as u64),
            StepRef::Key(key) => (Step::Key(key.clone()), key_hash(key)?),
        };
        let up = &self.frames[parent as usize];
        let frame = Frame {
            parent,
            depth: up.depth + 1,
            hash: mix(up.hash, step_hash),
            step,
        };
        self.frames.push(frame);
        Ok((self.frames.len() - 1) as u32)
    }

    #[inline]
    pub fn hash(&self, frame: u32) -> u64 {
        self.frames[frame as usize].hash
    }

    fn step_eq(left: &Step<'py>, right: &Step<'py>) -> PyResult<bool> {
        match (left, right) {
            (Step::Index(a), Step::Index(b)) => Ok(a == b),
            (Step::Key(a), Step::Key(b)) => py_eq(a, b),
            (Step::Index(a), Step::Key(b)) | (Step::Key(b), Step::Index(a)) => index_matches(*a, b),
        }
    }

    /// Whether two frames name the same path.
    pub fn same(&self, mut left: u32, mut right: u32) -> PyResult<bool> {
        if self.frames[left as usize].depth != self.frames[right as usize].depth {
            return Ok(false);
        }
        while left != right {
            let (a, b) = (&self.frames[left as usize], &self.frames[right as usize]);
            if a.hash != b.hash || !Self::step_eq(&a.step, &b.step)? {
                return Ok(false);
            }
            left = a.parent;
            right = b.parent;
        }
        Ok(true)
    }

    /// The hash a frame for ``steps`` would have; steps are str or int.
    pub fn hash_of(steps: &[B<'py>]) -> PyResult<u64> {
        steps
            .iter()
            .try_fold(SEED, |hash, step| Ok(mix(hash, key_hash(step)?)))
    }

    /// Whether ``frame`` names exactly ``steps``.
    pub fn matches(&self, mut frame: u32, steps: &[B<'py>]) -> PyResult<bool> {
        if self.frames[frame as usize].depth as usize != steps.len() {
            return Ok(false);
        }
        for step in steps.iter().rev() {
            let here = &self.frames[frame as usize];
            let equal = match &here.step {
                Step::Index(index) => index_matches(*index, step)?,
                Step::Key(key) => py_eq(key, step)?,
            };
            if !equal {
                return Ok(false);
            }
            frame = here.parent;
        }
        Ok(true)
    }

    /// The path tuple Python code sees for ``frame``.
    pub fn tuple(
        &self,
        py: Python<'py>,
        mut frame: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let mut steps: Vec<B<'py>> = Vec::with_capacity(self.frames[frame as usize].depth as usize);
        while frame != ROOT {
            let here = &self.frames[frame as usize];
            steps.push(match &here.step {
                Step::Index(index) => index.into_pyobject(py)?.into_any(),
                Step::Key(key) => key.clone(),
            });
            frame = here.parent;
        }
        steps.reverse();
        pyo3::types::PyTuple::new(py, steps)
    }
}

pub struct Placed<'py> {
    hash: u64,
    frame: u32,
    value: B<'py>,
}

/// Every placed value by path, with the order new paths arrived for rollback.
pub struct PlacedTable<'py> {
    table: HashTable<Placed<'py>>,
    log: Vec<(u64, u32)>,
}

impl<'py> PlacedTable<'py> {
    pub fn new() -> Self {
        Self {
            table: HashTable::new(),
            log: Vec::new(),
        }
    }

    // The table's equality test cannot fail, so in ``set`` and ``get`` the
    // first error a key raises ends the search and is returned after it.
    pub fn set(&mut self, paths: &Paths<'py>, frame: u32, value: &B<'py>) -> PyResult<()> {
        let hash = paths.hash(frame);
        let mut failed = None;
        let found = self.table.find_mut(hash, |entry| {
            entry.hash == hash
                && failed.is_none()
                && paths.same(entry.frame, frame).unwrap_or_else(|error| {
                    failed = Some(error);
                    false
                })
        });
        if let Some(error) = failed {
            return Err(error);
        }
        if let Some(entry) = found {
            entry.value = value.clone();
            return Ok(());
        }
        self.table.insert_unique(
            hash,
            Placed {
                hash,
                frame,
                value: value.clone(),
            },
            |entry| entry.hash,
        );
        self.log.push((hash, frame));
        Ok(())
    }

    pub fn get(&self, paths: &Paths<'py>, steps: &[B<'py>]) -> PyResult<Option<&B<'py>>> {
        let hash = Paths::hash_of(steps)?;
        let mut failed = None;
        let found = self.table.find(hash, |entry| {
            entry.hash == hash
                && failed.is_none()
                && paths.matches(entry.frame, steps).unwrap_or_else(|error| {
                    failed = Some(error);
                    false
                })
        });
        match failed {
            Some(error) => Err(error),
            None => Ok(found.map(|entry| &entry.value)),
        }
    }

    #[inline]
    pub fn mark(&self) -> usize {
        self.log.len()
    }

    /// Forget every path first placed since ``mark``.
    pub fn rollback(&mut self, mark: usize) {
        for (hash, frame) in self.log.drain(mark..) {
            if let Ok(entry) = self.table.find_entry(hash, |entry| entry.frame == frame) {
                entry.remove();
            }
        }
    }
}
