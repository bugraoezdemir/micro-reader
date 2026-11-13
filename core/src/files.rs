//! The files the engine reads, owned by the engine and shared by all its threads.
//!
//! Positional reads (`pread`) on one handle run concurrently on Unix: one handle
//! per file.  On Windows the system serialises reads on a synchronous handle, even
//! positional ones, so a file keeps a few handles: as many as threads read it at
//! the same time, opened when that first happens (a file read by one thread at a
//! time -- one plane of a file set -- keeps one).

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::sync::{Arc, RwLock};

/// One open file.
pub struct Source {
    pub path: String,
    #[cfg(unix)]
    file: File,
    /// handles not in use (Windows: one read at a time each)
    #[cfg(windows)]
    idle: std::sync::Mutex<Vec<File>>,
}

impl Source {
    #[cfg(unix)]
    pub fn open(path: &str) -> io::Result<Self> {
        Ok(Self { path: path.to_owned(), file: File::open(path)? })
    }

    #[cfg(windows)]
    pub fn open(path: &str) -> io::Result<Self> {
        let first = File::open(path)?;
        Ok(Self { path: path.to_owned(), idle: std::sync::Mutex::new(vec![first]) })
    }

    /// Fill `buf` from byte `offset`; Ok(false) when the file ends first.
    pub fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<bool> {
        let mut done = 0usize;
        while done < buf.len() {
            let n = self.read_at(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Ok(false);
            }
            done += n;
        }
        Ok(true)
    }

    #[cfg(unix)]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        use std::os::unix::fs::FileExt;
        loop {
            match self.file.read_at(buf, offset) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                other => return other,
            }
        }
    }

    #[cfg(windows)]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        // ReadFile with an OVERLAPPED offset on a handle no other thread is using
        use std::os::windows::fs::FileExt;
        let taken = self.idle.lock().unwrap().pop();
        let file = match taken {
            Some(f) => f,
            None => File::open(&self.path)?,      // another thread reads this file now
        };
        let result = loop {
            match file.seek_read(buf, offset) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                other => break other,
            }
        };
        self.idle.lock().unwrap().push(file);
        result
    }

    /// The error of a read that met the end of the file, as micro-reader words it.
    pub fn ends_early(&self, offset: u64) -> String {
        format!("{}: data at byte {offset} ends early (truncated file?)", quoted(&self.path))
    }
}

/// `path` as Python's repr() prints a str (single quotes, backslashes doubled), so
/// the engine's messages read as micro-reader's own.
pub fn quoted(path: &str) -> String {
    format!("'{}'", path.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// The open files, by the number `open` gave them (never reused while the engine
/// lives, so a closed file's cached segments can never be taken for another's).
#[derive(Default)]
pub struct Files {
    open: RwLock<HashMap<u64, Arc<Source>>>,
    next: std::sync::atomic::AtomicU64,
}

impl Files {
    pub fn open(&self, path: &str) -> io::Result<u64> {
        let source = Arc::new(Source::open(path)?);
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        self.open.write().unwrap().insert(id, source);
        Ok(id)
    }

    /// Forget file `id`; reads still running keep its handle until they end.
    pub fn close(&self, id: u64) -> bool {
        self.open.write().unwrap().remove(&id).is_some()
    }

    pub fn get(&self, id: u64) -> Option<Arc<Source>> {
        self.open.read().unwrap().get(&id).cloned()
    }

    pub fn count(&self) -> usize {
        self.open.read().unwrap().len()
    }
}
