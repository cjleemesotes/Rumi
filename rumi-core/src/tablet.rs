//! Tablet — the filesystem layer of rumi's per-(agent_type, agent_id) store.
//!
//! Owns the filename codec, path/scope management, atomic byte writes, reads,
//! and listing. It stores OPAQUE BYTES: JSON (de)serialization stays in Python
//! (the caller's concern), which makes file-content byte-compatibility with the
//! Python `FilesystemTablet` automatic.
//!
//! Single-owner by design (not `Sync`): scale by spawning worker processes, not
//! by sharing one tablet across threads. (Locked Phase-0 decision.)

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

use crate::codec::{decode_safe_filename, safe_filename};

#[pyclass]
pub struct Tablet {
    scope_dir: PathBuf,
    #[pyo3(get)]
    agent_type: String,
    #[pyo3(get)]
    agent_id: String,
}

#[pymethods]
impl Tablet {
    #[new]
    fn new(data_dir: &str, agent_type: &str, agent_id: &str) -> PyResult<Self> {
        // Same validation + messages as Python's TabletScopeMixin.
        if agent_type.is_empty() || agent_id.is_empty() {
            return Err(PyValueError::new_err(
                "agent_type and agent_id must both be non-empty",
            ));
        }
        if agent_type.contains('/') || agent_id.contains('/') {
            return Err(PyValueError::new_err(
                "agent_type and agent_id must not contain '/'",
            ));
        }
        let scope_dir = Path::new(data_dir).join(agent_type).join(agent_id);
        fs::create_dir_all(&scope_dir).map_err(|e| PyOSError::new_err(e.to_string()))?;
        Ok(Tablet {
            scope_dir,
            agent_type: agent_type.to_string(),
            agent_id: agent_id.to_string(),
        })
    }

    /// Absolute scope directory (`{data_dir}/{agent_type}/{agent_id}`).
    fn scope_dir(&self) -> String {
        self.scope_dir.to_string_lossy().into_owned()
    }

    /// Absolute path of the file backing `key`.
    fn path(&self, key: &str) -> String {
        self.file_for(key).to_string_lossy().into_owned()
    }

    /// Store a pre-serialized JSON string atomically (tempfile + atomic rename,
    /// mirroring Python's `mkstemp` + `os.replace`). On failure the temp file is
    /// dropped (auto-unlinked) and the error propagates.
    fn set(&self, key: &str, json_str: &str) -> PyResult<()> {
        let target = self.file_for(key);
        let parent = target.parent().unwrap_or(self.scope_dir.as_path());
        fs::create_dir_all(parent).map_err(|e| PyOSError::new_err(e.to_string()))?;
        let mut tmp = NamedTempFile::new_in(parent).map_err(|e| PyOSError::new_err(e.to_string()))?;
        tmp.write_all(json_str.as_bytes())
            .map_err(|e| PyOSError::new_err(e.to_string()))?;
        tmp.flush().map_err(|e| PyOSError::new_err(e.to_string()))?;
        tmp.persist(&target)
            .map_err(|e| PyOSError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Raw stored JSON string, or None if the key is missing or unreadable.
    /// (The Python adapter applies `json.loads`; a corrupt payload → its
    /// caller-supplied default, matching the Python tablet.)
    fn get(&self, key: &str) -> Option<String> {
        fs::read_to_string(self.file_for(key)).ok()
    }

    /// Delete a key. True iff it existed and was removed.
    fn delete(&self, key: &str) -> bool {
        let target = self.file_for(key);
        if !target.exists() {
            return false;
        }
        fs::remove_file(&target).is_ok()
    }

    /// Decoded keys under this scope, optionally prefix-filtered, sorted on the
    /// DECODED string (matching Python's `sorted()` over decoded keys).
    fn list_keys(&self, prefix: &str) -> Vec<String> {
        let mut keys: Vec<String> = Vec::new();
        let rd = match fs::read_dir(&self.scope_dir) {
            Ok(rd) => rd,
            Err(_) => return keys,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let fname = match p.file_name().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            if !fname.ends_with(".json") {
                continue;
            }
            let stem = &fname[..fname.len() - ".json".len()];
            let decoded = decode_safe_filename(stem);
            if decoded.starts_with(prefix) {
                keys.push(decoded);
            }
        }
        keys.sort();
        keys
    }
}

impl Tablet {
    fn file_for(&self, key: &str) -> PathBuf {
        self.scope_dir.join(format!("{}.json", safe_filename(key)))
    }
}
