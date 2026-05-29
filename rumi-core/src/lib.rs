//! rumi._core — the native core extension.
//!
//! The Rust core owns the filesystem Tablet (codec + atomic byte I/O) and the
//! whirl-pipeline (orchestration; Ideas and the Heart stay Python). Built as an
//! abi3 extension (`abi3-py311`) so one wheel spans CPython 3.11+.

use pyo3::prelude::*;

mod codec;
mod errors;
mod pipeline;
mod tablet;

/// Module-level codec functions, exposed so the test suite can assert
/// byte-identity against the Python encoder via committed golden vectors.
#[pyfunction]
fn safe_filename(key: &str) -> String {
    codec::safe_filename(key)
}

#[pyfunction]
fn decode_safe_filename(name: &str) -> String {
    codec::decode_safe_filename(name)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tablet::Tablet>()?;
    m.add_function(wrap_pyfunction!(safe_filename, m)?)?;
    m.add_function(wrap_pyfunction!(decode_safe_filename, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::run_whirl_pipeline, m)?)?;
    Ok(())
}
