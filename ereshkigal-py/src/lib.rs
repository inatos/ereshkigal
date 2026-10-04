use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
struct Engine {}

#[pymethods]
impl Engine {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

#[pyfunction]
fn schema() -> PyResult<String> {
    ereshkigal_core::schema::library_schema_json().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn ereshkigal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    Ok(())
}
