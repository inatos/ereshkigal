use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::Value;
use std::path::PathBuf;

#[pyclass]
struct Engine {
    gguf: Option<String>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (gguf=None))]
    fn new(gguf: Option<String>) -> Self {
        Self { gguf }
    }

    /// JSON Schema for decree libraries (no GGUF required).
    fn schema(&self) -> PyResult<String> {
        ereshkigal_core::library_schema_json()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Score one decision row when a GGUF was provided at construction.
    #[pyo3(signature = (state, question, options, id=None))]
    fn decide(
        &self,
        state: &Bound<'_, PyAny>,
        question: &str,
        options: &Bound<'_, PyAny>,
        id: Option<&str>,
    ) -> PyResult<String> {
        let Some(gguf) = self.gguf.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "Engine(gguf=...) required for decide; use schema() for offline schema",
            ));
        };
        let state_v: Value = pythonize_any(state)?;
        let options_v: Value = pythonize_any(options)?;
        let opts = options_v.as_array().ok_or_else(|| {
            PyRuntimeError::new_err("options must be a list of {id, description}")
        })?;
        let mut option_specs = Vec::new();
        for o in opts {
            let oid = o
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| PyRuntimeError::new_err("option missing id"))?
                .to_string();
            let description = o
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            option_specs.push(ereshkigal_core::OptionSpec {
                id: oid,
                description,
            });
        }
        let row = ereshkigal_core::DecisionRow {
            id: id.unwrap_or("py-decide").to_string(),
            state: state_v,
            question: question.to_string(),
            options: option_specs,
        };
        let (tok_src, tok_rev) = tokenizer_for_gguf(gguf);
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let n_gpu_layers = std::env::var("N_GPU_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let cfg = ereshkigal_core::EngineConfig {
            gguf: PathBuf::from(gguf),
            tokenizer_source: tok_src,
            tokenizer_revision: tok_rev,
            max_prompt_tokens: 4096,
            threads,
            n_gpu_layers,
            n_seq_max: 8,
            embeddings: false,
            adapter: None,
            adapter_scale: 1.0,
        };
        let (engine, tokenizer) = ereshkigal_core::EngineOwned::load(cfg)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut scorer = ereshkigal_core::Scorer::new(engine, tokenizer);
        scorer.replay_enabled = false;
        let scored = scorer
            .score_direct(&row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string_pretty(&scored).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

fn tokenizer_for_gguf(gguf: &str) -> (String, String) {
    let name = gguf.to_ascii_lowercase();
    if name.contains("0.6b") || name.contains("0_6b") {
        (
            "Qwen/Qwen3-0.6B".into(),
            "c1899de289a04d12100db370d81485cdf75e47ca".into(),
        )
    } else {
        (
            "Qwen/Qwen3.5-4B".into(),
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
        )
    }
}

fn pythonize_any(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = pyo3::types::PyModule::import(obj.py(), "json")?;
    let s: String = json.getattr("dumps")?.call1((obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn schema() -> PyResult<String> {
    ereshkigal_core::library_schema_json().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn ereshkigal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    Ok(())
}
