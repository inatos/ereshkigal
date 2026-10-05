//! Shared JSON-RPC dispatch for `serve --stdio` and `serve --http`.

use anyhow::Result;
use ereshkigal_core::lint;
use ereshkigal_core::{Library, Runtime};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::{load_runtime, score_library_tests};

pub struct ServeState {
    pub library: Library,
    pub lib_path: PathBuf,
    pub runtime: Mutex<Option<Runtime>>,
    pub gguf_path: Option<PathBuf>,
}

impl ServeState {
    pub fn new(lib_path: &Path, gguf: Option<PathBuf>) -> Result<Self> {
        let library = Library::load(lib_path)?;
        let gguf_path = gguf
            .clone()
            .or_else(|| std::env::var_os("ERESHKIGAL_GGUF").map(PathBuf::from));
        let runtime = match &gguf_path {
            Some(p) => Some(load_runtime(Some(p.clone()))?),
            None => None,
        };
        Ok(Self {
            library,
            lib_path: lib_path.to_path_buf(),
            runtime: Mutex::new(runtime),
            gguf_path,
        })
    }

    pub fn dispatch(&self, method: &str, params: &Value) -> Value {
        match method {
            "lint" => match lint::lint_library(&self.library) {
                Ok(()) => json!({"ok": true}),
                Err(e) => json!({"ok": false, "error": e.to_string()}),
            },
            "stats" => json!({
                "backend": "llamacpp",
                "lib": self.lib_path.display().to_string(),
                "decrees": self.library.decrees.len(),
                "programs": self.library.programs.len(),
                "gguf_loaded": self.runtime.lock().map(|g| g.is_some()).unwrap_or(false),
                "gguf": self.gguf_path.as_ref().map(|p| p.display().to_string()),
                "recipe": self.library.recipe,
            }),
            "decide" => self.with_rt(|rt| {
                let decree = params.get("decree").and_then(|v| v.as_str()).unwrap_or("");
                let state = params.get("state").cloned().unwrap_or(json!(""));
                match rt.decide(&self.library, decree, &state) {
                    Ok(d) => serde_json::to_value(d).unwrap_or(json!({"error": "serialize"})),
                    Err(e) => json!({"error": e.to_string()}),
                }
            }),
            "run" => self.with_rt(|rt| {
                let program = params.get("program").and_then(|v| v.as_str()).unwrap_or("");
                let state = params.get("state").cloned().unwrap_or(json!(""));
                match rt.run(&self.library, program, &state) {
                    Ok(d) => serde_json::to_value(d).unwrap_or(json!({"error": "serialize"})),
                    Err(e) => json!({"error": e.to_string()}),
                }
            }),
            "test" => self.with_rt(|rt| {
                let split = params.get("split").and_then(|v| v.as_str()).unwrap_or("all");
                match score_library_tests(rt, &self.library, split) {
                    Ok(v) => v,
                    Err(e) => json!({"error": e.to_string()}),
                }
            }),
            _ => json!({"error": format!("unknown method {method}")}),
        }
    }

    fn with_rt(&self, f: impl FnOnce(&mut Runtime) -> Value) -> Value {
        match self.runtime.lock() {
            Ok(mut g) => match g.as_mut() {
                None => json!({"error": "gguf not loaded; pass --gguf or ERESHKIGAL_GGUF"}),
                Some(rt) => f(rt),
            },
            Err(e) => json!({"error": format!("runtime lock: {e}")}),
        }
    }
}

pub fn serve_stdio(lib_path: &Path, gguf: Option<PathBuf>) -> Result<()> {
    use std::io::{BufRead, Write};
    let state = ServeState::new(lib_path, gguf)?;
    eprintln!(
        "[ereshkigal] serve stdio lib={} decrees={} programs={} gguf={}",
        state.lib_path.display(),
        state.library.decrees.len(),
        state.library.programs.len(),
        state
            .gguf_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none — decide/run/test need --gguf)".into())
    );
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = serde_json::from_str(&line).unwrap_or(json!({}));
        let id = req.get("id").cloned().unwrap_or(json!(null));
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));
        let result = state.dispatch(method, &params);
        let resp = json!({"jsonrpc": "2.0", "id": id, "result": result});
        writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        stdout.flush()?;
    }
    Ok(())
}

pub fn serve_http(port: u16, lib_path: &Path, gguf: Option<PathBuf>) -> Result<()> {
    let state = ServeState::new(lib_path, gguf)?;
    let server = tiny_http::Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow::anyhow!("http bind: {e}"))?;
    eprintln!(
        "[ereshkigal] serve http://127.0.0.1:{port} lib={} gguf={}",
        state.lib_path.display(),
        state
            .gguf_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    for mut req in server.incoming_requests() {
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let (method, params, id) = if req.url() == "/rpc" || req.url().starts_with("/rpc?") {
            let v: Value = serde_json::from_str(&body).unwrap_or(json!({}));
            (
                v.get("method")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string(),
                v.get("params").cloned().unwrap_or(json!({})),
                v.get("id").cloned().unwrap_or(json!(null)),
            )
        } else {
            let m = req.url().trim_start_matches('/').split('?').next().unwrap_or("");
            (m.to_string(), serde_json::from_str(&body).unwrap_or(json!({})), json!(null))
        };
        let result = state.dispatch(&method, &params);
        let resp = json!({"jsonrpc": "2.0", "id": id, "result": result});
        let bytes = serde_json::to_vec(&resp)?;
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
            .unwrap_or_else(|_| unreachable!());
        let response = tiny_http::Response::from_data(bytes).with_header(header);
        let _ = req.respond(response);
    }
    Ok(())
}
