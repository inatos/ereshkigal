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
    pub mmproj_path: Option<PathBuf>,
    pub cosine_head_path: Option<PathBuf>,
    pub cosine_head: Option<ereshkigal_core::CosineHead>,
    #[cfg(feature = "vision")]
    pub vision_ctx:
        Mutex<Option<std::result::Result<ereshkigal_core::vision::VisionContext, String>>>,
}

impl ServeState {
    pub fn new(lib_path: &Path, gguf: Option<PathBuf>) -> Result<Self> {
        let mmproj = std::env::var_os("ERESHKIGAL_MMPROJ")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty());
        let cosine_head = std::env::var_os("ERESHKIGAL_COSINE_HEAD")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty());
        Self::new_with_options(lib_path, gguf, mmproj, cosine_head)
    }

    pub fn new_with_options(
        lib_path: &Path,
        gguf: Option<PathBuf>,
        mmproj_path: Option<PathBuf>,
        cosine_head_path: Option<PathBuf>,
    ) -> Result<Self> {
        let library = Library::load(lib_path)?;
        let gguf_path = gguf
            .clone()
            .or_else(|| std::env::var_os("ERESHKIGAL_GGUF").map(PathBuf::from));
        let cosine_head = match &cosine_head_path {
            Some(p) => Some(
                ereshkigal_core::CosineHead::load_file(p)
                    .map_err(|e| anyhow::anyhow!("load cosine head {}: {e}", p.display()))?,
            ),
            None => None,
        };
        let runtime = match &gguf_path {
            Some(p) => Some(load_runtime(Some(p.clone()), None)?),
            None => None,
        };
        Ok(Self {
            library,
            lib_path: lib_path.to_path_buf(),
            runtime: Mutex::new(runtime),
            gguf_path,
            mmproj_path,
            cosine_head_path,
            cosine_head,
            #[cfg(feature = "vision")]
            vision_ctx: Mutex::new(None),
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
                "mmproj": self.mmproj_path.as_ref().map(|p| p.display().to_string()),
                "vision_compiled": cfg!(feature = "vision"),
                "cosine_head_loaded": self.cosine_head.is_some(),
                "cosine_head": self.cosine_head_path.as_ref().map(|p| p.display().to_string()),
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
            "score" | "systemone" => self.with_rt(|rt| match score_freeform(self, rt, params) {
                Ok(v) => v,
                Err(e) => json!({"error": e.to_string()}),
            }),
            "health" => self.health_json(),
            _ => json!({"error": format!("unknown method {method}")}),
        }
    }

    fn health_json(&self) -> Value {
        json!({
            "ok": true,
            "service": "ereshkigal",
            "gguf_loaded": self.runtime.lock().map(|g| g.is_some()).unwrap_or(false),
            "gguf": self.gguf_path.as_ref().map(|p| p.display().to_string()),
            "mmproj": self.mmproj_path.as_ref().map(|p| p.display().to_string()),
            "vision_compiled": cfg!(feature = "vision"),
            "cosine_head_loaded": self.cosine_head.is_some(),
            "cosine_head": self.cosine_head_path.as_ref().map(|p| p.display().to_string()),
            "decrees": self.library.decrees.len(),
            "lib": self.lib_path.display().to_string(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn decrees_lib() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../decrees")
    }

    #[test]
    fn dispatch_stats_without_gguf() {
        let lib = decrees_lib();
        if !lib.is_dir() {
            return;
        }
        let state = ServeState::new(&lib, None).expect("load lib");
        let v = state.dispatch("stats", &json!({}));
        assert!(v["decrees"].as_u64().unwrap_or(0) >= 1, "{v}");
        assert_eq!(v["gguf_loaded"], false, "{v}");
    }

    #[test]
    fn http_rpc_stats_localhost() {
        let lib = decrees_lib();
        if !lib.is_dir() {
            return;
        }
        // Bind an ephemeral port, then hand it to serve_http in a thread.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let lib2 = lib.clone();
        let handle = thread::spawn(move || {
            let _ = serve_http(port, &lib2, None);
        });
        thread::sleep(Duration::from_millis(150));
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"stats","params":{}}"#;
        let req = format!(
            "POST /rpc HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(req.as_bytes()).unwrap();
        let mut resp = String::new();
        let _ = stream.read_to_string(&mut resp);
        assert!(resp.contains("\"result\""), "{resp}");
        assert!(resp.contains("decrees") || resp.contains("gguf"), "{resp}");

        // Verify /v1/systemone routes cleanly to systemone handler (not unknown method).
        let mut stream2 = TcpStream::connect(("127.0.0.1", port)).expect("connect systemone");
        stream2
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let s1_body = r#"{"state":"ok","packing":"auto","think":{"mode":"auto","budget":128},"questions":[{"question":"Q1?","options":["yes","no"]},{"question":"Q2?","options":["yes","no"]}]}"#;
        let s1_req = format!(
            "POST /v1/systemone HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            s1_body.len(),
            s1_body
        );
        stream2.write_all(s1_req.as_bytes()).unwrap();
        let mut s1_resp = String::new();
        let _ = stream2.read_to_string(&mut s1_resp);
        assert!(s1_resp.contains("gguf not loaded"), "{s1_resp}");
        assert!(!s1_resp.contains("unknown method"), "{s1_resp}");

        // serve_http loops forever; detach by dropping the join handle.
        let _ = Arc::new(handle);
    }

    #[test]
    fn freeform_packing_resolution_and_think_parsing() {
        // Auto with 2+ shared-state questions selects joint packing.
        let batch_params = json!({
            "id": "b1",
            "state": "Shared evidence text.",
            "packing": "auto",
            "think": { "mode": "auto", "gate": 0.9, "budget": 256, "alpha": 0.75, "post_think_t": 1.25 },
            "questions": [
                { "question": "Q1?", "options": ["yes", "no"] },
                { "question": "Q2?", "type": "noul", "options": [
                    {"id": "true", "description": "True"},
                    {"id": "false", "description": "False"}
                ]}
            ]
        });
        let req = parse_freeform_request(&batch_params).expect("parse batch");
        assert!(req.is_batch);
        assert_eq!(req.items.len(), 2);
        assert_eq!(req.packing_choice, PackingChoice::Joint);
        assert_eq!(req.packing_choice.as_str(), "joint");
        let th = req.items[0].think.as_ref().expect("think config");
        assert_eq!(th.mode, ereshkigal_core::ThinkMode::Auto);
        assert_eq!(th.budget, 256);
        assert_eq!(req.items[0].post_think_t, Some(1.25));

        // Explicit shared_kv on batch alias.
        let shared_params = json!({
            "state": "Shared evidence text.",
            "packing": "shared_kv",
            "batch": [
                { "question": "Q1?", "options": ["a", "b"] },
                { "question": "Q2?", "options": ["c", "d"] }
            ]
        });
        let req_shared = parse_freeform_request(&shared_params).expect("parse shared");
        assert_eq!(req_shared.packing_choice, PackingChoice::SharedKv);
        assert_eq!(req_shared.packing_choice.as_str(), "shared_kv");

        // Auto with 1 question selects direct; explicit joint selects joint.
        let single_auto = json!({
            "state": "Single evidence.",
            "question": "Q?",
            "options": ["a", "b"]
        });
        let req_single = parse_freeform_request(&single_auto).expect("parse single");
        assert!(!req_single.is_batch);
        assert_eq!(req_single.packing_choice, PackingChoice::Direct);

        let single_joint = json!({
            "state": "Single evidence.",
            "question": "Q?",
            "packing": "joint",
            "options": ["a", "b"]
        });
        assert_eq!(
            parse_freeform_request(&single_joint).unwrap().packing_choice,
            PackingChoice::Joint
        );

        // Distinct per-item states under auto fall back to direct.
        let diff_states = json!({
            "packing": "auto",
            "questions": [
                { "state": "S1", "question": "Q1?", "options": ["a", "b"] },
                { "state": "S2", "question": "Q2?", "options": ["a", "b"] }
            ]
        });
        assert_eq!(
            parse_freeform_request(&diff_states).unwrap().packing_choice,
            PackingChoice::Direct
        );

        assert_eq!(
            packing_used_from_serving_config("llamacpp-joint-options-v1", "joint"),
            "joint"
        );
        assert_eq!(
            packing_used_from_serving_config("llamacpp-state-restore-shared-v1", "joint"),
            "shared_kv"
        );
    }

    #[test]
    fn serve_state_loads_conduit_autothink_and_lints() {
        let lib = decrees_lib();
        if !lib.is_dir() {
            return;
        }
        let state = ServeState::new(&lib, None).expect("load lib");
        assert_eq!(state.dispatch("lint", &json!({}))["ok"], true);
        for name in ["promote_hold", "continuity_check", "conduit_route"] {
            let d = state.library.decree(name).expect("decree present");
            let th = d.think.as_ref().expect("think configured");
            assert_eq!(th.mode, "auto");
            assert_eq!(th.budget, 256);
        }
    }

    #[test]
    fn serve_state_loads_cosine_head_and_parses_readout_and_visual_fences() {
        let lib = decrees_lib();
        if !lib.is_dir() {
            return;
        }
        let head = ereshkigal_core::CosineHead::random_init(16, 42);
        let ehead_path = std::env::temp_dir().join(format!(
            "ereshkigal-serve-test-{}.ehead",
            std::process::id()
        ));
        head.save_file(&ehead_path).expect("save ehead");
        let state = ServeState::new_with_options(
            &lib,
            None,
            Some(PathBuf::from("/tmp/missing-mmproj.gguf")),
            Some(ehead_path.clone()),
        )
        .expect("load serve state with cosine head");
        let stats = state.dispatch("stats", &json!({}));
        assert_eq!(stats["cosine_head_loaded"], true, "{stats}");
        assert_eq!(stats["mmproj"], "/tmp/missing-mmproj.gguf", "{stats}");
        let health = state.dispatch("health", &json!({}));
        assert_eq!(health["cosine_head_loaded"], true, "{health}");

        let cosine_req = parse_freeform_request(&json!({
            "state": "Evidence.",
            "question": "Which?",
            "readout": "cosine",
            "options": ["a", "b"]
        }))
        .expect("parse cosine request");
        assert_eq!(cosine_req.readout, "cosine");

        // Unversioned or clipped visual_evidence must be rejected (protocol_version:1 fence).
        assert!(parse_freeform_request(&json!({
            "state": "Evidence.",
            "question": "Which?",
            "options": ["a", "b"],
            "visual_evidence": { "png_paths": ["/tmp/a.png"] }
        }))
        .is_err());
        assert!(parse_freeform_request(&json!({
            "state": "Evidence.",
            "question": "Which?",
            "options": ["a", "b"],
            "visual_evidence": { "protocol_version": 1, "clipped": true, "png_paths": ["/tmp/a.png"] }
        }))
        .is_err());
        let ok_vis = parse_freeform_request(&json!({
            "state": "Evidence.",
            "question": "Which?",
            "options": ["a", "b"],
            "visual_evidence": { "protocol_version": 1, "clipped": false, "png_paths": ["/tmp/a.png"] }
        }))
        .expect("valid v1 visual evidence");
        assert_eq!(ok_vis.visual_pngs.len(), 1);
        let _ = std::fs::remove_file(&ehead_path);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackingChoice {
    Direct,
    Joint,
    SharedKv,
}

impl PackingChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Joint => "joint",
            Self::SharedKv => "shared_kv",
        }
    }
}

fn resolve_packing(packing: &str, n_questions: usize, shared_state: bool) -> PackingChoice {
    match packing {
        "joint" | "joint-options-v1" => PackingChoice::Joint,
        "shared_kv" | "shared-kv" | "shared" => PackingChoice::SharedKv,
        "direct" | "none" => PackingChoice::Direct,
        _ => {
            if n_questions >= 2 && shared_state {
                PackingChoice::Joint
            } else {
                PackingChoice::Direct
            }
        }
    }
}

fn packing_used_from_serving_config<'a>(serving_config: &str, default_label: &'a str) -> &'a str {
    if serving_config.contains("joint") {
        PackingChoice::Joint.as_str()
    } else if serving_config.contains("shared") {
        PackingChoice::SharedKv.as_str()
    } else {
        default_label
    }
}

fn parse_options_array(
    opts: &[Value],
    include_none: bool,
) -> Result<Vec<ereshkigal_core::OptionSpec>, String> {
    use ereshkigal_core::OptionSpec;
    let mut options = Vec::with_capacity(opts.len());
    for (i, o) in opts.iter().enumerate() {
        if let Some(s) = o.as_str() {
            options.push(OptionSpec {
                id: s.to_string(),
                description: s.to_string(),
            });
            continue;
        }
        let oid = o
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("options[{i}].id required"))?
            .to_string();
        let desc = o
            .get("description")
            .or_else(|| o.get("label"))
            .or_else(|| o.get("desc"))
            .and_then(|v| v.as_str())
            .unwrap_or(oid.as_str())
            .to_string();
        options.push(OptionSpec {
            id: oid,
            description: desc,
        });
    }
    if include_none && !options.iter().any(|o| o.id == "none") {
        options.push(OptionSpec {
            id: "none".into(),
            description: "None of the above".into(),
        });
    }
    if options.is_empty() {
        return Err("options must be non-empty".into());
    }
    Ok(options)
}

struct FreeformItem {
    row: ereshkigal_core::DecisionRow,
    qtype: String,
    think: Option<ereshkigal_core::ThinkConfig>,
    conformal_set_size: usize,
    post_think_t: Option<f64>,
}

struct FreeformRequest {
    is_batch: bool,
    packing: String,
    packing_choice: PackingChoice,
    readout: String,
    visual_pngs: Vec<PathBuf>,
    items: Vec<FreeformItem>,
    temps: ereshkigal_core::TypeTemperatures,
}

fn parse_visual_pngs(params: &Value) -> Result<Vec<PathBuf>, String> {
    if let Some(ev) = params.get("visual_evidence") {
        let obj = ev
            .as_object()
            .ok_or_else(|| "visual_evidence must be an object".to_string())?;
        let pv = obj
            .get("protocol_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if pv != 1 {
            return Err("visual_evidence.protocol_version=1 required; abstain to text".into());
        }
        if obj.get("clipped").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err("visual_evidence clipped; abstain to text".into());
        }
        let paths: Vec<PathBuf> = obj
            .get("png_paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(PathBuf::from))
                    .collect()
            })
            .unwrap_or_default();
        return Ok(paths);
    }
    if let Some(arr) = params.get("png_paths").and_then(|v| v.as_array()) {
        let paths: Vec<PathBuf> = arr
            .iter()
            .filter_map(|x| x.as_str().map(PathBuf::from))
            .collect();
        return Ok(paths);
    }
    Ok(Vec::new())
}

fn parse_post_think_t(v: &Value) -> Option<f64> {
    v.get("post_think_t")
        .and_then(|x| x.as_f64())
        .or_else(|| {
            v.get("think")
                .and_then(|t| t.get("post_think_t"))
                .and_then(|x| x.as_f64())
        })
        .filter(|t| t.is_finite() && *t > 0.0)
}

fn parse_conformal_set_size(v: &Value) -> Option<usize> {
    v.get("conformal_set_size")
        .and_then(|x| x.as_u64())
        .or_else(|| {
            v.get("think")
                .and_then(|t| t.get("conformal_set_size"))
                .and_then(|x| x.as_u64())
        })
        .map(|n| n as usize)
}

fn parse_freeform_request(params: &Value) -> Result<FreeformRequest, String> {
    use ereshkigal_core::{DecisionRow, ThinkConfig, TypeTemperatures};

    let base_id = params
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("score")
        .to_string();
    let base_qtype = params
        .get("type")
        .or_else(|| params.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("choice")
        .to_string();
    let base_state = params.get("state").cloned().unwrap_or(json!(""));
    let base_include_none = params
        .get("include_none")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let packing = params
        .get("packing")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let readout = params
        .get("readout")
        .and_then(|v| v.as_str())
        .unwrap_or("letter")
        .to_ascii_lowercase();
    if !matches!(readout.as_str(), "letter" | "cosine" | "vision") {
        return Err(format!("unsupported readout {readout:?} (expected letter|cosine|vision)"));
    }
    let visual_pngs = parse_visual_pngs(params)?;
    let temps = params
        .get("calibration")
        .map(TypeTemperatures::from_json)
        .unwrap_or_default();
    let base_think = params.get("think").and_then(ThinkConfig::from_json);
    let base_csize = parse_conformal_set_size(params).unwrap_or(1);
    let base_post_t = parse_post_think_t(params);

    if let Some(arr) = params
        .get("questions")
        .or_else(|| params.get("batch"))
        .and_then(|v| v.as_array())
    {
        if arr.is_empty() {
            return Err("questions/batch array must be non-empty".into());
        }
        let shared_opts = params.get("options").and_then(|v| v.as_array());
        let mut items = Vec::with_capacity(arr.len());
        for (idx, item) in arr.iter().enumerate() {
            let item_id = item
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{base_id}-{idx}"));
            let item_qtype = item
                .get("type")
                .or_else(|| item.get("kind"))
                .and_then(|v| v.as_str())
                .unwrap_or(base_qtype.as_str())
                .to_string();
            let item_state = item
                .get("state")
                .or_else(|| item.get("evidence"))
                .cloned()
                .unwrap_or_else(|| base_state.clone());
            let item_question = item
                .get("question")
                .or_else(|| item.get("criterion"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let item_include_none = item
                .get("include_none")
                .and_then(|v| v.as_bool())
                .unwrap_or(base_include_none);
            let raw_opts = item
                .get("options")
                .and_then(|v| v.as_array())
                .or(shared_opts)
                .ok_or_else(|| format!("questions[{idx}].options array required"))?;
            let options = parse_options_array(raw_opts, item_include_none)?;
            let item_think = item
                .get("think")
                .and_then(ThinkConfig::from_json)
                .or_else(|| base_think.clone());
            let item_csize = parse_conformal_set_size(item).unwrap_or(base_csize);
            let item_post_t = parse_post_think_t(item).or(base_post_t);
            items.push(FreeformItem {
                row: DecisionRow {
                    id: item_id,
                    state: item_state,
                    question: item_question,
                    options,
                },
                qtype: item_qtype,
                think: item_think,
                conformal_set_size: item_csize,
                post_think_t: item_post_t,
            });
        }
        let shared_state = items
            .first()
            .map(|first| items.iter().all(|it| it.row.state == first.row.state))
            .unwrap_or(true);
        let packing_choice = resolve_packing(&packing, items.len(), shared_state);
        return Ok(FreeformRequest {
            is_batch: true,
            packing,
            packing_choice,
            readout,
            visual_pngs,
            items,
            temps,
        });
    }

    let question = params
        .get("question")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let raw_opts = params
        .get("options")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "options array required".to_string())?;
    let options = parse_options_array(raw_opts, base_include_none)?;
    let packing_choice = resolve_packing(&packing, 1, true);
    Ok(FreeformRequest {
        is_batch: false,
        packing,
        packing_choice,
        readout,
        visual_pngs,
        items: vec![FreeformItem {
            row: DecisionRow {
                id: base_id,
                state: base_state,
                question,
                options,
            },
            qtype: base_qtype,
            think: base_think,
            conformal_set_size: base_csize,
            post_think_t: base_post_t,
        }],
        temps,
    })
}

fn format_scored_payload(
    item: &FreeformItem,
    result: &ereshkigal_core::ScoreResult,
    think_outcome: Option<&ereshkigal_core::AutoThinkOutcome>,
    tournament: Option<usize>,
    calib_t: f64,
    eff_post_t: Option<f64>,
) -> Result<Value, String> {
    let think_invoked = think_outcome.map(|o| o.think_invoked).unwrap_or(false);
    let think_tokens = think_outcome.map(|o| o.think_tokens).unwrap_or(0);
    let p0_vs_p1_flipped = think_outcome.map(|o| o.p0_vs_p1_flipped).unwrap_or(false);

    let (probs, used_t) = if let Some(out) = think_outcome.filter(|o| o.think_invoked) {
        (out.probs.clone(), eff_post_t.unwrap_or(1.0))
    } else if (calib_t - 1.0).abs() > 1e-12 {
        (
            ereshkigal_core::apply_temperature(&result.option_logits, calib_t)
                .map_err(|e| e.to_string())?,
            calib_t,
        )
    } else {
        (result.probabilities.clone(), calib_t)
    };

    let chosen_idx = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let none_probability = item
        .row
        .options
        .iter()
        .position(|o| o.id == "none")
        .and_then(|i| probs.get(i).copied());
    let noul_p_true = if matches!(item.qtype.as_str(), "noul" | "bool") {
        probs.first().copied()
    } else {
        None
    };
    let scores: Vec<Value> = result
        .option_ids
        .iter()
        .enumerate()
        .map(|(i, oid)| {
            json!([
                oid,
                result.option_logits.get(i).copied().unwrap_or(0.0)
            ])
        })
        .collect();
    Ok(json!({
        "id": result.id,
        "type": item.qtype,
        "scores": scores,
        "probabilities": probs,
        "prompt_sha256": result.prompt_sha256,
        "chosen": result.option_ids.get(chosen_idx).cloned()
            .or_else(|| tournament.map(|w| item.row.options[w].id.clone())),
        "none_probability": none_probability,
        "noul_p_true": noul_p_true,
        "calibrated": (used_t - 1.0).abs() > 1e-12,
        "temperature": used_t,
        "tournament": tournament.is_some(),
        "think_invoked": think_invoked,
        "think_tokens": think_tokens,
        "p0_vs_p1_flipped": p0_vs_p1_flipped,
    }))
}

#[cfg(feature = "vision")]
fn score_visual_row(
    state: &ServeState,
    rt: &mut Runtime,
    png_paths: &[PathBuf],
    row: &ereshkigal_core::DecisionRow,
) -> Result<ereshkigal_core::ScoreResult, String> {
    let Some(mmproj) = state.mmproj_path.as_ref() else {
        return Err("mmproj not configured (--mmproj or ERESHKIGAL_MMPROJ); abstain to text".into());
    };
    let mut guard = state.vision_ctx.lock().map_err(|e| e.to_string())?;
    if guard.is_none() {
        let mut cfg = ereshkigal_core::vision::VisionConfig::from_mmproj(mmproj.clone());
        cfg.use_gpu = rt.scorer.meta().n_gpu_layers > 0;
        cfg.n_threads = rt.scorer.meta().threads.max(1);
        *guard = Some(rt.scorer.load_vision(&cfg).map_err(|e| e.to_string()));
    }
    let vctx = match guard.as_ref().expect("vision_ctx populated") {
        Ok(v) => v,
        Err(e) => return Err(e.clone()),
    };
    let refs: Vec<&Path> = png_paths.iter().map(PathBuf::as_path).collect();
    rt.scorer
        .score_visual(vctx, &refs, row)
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "vision"))]
fn score_visual_row(
    _state: &ServeState,
    _rt: &mut Runtime,
    _png_paths: &[PathBuf],
    _row: &ereshkigal_core::DecisionRow,
) -> Result<ereshkigal_core::ScoreResult, String> {
    Err("vision feature not compiled into ereshkigal binary; abstain to text".into())
}

fn score_freeform(state: &ServeState, rt: &mut Runtime, params: &Value) -> Result<Value, String> {
    use ereshkigal_core::{run_tournament, should_think, DecisionRow, ThinkMode};

    let req = parse_freeform_request(params)?;

    if req.readout == "cosine" {
        let head = state.cosine_head.as_ref().ok_or_else(|| {
            "readout=cosine requires --cosine-head or ERESHKIGAL_COSINE_HEAD".to_string()
        })?;
        if req.is_batch {
            let mut results_json = Vec::with_capacity(req.items.len());
            for item in &req.items {
                let calib_t = req.temps.for_kind(&item.qtype);
                let res = rt
                    .scorer
                    .score_cosine(&item.row, head)
                    .map_err(|e| e.to_string())?;
                results_json.push(format_scored_payload(
                    item,
                    &res,
                    None,
                    None,
                    calib_t,
                    item.post_think_t,
                )?);
            }
            return Ok(json!({
                "ok": true,
                "readout": "cosine",
                "packing": req.packing,
                "packing_used": "cosine",
                "count": results_json.len(),
                "results": results_json,
            }));
        }
        let item = &req.items[0];
        let calib_t = req.temps.for_kind(&item.qtype);
        let res = rt
            .scorer
            .score_cosine(&item.row, head)
            .map_err(|e| e.to_string())?;
        let mut payload =
            format_scored_payload(item, &res, None, None, calib_t, item.post_think_t)?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("ok".into(), json!(true));
            obj.insert("readout".into(), json!("cosine"));
            obj.insert("packing".into(), json!(req.packing));
            obj.insert("packing_used".into(), json!("cosine"));
        }
        return Ok(payload);
    }

    if req.readout == "vision" || !req.visual_pngs.is_empty() {
        if req.visual_pngs.is_empty() {
            return Err("readout=vision requires png_paths or visual_evidence.png_paths".into());
        }
        if req.is_batch {
            let mut results_json = Vec::with_capacity(req.items.len());
            for item in &req.items {
                let calib_t = req.temps.for_kind(&item.qtype);
                let res = score_visual_row(state, rt, &req.visual_pngs, &item.row)?;
                results_json.push(format_scored_payload(
                    item,
                    &res,
                    None,
                    None,
                    calib_t,
                    item.post_think_t,
                )?);
            }
            return Ok(json!({
                "ok": true,
                "readout": "vision",
                "packing": req.packing,
                "packing_used": "vision-mtmd",
                "count": results_json.len(),
                "results": results_json,
            }));
        }
        let item = &req.items[0];
        let calib_t = req.temps.for_kind(&item.qtype);
        let res = score_visual_row(state, rt, &req.visual_pngs, &item.row)?;
        let mut payload =
            format_scored_payload(item, &res, None, None, calib_t, item.post_think_t)?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("ok".into(), json!(true));
            obj.insert("readout".into(), json!("vision"));
            obj.insert("packing".into(), json!(req.packing));
            obj.insert("packing_used".into(), json!("vision-mtmd"));
        }
        return Ok(payload);
    }

    if req.is_batch {
        let rows: Vec<DecisionRow> = req.items.iter().map(|it| it.row.clone()).collect();
        let (mut scored, timing, packing_used) = match req.packing_choice {
            PackingChoice::Joint => {
                let (res, tm) = rt
                    .scorer
                    .score_joint_prompt(&rows)
                    .map_err(|e| e.to_string())?;
                let used = res
                    .first()
                    .map(|r| {
                        packing_used_from_serving_config(
                            &r.model.serving_config,
                            PackingChoice::Joint.as_str(),
                        )
                    })
                    .unwrap_or(PackingChoice::Joint.as_str());
                (res, Some(tm), used)
            }
            PackingChoice::SharedKv => {
                let (res, tm) = rt.scorer.score_shared(&rows).map_err(|e| e.to_string())?;
                (res, Some(tm), PackingChoice::SharedKv.as_str())
            }
            PackingChoice::Direct => {
                let mut res = Vec::with_capacity(rows.len());
                for row in &rows {
                    res.push(rt.scorer.score_direct(row).map_err(|e| e.to_string())?);
                }
                (res, None, PackingChoice::Direct.as_str())
            }
        };

        let mut results_json = Vec::with_capacity(req.items.len());
        let mut any_think = false;
        let mut total_think_tokens = 0usize;
        let mut any_flipped = false;

        for (item, base_res) in req.items.iter().zip(scored.drain(..)) {
            let calib_t = req.temps.for_kind(&item.qtype);
            let eff_post_t = item
                .post_think_t
                .or_else(|| ((calib_t - 1.0).abs() > 1e-12).then_some(calib_t));
            let (final_res, think_outcome) = if let Some(cfg) = item
                .think
                .as_ref()
                .filter(|c| c.mode != ThinkMode::Never)
            {
                let max_p0 = base_res
                    .probabilities
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
                if should_think(cfg, item.conformal_set_size, max_p0) {
                    let (r, out) = rt
                        .scorer
                        .score_autothink(&item.row, cfg, item.conformal_set_size, eff_post_t)
                        .map_err(|e| e.to_string())?;
                    (r, Some(out))
                } else {
                    (base_res, None)
                }
            } else {
                (base_res, None)
            };
            if let Some(ref out) = think_outcome {
                if out.think_invoked {
                    any_think = true;
                    total_think_tokens += out.think_tokens;
                    if out.p0_vs_p1_flipped {
                        any_flipped = true;
                    }
                }
            }
            results_json.push(format_scored_payload(
                item,
                &final_res,
                think_outcome.as_ref(),
                None,
                calib_t,
                eff_post_t,
            )?);
        }

        return Ok(json!({
            "ok": true,
            "packing": req.packing,
            "packing_used": packing_used,
            "count": results_json.len(),
            "timing": timing,
            "think_invoked": any_think,
            "think_tokens": total_think_tokens,
            "p0_vs_p1_flipped": any_flipped,
            "results": results_json,
        }));
    }

    let item = &req.items[0];
    let calib_t = req.temps.for_kind(&item.qtype);
    let eff_post_t = item
        .post_think_t
        .or_else(|| ((calib_t - 1.0).abs() > 1e-12).then_some(calib_t));

    let (result, think_outcome, tournament, packing_used) = if item.row.options.len() > 16 {
        let winner = run_tournament(&item.row, |bucket| {
            let scored = rt.scorer.score_direct(bucket)?;
            Ok(scored
                .probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0))
        })
        .map_err(|e| e.to_string())?;
        let final_opts = vec![item.row.options[winner].clone()];
        let final_row = DecisionRow {
            id: format!("{}-final", item.row.id),
            state: item.row.state.clone(),
            question: item.row.question.clone(),
            options: final_opts,
        };
        let r = rt
            .scorer
            .score_direct(&final_row)
            .map_err(|e| e.to_string())?;
        (r, None, Some(winner), "direct")
    } else {
        match req.packing_choice {
            PackingChoice::Joint => {
                let (mut res_vec, _) = rt
                    .scorer
                    .score_joint_prompt(std::slice::from_ref(&item.row))
                    .map_err(|e| e.to_string())?;
                let base_res = res_vec.remove(0);
                let used =
                    packing_used_from_serving_config(&base_res.model.serving_config, "joint");
                if let Some(cfg) = item
                    .think
                    .as_ref()
                    .filter(|c| c.mode != ThinkMode::Never)
                {
                    let max_p0 = base_res
                        .probabilities
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max);
                    if should_think(cfg, item.conformal_set_size, max_p0) {
                        let (r, out) = rt
                            .scorer
                            .score_autothink(&item.row, cfg, item.conformal_set_size, eff_post_t)
                            .map_err(|e| e.to_string())?;
                        (r, Some(out), None, used)
                    } else {
                        (base_res, None, None, used)
                    }
                } else {
                    (base_res, None, None, used)
                }
            }
            PackingChoice::SharedKv => {
                let (mut res_vec, _) = rt
                    .scorer
                    .score_shared(std::slice::from_ref(&item.row))
                    .map_err(|e| e.to_string())?;
                let base_res = res_vec.remove(0);
                if let Some(cfg) = item
                    .think
                    .as_ref()
                    .filter(|c| c.mode != ThinkMode::Never)
                {
                    let max_p0 = base_res
                        .probabilities
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max);
                    if should_think(cfg, item.conformal_set_size, max_p0) {
                        let (r, out) = rt
                            .scorer
                            .score_autothink(&item.row, cfg, item.conformal_set_size, eff_post_t)
                            .map_err(|e| e.to_string())?;
                        (r, Some(out), None, "shared_kv")
                    } else {
                        (base_res, None, None, "shared_kv")
                    }
                } else {
                    (base_res, None, None, "shared_kv")
                }
            }
            PackingChoice::Direct => {
                if let Some(cfg) = item
                    .think
                    .as_ref()
                    .filter(|c| c.mode != ThinkMode::Never)
                {
                    let (r, out) = rt
                        .scorer
                        .score_autothink(&item.row, cfg, item.conformal_set_size, eff_post_t)
                        .map_err(|e| e.to_string())?;
                    (r, Some(out), None, "direct")
                } else {
                    let r = rt
                        .scorer
                        .score_direct(&item.row)
                        .map_err(|e| e.to_string())?;
                    (r, None, None, "direct")
                }
            }
        }
    };

    let mut payload = format_scored_payload(
        item,
        &result,
        think_outcome.as_ref(),
        tournament,
        calib_t,
        eff_post_t,
    )?;
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("ok".into(), json!(true));
        obj.insert("packing".into(), json!(req.packing));
        obj.insert("packing_used".into(), json!(packing_used));
    }
    Ok(payload)
}

pub fn serve_http(port: u16, lib_path: &Path, gguf: Option<PathBuf>) -> Result<()> {
    serve_http_bind(("127.0.0.1", port), lib_path, gguf)
}

/// Bind host:port form of [`serve_http`] (Phase 2 uses `10.55.0.2`).
pub fn serve_http_bind(
    addr: (impl Into<String>, u16),
    lib_path: &Path,
    gguf: Option<PathBuf>,
) -> Result<()> {
    let (host, port) = (addr.0.into(), addr.1);
    let state = ServeState::new(lib_path, gguf)?;
    let server = tiny_http::Server::http((host.as_str(), port))
        .map_err(|e| anyhow::anyhow!("http bind: {e}"))?;
    eprintln!(
        "[ereshkigal] serve http://{host}:{port} lib={} gguf={}",
        state.lib_path.display(),
        state
            .gguf_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        if url == "/health" || url.starts_with("/health?") {
            let bytes = serde_json::to_vec(&state.health_json())?;
            let header =
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                    .unwrap_or_else(|_| unreachable!());
            let _ = req.respond(tiny_http::Response::from_data(bytes).with_header(header));
            continue;
        }
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let (method, params, id) = if url == "/rpc" || url.starts_with("/rpc?") || url.starts_with("/v1/") {
            let v: Value = serde_json::from_str(&body).unwrap_or(json!({}));
            let method = if url.starts_with("/v1/systemone") {
                "systemone".to_string()
            } else {
                v.get("method")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            (
                method,
                v.get("params").cloned().unwrap_or_else(|| {
                    if url.starts_with("/v1/") {
                        v.clone()
                    } else {
                        json!({})
                    }
                }),
                v.get("id").cloned().unwrap_or(json!(null)),
            )
        } else {
            let m = url.trim_start_matches('/').split('?').next().unwrap_or("");
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
