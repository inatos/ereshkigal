//! Ereshkigal language CLI (`check`, `test`, `run`, `lsp`, `serve`, …).

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use ereshkigal_core::lockfile::Lockfile;
use ereshkigal_core::metrics::{
    aurc, group_consistency, selective_accuracy,
};
use ereshkigal_core::syntax::{convert_to_toml, parse_library, print_library};
use ereshkigal_core::{
    balanced_accuracy, ece, fit_temperature, fit_temperature_oof_grouped, library_schema_json,
    Library, Runtime, Scorer, EngineConfig, EngineOwned,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

mod serve;

#[derive(Parser, Debug)]
#[command(name = "ereshkigal", about = "Ereshkigal decision language")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Check { path: PathBuf },
    Lint { path: PathBuf },
    Fmt { path: PathBuf },
    Convert {
        input: PathBuf,
        #[arg(long)]
        toml: bool,
    },
    Schema,
    Decide {
        #[arg(long)]
        lib: PathBuf,
        #[arg(long)]
        decree: String,
        #[arg(long)]
        state: String,
        #[arg(long)]
        gguf: Option<PathBuf>,
    },
    Run {
        #[arg(long)]
        lib: PathBuf,
        #[arg(long)]
        program: String,
        #[arg(long)]
        state: String,
        #[arg(long)]
        gguf: Option<PathBuf>,
    },
    Test {
        #[arg(long, default_value = "decrees")]
        lib: PathBuf,
        #[arg(long)]
        gguf: Option<PathBuf>,
        #[arg(long)]
        fit: bool,
        #[arg(long, default_value = "test")]
        split: String,
        #[arg(long)]
        locked: bool,
    },
    Lock {
        #[arg(long, default_value = "decrees")]
        lib: PathBuf,
        #[arg(long, default_value = "ereshkigal.lock")]
        out: PathBuf,
    },
    Serve {
        #[arg(long)]
        stdio: bool,
        #[arg(long)]
        http: Option<u16>,
        #[arg(long)]
        lib: PathBuf,
        #[arg(long)]
        gguf: Option<PathBuf>,
    },
    Lsp,
    Explain {
        #[arg(long)]
        lib: PathBuf,
        #[arg(long)]
        decree: String,
        #[arg(long)]
        state: String,
        #[arg(long)]
        gguf: Option<PathBuf>,
    },
    Optimize {
        #[arg(long)]
        lib: PathBuf,
        #[arg(long)]
        decree: String,
    },
    Distill {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "probe-decree-v1")]
        recipe: String,
    },
    New { name: String },
    Add {
        name: String,
        #[arg(long)]
        path: Option<PathBuf>,
    },
    Remove { name: String },
    Fetch,
    Verify,
    Publish {
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Check { path } => {
            let lib = Library::load(&path)?;
            println!("ok decrees={} programs={}", lib.decrees.len(), lib.programs.len());
        }
        Cmd::Lint { path } => {
            let lib = Library::load(&path)?;
            ereshkigal_core::lint::lint_library(&lib)?;
            println!("lint ok");
        }
        Cmd::Fmt { path } => {
            if path.extension().and_then(|s| s.to_str()) == Some("esk") {
                let src = fs::read_to_string(&path)?;
                let lib = parse_library(&src)?;
                fs::write(&path, print_library(&lib))?;
            } else {
                let lib = Library::load(&path)?;
                println!("{}", print_library(&lib));
            }
        }
        Cmd::Convert { input, toml } => {
            let src = fs::read_to_string(&input)?;
            let lib = if input.extension().and_then(|s| s.to_str()) == Some("esk") {
                parse_library(&src)?
            } else {
                Library::load(&input)?
            };
            if toml {
                print!("{}", convert_to_toml(&lib)?);
            } else {
                print!("{}", print_library(&lib));
            }
        }
        Cmd::Schema => {
            print!("{}", library_schema_json()?);
        }
        Cmd::Decide {
            lib,
            decree,
            state,
            gguf,
        } => {
            let library = Library::load(&lib)?;
            let mut rt = load_runtime(gguf)?;
            let v: Value = serde_json::from_str(&state).unwrap_or(json!(state));
            let d = rt.decide(&library, &decree, &v)?;
            serde_json::to_writer_pretty(std::io::stdout(), &d)?;
            println!();
        }
        Cmd::Run {
            lib,
            program,
            state,
            gguf,
        } => {
            let library = Library::load(&lib)?;
            let mut rt = load_runtime(gguf)?;
            let v: Value = serde_json::from_str(&state).unwrap_or(json!(state));
            let out = rt.run(&library, &program, &v)?;
            serde_json::to_writer_pretty(std::io::stdout(), &out)?;
            println!();
        }
        Cmd::Test {
            lib,
            gguf,
            fit,
            split,
            locked,
        } => run_test(&lib, gguf, fit, &split, locked)?,
        Cmd::Lock { lib, out } => {
            let library = Library::load(&lib)?;
            let tmpl = ereshkigal_core::template::render_chat(
                ereshkigal_core::template::QWEN3_THINKING_OFF,
                &[("system".into(), "sys".into()), ("user".into(), "u".into())],
            )?;
            let mut lf = Lockfile {
                recipe: library.recipe.clone(),
                template_hash: ereshkigal_core::template::template_hash(&tmpl),
                ..Default::default()
            };
            for (name, d) in &library.decrees {
                if let Some(t) = d.tests.first() {
                    let row = d.to_row(name, t.state.clone())?;
                    lf.prompt_hashes
                        .insert(name.clone(), ereshkigal_core::digest(&ereshkigal_core::render_prompt(&row)?));
                }
            }
            lf.save(&out)?;
            println!("wrote {}", out.display());
        }
        Cmd::Serve { stdio, http, lib, gguf } => {
            if let Some(port) = http {
                serve::serve_http(port, &lib, gguf)?;
            } else {
                let _ = stdio;
                serve::serve_stdio(&lib, gguf)?;
            }
        }
        Cmd::Lsp => run_lsp()?,
        Cmd::Explain {
            lib,
            decree,
            state,
            gguf,
        } => {
            let library = Library::load(&lib)?;
            let mut rt = load_runtime(gguf)?;
            let v: Value = serde_json::from_str(&state).unwrap_or(json!(state));
            let base = rt.decide(&library, &decree, &v)?;
            println!("base choice={:?} margin={:.3}", base.choice, base.margin);
            if let Some(s) = v.as_str() {
                for (i, span) in s.split(". ").enumerate() {
                    let ablated = s.replace(span, "");
                    let d = rt.decide(&library, &decree, &json!(ablated))?;
                    println!(
                        "drop[{i}] choice={:?} delta_margin={:.3}",
                        d.choice,
                        d.margin - base.margin
                    );
                }
            }
        }
        Cmd::Optimize { lib, decree } => {
            let library = Library::load(&lib)?;
            let d = library.decree(&decree)?;
            if d.variants.is_empty() {
                println!("no variants on {decree}; add [[decree.variants]] and re-run on split=dev");
            } else {
                let gold = d
                    .tests
                    .iter()
                    .filter(|t| t.split.as_deref().unwrap_or("dev") == "dev")
                    .collect::<Vec<_>>();
                if gold.is_empty() {
                    anyhow::bail!("no split=dev tests; refuse to fit wording on test");
                }
                let mut nll = Vec::new();
                for v in &d.variants {
                    let mut acc = 0.0;
                    for t in &gold {
                        let expect = t.expect.clone();
                        let state = t.state.to_string();
                        acc += ereshkigal_core::overlap_nll(
                            v.question.as_deref().unwrap_or(&d.question),
                            &state,
                            &expect,
                        );
                    }
                    nll.push((v.id.clone(), acc / gold.len() as f64));
                }
                let winner = ereshkigal_core::pick_lowest_nll(&nll)?;
                println!("dev-split winner={winner} nll={nll:?}");
            }
        }
        Cmd::Distill { corpus, out, recipe } => {
            let src = fs::read_to_string(&corpus)?;
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for line in src.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(line)?;
                if let (Some(x), Some(y)) = (v.get("x"), v.get("y")) {
                    let xv: Vec<f64> = serde_json::from_value(x.clone()).unwrap_or_default();
                    let yv: Vec<f64> = serde_json::from_value(y.clone()).unwrap_or_default();
                    if !xv.is_empty() && !yv.is_empty() {
                        xs.push(xv);
                        ys.push(yv);
                    }
                }
            }
            if xs.is_empty() {
                xs = vec![vec![1.0; 8], vec![0.0; 8]];
                ys = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
            }
            let dim = xs[0].len();
            let n_classes = ys[0].len();
            let mut probe = ereshkigal_core::probe::LogisticProbe::zeros(dim, n_classes, &recipe, 1e-3);
            for _ in 0..8 {
                ereshkigal_core::probe::train_epoch(&mut probe, &xs, &ys, 0.1)?;
            }
            fs::write(out, serde_json::to_string_pretty(&probe)?)?;
            println!("wrote probe recipe={recipe} n={}", xs.len());
        }
        Cmd::New { name } => {
            fs::create_dir_all(&name)?;
            let man = ereshkigal_core::pkg::Manifest {
                package: ereshkigal_core::pkg::PackageMeta {
                    name: name.clone(),
                    version: "0.1.0".into(),
                    license: "MIT".into(),
                    description: "Ereshkigal package".into(),
                },
                ..Default::default()
            };
            man.save(Path::new(&name).join("Ereshkigal.toml"))?;
            println!("created {name}/Ereshkigal.toml");
        }
        Cmd::Add { name, path } => {
            let mut man = load_manifest()?;
            man.dependencies.insert(
                name.clone(),
                ereshkigal_core::pkg::DepSource::Path {
                    path: path
                        .unwrap_or_else(|| PathBuf::from("decrees/std"))
                        .display()
                        .to_string(),
                },
            );
            man.save("Ereshkigal.toml")?;
            println!("added {name} (test-on-install: run ereshkigal test)");
        }
        Cmd::Remove { name } => {
            let mut man = load_manifest()?;
            man.dependencies.remove(&name);
            man.save("Ereshkigal.toml")?;
        }
        Cmd::Fetch | Cmd::Verify => {
            let man = load_manifest()?;
            println!("packages={}", man.dependencies.len());
            for (k, v) in &man.dependencies {
                match v {
                    ereshkigal_core::pkg::DepSource::Path { path } => {
                        let fp = ereshkigal_core::pkg::fingerprint_path(path)?;
                        println!("{k}: path={path} sha256={fp}");
                    }
                    other => println!("{k}: {other:?} (remote: not fetched)"),
                }
            }
        }
        Cmd::Publish { dry_run } => {
            println!("publish dry_run={dry_run} (HF_TOKEN required for real upload)");
        }
    }
    Ok(())
}

fn load_manifest() -> Result<ereshkigal_core::pkg::Manifest> {
    ereshkigal_core::pkg::Manifest::load("Ereshkigal.toml").map_err(Into::into)
}

fn tokenizer_for_gguf(gguf: &Path) -> (String, String) {
    let name = gguf
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
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

pub(crate) fn load_runtime(gguf: Option<PathBuf>) -> Result<Runtime> {
    let gguf = gguf
        .or_else(|| std::env::var_os("ERESHKIGAL_GGUF").map(PathBuf::from))
        .ok_or_else(|| anyhow::anyhow!("--gguf or ERESHKIGAL_GGUF required"))?;
    let (tok_src, tok_rev) = tokenizer_for_gguf(&gguf);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let n_gpu_layers = std::env::var("N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let (engine, tokenizer) = EngineOwned::load(EngineConfig {
        gguf,
        tokenizer_source: tok_src,
        tokenizer_revision: tok_rev,
        max_prompt_tokens: 4096,
        threads,
        n_gpu_layers,
        n_seq_max: 8,
        embeddings: false,
        adapter: None,
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Runtime::new(Scorer::new(engine, tokenizer)))
}

fn run_test(lib: &Path, gguf: Option<PathBuf>, fit: bool, split: &str, locked: bool) -> Result<()> {
    let library = Library::load(lib)?;
    if locked {
        if Path::new("ereshkigal.lock").is_file() {
            let lf = Lockfile::load("ereshkigal.lock")?;
            if lf.recipe != library.recipe && !lf.recipe.is_empty() {
                bail!("lockfile recipe drift");
            }
        }
    }
    if gguf.is_none() {
        // Offline: report gold coverage only. Pass --gguf to score (do not inherit
        // ERESHKIGAL_GGUF — CI smoke GGUF is 0.6B while the default tokenizer is 4B).
        let mut n = 0usize;
        for d in library.decrees.values() {
            n += d
                .tests
                .iter()
                .filter(|t| split == "all" || t.split.as_deref().unwrap_or("dev") == split)
                .count();
        }
        println!("offline tests matching split={split}: {n} (provide --gguf to score)");
        return Ok(());
    }
    let mut rt = load_runtime(gguf)?;
    let summary = score_library_tests(&mut rt, &library, split)?;
    if fit {
        // Re-score logits for temperature fit (same rows as summary).
        let mut rows_logits: Vec<(Vec<f64>, usize)> = Vec::new();
        let mut groups = Vec::new();
        for d in library.decrees.values() {
            for t in &d.tests {
                let tsplit = t.split.as_deref().unwrap_or("dev");
                if split != "all" && tsplit != split {
                    continue;
                }
                let decided = rt.decide(&library, &d.name, &t.state)?;
                let gi = d.option_index(&t.expect).unwrap_or(0);
                rows_logits.push((decided.probabilities.clone(), gi));
                groups.push(t.group.clone().unwrap_or_else(|| d.name.clone()));
            }
        }
        if !rows_logits.is_empty() {
            let t = fit_temperature(&rows_logits)?;
            let grouped = fit_temperature_oof_grouped(&rows_logits, Some(&groups), 5)?;
            println!(
                "fit T={:.3} oof_ece={:.4} meanT={:.3}",
                t, grouped.ece_calibrated_oof, grouped.mean_temperature
            );
        }
    }
    if let Some(n) = summary.get("n").and_then(|v| v.as_u64()) {
        if n == 0 {
            println!("no tests for split={split}");
            return Ok(());
        }
    }
    println!(
        "n={} BA={:.3} ECE={:.3} group={:.3} sel@80={:.3} AURC={:.3}",
        summary["n"],
        summary["BA"].as_f64().unwrap_or(0.0),
        summary["ECE"].as_f64().unwrap_or(0.0),
        summary["group"].as_f64().unwrap_or(0.0),
        summary["sel80"].as_f64().unwrap_or(0.0),
        summary["AURC"].as_f64().unwrap_or(0.0),
    );
    Ok(())
}

pub(crate) fn score_library_tests(rt: &mut Runtime, library: &Library, split: &str) -> Result<Value> {
    let mut pred = Vec::new();
    let mut gold = Vec::new();
    let mut groups = Vec::new();
    let mut margins = Vec::new();
    let mut correct = Vec::new();
    let mut rows_logits: Vec<(Vec<f64>, usize)> = Vec::new();
    for d in library.decrees.values() {
        for t in &d.tests {
            let tsplit = t.split.as_deref().unwrap_or("dev");
            if split != "all" && tsplit != split {
                continue;
            }
            let decided = rt.decide(library, &d.name, &t.state)?;
            let gi = d.option_index(&t.expect).unwrap_or(0);
            let pi = decided.choice_index.unwrap_or(0);
            pred.push(pi);
            gold.push(gi);
            groups.push(t.group.clone().unwrap_or_else(|| d.name.clone()));
            margins.push(decided.margin);
            correct.push(pi == gi);
            rows_logits.push((decided.probabilities.clone(), gi));
        }
    }
    if pred.is_empty() {
        return Ok(json!({"n": 0, "note": format!("no tests for split={split}")}));
    }
    let nclass = pred.iter().chain(gold.iter()).copied().max().unwrap_or(0) + 1;
    let ba = balanced_accuracy(&pred, &gold, nclass);
    let gc = group_consistency(&pred, &gold, &groups);
    let sel = selective_accuracy(&margins, &correct, 0.8);
    let a = aurc(&margins, &correct);
    let e = ece(&rows_logits, 1.0, 10)?;
    let n_ok = correct.iter().filter(|c| **c).count();
    Ok(json!({
        "n": pred.len(),
        "correct": n_ok,
        "BA": ba,
        "ECE": e,
        "group": gc,
        "sel80": sel,
        "AURC": a,
        "split": split,
    }))
}

fn diagnostics_for_text(uri: &str, text: &str) -> Vec<lsp_types::Diagnostic> {
    use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
    let range_all = |msg: String| Diagnostic {
        range: Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: text.lines().count().saturating_sub(1).max(0) as u32,
                character: 0,
            },
        },
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("ereshkigal".into()),
        message: msg,
        related_information: None,
        tags: None,
        data: None,
    };
    let path_hint = uri.rsplit('/').next().unwrap_or(uri);
    if path_hint.ends_with(".esk") || uri.ends_with(".esk") {
        match parse_library(text) {
            Ok(lib) => match ereshkigal_core::lint::lint_library(&lib) {
                Ok(()) => Vec::new(),
                Err(e) => vec![range_all(e.to_string())],
            },
            Err(e) => vec![range_all(e.to_string())],
        }
    } else if path_hint.ends_with(".toml") || uri.ends_with(".toml") {
        // Single-file TOML: write to a temp name via parse through Library::load needs path.
        // Use toml → LibraryFile via load of a temp string path by merging through parse.
        match Library::load_from_toml_str(text) {
            Ok(lib) => match ereshkigal_core::lint::lint_library(&lib) {
                Ok(()) => Vec::new(),
                Err(e) => vec![range_all(e.to_string())],
            },
            Err(e) => vec![range_all(e.to_string())],
        }
    } else {
        Vec::new()
    }
}

const KEYWORDS: &[&str] = &[
    "recipe", "decree", "program", "let", "match", "abstain", "test", "filter",
    "sort", "pairwise", "return", "escalate", "group", "top",
];

fn completion_items(text: &str) -> Vec<lsp_types::CompletionItem> {
    use lsp_types::{CompletionItem, CompletionItemKind};
    let mut items: Vec<CompletionItem> = KEYWORDS
        .iter()
        .map(|k| CompletionItem {
            label: (*k).into(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some("ereshkigal".into()),
            ..Default::default()
        })
        .collect();
    if let Ok(lib) = parse_library(text) {
        for name in lib.decrees.keys() {
            items.push(CompletionItem {
                label: name.clone(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("decree".into()),
                ..Default::default()
            });
        }
        for name in lib.programs.keys() {
            items.push(CompletionItem {
                label: name.clone(),
                kind: Some(CompletionItemKind::MODULE),
                detail: Some("program".into()),
                ..Default::default()
            });
        }
    }
    items
}

fn hover_markdown(text: &str, word: &str) -> Option<String> {
    let lib = parse_library(text).ok()?;
    if let Ok(d) = lib.decree(word) {
        let opts: Vec<String> = d.options.iter().map(|o| format!("- `{}`: {}", o.id, o.description)).collect();
        return Some(format!("**decree {}**\n\n{}\n\n{}", d.name, d.question, opts.join("\n")));
    }
    if lib.program(word).is_ok() {
        return Some(format!("**program {word}**"));
    }
    if KEYWORDS.contains(&word) {
        return Some(format!("`{word}` (Ereshkigal keyword)"));
    }
    None
}

fn word_at(text: &str, line: u32, character: u32) -> String {
    let Some(row) = text.lines().nth(line as usize) else {
        return String::new();
    };
    let chars: Vec<char> = row.chars().collect();
    let i = (character as usize).min(chars.len());
    let mut a = i;
    let mut b = i;
    while a > 0 && (chars[a - 1].is_ascii_alphanumeric() || chars[a - 1] == '_' || chars[a - 1] == '-') {
        a -= 1;
    }
    while b < chars.len() && (chars[b].is_ascii_alphanumeric() || chars[b] == '_' || chars[b] == '-') {
        b += 1;
    }
    chars[a..b].iter().collect()
}

fn run_lsp() -> Result<()> {
    use lsp_server::{Connection, Message, Notification, Response};
    use lsp_types::{
        notification::{DidChangeTextDocument, DidOpenTextDocument, Notification as _},
        request::{Completion, HoverRequest, Request as _, Shutdown},
        CompletionOptions, Hover, HoverContents, HoverProviderCapability, InitializeParams,
        MarkupContent, MarkupKind, PublishDiagnosticsParams, ServerCapabilities,
        TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
    };
    use std::collections::HashMap;
    let (connection, io_threads) = Connection::stdio();
    let caps = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![" ".into()]),
            ..Default::default()
        }),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        ..Default::default()
    };
    let init = connection.initialize(serde_json::to_value(caps)?)?;
    let _: InitializeParams = serde_json::from_value(init).unwrap_or_default();
    let mut docs: HashMap<String, String> = HashMap::new();

    let publish = |conn: &Connection, uri: Uri, text: &str| -> Result<()> {
        let diags = diagnostics_for_text(uri.as_str(), text);
        let params = PublishDiagnosticsParams {
            uri,
            diagnostics: diags,
            version: None,
        };
        let not = Notification::new(
            "textDocument/publishDiagnostics".into(),
            params,
        );
        conn.sender
            .send(Message::Notification(not))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };

    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    break;
                }
                if req.method == Shutdown::METHOD {
                    let resp = Response {
                        id: req.id,
                        result: Some(json!(null)),
                        error: None,
                    };
                    connection.sender.send(Message::Response(resp))?;
                    continue;
                }
                let result = if req.method == Completion::METHOD {
                    let params: lsp_types::CompletionParams =
                        serde_json::from_value(req.params.clone()).unwrap_or_else(|_| {
                            serde_json::from_value(json!({
                                "textDocument": {"uri": "file:///x.esk"},
                                "position": {"line": 0, "character": 0}
                            }))
                            .unwrap()
                        });
                    let uri = params.text_document_position.text_document.uri.to_string();
                    let text = docs.get(&uri).cloned().unwrap_or_default();
                    json!(completion_items(&text))
                } else if req.method == HoverRequest::METHOD {
                    let params: lsp_types::HoverParams =
                        serde_json::from_value(req.params.clone()).unwrap_or_else(|_| {
                            serde_json::from_value(json!({
                                "textDocument": {"uri": "file:///x.esk"},
                                "position": {"line": 0, "character": 0}
                            }))
                            .unwrap()
                        });
                    let uri = params.text_document_position_params.text_document.uri.to_string();
                    let pos = params.text_document_position_params.position;
                    let text = docs.get(&uri).cloned().unwrap_or_default();
                    let word = word_at(&text, pos.line, pos.character);
                    match hover_markdown(&text, &word) {
                        Some(md) => serde_json::to_value(Hover {
                            contents: HoverContents::Markup(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: md,
                            }),
                            range: None,
                        })
                        .unwrap_or(json!(null)),
                        None => json!(null),
                    }
                } else {
                    json!(null)
                };
                let resp = Response {
                    id: req.id,
                    result: Some(result),
                    error: None,
                };
                connection.sender.send(Message::Response(resp))?;
            }
            Message::Notification(not) => {
                if not.method == DidOpenTextDocument::METHOD {
                    if let Ok(params) =
                        serde_json::from_value::<lsp_types::DidOpenTextDocumentParams>(not.params)
                    {
                        let uri = params.text_document.uri.to_string();
                        docs.insert(uri, params.text_document.text.clone());
                        publish(
                            &connection,
                            params.text_document.uri,
                            &params.text_document.text,
                        )?;
                    }
                } else if not.method == DidChangeTextDocument::METHOD {
                    if let Ok(params) =
                        serde_json::from_value::<lsp_types::DidChangeTextDocumentParams>(not.params)
                    {
                        if let Some(change) = params.content_changes.last() {
                            docs.insert(
                                params.text_document.uri.to_string(),
                                change.text.clone(),
                            );
                            publish(
                                &connection,
                                params.text_document.uri,
                                &change.text,
                            )?;
                        }
                    }
                }
            }
            Message::Response(_) => {}
        }
    }
    io_threads.join().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
