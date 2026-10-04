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
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

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
            let mut lf = Lockfile {
                recipe: library.recipe.clone(),
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
            let _ = (stdio, http, lib, gguf);
            serve_stdio()?;
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
                println!("no variants on {decree}");
            } else {
                println!("{} variants; pick on dev with `ereshkigal test --fit`", d.variants.len());
            }
        }
        Cmd::Distill { corpus, out, recipe } => {
            println!(
                "distill corpus={} out={} recipe={} (train logistic probe; see training/README.md)",
                corpus.display(),
                out.display(),
                recipe
            );
            let mut probe = ereshkigal_core::probe::LogisticProbe::zeros(8, 3, &recipe, 1e-3);
            let xs = vec![vec![1.0; 8], vec![0.0; 8]];
            let ys = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
            ereshkigal_core::probe::train_epoch(&mut probe, &xs, &ys, 0.1)?;
            fs::write(out, serde_json::to_string_pretty(&probe)?)?;
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
                println!("{k}: {v:?}");
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

fn load_runtime(gguf: Option<PathBuf>) -> Result<Runtime> {
    let gguf = gguf
        .or_else(|| std::env::var_os("ERESHKIGAL_GGUF").map(PathBuf::from))
        .ok_or_else(|| anyhow::anyhow!("--gguf or ERESHKIGAL_GGUF required"))?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let (engine, tokenizer) = EngineOwned::load(EngineConfig {
        gguf,
        tokenizer_source: "Qwen/Qwen3.5-4B".into(),
        tokenizer_revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
        max_prompt_tokens: 4096,
        threads,
        n_gpu_layers: 0,
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
                .filter(|t| t.split.as_deref().unwrap_or("dev") == split || split == "all")
                .count();
        }
        println!("offline tests matching split={split}: {n} (provide --gguf to score)");
        return Ok(());
    }
    let mut rt = load_runtime(gguf)?;
    let mut pred = Vec::new();
    let mut gold = Vec::new();
    let mut groups = Vec::new();
    let mut margins = Vec::new();
    let mut correct = Vec::new();
    let mut rows_logits: Vec<(Vec<f64>, usize)> = Vec::new();
    for d in library.decrees.values() {
        for t in &d.tests {
            if t.split.as_deref().unwrap_or("dev") != split && split != "all" && split != "dev" {
                continue;
            }
            let decided = rt.decide(&library, &d.name, &t.state)?;
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
        println!("no tests for split={split}");
        return Ok(());
    }
    let nclass = pred.iter().chain(gold.iter()).copied().max().unwrap_or(0) + 1;
    let ba = balanced_accuracy(&pred, &gold, nclass);
    let gc = group_consistency(&pred, &gold, &groups);
    let sel = selective_accuracy(&margins, &correct, 0.8);
    let a = aurc(&margins, &correct);
    if fit {
        let t = fit_temperature(&rows_logits)?;
        let grouped = fit_temperature_oof_grouped(&rows_logits, Some(&groups), 5)?;
        println!(
            "fit T={:.3} oof_ece={:.4} meanT={:.3}",
            t, grouped.ece_calibrated_oof, grouped.mean_temperature
        );
    }
    let e = ece(&rows_logits, 1.0, 10)?;
    println!(
        "n={} BA={:.3} ECE={:.3} group={:.3} sel@80={:.3} AURC={:.3}",
        pred.len(),
        ba,
        e,
        gc,
        sel,
        a
    );
    Ok(())
}

fn serve_stdio() -> Result<()> {
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
        let result = match method {
            "lint" => json!({"ok": true}),
            "stats" => json!({"backend": "llamacpp"}),
            "decide" | "run" | "test" => json!({"error": "load library+gguf then retry"}),
            _ => json!({"error": format!("unknown method {method}")}),
        };
        let resp = json!({"jsonrpc": "2.0", "id": id, "result": result});
        writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        stdout.flush()?;
    }
    Ok(())
}

fn run_lsp() -> Result<()> {
    use lsp_server::{Connection, Message, Response};
    use lsp_types::{InitializeParams, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind};
    let (connection, io_threads) = Connection::stdio();
    let caps = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        ..Default::default()
    };
    let init = connection.initialize(serde_json::to_value(caps)?)?;
    let _: InitializeParams = serde_json::from_value(init).unwrap_or(InitializeParams::default());
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    break;
                }
                let resp = Response {
                    id: req.id,
                    result: Some(json!(null)),
                    error: None,
                };
                connection.sender.send(Message::Response(resp))?;
            }
            Message::Notification(_) | Message::Response(_) => {}
        }
    }
    io_threads.join().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
