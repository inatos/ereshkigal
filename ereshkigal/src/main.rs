use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ereshkigal_core::{
    apply_temperature, decision_argmax, ece, fit_temperature, DecisionRow, EngineConfig,
    EngineOwned, ScoreResult, Scorer,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Direct,
    Serial,
    Shared,
}

#[derive(Parser, Debug)]
#[command(
    name = "semif-score",
    about = "Ereshkigal — typed option logits from open GGUF models (SemIf-compatible)",
    long_about = "Independent Rust port of the SemIf/OpenJev scoring interface.\nNot affiliated with Jev or TypeSafe."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Scoring mode (default command when no subcommand)
    #[arg(long, value_enum, default_value_t = Mode::Direct, global = true)]
    mode: Mode,

    /// Hugging Face model id for the reference tokenizer
    #[arg(long, default_value = "Qwen/Qwen3.5-4B", global = true)]
    model: String,

    /// Pinned tokenizer revision (40-char commit)
    #[arg(
        long,
        default_value = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        global = true
    )]
    revision: String,

    /// Path to a local GGUF checkpoint
    #[arg(long, env = "ERESHKIGAL_GGUF", global = true)]
    gguf: Option<PathBuf>,

    /// JSONL decisions input
    #[arg(long, global = true)]
    input: Option<PathBuf>,

    /// JSONL results output (default: stdout)
    #[arg(long, global = true)]
    output: Option<PathBuf>,

    /// Max prompt tokens (no truncation)
    #[arg(long, default_value_t = 4096, global = true)]
    max_tokens: usize,

    /// CPU threads for llama.cpp
    #[arg(long = "llama-threads", global = true)]
    llama_threads: Option<i32>,

    /// GPU layers to offload (0 = CPU only)
    #[arg(long, default_value_t = 0, global = true)]
    n_gpu_layers: u32,

    /// Optional temperature for calibrated_probabilities (argmax unchanged)
    #[arg(long, global = true)]
    temperature: Option<f64>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Score decisions (default if omitted)
    Score,
    /// Fit / apply temperature from labeled gold + prediction JSONL
    Calibrate {
        /// Gold JSONL with `id` and integer `label` (option index)
        #[arg(long)]
        gold: PathBuf,
        /// Predictions JSONL from semif-score (needs option_logits)
        #[arg(long)]
        predictions: PathBuf,
        /// Write calibration report JSON here
        #[arg(long)]
        report: PathBuf,
        /// Optional calibrated predictions JSONL
        #[arg(long)]
        calibrated_out: Option<PathBuf>,
        /// Skip fitting; apply this temperature
        #[arg(long)]
        temperature: Option<f64>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Calibrate {
            gold,
            predictions,
            report,
            calibrated_out,
            temperature,
        }) => run_calibrate(gold, predictions, report, calibrated_out, temperature),
        Some(Commands::Score) | None => run_score(cli),
    }
}

fn run_score(cli: Cli) -> Result<()> {
    let gguf = cli
        .gguf
        .ok_or_else(|| anyhow::anyhow!("--gguf or ERESHKIGAL_GGUF is required"))?;
    let input = cli
        .input
        .ok_or_else(|| anyhow::anyhow!("--input is required"))?;
    let threads = cli.llama_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
    });

    let (engine, tokenizer) = EngineOwned::load(EngineConfig {
        gguf,
        tokenizer_source: cli.model,
        tokenizer_revision: cli.revision,
        max_prompt_tokens: cli.max_tokens,
        threads,
        n_gpu_layers: cli.n_gpu_layers,
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut scorer = Scorer::new(engine, tokenizer);
    let rows = read_jsonl(&input)?;
    if rows.is_empty() {
        bail!("input contained no decisions");
    }

    let mut out: Box<dyn Write> = if let Some(path) = &cli.output {
        Box::new(File::create(path).with_context(|| format!("create {}", path.display()))?)
    } else {
        Box::new(std::io::stdout())
    };

    let temp = cli.temperature;
    match cli.mode {
        Mode::Direct => {
            for row in &rows {
                let mut result = scorer
                    .score_direct(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                apply_temp_fields(&mut result, temp)?;
                write_result(&mut out, &result)?;
            }
        }
        Mode::Serial => {
            for row in &rows {
                let mut result = scorer
                    .score_serial(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                apply_temp_fields(&mut result, temp)?;
                write_result(&mut out, &result)?;
            }
        }
        Mode::Shared => {
            let mut i = 0;
            while i < rows.len() {
                let state = &rows[i].state;
                let mut j = i + 1;
                while j < rows.len() && &rows[j].state == state {
                    j += 1;
                }
                let batch = &rows[i..j];
                let (results, timing) = scorer
                    .score_shared(batch)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                for mut result in results {
                    apply_temp_fields(&mut result, temp)?;
                    write_result(&mut out, &result)?;
                }
                eprintln!(
                    "shared batch size={} total_s={:.3} prefill_s={:.3} suffix_s={:.3}",
                    timing.batch_size,
                    timing.total_seconds,
                    timing.prefill_seconds,
                    timing.suffix_forward_seconds
                );
                i = j;
            }
        }
    }

    Ok(())
}

fn apply_temp_fields(result: &mut ScoreResult, temperature: Option<f64>) -> Result<()> {
    if let Some(t) = temperature {
        let calibrated = apply_temperature(&result.option_logits, t)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Argmax must match raw.
        let raw_i = decision_argmax(result);
        let cal_i = ereshkigal_core::argmax(&calibrated);
        if raw_i != cal_i {
            bail!("temperature changed argmax — refuse to emit");
        }
        result.calibrated_probabilities = Some(calibrated);
        result.temperature = Some(t);
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct GoldRow {
    id: String,
    label: usize,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct PredRow {
    id: String,
    option_logits: Vec<f64>,
    #[serde(default)]
    probabilities: Vec<f64>,
}

fn run_calibrate(
    gold_path: PathBuf,
    pred_path: PathBuf,
    report_path: PathBuf,
    calibrated_out: Option<PathBuf>,
    fixed_t: Option<f64>,
) -> Result<()> {
    let gold: Vec<GoldRow> = read_any_jsonl(&gold_path)?;
    let preds: Vec<PredRow> = read_any_jsonl(&pred_path)?;
    let mut by_id = std::collections::HashMap::new();
    for p in &preds {
        by_id.insert(p.id.as_str(), p);
    }
    let mut rows: Vec<(Vec<f64>, usize)> = Vec::new();
    let mut ordered_preds: Vec<&PredRow> = Vec::new();
    for g in &gold {
        let Some(p) = by_id.get(g.id.as_str()) else {
            bail!("missing prediction for gold id {}", g.id);
        };
        if g.label >= p.option_logits.len() {
            bail!("gold label OOB for {}", g.id);
        }
        rows.push((p.option_logits.clone(), g.label));
        ordered_preds.push(*p);
    }
    let t = if let Some(t) = fixed_t {
        t
    } else {
        fit_temperature(&rows).map_err(|e| anyhow::anyhow!("{e}"))?
    };
    let ece_raw = ece(&rows, 1.0, 10).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ece_cal = ece(&rows, t, 10).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut correct = 0usize;
    for (logits, gold) in &rows {
        let probs = apply_temperature(logits, 1.0).map_err(|e| anyhow::anyhow!("{e}"))?;
        if ereshkigal_core::argmax(&probs) == Some(*gold) {
            correct += 1;
        }
    }
    let report = json!({
        "rows": rows.len(),
        "temperature": t,
        "accuracy": correct as f64 / rows.len() as f64,
        "ece_t1": ece_raw,
        "ece_calibrated": ece_cal,
        "note": "argmax is unchanged by temperature; only confidence is rescaled",
    });
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", report_path.display());

    if let Some(out_path) = calibrated_out {
        let mut out = File::create(&out_path)?;
        for (p, (logits, _)) in ordered_preds.iter().zip(rows.iter()) {
            let calibrated = apply_temperature(logits, t).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut v: Value = serde_json::to_value(p)?;
            if let Some(obj) = v.as_object_mut() {
                obj.insert("calibrated_probabilities".into(), json!(calibrated));
                obj.insert("temperature".into(), json!(t));
            }
            serde_json::to_writer(&mut out, &v)?;
            out.write_all(b"\n")?;
        }
        eprintln!("wrote {}", out_path.display());
    }
    Ok(())
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<DecisionRow>> {
    read_any_jsonl(path)
}

fn read_any_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: T = serde_json::from_str(line)
            .with_context(|| format!("{}:{} invalid JSONL row", path.display(), lineno + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn write_result(out: &mut dyn Write, result: &ScoreResult) -> Result<()> {
    serde_json::to_writer(&mut *out, result)?;
    out.write_all(b"\n")?;
    Ok(())
}
