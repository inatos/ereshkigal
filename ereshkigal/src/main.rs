use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ereshkigal_core::{
    apply_pride, apply_temperature, cascade_select, cascade_select_conformal, cycle_options,
    decision_argmax, ece, fit_qhat, fit_temperature, fit_temperature_oof, permute_debias,
    pride_frac_count, pride_log_prior, softmax, top2_margin, DecisionRow, EngineConfig,
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

    /// Optional LoRA adapter GGUF (draft student). Env: ERESHKIGAL_ADAPTER.
    #[arg(long, env = "ERESHKIGAL_ADAPTER", global = true)]
    adapter: Option<PathBuf>,

    /// LoRA adapter scale (default 1.0). Env: ERESHKIGAL_ADAPTER_SCALE.
    #[arg(long, env = "ERESHKIGAL_ADAPTER_SCALE", default_value_t = 1.0, global = true)]
    adapter_scale: f32,

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

    /// GPU layers to offload (0 = CPU only). Use a large number to offload all.
    #[arg(long, default_value_t = 0, global = true)]
    n_gpu_layers: u32,

    /// Max llama.cpp sequences (parallel shared suffixes)
    #[arg(long, default_value_t = 32, global = true)]
    n_seq_max: u32,

    /// Prompt recipe (`direct-options-v1` or `state-outline-v1`)
    #[arg(long, default_value = "direct-options-v1", global = true)]
    prompt_version: String,

    /// Disable prompt_sha256 replay cache
    #[arg(long, global = true)]
    no_replay: bool,

    /// Disable token-prefix radix KV cache
    #[arg(long, global = true)]
    no_radix: bool,

    /// Disable parallel suffix decode in shared mode
    #[arg(long, global = true)]
    no_parallel_suffixes: bool,

    /// Optional temperature for calibrated_probabilities (argmax unchanged)
    #[arg(long, global = true)]
    temperature: Option<f64>,

    /// Letter-prior correction after scoring
    #[arg(long, value_enum, default_value_t = DebiasCli::None, global = true)]
    debias: DebiasCli,

    /// Fraction of rows used to estimate PriDe letter prior (cyclic perms)
    #[arg(long, default_value_t = 0.05, global = true)]
    pride_frac: f64,
}

#[derive(Debug, Clone, ValueEnum)]
enum DebiasCli {
    None,
    Pride,
    Permute,
}

#[derive(Debug, Clone, ValueEnum)]
enum CascadeRouting {
    Conformal,
    Margin,
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
        /// K-fold OOF ECE (0 = in-sample only)
        #[arg(long, default_value_t = 5)]
        oof_folds: usize,
    },
    /// Draft/verify letter-slot cascade (0.6B logits JSONL + optional 4B GGUF)
    Cascade {
        /// Gold or input decisions JSONL
        #[arg(long)]
        input: PathBuf,
        /// Draft predictions JSONL (option_logits)
        #[arg(long)]
        draft: PathBuf,
        /// Output JSONL
        #[arg(long)]
        output: PathBuf,
        /// Commit draft when top-1 − top-2 > tau (only with --routing margin)
        #[arg(long, default_value_t = 0.35)]
        tau: f64,
        /// Verify GGUF for deferred rows (required unless --draft-only or --verify-predictions)
        #[arg(long)]
        verify_gguf: Option<PathBuf>,
        /// Existing verify JSONL (same ids) — skip a second GGUF load
        #[arg(long)]
        verify_predictions: Option<PathBuf>,
        /// Skip verify (measure skip rate only; deferred rows keep draft)
        #[arg(long)]
        draft_only: bool,
        /// Default conformal: accept draft iff |C|=1. `margin` is the tau debug path.
        #[arg(long, value_enum, default_value_t = CascadeRouting::Conformal)]
        routing: CascadeRouting,
        /// Conformal miscoverage (split THR scores)
        #[arg(long, default_value_t = 0.1)]
        alpha: f64,
        /// Pre-fit q-hat (skips --gold)
        #[arg(long)]
        qhat: Option<f64>,
        /// Gold JSONL with integer `label` to fit q-hat
        #[arg(long)]
        gold: Option<PathBuf>,
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
            oof_folds,
        }) => run_calibrate(
            gold,
            predictions,
            report,
            calibrated_out,
            temperature,
            oof_folds,
        ),
        Some(Commands::Cascade {
            input,
            draft,
            output,
            tau,
            verify_gguf,
            verify_predictions,
            draft_only,
            routing,
            alpha,
            qhat,
            gold,
        }) => run_cascade(
            EngineConfig {
                gguf: verify_gguf
                    .clone()
                    .or(cli.gguf.clone())
                    .unwrap_or_else(|| PathBuf::from("")),
                tokenizer_source: cli.model.clone(),
                tokenizer_revision: cli.revision.clone(),
                max_prompt_tokens: cli.max_tokens,
                threads: cli.llama_threads.unwrap_or(4),
                n_gpu_layers: cli.n_gpu_layers,
                n_seq_max: cli.n_seq_max,
                embeddings: false,
                // Cascade verify path: never attach student LoRA.
                adapter: None,
                adapter_scale: 1.0,
            },
            cli.prompt_version.clone(),
            draft_only,
            verify_gguf.or(cli.gguf.clone()),
            verify_predictions,
            input,
            draft,
            output,
            tau,
            routing,
            alpha,
            qhat,
            gold,
        ),
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
        n_seq_max: cli.n_seq_max,
        embeddings: false,
        adapter: cli.adapter.clone(),
        adapter_scale: if cli.adapter_scale.is_finite() && cli.adapter_scale > 0.0 {
            cli.adapter_scale
        } else {
            1.0
        },
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut scorer = Scorer::new(engine, tokenizer);
    scorer.prompt_version = cli.prompt_version.clone();
    scorer.replay_enabled = !cli.no_replay;
    scorer.radix_enabled = !cli.no_radix;
    scorer.parallel_suffixes = !cli.no_parallel_suffixes;
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
    let mut scored: Vec<ScoreResult> = Vec::with_capacity(rows.len());
    match cli.mode {
        Mode::Direct => {
            for row in &rows {
                let mut result = scorer
                    .score_direct(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                apply_temp_fields(&mut result, temp)?;
                scored.push(result);
            }
        }
        Mode::Serial => {
            for row in &rows {
                let mut result = scorer
                    .score_serial(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                apply_temp_fields(&mut result, temp)?;
                scored.push(result);
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
                eprintln!(
                    "shared batch size={} total_s={:.3} prefill_s={:.3} suffix_s={:.3}",
                    timing.batch_size,
                    timing.total_seconds,
                    timing.prefill_seconds,
                    timing.suffix_forward_seconds
                );
                for mut result in results {
                    apply_temp_fields(&mut result, temp)?;
                    scored.push(result);
                }
                i = j;
            }
        }
    }
    apply_debias(&mut scorer, &rows, &mut scored, cli.debias, cli.pride_frac)?;
    for result in &scored {
        write_result(&mut out, result)?;
    }

    Ok(())
}

fn apply_debias(
    scorer: &mut Scorer,
    rows: &[DecisionRow],
    scored: &mut [ScoreResult],
    mode: DebiasCli,
    pride_frac: f64,
) -> Result<()> {
    match mode {
        DebiasCli::None => Ok(()),
        DebiasCli::Pride => {
            let n_fit = pride_frac_count(rows.len(), pride_frac).max(1);
            let mut letter_ps: Vec<Vec<f64>> = Vec::new();
            for row in rows.iter().take(n_fit) {
                let n = row.options.len().max(1);
                for k in 0..n {
                    if k == 0 {
                        if let Some(r) = scored.iter().find(|s| s.id == row.id) {
                            letter_ps.push(r.probabilities.clone());
                            continue;
                        }
                    }
                    let cyc = cycle_options(row, k);
                    let r = scorer
                        .score_direct(&cyc)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    letter_ps.push(r.probabilities);
                }
            }
            let prior = pride_log_prior(&letter_ps).map_err(|e| anyhow::anyhow!("{e}"))?;
            for r in scored.iter_mut() {
                r.probabilities =
                    apply_pride(&r.probabilities, &prior).map_err(|e| anyhow::anyhow!("{e}"))?;
                r.probability_status =
                    "PriDe letter-prior corrected (Zheng et al. ICLR 2024)".into();
            }
            eprintln!("PriDe prior={prior:?} fit_rows={n_fit}");
            Ok(())
        }
        DebiasCli::Permute => {
            for (row, r) in rows.iter().zip(scored.iter_mut()) {
                let n = row.options.len();
                let mut perms = Vec::with_capacity(n);
                perms.push(r.option_logits.clone());
                for k in 1..n {
                    let cyc = cycle_options(row, k);
                    let scored_k = scorer
                        .score_direct(&cyc)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    perms.push(scored_k.option_logits);
                }
                r.probabilities = permute_debias(&perms).map_err(|e| anyhow::anyhow!("{e}"))?;
                r.probability_status = "cyclic permutation-averaged option probabilities".into();
            }
            Ok(())
        }
    }
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
    oof_folds: usize,
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
    let oof = if oof_folds >= 2 {
        Some(fit_temperature_oof(&rows, oof_folds).map_err(|e| anyhow::anyhow!("{e}"))?)
    } else {
        None
    };

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
        "oof": oof,
        "note": "argmax is unchanged by temperature; only confidence is rescaled. oof is k-fold holdout ECE.",
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

fn run_cascade(
    cfg: EngineConfig,
    prompt_version: String,
    draft_only: bool,
    verify_gguf: Option<PathBuf>,
    verify_predictions: Option<PathBuf>,
    input: PathBuf,
    draft_path: PathBuf,
    output: PathBuf,
    tau: f64,
    routing: CascadeRouting,
    alpha: f64,
    qhat: Option<f64>,
    gold: Option<PathBuf>,
) -> Result<()> {
    let rows: Vec<DecisionRow> = read_jsonl(&input)?;
    let drafts: Vec<PredRow> = read_any_jsonl(&draft_path)?;
    let mut by_id = std::collections::HashMap::new();
    for d in &drafts {
        by_id.insert(d.id.clone(), d);
    }
    let verify_by_id = if let Some(path) = &verify_predictions {
        let v: Vec<PredRow> = read_any_jsonl(path)?;
        let mut m = std::collections::HashMap::new();
        for r in v {
            m.insert(r.id.clone(), r);
        }
        Some(m)
    } else {
        None
    };
    let mut scorer = if draft_only || verify_by_id.is_some() {
        None
    } else {
        let gguf = verify_gguf
            .filter(|p| p.as_os_str().len() > 0)
            .or_else(|| {
                if cfg.gguf.as_os_str().is_empty() {
                    None
                } else {
                    Some(cfg.gguf.clone())
                }
            })
            .ok_or_else(|| anyhow::anyhow!("cascade needs --verify-gguf, --gguf, or --verify-predictions"))?;
        let mut load = cfg;
        load.gguf = gguf;
        let (engine, tokenizer) = EngineOwned::load(load).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut s = Scorer::new(engine, tokenizer);
        s.prompt_version = prompt_version;
        Some(s)
    };
    let qhat_val = match routing {
        CascadeRouting::Margin => None,
        CascadeRouting::Conformal => {
            if let Some(q) = qhat {
                Some(q)
            } else {
                let gold_path = gold.ok_or_else(|| {
                    anyhow::anyhow!("conformal cascade needs --gold or --qhat")
                })?;
                let gold_rows: Vec<GoldRow> = read_any_jsonl(&gold_path)?;
                let mut scores = Vec::new();
                for g in &gold_rows {
                    let d = by_id
                        .get(&g.id)
                        .ok_or_else(|| anyhow::anyhow!("missing draft for gold {}", g.id))?;
                    let p = if d.probabilities.len() == d.option_logits.len()
                        && !d.probabilities.is_empty()
                    {
                        d.probabilities.clone()
                    } else {
                        softmax(&d.option_logits).map_err(|e| anyhow::anyhow!("{e}"))?
                    };
                    if g.label >= p.len() {
                        bail!("gold label OOB for {}", g.id);
                    }
                    scores.push(1.0 - p[g.label]);
                }
                let q = fit_qhat(&scores, alpha).map_err(|e| anyhow::anyhow!("{e}"))?;
                eprintln!("fitted conformal qhat={q:.6} alpha={alpha} n={}", scores.len());
                Some(q)
            }
        }
    };
    let mut out = File::create(&output)?;
    let mut n_draft = 0usize;
    let mut n_verify = 0usize;
    let mut n_forced = 0usize;
    for row in &rows {
        let draft = by_id
            .get(&row.id)
            .ok_or_else(|| anyhow::anyhow!("missing draft for {}", row.id))?;
        let need_verify = match routing {
            CascadeRouting::Margin => {
                top2_margin(&draft.option_logits).map_err(|e| anyhow::anyhow!("{e}"))? <= tau
            }
            CascadeRouting::Conformal => {
                let q = qhat_val.unwrap();
                let p = if draft.probabilities.is_empty() {
                    softmax(&draft.option_logits).map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    draft.probabilities.clone()
                };
                ereshkigal_core::option_set(&p, q).len() != 1
            }
        };
        let verify_logits: Option<Vec<f64>> = if !need_verify {
            None
        } else if let Some(map) = &verify_by_id {
            let v = map
                .get(&row.id)
                .ok_or_else(|| anyhow::anyhow!("missing verify prediction for {}", row.id))?;
            Some(v.option_logits.clone())
        } else if let Some(s) = scorer.as_mut() {
            let verified = s.score_direct(row).map_err(|e| anyhow::anyhow!("{e}"))?;
            Some(verified.option_logits)
        } else {
            None
        };
        let outcome = match routing {
            CascadeRouting::Margin => {
                if !need_verify {
                    n_draft += 1;
                    cascade_select(&draft.option_logits, None, tau)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else if let Some(v) = verify_logits.as_deref() {
                    n_verify += 1;
                    cascade_select(&draft.option_logits, Some(v), tau)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    n_forced += 1;
                    let mut o = cascade_select(&draft.option_logits, None, 0.0)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    o.source = "cascade-draft-forced";
                    o
                }
            }
            CascadeRouting::Conformal => {
                let q = qhat_val.unwrap();
                if !need_verify {
                    n_draft += 1;
                    cascade_select_conformal(&draft.option_logits, None, q)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else if let Some(v) = verify_logits.as_deref() {
                    n_verify += 1;
                    cascade_select_conformal(&draft.option_logits, Some(v), q)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    n_forced += 1;
                    let mut o = cascade_select(&draft.option_logits, None, 0.0)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    o.source = "cascade-draft-forced";
                    o
                }
            }
        };
        let v = json!({
            "id": row.id,
            "probabilities": outcome.probabilities,
            "option_logits": draft.option_logits,
            "cascade_source": outcome.source,
            "used_verify": outcome.used_verify,
            "draft_margin": outcome.draft_margin,
            "set_size": outcome.set_size,
            "tau": tau,
            "routing": format!("{routing:?}").to_lowercase(),
            "qhat": qhat_val,
            "alpha": alpha,
        });
        serde_json::to_writer(&mut out, &v)?;
        out.write_all(b"\n")?;
    }
    eprintln!(
        "cascade routing={routing:?} draft_commit={n_draft} verify={n_verify} draft_forced={n_forced} total={}",
        rows.len()
    );
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
