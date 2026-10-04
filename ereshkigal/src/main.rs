use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use ereshkigal_core::{
    DecisionRow, EngineConfig, EngineOwned, Scorer, ScoreResult,
};
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
struct Args {
    /// Scoring mode
    #[arg(long, value_enum, default_value_t = Mode::Direct)]
    mode: Mode,

    /// Hugging Face model id for the reference tokenizer
    #[arg(long, default_value = "Qwen/Qwen3.5-4B")]
    model: String,

    /// Pinned tokenizer revision (40-char commit)
    #[arg(long, default_value = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")]
    revision: String,

    /// Path to a local GGUF checkpoint
    #[arg(long, env = "ERESHKIGAL_GGUF")]
    gguf: PathBuf,

    /// JSONL decisions input
    #[arg(long)]
    input: PathBuf,

    /// JSONL results output (default: stdout)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Max prompt tokens (no truncation)
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// CPU threads for llama.cpp
    #[arg(long = "llama-threads")]
    llama_threads: Option<i32>,

    /// GPU layers to offload (0 = CPU only)
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let threads = args
        .llama_threads
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4));

    let (engine, tokenizer) = EngineOwned::load(EngineConfig {
        gguf: args.gguf.clone(),
        tokenizer_source: args.model.clone(),
        tokenizer_revision: args.revision.clone(),
        max_prompt_tokens: args.max_tokens,
        threads,
        n_gpu_layers: args.n_gpu_layers,
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut scorer = Scorer::new(engine, tokenizer);
    let rows = read_jsonl(&args.input)?;
    if rows.is_empty() {
        bail!("input contained no decisions");
    }

    let mut out: Box<dyn Write> = if let Some(path) = &args.output {
        Box::new(File::create(path).with_context(|| format!("create {}", path.display()))?)
    } else {
        Box::new(std::io::stdout())
    };

    match args.mode {
        Mode::Direct => {
            for row in &rows {
                let result = scorer
                    .score_direct(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                write_result(&mut out, &result)?;
            }
        }
        Mode::Serial => {
            for row in &rows {
                let result = scorer
                    .score_serial(row)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                write_result(&mut out, &result)?;
            }
        }
        Mode::Shared => {
            // Group consecutive rows that share an identical state.
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
                for result in results {
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

fn read_jsonl(path: &PathBuf) -> Result<Vec<DecisionRow>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: DecisionRow = serde_json::from_str(line)
            .with_context(|| format!("{}:{} invalid decision row", path.display(), lineno + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn write_result(out: &mut dyn Write, result: &ScoreResult) -> Result<()> {
    serde_json::to_writer(&mut *out, result)?;
    out.write_all(b"\n")?;
    Ok(())
}
