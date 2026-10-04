//! GGUF integration / parity tests. Requires `ERESHKIGAL_GGUF`.
//!
//! One test owns the process-scoped llama backend (llama.cpp init is once-per-process).

use ereshkigal_core::{
    decision_argmax, probs_within_tol, DecisionRow, EngineConfig, EngineOwned, Scorer,
};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const PROB_TOL: f64 = 1e-4;

#[derive(Debug, Deserialize)]
struct ExpectedRow {
    id: String,
    option_ids: Vec<String>,
    probabilities: Vec<f64>,
    prompt_sha256: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace")
        .to_path_buf()
}

fn path(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

fn gguf_path() -> Option<PathBuf> {
    std::env::var_os("ERESHKIGAL_GGUF").map(PathBuf::from)
}

fn load_rows(path: &Path) -> Vec<DecisionRow> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    BufReader::new(file)
        .lines()
        .map(|l| l.expect("line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("row"))
        .collect()
}

fn load_expected(path: &Path) -> Vec<ExpectedRow> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    BufReader::new(file)
        .lines()
        .map(|l| l.expect("line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("expected"))
        .collect()
}

#[test]
fn gguf_direct_serial_shared_parity() {
    let Some(gguf) = gguf_path() else {
        eprintln!("skip: set ERESHKIGAL_GGUF");
        return;
    };
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
    })
    .expect("load engine");
    let mut scorer = Scorer::new(engine, tokenizer);

    let rows = load_rows(&path("examples/decisions.jsonl"));
    let expected = load_expected(&path("fixtures/expected_direct.jsonl"));
    assert_eq!(rows.len(), expected.len());
    for (row, exp) in rows.iter().zip(expected.iter()) {
        let got = scorer.score_direct(row).expect("score");
        assert_eq!(got.id, exp.id);
        assert_eq!(got.prompt_sha256, exp.prompt_sha256);
        assert_eq!(got.option_ids, exp.option_ids);
        let exp_argmax = exp
            .probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(decision_argmax(&got), Some(exp_argmax));
        assert!(
            probs_within_tol(&got.probabilities, &exp.probabilities, PROB_TOL),
            "prob mismatch for {}: {:?} vs {:?}",
            got.id,
            got.probabilities,
            exp.probabilities
        );
    }

    let shared_rows = load_rows(&path("fixtures/shared_state.jsonl"));
    let mut serial = Vec::new();
    for row in &shared_rows {
        serial.push(scorer.score_serial(row).expect("serial"));
    }
    let (shared, _) = scorer.score_shared(&shared_rows).expect("shared");
    assert_eq!(serial.len(), shared.len());
    for (a, b) in serial.iter().zip(shared.iter()) {
        assert_eq!(a.prompt_sha256, b.prompt_sha256);
        assert_eq!(decision_argmax(a), decision_argmax(b));
        assert!(
            probs_within_tol(&a.probabilities, &b.probabilities, PROB_TOL),
            "{} serial vs shared probs",
            a.id
        );
    }
    assert_eq!(serial[0].cache_hit, Some(false));
    assert_eq!(serial[1].cache_hit, Some(true));
}
