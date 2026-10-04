//! GGUF integration / parity tests. Requires `ERESHKIGAL_GGUF`.
//!
//! Set `ERESHKIGAL_SMOKE=1` to use the CI smoke fixture + tokenizer from
//! `manifests/ci-smoke.json` (Qwen3-0.6B). Default uses 4B expected_direct.

use ereshkigal_core::{
    decision_argmax, probs_within_tol, DecisionRow, EngineConfig, EngineOwned, Scorer,
};
use serde::Deserialize;
use serde_json::Value;
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
    let raw = PathBuf::from(std::env::var_os("ERESHKIGAL_GGUF")?);
    if raw.is_absolute() {
        Some(raw)
    } else {
        let from_root = repo_root().join(&raw);
        if from_root.is_file() {
            Some(from_root)
        } else {
            Some(raw)
        }
    }
}

fn smoke_mode() -> bool {
    matches!(
        std::env::var("ERESHKIGAL_SMOKE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
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

fn tokenizer_cfg() -> (String, String) {
    if smoke_mode() {
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(path("manifests/ci-smoke.json")).expect("ci-smoke"),
        )
        .expect("json");
        (
            v["tokenizer"]["source"].as_str().unwrap().into(),
            v["tokenizer"]["revision"].as_str().unwrap().into(),
        )
    } else {
        (
            "Qwen/Qwen3.5-4B".into(),
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
        )
    }
}

#[test]
fn gguf_direct_serial_shared_parity() {
    let Some(gguf) = gguf_path() else {
        eprintln!("skip: set ERESHKIGAL_GGUF");
        return;
    };
    let (tok_source, tok_rev) = tokenizer_cfg();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let (engine, tokenizer) = EngineOwned::load(EngineConfig {
        gguf,
        tokenizer_source: tok_source,
        tokenizer_revision: tok_rev,
        max_prompt_tokens: 4096,
        threads,
        n_gpu_layers: 0,
        n_seq_max: 8,
        embeddings: false,
        adapter: None,
    })
    .expect("load engine");
    let mut scorer = Scorer::new(engine, tokenizer);

    if smoke_mode() {
        // Lightweight: examples + shared_state; prompt hashes + serial≡shared.
        let rows = load_rows(&path("examples/decisions.jsonl"));
        let mut directs = Vec::new();
        for row in &rows {
            let got = scorer.score_direct(row).expect("direct");
            assert!(!got.prompt_sha256.is_empty());
            assert!(got.probabilities.iter().all(|p| p.is_finite()));
            assert!((got.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-5);
            directs.push(got);
        }
        // Persist expected_smoke on first run if missing (CI commits it).
        let smoke_exp = path("fixtures/expected_smoke.jsonl");
        if smoke_exp.is_file() {
            let expected = load_expected(&smoke_exp);
            assert_eq!(directs.len(), expected.len());
            for (got, exp) in directs.iter().zip(expected.iter()) {
                assert_eq!(got.prompt_sha256, exp.prompt_sha256);
                assert_eq!(got.option_ids, exp.option_ids);
                assert_eq!(
                    decision_argmax(got),
                    Some(
                        exp.probabilities
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap()
                            .0
                    )
                );
                assert!(probs_within_tol(
                    &got.probabilities,
                    &exp.probabilities,
                    PROB_TOL
                ));
            }
        }

        let shared_rows = load_rows(&path("fixtures/shared_state.jsonl"));
        let mut serial = Vec::new();
        for row in &shared_rows {
            serial.push(scorer.score_serial(row).expect("serial"));
        }
        let (shared, _) = scorer.score_shared(&shared_rows).expect("shared");
        for (a, b) in serial.iter().zip(shared.iter()) {
            assert_eq!(a.prompt_sha256, b.prompt_sha256);
            assert_eq!(decision_argmax(a), decision_argmax(b));
            assert!(probs_within_tol(
                &a.probabilities,
                &b.probabilities,
                PROB_TOL
            ));
        }
        assert_eq!(serial[0].cache_hit, Some(false));
        assert_eq!(serial[1].cache_hit, Some(true));
        return;
    }

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
        assert!(probs_within_tol(
            &a.probabilities,
            &b.probabilities,
            PROB_TOL
        ));
    }
    assert_eq!(serial[0].cache_hit, Some(false));
    assert_eq!(serial[1].cache_hit, Some(true));
}
