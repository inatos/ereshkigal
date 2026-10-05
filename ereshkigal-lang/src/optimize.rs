//! Wording optimizer: pick a decree variant on a **dev** split only.
//!
//! Never fit on authored144 `split=test`. Callers pass precomputed per-variant
//! NLL (from GGUF) or a token-overlap proxy for offline tests.

use crate::decree::Variant;
use crate::error::{Error, Result};

/// Select the variant with lowest mean NLL. Ties keep the first (identity) variant.
pub fn pick_lowest_nll(nll: &[(String, f64)]) -> Result<&str> {
    if nll.is_empty() {
        return Err(Error::Validation("no variants to pick".into()));
    }
    let mut best_i = 0usize;
    let mut best = f64::INFINITY;
    for (i, (_, v)) in nll.iter().enumerate() {
        if v.is_finite() && *v < best {
            best = *v;
            best_i = i;
        }
    }
    Ok(nll[best_i].0.as_str())
}

/// Offline proxy: overlap of variant question tokens vs gold state + expected id.
/// Higher is better; invert to NLL-shaped scores via `-ln(p)`.
pub fn overlap_nll(question: &str, state: &str, expect: &str) -> f64 {
    let q: Vec<&str> = tokenize(question);
    let hay = format!("{state} {expect}");
    let h: Vec<&str> = tokenize(&hay);
    if q.is_empty() {
        return 10.0;
    }
    let hits = q.iter().filter(|t| h.contains(t)).count() as f64;
    let p = (hits / q.len() as f64).clamp(1e-6, 1.0 - 1e-6);
    -p.ln()
}

fn tokenize(s: &str) -> Vec<&str> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 1)
        .collect()
}

/// Rank `variants` using overlap against one gold (state, expect) row.
pub fn rank_variants_overlap(variants: &[Variant], state: &str, expect: &str) -> Vec<(String, f64)> {
    variants
        .iter()
        .map(|v| {
            (
                v.id.clone(),
                overlap_nll(v.question.as_deref().unwrap_or(""), state, expect),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decree::Variant;

    #[test]
    fn picks_lowest_nll() {
        let rows = vec![
            ("identity".into(), 1.2),
            ("short".into(), 0.4),
            ("long".into(), 0.9),
        ];
        assert_eq!(pick_lowest_nll(&rows).unwrap(), "short");
    }

    #[test]
    fn overlap_prefers_matching_question() {
        let vars = vec![
            Variant {
                id: "a".into(),
                question: Some("Which dessert is this?".into()),
                options: None,
            },
            Variant {
                id: "b".into(),
                question: Some("Did the deploy succeed with evidence?".into()),
                options: None,
            },
        ];
        let ranked = rank_variants_overlap(&vars, "deploy succeeded with logs as evidence", "yes");
        let winner = pick_lowest_nll(&ranked).unwrap();
        assert_eq!(winner, "b");
    }
}
