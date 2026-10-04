//! Per-workload temperature scaling (SemIf-style).
//!
//! Calibrated probs = softmax(option_logits / T). Argmax is invariant for T > 0.

use crate::error::{Error, Result};
use crate::softmax::{argmax, softmax};

/// Apply temperature scaling to option logits.
pub fn apply_temperature(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(Error::Validation(
            "temperature must be a finite positive number".into(),
        ));
    }
    let scaled: Vec<f64> = logits.iter().map(|v| v / temperature).collect();
    softmax(&scaled)
}

/// Fit a scalar temperature by minimizing mean NLL on labeled rows.
///
/// `rows` is `(option_logits, gold_index)` pairs. Gold index is into the option list.
pub fn fit_temperature(rows: &[(Vec<f64>, usize)]) -> Result<f64> {
    if rows.is_empty() {
        return Err(Error::Validation("need at least one labeled row".into()));
    }
    for (logits, gold) in rows {
        if *gold >= logits.len() || logits.len() < 2 {
            return Err(Error::Validation(
                "gold index out of range or too few options".into(),
            ));
        }
    }

    // Bracket search on log T in [log(0.05), log(10)], then refine.
    let mut best_t = 1.0;
    let mut best_nll = f64::INFINITY;
    for i in 0..=40 {
        let log_t = (0.05_f64).ln() + (i as f64) * ((10.0_f64).ln() - (0.05_f64).ln()) / 40.0;
        let t = log_t.exp();
        let nll = mean_nll(rows, t)?;
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
    }
    // Local refine around best.
    let mut t = best_t;
    let mut step = best_t * 0.15;
    for _ in 0..25 {
        let mut improved = false;
        for dir in [-1.0, 1.0] {
            let cand = (t + dir * step).clamp(0.05, 10.0);
            let nll = mean_nll(rows, cand)?;
            if nll < best_nll {
                best_nll = nll;
                t = cand;
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    Ok(t)
}

fn mean_nll(rows: &[(Vec<f64>, usize)], temperature: f64) -> Result<f64> {
    let mut total = 0.0;
    for (logits, gold) in rows {
        let probs = apply_temperature(logits, temperature)?;
        let p = probs[*gold].max(1e-12);
        total -= p.ln();
    }
    Ok(total / rows.len() as f64)
}

/// Expected calibration error (equal-width bins on max probability).
pub fn ece(rows: &[(Vec<f64>, usize)], temperature: f64, bins: usize) -> Result<f64> {
    let bins = bins.max(1);
    let mut bin_correct = vec![0.0; bins];
    let mut bin_conf = vec![0.0; bins];
    let mut bin_count = vec![0.0; bins];
    for (logits, gold) in rows {
        let probs = apply_temperature(logits, temperature)?;
        let pred = argmax(&probs).unwrap();
        let conf = probs[pred];
        let b = ((conf * bins as f64).floor() as usize).min(bins - 1);
        bin_count[b] += 1.0;
        bin_conf[b] += conf;
        if pred == *gold {
            bin_correct[b] += 1.0;
        }
    }
    let n = rows.len() as f64;
    let mut ece = 0.0;
    for b in 0..bins {
        if bin_count[b] == 0.0 {
            continue;
        }
        let acc = bin_correct[b] / bin_count[b];
        let avg_conf = bin_conf[b] / bin_count[b];
        ece += (bin_count[b] / n) * (acc - avg_conf).abs();
    }
    Ok(ece)
}

/// Balanced accuracy: mean per-class recall.
pub fn balanced_accuracy(pred: &[usize], gold: &[usize], n_classes: usize) -> f64 {
    assert_eq!(pred.len(), gold.len());
    if pred.is_empty() || n_classes == 0 {
        return 0.0;
    }
    let mut tp = vec![0.0; n_classes];
    let mut support = vec![0.0; n_classes];
    for (&p, &g) in pred.iter().zip(gold.iter()) {
        if g < n_classes {
            support[g] += 1.0;
            if p == g {
                tp[g] += 1.0;
            }
        }
    }
    let mut sum = 0.0;
    let mut used = 0.0;
    for c in 0..n_classes {
        if support[c] > 0.0 {
            sum += tp[c] / support[c];
            used += 1.0;
        }
    }
    if used == 0.0 {
        0.0
    } else {
        sum / used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_one_matches_softmax() {
        let logits = vec![1.0, 2.0, 0.5];
        let a = apply_temperature(&logits, 1.0).unwrap();
        let b = softmax(&logits).unwrap();
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    #[test]
    fn argmax_invariant_under_temperature() {
        let logits = vec![0.1, 3.0, 0.2];
        let p1 = apply_temperature(&logits, 1.0).unwrap();
        let p2 = apply_temperature(&logits, 2.5).unwrap();
        assert_eq!(argmax(&p1), argmax(&p2));
    }

    #[test]
    fn fit_raises_t_when_overconfident() {
        // Gold always class 0, but logit strongly prefers 0 with huge margin → overconfident.
        // With noisy gold (half wrong), fit should push T up.
        let mut rows = Vec::new();
        for i in 0..40 {
            let gold = if i % 2 == 0 { 0 } else { 1 };
            // Model always very confident on class 0
            rows.push((vec![5.0, 0.0], gold));
        }
        let t = fit_temperature(&rows).unwrap();
        assert!(t > 1.2, "expected T>1.2 for overconfident wrong-half case, got {t}");
    }

    #[test]
    fn balanced_accuracy_perfect() {
        let pred = vec![0, 1, 0, 1];
        let gold = vec![0, 1, 0, 1];
        assert!((balanced_accuracy(&pred, &gold, 2) - 1.0).abs() < 1e-12);
    }
}
