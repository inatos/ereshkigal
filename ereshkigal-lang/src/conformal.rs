//! Split conformal option sets (THR / 1 − p_y scores).

use crate::error::{Error, Result};

/// Fit q-hat on calibration scores `s_i = 1 - p_gold`.
pub fn fit_qhat(scores: &[f64], alpha: f64) -> Result<f64> {
    if scores.is_empty() {
        return Err(Error::Validation("conformal needs calibration scores".into()));
    }
    let alpha = alpha.clamp(1e-6, 0.99);
    let mut s = scores.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len() as f64;
    let q = ((1.0 - alpha) * (n + 1.0)).ceil() / (n + 1.0);
    let idx = ((q * n).floor() as usize).min(s.len() - 1);
    Ok(s[idx])
}

/// Include labels with `1 - p_i <= qhat`.
pub fn option_set(probs: &[f64], qhat: f64) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..probs.len())
        .filter(|&i| 1.0 - probs[i] <= qhat + 1e-12)
        .collect();
    if ids.is_empty() {
        if let Some(i) = crate::softmax::argmax(probs) {
            ids.push(i);
        }
    }
    ids
}

pub fn coverage(sets: &[Vec<usize>], gold: &[usize]) -> f64 {
    assert_eq!(sets.len(), gold.len());
    if sets.is_empty() {
        return 0.0;
    }
    let ok = sets
        .iter()
        .zip(gold)
        .filter(|(s, g)| s.contains(g))
        .count();
    ok as f64 / sets.len() as f64
}

pub fn mean_set_size(sets: &[Vec<usize>]) -> f64 {
    if sets.is_empty() {
        return 0.0;
    }
    sets.iter().map(|s| s.len() as f64).sum::<f64>() / sets.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_includes_confident_label() {
        let q = fit_qhat(&[0.05, 0.1, 0.08], 0.1).unwrap();
        let set = option_set(&[0.92, 0.05, 0.03], q);
        assert!(set.contains(&0));
    }
}
