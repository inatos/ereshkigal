//! Label-free letter-prior correction (content-free / permutation / PriDe).

use crate::error::{Error, Result};
use crate::softmax::softmax;
use crate::types::{DecisionRow, OptionSpec};

/// Subtract log prior (content-free calibration, Zhao et al. 2021).
pub fn apply_content_free(logits: &[f64], log_prior: &[f64]) -> Result<Vec<f64>> {
    if logits.len() != log_prior.len() {
        return Err(Error::Validation("prior length mismatch".into()));
    }
    Ok(logits
        .iter()
        .zip(log_prior)
        .map(|(l, p)| l - p)
        .collect())
}

/// Estimate log-prior from content-free (neutral evidence) option logits.
pub fn prior_from_content_free(logits: &[f64]) -> Result<Vec<f64>> {
    let p = softmax(logits)?;
    Ok(p.iter().map(|x| x.max(1e-12).ln()).collect())
}

/// Average probabilities across a cyclic permutation of options, mapped back.
pub fn permute_debias(logit_perms: &[Vec<f64>]) -> Result<Vec<f64>> {
    if logit_perms.is_empty() {
        return Err(Error::Validation("need at least one permutation".into()));
    }
    let n = logit_perms[0].len();
    let mut acc = vec![0.0; n];
    for (k, logits) in logit_perms.iter().enumerate() {
        if logits.len() != n {
            return Err(Error::Validation("ragged permutations".into()));
        }
        let p = softmax(logits)?;
        for i in 0..n {
            let orig = (i + (k % n)) % n;
            acc[orig] += p[i];
        }
    }
    let m = logit_perms.len() as f64;
    for x in &mut acc {
        *x /= m;
    }
    Ok(acc)
}

/// Mean letter probability → log prior (PriDe; Zheng et al. ICLR 2024).
/// `letter_probs` are softmax vectors in **letter order** (not remapped to content).
pub fn pride_log_prior(letter_probs: &[Vec<f64>]) -> Result<Vec<f64>> {
    if letter_probs.is_empty() {
        return Err(Error::Validation("PriDe prior needs at least one row".into()));
    }
    let n = letter_probs[0].len();
    let mut acc = vec![0.0; n];
    for p in letter_probs {
        if p.len() != n {
            return Err(Error::Validation("ragged PriDe prior rows".into()));
        }
        for (i, x) in p.iter().enumerate() {
            acc[i] += *x;
        }
    }
    let m = letter_probs.len() as f64;
    Ok(acc
        .iter()
        .map(|x| (x / m).max(1e-12).ln())
        .collect())
}

/// `p' ∝ p / π` then renormalize (PriDe apply step).
pub fn apply_pride(probs: &[f64], log_prior: &[f64]) -> Result<Vec<f64>> {
    if probs.len() != log_prior.len() {
        return Err(Error::Validation("PriDe length mismatch".into()));
    }
    let mut w: Vec<f64> = probs
        .iter()
        .zip(log_prior)
        .map(|(p, lp)| p / lp.exp().max(1e-12))
        .collect();
    let z: f64 = w.iter().sum();
    if !(z.is_finite() && z > 0.0) {
        return Err(Error::Validation("PriDe renormalize failed".into()));
    }
    for x in &mut w {
        *x /= z;
    }
    Ok(w)
}

/// Cycle option contents by `k` slots (letters stay A,B,C…; content rotates).
pub fn cycle_options(row: &DecisionRow, k: usize) -> DecisionRow {
    let n = row.options.len();
    if n == 0 {
        return row.clone();
    }
    let k = k % n;
    let mut options: Vec<OptionSpec> = Vec::with_capacity(n);
    for i in 0..n {
        options.push(row.options[(i + k) % n].clone());
    }
    DecisionRow {
        id: format!("{}#cyc{k}", row.id),
        state: row.state.clone(),
        question: row.question.clone(),
        options,
    }
}

pub fn pride_frac_count(n: usize, frac: f64) -> usize {
    let f = frac.clamp(0.02, 1.0);
    ((n as f64) * f).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DecisionRow, OptionSpec};
    use serde_json::json;

    #[test]
    fn permute_maps_cycled_letter_back_to_content() {
        // k=0: letter B wins (index 1). k=1 cycle: letter A holds original B and wins.
        let p = permute_debias(&[vec![0.0, 5.0], vec![5.0, 0.0]]).unwrap();
        assert!(p[1] > p[0], "{p:?}");
    }

    #[test]
    fn pride_downweights_letter_a_prior() {
        let prior = pride_log_prior(&[vec![0.8, 0.1, 0.1], vec![0.7, 0.2, 0.1]]).unwrap();
        let p = apply_pride(&[0.8, 0.1, 0.1], &prior).unwrap();
        assert!(p[0] < 0.8);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cycle_rotates_ids() {
        let row = DecisionRow {
            id: "r".into(),
            state: json!("s"),
            question: "q".into(),
            options: vec![
                OptionSpec {
                    id: "A".into(),
                    description: "aa".into(),
                },
                OptionSpec {
                    id: "B".into(),
                    description: "bb".into(),
                },
            ],
        };
        let c = cycle_options(&row, 1);
        assert_eq!(c.options[0].id, "B");
        assert_eq!(c.options[1].id, "A");
    }
}
