//! Letter-slot cascade: draft model commits when margin is large, else verify.
//!
//! Leviathan-style speculative decoding adapted to option logits (no generated
//! answer tokens). Argmax of the committed distribution is the decision.

use crate::error::{Error, Result};
use crate::softmax::{argmax, softmax};
use ereshkigal_lang::conformal::option_set;

#[derive(Debug, Clone, PartialEq)]
pub struct CascadeOutcome {
    pub probabilities: Vec<f64>,
    pub used_verify: bool,
    pub draft_margin: f64,
    pub source: &'static str,
    pub set_size: usize,
}

/// Top-1 minus top-2 probability from logits.
pub fn top2_margin(logits: &[f64]) -> Result<f64> {
    let p = softmax(logits)?;
    let mut s = p.clone();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(s[0] - s[1])
}

/// If draft margin > `tau`, commit draft softmax; else use `verify_logits`.
pub fn cascade_select(
    draft_logits: &[f64],
    verify_logits: Option<&[f64]>,
    tau: f64,
) -> Result<CascadeOutcome> {
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(Error::Validation("cascade tau must be finite and >= 0".into()));
    }
    let margin = top2_margin(draft_logits)?;
    if margin > tau {
        return Ok(CascadeOutcome {
            probabilities: softmax(draft_logits)?,
            used_verify: false,
            draft_margin: margin,
            source: "cascade-draft",
            set_size: 1,
        });
    }
    let verify = verify_logits.ok_or_else(|| {
        Error::Validation("cascade requires verify logits when margin <= tau".into())
    })?;
    if verify.len() != draft_logits.len() {
        return Err(Error::Validation(
            "draft and verify option counts differ".into(),
        ));
    }
    Ok(CascadeOutcome {
        probabilities: softmax(verify)?,
        used_verify: true,
        draft_margin: margin,
        source: "cascade-verify",
        set_size: 0,
    })
}

/// Commit draft iff conformal set is a singleton (`|C|=1`); else verify.
/// `qhat` from [`crate::conformal::fit_qhat`] on `s_i = 1 - p_gold`.
pub fn cascade_select_conformal(
    draft_logits: &[f64],
    verify_logits: Option<&[f64]>,
    qhat: f64,
) -> Result<CascadeOutcome> {
    let margin = top2_margin(draft_logits)?;
    let probs = softmax(draft_logits)?;
    let set = option_set(&probs, qhat);
    if set.len() == 1 {
        return Ok(CascadeOutcome {
            probabilities: probs,
            used_verify: false,
            draft_margin: margin,
            source: "cascade-draft",
            set_size: 1,
        });
    }
    let verify = verify_logits.ok_or_else(|| {
        Error::Validation("cascade requires verify logits when conformal set is not singleton".into())
    })?;
    if verify.len() != draft_logits.len() {
        return Err(Error::Validation(
            "draft and verify option counts differ".into(),
        ));
    }
    Ok(CascadeOutcome {
        probabilities: softmax(verify)?,
        used_verify: true,
        draft_margin: margin,
        source: "cascade-verify",
        set_size: set.len(),
    })
}

/// Whether a draft-only commit would keep the same argmax as verify (diagnostic).
pub fn draft_agrees_argmax(draft_logits: &[f64], verify_logits: &[f64]) -> Result<bool> {
    let d = softmax(draft_logits)?;
    let v = softmax(verify_logits)?;
    Ok(argmax(&d) == argmax(&v))
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeTreeOutcome {
    /// Longest prefix of draft tokens accepted by the verify pass.
    pub accepted_prefix: usize,
    pub used_verify: bool,
    pub source: &'static str,
}

/// Batched MoE-style speculative tree verify.
///
/// `draft_token_ids` is the K-token candidate branch from a resident draft model.
/// `verify_token_ids` is the MoE (or verify model) argmax sequence for the same
/// positions in one amortized sweep. Commits the longest matching prefix.
pub fn speculative_tree_verify(
    draft_token_ids: &[u32],
    verify_token_ids: &[u32],
) -> Result<SpeculativeTreeOutcome> {
    if draft_token_ids.is_empty() {
        return Err(Error::Validation(
            "speculative_tree_verify: empty draft branch".into(),
        ));
    }
    if verify_token_ids.len() != draft_token_ids.len() {
        return Err(Error::Validation(
            "speculative_tree_verify: draft and verify lengths differ".into(),
        ));
    }
    let mut accepted = 0usize;
    for (d, v) in draft_token_ids.iter().zip(verify_token_ids.iter()) {
        if d == v {
            accepted += 1;
        } else {
            break;
        }
    }
    Ok(SpeculativeTreeOutcome {
        accepted_prefix: accepted,
        used_verify: true,
        source: if accepted == draft_token_ids.len() {
            "speculative-tree-full"
        } else if accepted == 0 {
            "speculative-tree-reject"
        } else {
            "speculative-tree-prefix"
        },
    })
}

/// Per-position letter-logit cascade over a speculative tree (option space).
/// For each draft position, commit draft when margin > tau; else take verify.
pub fn speculative_tree_cascade(
    draft_logits_per_step: &[Vec<f64>],
    verify_logits_per_step: &[Vec<f64>],
    tau: f64,
) -> Result<(Vec<CascadeOutcome>, usize)> {
    if draft_logits_per_step.is_empty() {
        return Err(Error::Validation(
            "speculative_tree_cascade: empty draft".into(),
        ));
    }
    if verify_logits_per_step.len() != draft_logits_per_step.len() {
        return Err(Error::Validation(
            "speculative_tree_cascade: length mismatch".into(),
        ));
    }
    let mut outs = Vec::with_capacity(draft_logits_per_step.len());
    let mut accepted = 0usize;
    for (d, v) in draft_logits_per_step
        .iter()
        .zip(verify_logits_per_step.iter())
    {
        let out = cascade_select(d, Some(v), tau)?;
        let agree = draft_agrees_argmax(d, v)?;
        if agree || !out.used_verify {
            accepted += 1;
            outs.push(out);
        } else {
            outs.push(out);
            break;
        }
    }
    Ok((outs, accepted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_margin_skips_verify() {
        let draft = vec![8.0, 0.0, 0.1];
        let verify = vec![0.0, 8.0, 0.0]; // would flip if used
        let out = cascade_select(&draft, Some(&verify), 0.2).unwrap();
        assert!(!out.used_verify);
        assert_eq!(argmax(&out.probabilities), Some(0));
        assert!(out.draft_margin > 0.2);
    }

    #[test]
    fn low_margin_uses_verify() {
        let draft = vec![1.0, 0.9, 0.8];
        let verify = vec![0.0, 3.0, 0.1];
        let out = cascade_select(&draft, Some(&verify), 0.5).unwrap();
        assert!(out.used_verify);
        assert_eq!(argmax(&out.probabilities), Some(1));
    }

    #[test]
    fn conformal_singleton_skips_verify() {
        let draft = vec![8.0, 0.0, 0.1];
        let verify = vec![0.0, 8.0, 0.0];
        let q = ereshkigal_lang::fit_qhat(&[0.05, 0.08, 0.1], 0.1).unwrap();
        let out = cascade_select_conformal(&draft, Some(&verify), q).unwrap();
        assert!(!out.used_verify);
        assert_eq!(out.set_size, 1);
    }

    #[test]
    fn speculative_tree_accepts_prefix() {
        let draft = [1u32, 2, 3, 4, 5];
        let verify = [1u32, 2, 3, 9, 9];
        let out = speculative_tree_verify(&draft, &verify).unwrap();
        assert_eq!(out.accepted_prefix, 3);
        assert_eq!(out.source, "speculative-tree-prefix");
    }

    #[test]
    fn speculative_tree_full_accept() {
        let ids = [7u32, 8, 9];
        let out = speculative_tree_verify(&ids, &ids).unwrap();
        assert_eq!(out.accepted_prefix, 3);
        assert_eq!(out.source, "speculative-tree-full");
    }

    #[test]
    fn speculative_tree_cascade_stops_on_disagree() {
        let drafts = vec![vec![8.0, 0.0], vec![1.0, 0.9], vec![5.0, 0.1]];
        let verifies = vec![vec![8.0, 0.0], vec![0.0, 3.0], vec![5.0, 0.1]];
        let (outs, accepted) = speculative_tree_cascade(&drafts, &verifies, 0.5).unwrap();
        assert!(accepted >= 1);
        assert!(!outs.is_empty());
    }
}
